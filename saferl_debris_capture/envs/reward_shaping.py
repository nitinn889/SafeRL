"""
reward_shaping.py
=================
Publication-grade reward and cost functions for the debris capture task.

Reward architecture (9 components):
  R(s, a) = w₁ · r_approach       (dense: −‖EE − target‖)
          + w₂ · r_exp_approach    (exponential bonus near target)
          + w₃ · r_alignment       (quaternion orientation matching)
          + w₄ · r_velocity_match  (match EE velocity to debris drift)
          + w₅ · r_grasp           (sparse: successful capture)
          + w₆ · r_collision       (penalty: dangerous impact)
          + w₇ · r_gentle_contact  (bonus: controlled contact force)
          + w₈ · r_effort          (penalty: action magnitude)
          + w₉ · r_jerk            (penalty: action discontinuity)

Cost structure (for CMDP / probabilistic shield):
  C(s, a) = c_collision + c_proximity + c_joint_pos + c_joint_vel

Potential-based reward shaping (Ng et al. 1999) is also provided as
an auxiliary function to accelerate learning without altering the
optimal policy.

All methods accept batched tensors (num_envs, ...) for GPU parallelism.

References
----------
- Ng, A.Y. et al. (1999). "Policy Invariance Under Reward Transformations."
  ICML 1999.
- Achiam, J. et al. (2017). "Constrained Policy Optimisation." ICML 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .debris_capture_env import DebrisCaptureEnvCfg
    from .debris_dynamics import TumblingDebrisModel


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaper
# ─────────────────────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Multi-component reward and cost computation engine.

    This class encapsulates all reward/cost logic separate from the
    environment class for modularity and testability.

    Parameters
    ----------
    cfg : DebrisCaptureEnvCfg
        Environment configuration with reward weights and thresholds.
    """

    def __init__(self, cfg: "DebrisCaptureEnvCfg"):
        self.cfg = cfg
        self._prev_action: Optional[torch.Tensor] = None

    # ── Main entry points ────────────────────────────────────────────────────

    def compute(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        debris_model: "TumblingDebrisModel",
        prev_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-environment reward and safety cost.

        Parameters
        ----------
        obs          : (N, OBS_DIM)   current observation
        action       : (N, ACT_DIM)   action applied this step
        debris_model : TumblingDebrisModel
        prev_obs     : (N, OBS_DIM) | None  previous observation (for shaping)

        Returns
        -------
        reward : (N,)
        cost   : (N,)   multi-component safety cost (≥ 0)
        """
        cfg = self.cfg
        N = obs.shape[0]

        # ── Extract state variables ──────────────────────────────────────────
        debris_pos   = obs[:, 0:3]
        debris_quat  = obs[:, 3:7]
        debris_omega = obs[:, 7:10]
        ee_pos       = obs[:, 10:13]
        ee_quat      = obs[:, 13:17]
        joint_pos    = obs[:, 17:24]
        joint_vel    = obs[:, 24:31]
        contact_f    = obs[:, 31:34]

        dist      = torch.linalg.norm(ee_pos - debris_pos, dim=-1)
        force_mag = torch.linalg.norm(contact_f, dim=-1)

        # ── Reward components ────────────────────────────────────────────────

        # 1. Linear approach reward (dense)
        r_approach = -dist / (cfg.debris_initial_pos_max[0] + 1e-6)

        # 2. Exponential approach bonus (sharper near target)
        r_exp = torch.exp(-5.0 * dist)

        # 3. Orientation alignment (squared inner product ∈ [0, 1])
        r_align = self._orientation_reward(debris_quat, ee_quat)

        # 4. Velocity matching
        r_vel = self._velocity_match_reward(obs, prev_obs, debris_model)

        # 5. Grasp (sparse)
        close = dist < cfg.grasp_dist_threshold_m
        aligned = self._raw_alignment(debris_quat, ee_quat) > cfg.grasp_align_threshold
        gentle = (force_mag > cfg.gentle_force_range_N[0]) & (force_mag < cfg.gentle_force_range_N[1])
        r_grasp = (close & aligned & gentle).float()

        # 6. Collision penalty
        r_collision = (force_mag > cfg.collision_force_threshold_N).float()

        # 7. Gentle contact bonus
        r_gentle = (
            (force_mag > cfg.gentle_force_range_N[0])
            & (force_mag < cfg.gentle_force_range_N[1])
        ).float() * close.float()

        # 8. Effort penalty
        r_effort = torch.sum(action.pow(2), dim=-1)

        # 9. Jerk penalty
        if self._prev_action is not None:
            r_jerk = torch.sum((action - self._prev_action).pow(2), dim=-1)
        else:
            r_jerk = torch.zeros(N, device=obs.device)

        # ── Sum ──────────────────────────────────────────────────────────────
        reward = (
            cfg.w_approach        * r_approach
            + cfg.w_approach_exp  * r_exp
            + cfg.w_align         * r_align
            + cfg.w_velocity_match * r_vel
            + cfg.w_grasp         * r_grasp
            + cfg.w_collision     * r_collision
            + cfg.w_contact_gentle * r_gentle
            + cfg.w_effort        * r_effort
            + cfg.w_jerk          * r_jerk
        )

        # ── Potential-based shaping ──────────────────────────────────────────
        if prev_obs is not None:
            shaping = potential_based_shaping(prev_obs, obs, gamma=0.99)
            reward = reward + shaping

        # ── Safety cost ──────────────────────────────────────────────────────
        cost = self._compute_cost(obs, force_mag, dist, joint_pos, joint_vel)

        self._prev_action = action.clone()
        return reward, cost

    # ── Orientation reward ────────────────────────────────────────────────────

    def _orientation_reward(
        self,
        q_target: torch.Tensor,
        q_ee: torch.Tensor,
    ) -> torch.Tensor:
        """
        Orientation alignment reward.

        Uses the squared absolute quaternion inner product:
          r = |q_target · q_ee|²

        This metric is:
          - Invariant to quaternion double cover (q ≡ −q)
          - Smooth and differentiable
          - = 1 when perfectly aligned, = 0 when 180° misaligned

        For grasping, we want the EE gripper Z-axis to be anti-parallel
        to the debris surface normal at the capture point. The quaternion
        similarity serves as a proxy for this requirement.
        """
        return self._raw_alignment(q_target, q_ee).pow(2)

    @staticmethod
    def _raw_alignment(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Absolute quaternion inner product |q1 · q2| ∈ [0, 1]."""
        q1_norm = q1 / (torch.linalg.norm(q1, dim=-1, keepdim=True) + 1e-8)
        q2_norm = q2 / (torch.linalg.norm(q2, dim=-1, keepdim=True) + 1e-8)
        return torch.sum(q1_norm * q2_norm, dim=-1).abs()

    # ── Velocity matching reward ──────────────────────────────────────────────

    def _velocity_match_reward(
        self,
        obs: torch.Tensor,
        prev_obs: Optional[torch.Tensor],
        debris_model: "TumblingDebrisModel",
    ) -> torch.Tensor:
        """
        Reward for matching EE velocity to debris linear drift.

        In orbital mechanics, relative velocity at contact must be
        near-zero for safe berthing. This reward penalises ΔV at
        the moment of approach.

        Uses exponential kernel: r = exp(−k·‖v_EE − v_debris‖)
        """
        if prev_obs is None:
            return torch.zeros(obs.shape[0], device=obs.device)

        dt = self.cfg.sim_dt * self.cfg.decimation
        ee_vel = (obs[:, 10:13] - prev_obs[:, 10:13]) / (dt + 1e-6)
        debris_vel = debris_model.velocity
        vel_error = torch.linalg.norm(ee_vel - debris_vel, dim=-1)
        return torch.exp(-2.0 * vel_error)

    # ── Multi-component safety cost ───────────────────────────────────────────

    def _compute_cost(
        self,
        obs: torch.Tensor,
        force_mag: torch.Tensor,
        dist: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """
        CMDP safety cost (consumed by the probabilistic shield).

        Four binary cost components:
          c₁ = 1 if contact_force > collision_threshold
          c₂ = 1 if proximity < safe_approach_dist
          c₃ = 1 if any joint near position limit
          c₄ = 1 if any joint near velocity limit
        """
        cfg = self.cfg

        c_collision = (force_mag > cfg.collision_force_threshold_N).float()
        c_proximity = (dist < cfg.safe_approach_dist_m).float()

        # Joint position limit violation (obs already normalised to [−1, 1])
        c_joint_pos = (joint_pos.abs() > cfg.joint_pos_soft_limits_frac).any(dim=-1).float()

        # Joint velocity limit violation
        c_joint_vel = (joint_vel.abs() > cfg.joint_velocity_limit_frac).any(dim=-1).float()

        return c_collision + c_proximity + c_joint_pos + c_joint_vel


# ─────────────────────────────────────────────────────────────────────────────
# Potential-based reward shaping (Ng et al. 1999)
# ─────────────────────────────────────────────────────────────────────────────

def potential_based_shaping(
    obs_prev,
    obs_curr,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Policy-invariant reward shaping:

        F(s, s') = γ · Φ(s') − Φ(s)

    where Φ(s) is a composite potential:
        Φ(s) = −α · distance(EE, debris) + β · alignment(EE, debris)

    This provides denser gradient without changing the optimal policy
    (Ng et al., ICML 1999, Theorem 1).

    Handles both numpy and torch inputs for testing convenience.
    """
    import numpy as _np
    if isinstance(obs_prev, _np.ndarray):
        obs_prev = torch.from_numpy(obs_prev.astype(_np.float32))
    if isinstance(obs_curr, _np.ndarray):
        obs_curr = torch.from_numpy(obs_curr.astype(_np.float32))

    def phi(obs):
        debris_pos = obs[:, 0:3]
        ee_pos = obs[:, 10:13]
        dist = torch.linalg.norm(ee_pos - debris_pos, dim=-1)

        # Orientation potential (inner product of quaternions)
        debris_q = obs[:, 3:7]
        ee_q = obs[:, 13:17]
        debris_q = debris_q / (torch.linalg.norm(debris_q, dim=-1, keepdim=True) + 1e-8)
        ee_q = ee_q / (torch.linalg.norm(ee_q, dim=-1, keepdim=True) + 1e-8)
        align = torch.sum(debris_q * ee_q, dim=-1).abs()

        return -2.0 * dist + 0.5 * align

    return gamma * phi(obs_curr) - phi(obs_prev)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive reward weighting (experimental)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveRewardWeighter:
    """
    Automatically adjusts reward component weights during training
    based on success rate and collision rate statistics.

    This is an experimental feature for curriculum-aware reward scaling.

    Strategy:
      - If success_rate < 10%: increase w_approach, decrease w_align
      - If success_rate > 60%: increase w_align, increase w_grasp
      - If collision_rate > 20%: increase |w_collision|
      - If collision_rate < 5%: slightly decrease |w_collision| (allow exploration)

    Parameters
    ----------
    cfg : DebrisCaptureEnvCfg
        Base configuration (weights are modified in-place).
    update_interval : int
        Number of episodes between weight updates.
    """

    def __init__(self, cfg: "DebrisCaptureEnvCfg", update_interval: int = 100):
        self.cfg = cfg
        self.update_interval = update_interval
        self._base_weights = {
            "w_approach": cfg.w_approach,
            "w_align": cfg.w_align,
            "w_grasp": cfg.w_grasp,
            "w_collision": cfg.w_collision,
        }
        self._episode_count = 0
        self._success_buffer = []
        self._collision_buffer = []

    def log_episode(self, success: bool, collision: bool):
        """Record episode outcome."""
        self._success_buffer.append(float(success))
        self._collision_buffer.append(float(collision))
        self._episode_count += 1

        if self._episode_count % self.update_interval == 0:
            self._update_weights()

    def _update_weights(self):
        """Adjust weights based on recent performance."""
        if len(self._success_buffer) < self.update_interval:
            return

        recent = self.update_interval
        success_rate = sum(self._success_buffer[-recent:]) / recent
        collision_rate = sum(self._collision_buffer[-recent:]) / recent

        base = self._base_weights

        # Approach emphasis when struggling to reach target
        if success_rate < 0.10:
            self.cfg.w_approach = base["w_approach"] * 3.0
            self.cfg.w_align = base["w_align"] * 0.5
        elif success_rate < 0.40:
            self.cfg.w_approach = base["w_approach"] * 2.0
            self.cfg.w_align = base["w_align"] * 0.8
        else:
            self.cfg.w_approach = base["w_approach"]
            self.cfg.w_align = base["w_align"] * 1.5
            self.cfg.w_grasp = base["w_grasp"] * 1.5

        # Safety emphasis when collisions are frequent
        if collision_rate > 0.20:
            self.cfg.w_collision = base["w_collision"] * 2.0
        elif collision_rate > 0.10:
            self.cfg.w_collision = base["w_collision"] * 1.5
        elif collision_rate < 0.05:
            self.cfg.w_collision = base["w_collision"] * 0.8
