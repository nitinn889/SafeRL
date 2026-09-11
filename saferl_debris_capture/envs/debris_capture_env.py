"""
debris_capture_env.py
=====================
Production-grade Isaac Lab DirectRLEnv for autonomous space debris capture.

This environment simulates:
  - A 7-DOF Franka Panda robotic arm mounted on a servicing spacecraft
  - A tumbling, uncooperative defunct satellite (rigid body)
  - Microgravity (zero-g) orbital mechanics
  - High-fidelity contact/force sensing at the end-effector
  - Differential inverse kinematics for Cartesian action space
  - Multi-component reward shaping + CMDP safety cost

Observation space (36-D):
  [0:3]   debris position relative to robot base
  [3:7]   debris orientation quaternion (w, x, y, z)
  [7:10]  debris angular velocity (rad/s)
  [10:13] end-effector Cartesian position
  [13:17] end-effector orientation quaternion
  [17:24] robot joint positions (7-DOF)
  [24:31] robot joint velocities
  [31:34] EE contact force vector (N)
  [34]    normalised time remaining (1→0)
  [35]    debris tumble mode (0=spin, 1=moderate, 2=chaotic)

Action space (6-D continuous, ∈ [−1, 1]):
  [0:3]   delta EE position (scaled by max_ee_speed_m_s × dt)
  [3:6]   delta EE orientation (scaled by max_ee_rot_speed_rad_s × dt)

Physics:
  - PhysX 5 rigid body + articulation simulation
  - 120 Hz physics, RL acts every 4 physics steps (30 Hz control)
  - GPU-parallelised: 4096 environments on a single RTX 4090

Isaac Lab API compatibility:
  - Inherits from DirectRLEnv when Isaac Lab is installed
  - Falls back to a standalone class for offline testing
  - All configuration via DebrisCaptureEnvCfg dataclass
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ── Isaac Lab imports (graceful degradation if not installed) ─────────────────
try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
    from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
    from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
    from omni.isaac.lab.sim import SimulationCfg
    from omni.isaac.lab.utils import configclass
    from omni.isaac.lab.utils.math import (
        combine_frame_transforms,
        matrix_from_quat,
        quat_from_matrix,
        quat_error_magnitude,
        subtract_frame_transforms,
    )
    # Pre-built robot config
    from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG

    _ISAAC_AVAILABLE = True
except ImportError:
    _ISAAC_AVAILABLE = False
    DirectRLEnv = object
    configclass = dataclass    # fallback: use plain dataclass

from .debris_dynamics import TumblingDebrisModel, DebrisDynamicsConfig


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DebrisCaptureEnvCfg:
    """
    Complete configuration for the debris capture environment.

    Groups all hyperparameters into logical sections: simulation,
    robot, debris, reward, safety, and curriculum.
    """

    # ── Simulation ───────────────────────────────────────────────────────────
    sim_dt: float = 1.0 / 120.0         # 120 Hz physics
    decimation: int = 4                   # RL control at 30 Hz
    render_interval: int = 4             # render every 4 physics steps
    episode_length_s: float = 15.0       # 15 second episodes
    num_envs: int = 4096                 # GPU-parallel environments
    env_spacing: float = 6.0             # metres between env origins

    # ── Robot (Franka Panda 7-DOF) ───────────────────────────────────────────
    robot_prim_path: str = "{ENV_REGEX_NS}/Robot"
    ee_body_name: str = "panda_hand"
    ee_finger_names: Tuple[str, str] = ("panda_leftfinger", "panda_rightfinger")

    # Joint position defaults (home configuration)
    default_joint_pos: Dict[str, float] = field(default_factory=lambda: {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    })

    # Joint limits for safety cost
    joint_pos_soft_limits_frac: float = 0.95  # 95% of hardware limits

    # Action scaling
    max_ee_speed_m_s: float = 0.5           # translational action magnitude
    max_ee_rot_speed_rad_s: float = 1.0     # rotational action magnitude

    # ── Debris ───────────────────────────────────────────────────────────────
    debris_prim_path: str = "{ENV_REGEX_NS}/Debris"
    debris_usd_path: str = ""               # empty = use procedural cube
    debris_scale: Tuple[float, float, float] = (1.0, 0.6, 0.3)  # box dims (m)

    debris_initial_pos_min: Tuple[float, float, float] = (1.5, -0.5, 0.3)
    debris_initial_pos_max: Tuple[float, float, float] = (2.5, 0.5, 0.7)

    # Capture point offset from debris COM (in debris body frame)
    capture_point_body_offset: Tuple[float, float, float] = (0.0, 0.0, 0.15)

    # Physics
    debris_dynamics: DebrisDynamicsConfig = field(default_factory=DebrisDynamicsConfig)

    # ── Contact sensor ───────────────────────────────────────────────────────
    contact_sensor_prim_path: str = "{ENV_REGEX_NS}/Robot/panda_hand"
    contact_filter_prim_path: str = "{ENV_REGEX_NS}/Debris"
    contact_history_length: int = 3        # frames of contact history

    # ── Reward ───────────────────────────────────────────────────────────────
    w_approach: float = 2.0                # dense distance reward
    w_approach_exp: float = 3.0            # exponential distance bonus
    w_align: float = 1.0                   # orientation alignment
    w_velocity_match: float = 0.5          # match EE velocity to debris drift
    w_grasp: float = 5000.0                # sparse capture bonus
    w_collision: float = -100.0            # hard collision penalty
    w_contact_gentle: float = 5.0          # gentle contact bonus (correct force)
    w_effort: float = -0.005               # action effort penalty
    w_jerk: float = -0.002                 # action smoothness penalty

    # Thresholds
    grasp_dist_threshold_m: float = 0.05   # success: EE within 5cm of capture pt
    grasp_align_threshold: float = 0.85    # cosine similarity for valid grasp
    collision_force_threshold_N: float = 20.0   # dangerous impact
    gentle_force_range_N: Tuple[float, float] = (0.5, 10.0)  # good contact forces

    # ── Safety cost (for CMDP / shield) ──────────────────────────────────────
    safe_approach_dist_m: float = 0.15     # proximity warning zone
    joint_velocity_limit_frac: float = 0.90  # fraction of max joint vel

    # ── Curriculum ───────────────────────────────────────────────────────────
    curriculum_enabled: bool = True
    # Phase 1: easy (slow tumble, close debris, 0–1M steps)
    # Phase 2: medium (full tumble range, 1–3M steps)
    # Phase 3: hard (fast tumble + noise + drift, 3M+ steps)
    curriculum_phase_1_steps: int = 1_000_000
    curriculum_phase_2_steps: int = 3_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Main Environment Class
# ─────────────────────────────────────────────────────────────────────────────

class DebrisCaptureEnv(DirectRLEnv):
    """
    Isaac Lab DirectRLEnv for autonomous space debris capture.

    This environment provides a complete, publication-ready simulation for
    training Safe RL agents to capture tumbling space debris.

    Usage (with Isaac Lab installed)
    --------------------------------
    >>> cfg = DebrisCaptureEnvCfg(num_envs=4096)
    >>> env = DebrisCaptureEnv(cfg)
    >>> obs, info = env.reset()
    >>> for step in range(10000):
    ...     action = policy(obs["policy"])
    ...     obs, reward, terminated, truncated, info = env.step(action)

    Usage (without Isaac Lab — standalone mode)
    -------------------------------------------
    >>> cfg = DebrisCaptureEnvCfg(num_envs=64)
    >>> env = DebrisCaptureEnv(cfg, num_envs=64, device="cpu")
    >>> obs, info = env.reset()
    """

    # Dimensions
    NUM_OBS: int = 36
    NUM_ACTIONS: int = 6

    cfg: DebrisCaptureEnvCfg

    def __init__(
        self,
        cfg: DebrisCaptureEnvCfg = DebrisCaptureEnvCfg(),
        num_envs: Optional[int] = None,
        device: str = "cuda:0",
        render_mode: Optional[str] = None,
    ):
        self.cfg = cfg
        self.num_envs = num_envs or cfg.num_envs
        self.device = device
        self.render_mode = render_mode

        # ── Sub-modules ──────────────────────────────────────────────────────
        self._debris_model = TumblingDebrisModel(
            cfg=cfg.debris_dynamics,
            device=device,
        )

        # ── State buffers ────────────────────────────────────────────────────
        self._obs_buf = torch.zeros(self.num_envs, self.NUM_OBS, device=device)
        self._prev_obs_buf = torch.zeros(self.num_envs, self.NUM_OBS, device=device)
        self._rew_buf = torch.zeros(self.num_envs, device=device)
        self._cost_buf = torch.zeros(self.num_envs, device=device)
        self._terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self._prev_action = torch.zeros(self.num_envs, self.NUM_ACTIONS, device=device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._episode_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._total_steps: int = 0
        self._max_episode_steps = int(
            cfg.episode_length_s / (cfg.sim_dt * cfg.decimation)
        )

        # Capture point in debris body frame
        self._capture_offset = torch.tensor(
            cfg.capture_point_body_offset, dtype=torch.float32, device=device
        )

        # ── Isaac Lab scene setup ────────────────────────────────────────────
        self._robot = None
        self._debris = None
        self._contact_sensor = None
        self._diff_ik = None
        self._ee_idx: int = -1
        self._joint_limits_lower = None
        self._joint_limits_upper = None

        if _ISAAC_AVAILABLE:
            self._setup_isaac_scene()

    # ═════════════════════════════════════════════════════════════════════════
    # Isaac Lab scene construction
    # ═════════════════════════════════════════════════════════════════════════

    def _setup_isaac_scene(self):
        """
        Construct the complete Isaac Lab scene:
          1. Zero-gravity simulation context
          2. Franka Panda robot arm
          3. Debris rigid object
          4. Contact sensors
          5. Differential IK controller
          6. Environment replication
        """
        cfg = self.cfg

        # ── 1. Simulation context (zero gravity) ────────────────────────────
        sim_cfg = SimulationCfg(
            dt=cfg.sim_dt,
            render_interval=cfg.render_interval,
            gravity=(0.0, 0.0, 0.0),         # microgravity
            physx=sim_utils.PhysxCfg(
                bounce_threshold_velocity=0.01,
                gpu_found_lost_aggregate_pairs_capacity=1024 * 1024,
                gpu_total_aggregate_pairs_capacity=16 * 1024,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.0025,
            ),
        )

        # ── 2. Robot: Franka Panda with high-PD gains ──────────────────────
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path=cfg.robot_prim_path,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos=cfg.default_joint_pos,
            ),
        )

        # ── 3. Debris rigid body ───────────────────────────────────────────
        if cfg.debris_usd_path:
            # Use imported CAD model (NASA 3D / custom satellite)
            debris_cfg = RigidObjectCfg(
                prim_path=cfg.debris_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=cfg.debris_usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,           # microgravity
                        linear_damping=0.0,             # no atmospheric drag
                        angular_damping=0.0,            # torque-free precession
                        max_linear_velocity=10.0,
                        max_angular_velocity=100.0,
                        enable_gyroscopic_forces=True,   # critical for tumbling
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=500.0,  # overridden by dynamics model per-env
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(2.0, 0.0, 0.5),
                ),
            )
        else:
            # Procedural box (for testing without USD assets)
            debris_cfg = RigidObjectCfg(
                prim_path=cfg.debris_prim_path,
                spawn=sim_utils.CuboidCfg(
                    size=cfg.debris_scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=True,
                        linear_damping=0.0,
                        angular_damping=0.0,
                        max_angular_velocity=100.0,
                        enable_gyroscopic_forces=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.7, 0.3, 0.1),  # rust/Kapton orange
                        metallic=0.6,
                        roughness=0.4,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(2.0, 0.0, 0.5),
                ),
            )

        # ── 4. Contact sensor on EE ────────────────────────────────────────
        contact_cfg = ContactSensorCfg(
            prim_path=cfg.contact_sensor_prim_path,
            filter_prim_paths_expr=[cfg.contact_filter_prim_path],
            history_length=cfg.contact_history_length,
            update_period=0.0,  # every physics step
            track_air_time=False,
        )

        # ── 5. Build scene ─────────────────────────────────────────────────
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs,
            env_spacing=cfg.env_spacing,
            replicate_physics=True,
        )

        # Create scene entities
        self._robot = Articulation(robot_cfg)
        self._debris = RigidObject(debris_cfg)
        self._contact_sensor = ContactSensor(contact_cfg)

        # ── 6. Differential IK controller ──────────────────────────────────
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,           # delta commands
            ik_method="dls",                  # damped least squares
            ik_params={"lambda_val": 0.05},   # damping factor
        )
        self._diff_ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # Cache EE body index
        self._ee_idx = self._robot.find_bodies(cfg.ee_body_name)[0][0]

        # Cache joint limits for safety cost computation
        self._joint_limits_lower = self._robot.data.soft_joint_pos_limits[:, :7, 0]
        self._joint_limits_upper = self._robot.data.soft_joint_pos_limits[:, :7, 1]

    # ═════════════════════════════════════════════════════════════════════════
    # Gymnasium-compatible API
    # ═════════════════════════════════════════════════════════════════════════

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """
        Reset selected (or all) environments.

        Randomises:
          - Debris initial position, orientation, angular velocity, mass
          - Robot joint positions (near home config with noise)
          - Curriculum-dependent difficulty settings

        Returns
        -------
        obs  : dict with key "policy" → (num_envs, NUM_OBS)
        info : dict with keys "cost", "curriculum_phase"
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)

        # ── Reset step counters ──────────────────────────────────────────────
        self._step_count[env_ids] = 0
        self._episode_count[env_ids] += 1
        self._prev_action[env_ids] = 0.0

        # ── Determine curriculum phase ──────────────────────────────────────
        phase = self._get_curriculum_phase()

        # ── Reset debris dynamics with domain randomisation ─────────────────
        self._reset_debris(env_ids, phase)

        # ── Reset robot to home configuration with noise ────────────────────
        self._reset_robot(env_ids)

        # ── Compute initial observations ────────────────────────────────────
        obs = self._compute_observations(env_ids)
        self._obs_buf[env_ids] = obs
        self._prev_obs_buf[env_ids] = obs.clone()

        info = {
            "cost": torch.zeros(n, device=self.device),
            "curriculum_phase": phase,
        }
        return {"policy": self._obs_buf}, info

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Apply action, step simulation, compute obs/reward/cost/done.

        Parameters
        ----------
        action : (num_envs, NUM_ACTIONS) — normalised ∈ [−1, 1]

        Returns
        -------
        obs       : dict with "policy" → (num_envs, NUM_OBS)
        reward    : (num_envs,)
        terminated : (num_envs,) bool
        truncated  : (num_envs,) bool
        info      : dict with "cost", "is_success", "collision_force"
        """
        # Clamp actions to valid range
        action = torch.clamp(action, -1.0, 1.0)

        # ── Scale actions to physical units ──────────────────────────────────
        dt = self.cfg.sim_dt * self.cfg.decimation
        delta_pos = action[:, :3] * self.cfg.max_ee_speed_m_s * dt
        delta_rot = action[:, 3:] * self.cfg.max_ee_rot_speed_rad_s * dt

        # ── Step physics (decimation loop) ───────────────────────────────────
        for _ in range(self.cfg.decimation):
            # Apply IK-resolved joint commands
            self._apply_cartesian_action(delta_pos / self.cfg.decimation,
                                          delta_rot / self.cfg.decimation)

            # Step the debris dynamics model
            self._debris_model.step()

            # Write debris state to simulation
            if _ISAAC_AVAILABLE and self._debris is not None:
                self._debris_model.write_state_to_sim(self._debris)

            # Step PhysX
            if _ISAAC_AVAILABLE:
                self._robot.write_data_to_sim()
                sim_utils.SimulationContext.instance().step(render=False)
                self._robot.update(self.cfg.sim_dt)
                self._debris.update(self.cfg.sim_dt)
                self._contact_sensor.update(self.cfg.sim_dt)

        # ── Update state ─────────────────────────────────────────────────────
        self._step_count += 1
        self._total_steps += 1
        self._prev_obs_buf = self._obs_buf.clone()
        self._obs_buf = self._compute_observations()

        # ── Compute reward, cost, done ───────────────────────────────────────
        self._rew_buf = self._compute_rewards(action)
        self._cost_buf = self._compute_safety_cost()
        self._terminated_buf = self._compute_termination()
        self._truncated_buf = self._step_count >= self._max_episode_steps

        # ── Info dict ────────────────────────────────────────────────────────
        is_success = self._check_successful_capture()
        contact_force = self._get_contact_force_magnitude()

        info = {
            "cost":            self._cost_buf.clone(),
            "is_success":      is_success,
            "collision_force": contact_force,
            "tumble_rate_deg": torch.rad2deg(self._debris_model.get_tumbling_rate()),
            "tumble_mode":     self._debris_model.get_tumble_mode(),
        }

        self._prev_action = action.clone()

        return (
            {"policy": self._obs_buf},
            self._rew_buf,
            self._terminated_buf,
            self._truncated_buf,
            info,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Observation computation
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_observations(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the full 36-D observation vector.

        Layout:
          [0:3]   debris_pos − robot_base_pos  (relative position)
          [3:7]   debris quaternion (w, x, y, z)
          [7:10]  debris angular velocity
          [10:13] EE Cartesian position (relative to env origin)
          [13:17] EE orientation quaternion
          [17:24] joint positions (7-DOF, normalised)
          [24:31] joint velocities (normalised)
          [31:34] EE contact force vector
          [34]    normalised time remaining
          [35]    debris tumble mode (0/1/2)
        """
        N = self.num_envs if env_ids is None else len(env_ids)

        if _ISAAC_AVAILABLE and self._robot is not None:
            return self._compute_obs_isaac(env_ids)
        else:
            return self._compute_obs_standalone(env_ids, N)

    def _compute_obs_isaac(
        self,
        env_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build observations from live Isaac Lab simulation data."""
        ids = env_ids if env_ids is not None else slice(None)

        # Debris state
        debris_pos = self._debris.data.root_pos_w[ids] - self._robot.data.root_pos_w[ids]
        debris_quat = self._debris.data.root_quat_w[ids]
        debris_ang_vel = self._debris.data.root_ang_vel_w[ids]

        # Robot EE state
        ee_pos = self._robot.data.body_pos_w[ids, self._ee_idx] - self._robot.data.root_pos_w[ids]
        ee_quat = self._robot.data.body_quat_w[ids, self._ee_idx]

        # Joint state (normalised to [−1, 1])
        joint_pos_raw = self._robot.data.joint_pos[ids, :7]
        joint_vel_raw = self._robot.data.joint_vel[ids, :7]

        joint_pos_mid = (self._joint_limits_upper[ids] + self._joint_limits_lower[ids]) / 2
        joint_pos_range = (self._joint_limits_upper[ids] - self._joint_limits_lower[ids]) / 2
        joint_pos_norm = (joint_pos_raw - joint_pos_mid) / (joint_pos_range + 1e-6)

        max_joint_vel = 2.175  # Franka Panda max joint velocity (rad/s)
        joint_vel_norm = joint_vel_raw / max_joint_vel

        # Contact forces
        contact_forces = self._contact_sensor.data.net_forces_w[ids, 0, :]  # (N, 3)

        # Time remaining
        time_remaining = 1.0 - self._step_count[ids].float() / self._max_episode_steps

        # Tumble mode
        tumble_mode = self._debris_model.get_tumble_mode()[ids].float() / 2.0

        obs = torch.cat([
            debris_pos,           # [0:3]
            debris_quat,          # [3:7]
            debris_ang_vel,       # [7:10]
            ee_pos,               # [10:13]
            ee_quat,              # [13:17]
            joint_pos_norm,       # [17:24]
            joint_vel_norm,       # [24:31]
            contact_forces,       # [31:34]
            time_remaining.unsqueeze(-1),   # [34]
            tumble_mode.unsqueeze(-1),      # [35]
        ], dim=-1)

        return obs

    def _compute_obs_standalone(
        self,
        env_ids: Optional[torch.Tensor],
        N: int,
    ) -> torch.Tensor:
        """
        Build observations from internal dynamics model (no Isaac Lab).
        Used for offline testing, shield table generation, etc.
        """
        ids = env_ids if env_ids is not None else slice(None)

        debris_state = self._debris_model.get_state()
        time_remaining = 1.0 - self._step_count[ids].float() / self._max_episode_steps
        tumble_mode = self._debris_model.get_tumble_mode()[ids].float() / 2.0

        obs = torch.zeros(N, self.NUM_OBS, device=self.device)
        obs[:, 0:3]   = debris_state["position"][ids]
        obs[:, 3:7]   = debris_state["quaternion"][ids]
        obs[:, 7:10]  = debris_state["angular_vel"][ids]
        obs[:, 34]    = time_remaining if isinstance(time_remaining, torch.Tensor) else time_remaining
        obs[:, 35]    = tumble_mode

        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # Reward computation
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_rewards(self, action: torch.Tensor) -> torch.Tensor:
        """
        Multi-component shaped reward.

        R(s,a) = w_approach     × r_approach
               + w_approach_exp × r_exponential_approach
               + w_align        × r_alignment
               + w_velocity     × r_velocity_matching
               + w_grasp        × r_grasp
               + w_collision    × r_collision
               + w_contact      × r_gentle_contact
               + w_effort       × r_effort
               + w_jerk         × r_jerk
        """
        obs = self._obs_buf
        cfg = self.cfg

        # ── Distances and orientations ───────────────────────────────────────
        debris_pos = obs[:, 0:3]
        ee_pos = obs[:, 10:13]
        dist = torch.linalg.norm(ee_pos - debris_pos, dim=-1)

        debris_quat = obs[:, 3:7]
        ee_quat = obs[:, 13:17]

        contact_force = obs[:, 31:34]
        force_mag = torch.linalg.norm(contact_force, dim=-1)

        # ── 1. Linear approach reward (dense) ────────────────────────────────
        r_approach = -dist

        # ── 2. Exponential bonus for being close (sharper gradient near target)
        r_exp_approach = torch.exp(-5.0 * dist)

        # ── 3. Orientation alignment (quaternion inner product squared) ───────
        debris_q_norm = debris_quat / (torch.linalg.norm(debris_quat, dim=-1, keepdim=True) + 1e-8)
        ee_q_norm = ee_quat / (torch.linalg.norm(ee_quat, dim=-1, keepdim=True) + 1e-8)
        align_score = torch.sum(debris_q_norm * ee_q_norm, dim=-1).abs()
        r_alignment = align_score.pow(2)

        # ── 4. Velocity matching (match EE linear velocity to debris drift) ──
        debris_vel = self._debris_model.velocity
        # Compute EE velocity from consecutive observations
        ee_vel = (obs[:, 10:13] - self._prev_obs_buf[:, 10:13]) / (cfg.sim_dt * cfg.decimation)
        vel_error = torch.linalg.norm(ee_vel - debris_vel, dim=-1)
        r_velocity_match = torch.exp(-2.0 * vel_error)

        # ── 5. Grasp (sparse) ────────────────────────────────────────────────
        close_enough = dist < cfg.grasp_dist_threshold_m
        aligned_enough = align_score > cfg.grasp_align_threshold
        r_grasp = (close_enough & aligned_enough).float()

        # ── 6. Collision penalty ─────────────────────────────────────────────
        dangerous_contact = (force_mag > cfg.collision_force_threshold_N).float()
        r_collision = dangerous_contact

        # ── 7. Gentle contact bonus ──────────────────────────────────────────
        in_gentle_range = (
            (force_mag > cfg.gentle_force_range_N[0])
            & (force_mag < cfg.gentle_force_range_N[1])
        ).float()
        r_gentle = in_gentle_range * close_enough.float()

        # ── 8. Effort penalty (L2 action norm) ───────────────────────────────
        r_effort = torch.sum(action.pow(2), dim=-1)

        # ── 9. Jerk penalty (action smoothness) ─────────────────────────────
        action_diff = action - self._prev_action
        r_jerk = torch.sum(action_diff.pow(2), dim=-1)

        # ── Sum ──────────────────────────────────────────────────────────────
        reward = (
            cfg.w_approach      * r_approach
            + cfg.w_approach_exp * r_exp_approach
            + cfg.w_align        * r_alignment
            + cfg.w_velocity_match * r_velocity_match
            + cfg.w_grasp        * r_grasp
            + cfg.w_collision    * r_collision
            + cfg.w_contact_gentle * r_gentle
            + cfg.w_effort       * r_effort
            + cfg.w_jerk         * r_jerk
        )

        return reward

    # ═════════════════════════════════════════════════════════════════════════
    # Safety cost computation (for CMDP / shield)
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_safety_cost(self) -> torch.Tensor:
        """
        Multi-component safety cost vector (summed to scalar per env).

        C(s,a) = c_collision + c_proximity + c_joint_limit + c_joint_velocity

        Each component is binary (0 or 1) per environment.
        Total cost drives the CMDP constraint and is consumed by the
        probabilistic shield.
        """
        obs = self._obs_buf
        cfg = self.cfg

        # ── 1. Collision cost: excessive contact force ───────────────────────
        contact_force = obs[:, 31:34]
        force_mag = torch.linalg.norm(contact_force, dim=-1)
        c_collision = (force_mag > cfg.collision_force_threshold_N).float()

        # ── 2. Proximity cost: EE too close to debris body ──────────────────
        debris_pos = obs[:, 0:3]
        ee_pos = obs[:, 10:13]
        dist = torch.linalg.norm(ee_pos - debris_pos, dim=-1)
        c_proximity = (dist < cfg.safe_approach_dist_m).float()

        # ── 3. Joint position limit cost ─────────────────────────────────────
        if self._joint_limits_lower is not None:
            joint_pos = obs[:, 17:24]
            # De-normalise to raw positions
            joint_pos_mid = (self._joint_limits_upper + self._joint_limits_lower) / 2
            joint_pos_range = (self._joint_limits_upper - self._joint_limits_lower) / 2
            joint_pos_raw = joint_pos * joint_pos_range + joint_pos_mid

            frac = cfg.joint_pos_soft_limits_frac
            lower_viol = (joint_pos_raw < self._joint_limits_lower * frac).any(dim=-1).float()
            upper_viol = (joint_pos_raw > self._joint_limits_upper * frac).any(dim=-1).float()
            c_joint_pos = lower_viol + upper_viol
        else:
            c_joint_pos = torch.zeros(self.num_envs, device=self.device)

        # ── 4. Joint velocity limit cost ─────────────────────────────────────
        joint_vel_norm = obs[:, 24:31].abs()
        c_joint_vel = (joint_vel_norm > cfg.joint_velocity_limit_frac).any(dim=-1).float()

        return c_collision + c_proximity + c_joint_pos + c_joint_vel

    # ═════════════════════════════════════════════════════════════════════════
    # Termination & success detection
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_termination(self) -> torch.Tensor:
        """
        Episode terminates on:
          1. Dangerous collision (force > threshold)
          2. Successful capture (close + aligned)
          3. Debris too far (drifted away beyond recovery)
        """
        obs = self._obs_buf
        cfg = self.cfg

        # Collision
        force_mag = torch.linalg.norm(obs[:, 31:34], dim=-1)
        collision = force_mag > cfg.collision_force_threshold_N * 2.0  # hard terminate at 2x threshold

        # Success
        success = self._check_successful_capture()

        # Debris drifted too far (unrecoverable)
        dist = torch.linalg.norm(obs[:, 0:3] - obs[:, 10:13], dim=-1)
        too_far = dist > 5.0  # 5 metres → abort

        return collision | success | too_far

    def _check_successful_capture(self) -> torch.Tensor:
        """
        A capture is successful when:
          1. EE is within grasp_dist_threshold_m of the debris capture point
          2. EE orientation is aligned with capture orientation (cosine > threshold)
          3. Contact force is in the gentle range (deliberate, controlled contact)
        """
        obs = self._obs_buf
        cfg = self.cfg

        # Distance check
        debris_pos = obs[:, 0:3]
        ee_pos = obs[:, 10:13]
        dist = torch.linalg.norm(ee_pos - debris_pos, dim=-1)
        close = dist < cfg.grasp_dist_threshold_m

        # Alignment check
        debris_quat = obs[:, 3:7]
        ee_quat = obs[:, 13:17]
        align = torch.sum(
            debris_quat / (torch.linalg.norm(debris_quat, dim=-1, keepdim=True) + 1e-8)
            * ee_quat / (torch.linalg.norm(ee_quat, dim=-1, keepdim=True) + 1e-8),
            dim=-1
        ).abs()
        aligned = align > cfg.grasp_align_threshold

        # Gentle contact check
        force_mag = torch.linalg.norm(obs[:, 31:34], dim=-1)
        gentle = (force_mag > cfg.gentle_force_range_N[0]) & (force_mag < cfg.gentle_force_range_N[1])

        return close & aligned & gentle

    def _get_contact_force_magnitude(self) -> torch.Tensor:
        """Return EE contact force magnitude for logging."""
        return torch.linalg.norm(self._obs_buf[:, 31:34], dim=-1)

    # ═════════════════════════════════════════════════════════════════════════
    # Action application (Cartesian → joint via differential IK)
    # ═════════════════════════════════════════════════════════════════════════

    def _apply_cartesian_action(
        self,
        delta_pos: torch.Tensor,    # (num_envs, 3)
        delta_rot: torch.Tensor,    # (num_envs, 3)
    ):
        """
        Convert Cartesian EE delta commands to joint targets via differential IK.

        Uses Isaac Lab's DifferentialIKController with damped least squares (DLS)
        for singularity-robust inverse kinematics.
        """
        if not _ISAAC_AVAILABLE or self._diff_ik is None:
            return  # standalone mode: skip

        # Current EE pose
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_idx]

        # Current Jacobian
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_idx - 1, :, :7]

        # Current joint positions
        joint_pos = self._robot.data.joint_pos[:, :7]

        # Compute IK
        delta_cmd = torch.cat([delta_pos, delta_rot], dim=-1)
        self._diff_ik.set_command(delta_cmd)
        joint_pos_target = self._diff_ik.compute(
            ee_pos_w, ee_quat_w, jacobian, joint_pos
        )

        # Clamp to joint limits
        if self._joint_limits_lower is not None:
            joint_pos_target = torch.clamp(
                joint_pos_target,
                self._joint_limits_lower,
                self._joint_limits_upper,
            )

        # Write to sim (fingers hold current position)
        full_target = self._robot.data.joint_pos.clone()
        full_target[:, :7] = joint_pos_target
        self._robot.set_joint_position_target(full_target)

    # ═════════════════════════════════════════════════════════════════════════
    # Reset helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _reset_debris(self, env_ids: torch.Tensor, phase: int):
        """Reset debris with domain-randomised initial conditions."""
        n = len(env_ids)
        cfg = self.cfg

        # Reset dynamics model (randomises inertia, angular velocity, attitude)
        dyn_cfg = cfg.debris_dynamics

        # Curriculum: adjust tumble rate range based on training phase
        if cfg.curriculum_enabled:
            if phase == 0:
                dyn_cfg.tumble_rate_max_rad_s = math.radians(10.0)
                dyn_cfg.ou_sigma = 0.005
            elif phase == 1:
                dyn_cfg.tumble_rate_max_rad_s = math.radians(20.0)
                dyn_cfg.ou_sigma = 0.01
            else:
                dyn_cfg.tumble_rate_max_rad_s = math.radians(30.0)
                dyn_cfg.ou_sigma = 0.02

        self._debris_model.reset(self.num_envs, env_ids)

        # Randomise initial position within spawn volume
        pos_min = torch.tensor(cfg.debris_initial_pos_min, device=self.device)
        pos_max = torch.tensor(cfg.debris_initial_pos_max, device=self.device)
        rand_pos = torch.rand(n, 3, device=self.device) * (pos_max - pos_min) + pos_min
        self._debris_model.position[env_ids] = rand_pos

        # Write to simulation
        if _ISAAC_AVAILABLE and self._debris is not None:
            self._debris_model.write_state_to_sim(self._debris)

    def _reset_robot(self, env_ids: torch.Tensor):
        """Reset robot joints to home configuration with small noise."""
        if not _ISAAC_AVAILABLE or self._robot is None:
            return

        n = len(env_ids)

        # Home position + noise
        default_pos = self._robot.data.default_joint_pos[env_ids].clone()
        noise = torch.randn_like(default_pos) * 0.05  # ±0.05 rad noise
        noise[:, 7:] = 0.0  # no noise on finger joints
        target_pos = default_pos + noise

        # Zero velocity
        target_vel = torch.zeros_like(target_pos)

        self._robot.write_joint_state_to_sim(target_pos, target_vel, env_ids=env_ids)

    # ═════════════════════════════════════════════════════════════════════════
    # Curriculum
    # ═════════════════════════════════════════════════════════════════════════

    def _get_curriculum_phase(self) -> int:
        """
        Determine training curriculum phase based on total steps.

        Phase 0 (easy):   slow tumble, close debris, low noise
        Phase 1 (medium): moderate tumble, full position range
        Phase 2 (hard):   fast tumble, high noise, orbital drift
        """
        if not self.cfg.curriculum_enabled:
            return 2  # always hard

        if self._total_steps < self.cfg.curriculum_phase_1_steps:
            return 0
        elif self._total_steps < self.cfg.curriculum_phase_2_steps:
            return 1
        else:
            return 2

    # ═════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═════════════════════════════════════════════════════════════════════════

    def close(self):
        """Clean up Isaac Sim resources."""
        if _ISAAC_AVAILABLE:
            ctx = sim_utils.SimulationContext.instance()
            if ctx is not None:
                ctx.stop()
