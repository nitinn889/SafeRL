"""
shield_wrapper.py
=================
Gymnasium-compatible wrapper that enforces the probabilistic shield at runtime.

The shield operates as follows at each step:
  1. Extract the current observation.
  2. Abstract obs → (d, f, t) via StateAbstraction.
  3. Query PRISMShieldQuery for the collision probability of the RL agent's
     proposed action.
  4a. If P(collision | state, action) ≤ threshold → pass through (allow).
  4b. Else → replace with the least-restrictive safe action (minimise
      probability of collision while still making progress).
  5. Record intervention statistics.

Supports both single-env (gym.Env) and vectorised (gym.vector.VectorEnv)
environments.

References
----------
- Jansen, N., et al. (2020). "Safe Reinforcement Learning Using Probabilistic
  Shields." CONCUR 2020.
- Hasanbeig, M., et al. (2020). "Cautious Reinforcement Learning with
  Logical Constraints." AAMAS 2020.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .abstraction import AbstractState, StateAbstraction
from .shield_query import ACTION_APPROACH, ACTION_HOLD, ACTION_RETREAT, PRISMShieldQuery

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Action mapping: RL action (continuous 6-D) → abstract action (0/1/2)
# ─────────────────────────────────────────────────────────────────────────────

def map_to_abstract_action(action: np.ndarray, obs: np.ndarray) -> int:
    """
    Coarsely classify a 6-D continuous Cartesian action as approach / hold / retreat.

    The classification is based on the component of the action vector that
    points toward or away from the debris.

    Parameters
    ----------
    action : (6,) array — delta EE command [dx, dy, dz, droll, dpitch, dyaw]
    obs    : (OBS_DIM,) array — current observation

    Returns
    -------
    int : ACTION_APPROACH (0), ACTION_HOLD (1), or ACTION_RETREAT (2)
    """
    debris_rel = obs[0:3]                   # debris relative position
    direction_to_debris = debris_rel / (np.linalg.norm(debris_rel) + 1e-8)
    delta_pos = action[:3]                   # translational component
    projection = float(np.dot(delta_pos, direction_to_debris))

    if projection > 0.1:
        return ACTION_APPROACH
    elif projection < -0.1:
        return ACTION_RETREAT
    else:
        return ACTION_HOLD


def abstract_action_to_continuous(
    abstract_action: int,
    obs: np.ndarray,
    magnitude: float = 0.5,
) -> np.ndarray:
    """
    Convert an abstract action (0/1/2) back to a 6-D continuous action.

    Used when the shield overrides the RL action with the safest abstract action.

    Parameters
    ----------
    abstract_action : int
    obs             : (OBS_DIM,)  current observation
    magnitude       : float       speed factor in [0, 1]

    Returns
    -------
    action : (6,) ndarray
    """
    debris_rel = obs[0:3]
    unit_toward = debris_rel / (np.linalg.norm(debris_rel) + 1e-8)

    if abstract_action == ACTION_APPROACH:
        return np.concatenate([unit_toward * magnitude, np.zeros(3)])
    elif abstract_action == ACTION_RETREAT:
        return np.concatenate([-unit_toward * magnitude, np.zeros(3)])
    else:  # HOLD
        return np.zeros(6, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Intervention record
# ─────────────────────────────────────────────────────────────────────────────

class ShieldIntervention:
    """Records a single shield intervention event."""
    __slots__ = (
        "step", "abstract_state", "proposed_action", "safe_action",
        "collision_prob", "threshold"
    )

    def __init__(
        self,
        step: int,
        abstract_state: AbstractState,
        proposed_action: int,
        safe_action: int,
        collision_prob: float,
        threshold: float,
    ):
        self.step            = step
        self.abstract_state  = abstract_state
        self.proposed_action = proposed_action
        self.safe_action     = safe_action
        self.collision_prob  = collision_prob
        self.threshold       = threshold

    def __repr__(self):
        return (
            f"Intervention(step={self.step}, state={self.abstract_state}, "
            f"proposed={self.proposed_action}, safe={self.safe_action}, "
            f"P(col)={self.collision_prob:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shield wrapper (single-env)
# ─────────────────────────────────────────────────────────────────────────────

class ShieldWrapper:
    """
    Wraps any Gymnasium-compatible environment with a probabilistic shield.

    The wrapper is transparent: if the shield does not intervene, the RL
    action is passed through unchanged. If it does intervene, the least
    restrictive safe action is substituted and the intervention is logged.

    Parameters
    ----------
    env : gym.Env
        Base environment to wrap.
    shield_query : PRISMShieldQuery
        Pre-configured query engine.
    safety_threshold : float
        Maximum P(collision) allowed before intervention (default 0.05).
    fallback_action : int
        Abstract action to use if ALL actions are unsafe (default=HOLD).
    log_interventions : bool
        Whether to record full ShieldIntervention objects.
    """

    def __init__(
        self,
        env,
        shield_query: PRISMShieldQuery,
        safety_threshold: float = 0.05,
        fallback_action: int = ACTION_HOLD,
        log_interventions: bool = True,
    ):
        self.env              = env
        self.shield           = shield_query
        self.safety_threshold = safety_threshold
        self.fallback_action  = fallback_action
        self.log_interventions = log_interventions

        self._abstraction = StateAbstraction()
        self._last_obs: Optional[np.ndarray] = None
        self._step_count: int = 0

        # Statistics
        self.total_steps: int = 0
        self.total_interventions: int = 0
        self.interventions_log: List[ShieldIntervention] = []

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = self._to_numpy(obs)
        self._step_count = 0
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply shield check then step the environment.

        Parameters
        ----------
        action : np.ndarray  (ACTION_DIM,) — RL agent's proposed action

        Returns
        -------
        obs, reward, done, truncated, info — standard Gymnasium tuple
        info['shield_intervened'] = True/False
        info['proposed_action']   = original action before override
        """
        obs_np = self._last_obs if self._last_obs is not None else np.zeros(36)
        action_np = self._to_numpy(action)

        # Abstract state
        abstract_state = self._abstraction.abstract(obs_np)

        # Classify the proposed continuous action
        proposed_abstract = map_to_abstract_action(action_np, obs_np)
        p_collision       = self.shield.collision_prob(abstract_state, proposed_abstract)

        intervened        = p_collision > self.safety_threshold
        safe_action_np    = action_np

        if intervened:
            # Find the least-restrictive safe abstract action
            safe_abstract = self._least_restrictive_safe(abstract_state, proposed_abstract)
            safe_action_np = abstract_action_to_continuous(safe_abstract, obs_np)
            self.total_interventions += 1

            if self.log_interventions:
                self.interventions_log.append(ShieldIntervention(
                    step            = self._step_count,
                    abstract_state  = abstract_state,
                    proposed_action = proposed_abstract,
                    safe_action     = safe_abstract,
                    collision_prob  = p_collision,
                    threshold       = self.safety_threshold,
                ))
            logger.debug(
                f"Shield intervened at step {self._step_count}: "
                f"state={abstract_state}, P(col)={p_collision:.3f}"
            )

        obs, reward, done, truncated, info = self.env.step(safe_action_np)
        self._last_obs = self._to_numpy(obs)
        self._step_count  += 1
        self.total_steps  += 1

        info["shield_intervened"] = intervened
        info["proposed_action"]   = action_np
        info["collision_prob"]    = p_collision
        return obs, reward, done, truncated, info

    # ── Delegation ────────────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def close(self):
        self.env.close()

    # ── Shield logic ──────────────────────────────────────────────────────────

    def _least_restrictive_safe(
        self,
        state: AbstractState,
        proposed: int,
    ) -> int:
        """
        Select the safest abstract action, preferring the one closest to
        the original proposal to minimise task disruption.

        Priority: try proposed → hold → retreat → approach → fallback.
        """
        safe_set = self.shield.safe_action_set(state)

        if not safe_set:
            # All actions violate the threshold — use fallback (hold)
            logger.warning(
                f"All actions unsafe at state {state}. Using fallback={self.fallback_action}."
            )
            return self.fallback_action

        # Prefer the action nearest to the proposed one that is safe
        preference = [proposed, ACTION_HOLD, ACTION_RETREAT, ACTION_APPROACH]
        for a in preference:
            if a in safe_set:
                return a
        return safe_set[0]

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def intervention_rate(self) -> float:
        """Fraction of steps where the shield intervened."""
        if self.total_steps == 0:
            return 0.0
        return self.total_interventions / self.total_steps

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary dict of shield statistics."""
        return {
            "total_steps":         self.total_steps,
            "total_interventions": self.total_interventions,
            "intervention_rate":   self.intervention_rate,
        }

    def reset_stats(self):
        """Reset cumulative statistics without resetting the environment."""
        self.total_steps         = 0
        self.total_interventions = 0
        self.interventions_log   = []

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)
