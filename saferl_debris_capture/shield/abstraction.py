"""
abstraction.py
==============
State space abstraction for Probabilistic Model Checking (PMC).

Maps the continuous 36-D observation vector from the simulation into the
small discrete state space used by the PRISM model (debris_capture.pm).

Abstraction scheme
------------------
Continuous → discrete:
  distance_bucket  d ∈ {0,1,2,3,4}   EE-to-debris Euclidean distance
  force_bucket     f ∈ {0,1,2}       contact force magnitude at EE
  tumble_bucket    t ∈ {0,1,2}       debris angular speed

These three integers form the abstract state (d, f, t) fed to PRISM.

The abstraction is designed to be *conservative*: when uncertain which
bucket applies, we round toward the more dangerous bucket so the shield
is never over-optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Abstract state
# ─────────────────────────────────────────────────────────────────────────────

class AbstractState(NamedTuple):
    """Discrete state tuple fed to the PRISM model checker."""
    d: int   # distance bucket  [0 = contact, 4 = far]
    f: int   # force bucket     [0 = safe, 1 = warning, 2 = overload]
    t: int   # tumble bucket    [0 = slow, 1 = medium, 2 = fast]

    def as_prism_state_string(self) -> str:
        """Format for PRISM steady-state / reachability query."""
        return f"d={self.d}&f={self.f}&t={self.t}"


# ─────────────────────────────────────────────────────────────────────────────
# Abstraction configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AbstractionConfig:
    """Thresholds for the continuous → discrete mapping."""

    # Distance thresholds (metres from EE to debris capture point)
    dist_thresholds: tuple = (0.05, 0.20, 0.50, 1.00)
    # d=0: <0.05m (contact)  d=1: 0.05-0.20m  d=2: 0.20-0.50m
    # d=3: 0.50-1.00m        d=4: >1.00m

    # Force thresholds (Newtons at the EE contact sensor)
    force_warning_N: float = 10.0    # f=1: warning zone
    force_overload_N: float = 20.0   # f=2: collision / overload

    # Tumbling rate thresholds (rad/s)
    tumble_slow_rad_s: float = 0.087  # 5 deg/s
    tumble_fast_rad_s: float = 0.262  # 15 deg/s


# ─────────────────────────────────────────────────────────────────────────────
# Abstraction class
# ─────────────────────────────────────────────────────────────────────────────

class StateAbstraction:
    """
    Maps the Isaac Lab observation tensor to an AbstractState.

    Observation layout (must match DebrisCaptureEnv.OBS_DIM = 36):
      [0:3]   debris_relative_position
      [3:7]   debris_quaternion
      [7:10]  debris_angular_velocity
      [10:13] ee_position
      [13:17] ee_quaternion
      [17:24] joint_positions
      [24:31] joint_velocities
      [31:34] ee_contact_force
      [34]    normalised_time_remaining
      [35]    curriculum_level

    Parameters
    ----------
    cfg : AbstractionConfig
        Thresholds for binning.
    """

    def __init__(self, cfg: AbstractionConfig = AbstractionConfig()):
        self.cfg = cfg

    # ── Main interface ────────────────────────────────────────────────────────

    def abstract(
        self,
        obs: Union[np.ndarray, torch.Tensor],
    ) -> AbstractState:
        """
        Map a single observation vector to an AbstractState.

        Parameters
        ----------
        obs : array-like of shape (OBS_DIM,)

        Returns
        -------
        AbstractState
        """
        obs = self._to_numpy(obs)
        d = self._distance_bucket(obs)
        f = self._force_bucket(obs)
        t = self._tumble_bucket(obs)
        return AbstractState(d=d, f=f, t=t)

    def abstract_batch(
        self,
        obs_batch: Union[np.ndarray, torch.Tensor],
    ) -> list[AbstractState]:
        """
        Vectorised abstraction for a batch (num_envs, OBS_DIM).

        Returns a list of AbstractState named tuples.
        """
        if isinstance(obs_batch, torch.Tensor):
            obs_batch = obs_batch.cpu().numpy()
        return [self.abstract(obs) for obs in obs_batch]

    # ── Bucket functions ──────────────────────────────────────────────────────

    def _distance_bucket(self, obs: np.ndarray) -> int:
        """Bin EE-to-debris Euclidean distance into {0,1,2,3,4}."""
        debris_pos = obs[0:3]
        ee_pos     = obs[10:13]
        dist = float(np.linalg.norm(ee_pos - debris_pos))
        thresholds = self.cfg.dist_thresholds
        for bucket, thresh in enumerate(thresholds):
            if dist < thresh:
                return bucket
        return 4   # > max threshold → far

    def _force_bucket(self, obs: np.ndarray) -> int:
        """Bin contact force magnitude at EE into {0=safe, 1=warning, 2=overload}."""
        force_vec = obs[31:34]
        force_mag = float(np.linalg.norm(force_vec))
        if force_mag >= self.cfg.force_overload_N:
            return 2
        elif force_mag >= self.cfg.force_warning_N:
            return 1
        return 0

    def _tumble_bucket(self, obs: np.ndarray) -> int:
        """Bin debris angular speed into {0=slow, 1=medium, 2=fast}."""
        omega = obs[7:10]
        speed = float(np.linalg.norm(omega))
        if speed >= self.cfg.tumble_fast_rad_s:
            return 2
        elif speed >= self.cfg.tumble_slow_rad_s:
            return 1
        return 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    # ── State-space size ──────────────────────────────────────────────────────

    @property
    def num_abstract_states(self) -> int:
        """Total number of abstract states: |D| × |F| × |T| = 5 × 3 × 3 = 45."""
        return 5 * 3 * 3

    def state_to_index(self, s: AbstractState) -> int:
        """Flatten (d, f, t) → single integer index."""
        return s.d * 9 + s.f * 3 + s.t

    def index_to_state(self, idx: int) -> AbstractState:
        """Inverse of state_to_index."""
        d, rem = divmod(idx, 9)
        f, t   = divmod(rem, 3)
        return AbstractState(d=d, f=f, t=t)
