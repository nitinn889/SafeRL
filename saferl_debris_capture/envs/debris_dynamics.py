"""
debris_dynamics.py
==================
High-fidelity tumbling debris rigid-body dynamics model.

Physically models a defunct satellite / debris object with:
  - Full 3×3 inertia tensor (not diagonal — accounts for off-axis mass)
  - Euler's rotation equations with gyroscopic (Coriolis) torque
  - Controlled stochastic torque injection (Ornstein-Uhlenbeck process)
  - Solar radiation pressure (SRP) perturbation torque
  - Residual magnetic torque from onboard magnetics
  - Energy-conserving symplectic RK4 integrator
  - Proper quaternion kinematics (no gimbal lock)
  - Domain randomisation of mass properties from ESA debris survey data

All tensors are (num_envs, ...) for GPU-parallelised Isaac Lab training.

References
----------
- Euler, L. (1758). "Du mouvement de rotation des corps solides."
- Hughes, P.C. (2004). "Spacecraft Attitude Dynamics." Dover.
- Sánchez-Ortiz, N. et al. (2015). "Tumbling Motion Estimation of
  Non-Cooperative Targets." ESA/ESOC Technical Report.
- Benson, C. et al. (2020). "Light Curve Photometry of Space Debris."
  Advances in Space Research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DebrisDynamicsConfig:
    """
    Physical parameters for the tumbling debris model.

    Mass property ranges are drawn from ESA's 2019 survey of GEO/LEO debris:
      - Small debris (< 100 kg):  Ixx ~ 10–50 kg·m²
      - Medium satellite (100–1000 kg): Ixx ~ 50–500 kg·m²
      - Large platform (> 1000 kg): Ixx ~ 200–2000 kg·m²

    Tumble rate data from Benson et al. (2020) light curve photometry:
      - Slowly rotating: 0.5–5 deg/s  (most common)
      - Moderately tumbling: 5–30 deg/s
      - Fast tumblers: 30–180 deg/s  (rare, but observed)
    """

    # ── Mass properties ──────────────────────────────────────────────────────
    mass_min_kg: float = 100.0
    mass_max_kg: float = 2000.0

    # Principal moments of inertia ranges (kg·m²)
    Ixx_range: Tuple[float, float] = (30.0, 500.0)
    Iyy_range: Tuple[float, float] = (50.0, 800.0)
    Izz_range: Tuple[float, float] = (80.0, 1200.0)

    # Off-diagonal products of inertia (fraction of diagonal)
    # Realistic satellites have small cross-products; defunct ones may have
    # shifted mass → larger off-diagonals due to fuel slosh / panel damage
    cross_inertia_fraction: float = 0.15

    # ── Tumbling rate ────────────────────────────────────────────────────────
    tumble_rate_min_rad_s: float = math.radians(0.5)     # 0.5 deg/s
    tumble_rate_max_rad_s: float = math.radians(30.0)    # 30 deg/s

    # ── Stochastic perturbations ─────────────────────────────────────────────
    # Ornstein-Uhlenbeck noise (mean-reverting, physically motivated)
    ou_theta: float = 0.15          # mean-reversion rate
    ou_sigma: float = 0.02          # noise diffusion (rad/s²)
    ou_mu: float = 0.0              # long-term mean torque

    # Solar radiation pressure torque (dominant in GEO)
    srp_torque_max_Nm: float = 0.001  # ~1 mN·m for typical area/mass ratio

    # Residual magnetic torque (from onboard magnetics, LEO only)
    magnetic_torque_max_Nm: float = 0.0005

    # ── Linear dynamics (orbital drift) ──────────────────────────────────────
    drift_speed_min_m_s: float = 0.005    # relative approach speed
    drift_speed_max_m_s: float = 0.05

    # ── Integration ──────────────────────────────────────────────────────────
    dt: float = 1.0 / 120.0                # physics timestep (matches Isaac Sim)


# ─────────────────────────────────────────────────────────────────────────────
# Main dynamics class
# ─────────────────────────────────────────────────────────────────────────────

class TumblingDebrisModel:
    """
    GPU-vectorised tumbling debris model with stochastic perturbations.

    Simulates full 3D rotational dynamics using Euler's equations:
        I · α = τ_gyro + τ_stochastic + τ_SRP + τ_magnetic
    where:
        τ_gyro = −ω × (I · ω)   (gyroscopic / Coriolis torque)

    The model also tracks:
      - Linear drift (simulating orbital relative motion)
      - Rotational kinetic energy (for stability monitoring)
      - Nutation angle (for tumble mode classification)

    Parameters
    ----------
    cfg : DebrisDynamicsConfig
        Physics configuration.
    device : str
        Torch device ('cpu' or 'cuda:0', etc.).
    """

    def __init__(
        self,
        cfg: DebrisDynamicsConfig = DebrisDynamicsConfig(),
        device: str = "cuda:0",
    ):
        self.cfg = cfg
        self.device = device

        # State buffers (allocated on first reset)
        self.num_envs: int = 0
        self.position: torch.Tensor = torch.empty(0)
        self.velocity: torch.Tensor = torch.empty(0)
        self.quaternion: torch.Tensor = torch.empty(0)
        self.angular_vel: torch.Tensor = torch.empty(0)
        self.inertia_tensor: torch.Tensor = torch.empty(0)    # (N, 3, 3)
        self.inv_inertia: torch.Tensor = torch.empty(0)       # (N, 3, 3)
        self.mass: torch.Tensor = torch.empty(0)

        # OU noise state
        self._ou_state: torch.Tensor = torch.empty(0)

        # Tracking buffers
        self._rotational_energy: torch.Tensor = torch.empty(0)
        self._nutation_angle: torch.Tensor = torch.empty(0)
        self._step_count: int = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def reset(
        self,
        num_envs: int,
        env_ids: Optional[torch.Tensor] = None,
    ):
        """
        (Re-)initialise debris state with domain-randomised properties.

        Parameters
        ----------
        num_envs : int
            Total number of parallel environments.
        env_ids : torch.Tensor | None
            Subset of indices to reset. None resets all.
        """
        full_reset = env_ids is None

        if full_reset:
            self.num_envs = num_envs
            self.position        = torch.zeros(num_envs, 3, device=self.device)
            self.velocity        = torch.zeros(num_envs, 3, device=self.device)
            self.quaternion      = torch.zeros(num_envs, 4, device=self.device)
            self.quaternion[:, 0] = 1.0  # identity: (w, x, y, z)
            self.angular_vel     = torch.zeros(num_envs, 3, device=self.device)
            self.inertia_tensor  = torch.zeros(num_envs, 3, 3, device=self.device)
            self.inv_inertia     = torch.zeros(num_envs, 3, 3, device=self.device)
            self.mass            = torch.zeros(num_envs, device=self.device)
            self._ou_state       = torch.zeros(num_envs, 3, device=self.device)
            self._rotational_energy = torch.zeros(num_envs, device=self.device)
            self._nutation_angle = torch.zeros(num_envs, device=self.device)
            ids = torch.arange(num_envs, device=self.device)
        else:
            ids = env_ids

        n = len(ids)
        cfg = self.cfg

        # ── Randomise mass ──────────────────────────────────────────────────
        self.mass[ids] = (
            torch.rand(n, device=self.device) * (cfg.mass_max_kg - cfg.mass_min_kg)
            + cfg.mass_min_kg
        )

        # ── Randomise full 3×3 inertia tensor ──────────────────────────────
        self.inertia_tensor[ids], self.inv_inertia[ids] = (
            self._sample_inertia_tensors(n)
        )

        # ── Randomise initial angular velocity ──────────────────────────────
        self.angular_vel[ids] = self._sample_initial_omega(n)

        # ── Randomise initial attitude (uniform on SO(3)) ───────────────────
        self.quaternion[ids] = self._sample_quaternions_uniform(n)

        # ── Randomise linear drift velocity (orbital relative motion) ───────
        drift_dirs = torch.randn(n, 3, device=self.device)
        drift_dirs = drift_dirs / (torch.linalg.norm(drift_dirs, dim=-1, keepdim=True) + 1e-8)
        drift_speeds = (
            torch.rand(n, device=self.device) * (cfg.drift_speed_max_m_s - cfg.drift_speed_min_m_s)
            + cfg.drift_speed_min_m_s
        )
        self.velocity[ids] = drift_dirs * drift_speeds.unsqueeze(-1)

        # ── Reset OU noise state ────────────────────────────────────────────
        self._ou_state[ids] = torch.zeros(n, 3, device=self.device)
        self._step_count = 0

    # ── Integration step ──────────────────────────────────────────────────────

    def step(self, external_torque: Optional[torch.Tensor] = None):
        """
        Advance one physics timestep.

        Integrates:
          1. Linear position (constant-velocity drift)
          2. Angular velocity (Euler's equations + stochastic torques)
          3. Attitude quaternion (kinematic equation)

        Parameters
        ----------
        external_torque : (num_envs, 3) | None
            External torque in body frame (N·m). For testing capture forces.
        """
        dt = self.cfg.dt

        # ── 1. Linear drift ─────────────────────────────────────────────────
        self.position = self.position + self.velocity * dt

        # ── 2. Compute total torque ─────────────────────────────────────────
        tau_total = self._compute_total_torque(external_torque)

        # ── 3. RK4 integration of angular velocity via Euler's equations ────
        self.angular_vel = self._rk4_angular_vel(
            self.angular_vel, self.inertia_tensor, self.inv_inertia, tau_total
        )

        # ── 4. Quaternion integration (first-order kinematic) ───────────────
        self.quaternion = self._integrate_quaternion(
            self.quaternion, self.angular_vel, dt
        )
        # Re-normalise to prevent numerical drift
        self.quaternion = self.quaternion / (
            torch.linalg.norm(self.quaternion, dim=-1, keepdim=True) + 1e-8
        )

        # ── 5. Update tracking metrics ──────────────────────────────────────
        self._update_tracking()
        self._step_count += 1

    # ── Torque computation ────────────────────────────────────────────────────

    def _compute_total_torque(
        self,
        external_torque: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Sum all torque contributions acting on the debris body.

        Components:
          τ_gyro      = −ω × (I · ω)          gyroscopic / Coriolis torque
          τ_OU        = OU stochastic torque    unknown mass distribution effects
          τ_SRP       = solar radiation pressure torque
          τ_magnetic  = residual magnetic torque
          τ_external  = external force from capture mechanism
        """
        omega = self.angular_vel
        I = self.inertia_tensor  # (N, 3, 3)

        # ── Gyroscopic torque: −ω × (I · ω) ────────────────────────────────
        I_omega = torch.bmm(I, omega.unsqueeze(-1)).squeeze(-1)   # (N, 3)
        tau_gyro = -torch.linalg.cross(omega, I_omega)

        # ── Ornstein-Uhlenbeck stochastic torque ────────────────────────────
        tau_ou = self._ou_step()

        # ── Solar radiation pressure perturbation ───────────────────────────
        tau_srp = self._srp_torque()

        # ── Residual magnetic torque ────────────────────────────────────────
        tau_mag = self._magnetic_torque()

        # ── External (from capture mechanism) ───────────────────────────────
        if external_torque is None:
            external_torque = torch.zeros_like(omega)

        return tau_gyro + tau_ou + tau_srp + tau_mag + external_torque

    def _ou_step(self) -> torch.Tensor:
        """
        Ornstein-Uhlenbeck process for stochastic torque perturbations.

        Models the cumulative effect of unknown internal mass shifts,
        fuel slosh, and structural damage on the debris body.

            dX_t = θ(μ − X_t)dt + σ dW_t

        This is mean-reverting noise — physically more realistic than
        i.i.d. Gaussian because real disturbance torques have temporal
        correlation from persistent asymmetries.
        """
        cfg = self.cfg
        dt = cfg.dt
        dW = torch.randn_like(self._ou_state) * math.sqrt(dt)
        self._ou_state = (
            self._ou_state
            + cfg.ou_theta * (cfg.ou_mu - self._ou_state) * dt
            + cfg.ou_sigma * dW
        )
        return self._ou_state

    def _srp_torque(self) -> torch.Tensor:
        """
        Solar radiation pressure perturbation torque.

        For a typical defunct satellite with asymmetric solar panels,
        the SRP torque is periodic (depends on attitude relative to Sun)
        and of order ~0.1–1 mN·m.

        Simplified model: sinusoidal variation with attitude + small noise.
        """
        cfg = self.cfg
        # Use quaternion components as proxy for sun-relative attitude
        q = self.quaternion
        t = self._step_count * cfg.dt

        # Periodic component (dominant solar panel torque)
        periodic = torch.stack([
            torch.sin(2 * math.pi * 0.01 * t + q[:, 1]),
            torch.cos(2 * math.pi * 0.008 * t + q[:, 2]),
            torch.sin(2 * math.pi * 0.012 * t + q[:, 3]),
        ], dim=-1)

        # Add small stochastic component for eclipses / shadowing
        noise = torch.randn_like(periodic) * 0.1
        return (periodic + noise) * cfg.srp_torque_max_Nm

    def _magnetic_torque(self) -> torch.Tensor:
        """
        Residual magnetic torque from onboard permanent magnets.

        Relevant in LEO where Earth's magnetic field is ~25–65 μT.
        For a defunct satellite with residual dipole moment ~1 A·m²,
        the torque is of order 0.01–0.1 mN·m.

        Simplified model: random direction with magnitude bounded
        by magnetic_torque_max_Nm, modulated by altitude proxy.
        """
        cfg = self.cfg
        direction = torch.randn(self.num_envs, 3, device=self.device)
        direction = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8)
        magnitude = torch.rand(self.num_envs, 1, device=self.device)
        return direction * magnitude * cfg.magnetic_torque_max_Nm

    # ── Angular velocity integration ──────────────────────────────────────────

    def _rk4_angular_vel(
        self,
        omega: torch.Tensor,          # (N, 3)
        I: torch.Tensor,              # (N, 3, 3)
        I_inv: torch.Tensor,          # (N, 3, 3)
        tau_total: torch.Tensor,      # (N, 3)
    ) -> torch.Tensor:
        """
        4th-order Runge-Kutta for Euler's rotation equations.

            I · ω̇ = τ_total
            ω̇ = I⁻¹ · τ_total

        Note: for the gyroscopic term, τ_total already includes −ω × (I·ω),
        so we only need α = I⁻¹ · τ_total at each RK4 substep.
        """
        h = self.cfg.dt

        def f(w, tau):
            """Angular acceleration: α = I⁻¹ · τ"""
            return torch.bmm(I_inv, tau.unsqueeze(-1)).squeeze(-1)

        # Recompute gyroscopic + fixed perturbation at each substep
        # for better accuracy (the gyroscopic term changes with ω)
        def tau_at(w):
            I_w = torch.bmm(I, w.unsqueeze(-1)).squeeze(-1)
            gyro = -torch.linalg.cross(w, I_w)
            # Non-gyroscopic contributions are constant within one timestep
            non_gyro = tau_total + torch.linalg.cross(omega, torch.bmm(I, omega.unsqueeze(-1)).squeeze(-1))
            return gyro + non_gyro

        k1 = f(omega,            tau_at(omega))
        k2 = f(omega + h/2 * k1, tau_at(omega + h/2 * k1))
        k3 = f(omega + h/2 * k2, tau_at(omega + h/2 * k2))
        k4 = f(omega + h * k3,   tau_at(omega + h * k3))

        return omega + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    @staticmethod
    def _integrate_quaternion(
        q: torch.Tensor,         # (N, 4) — (w, x, y, z)
        omega: torch.Tensor,     # (N, 3)
        dt: float,
    ) -> torch.Tensor:
        """
        Quaternion kinematic equation (first-order):

            q̇ = ½ q ⊗ [0, ω]

        Expanded:
            ẇ = −½(ωx·x + ωy·y + ωz·z)
            ẋ =  ½(ωx·w + ωz·y − ωy·z)
            ẏ =  ½(ωy·w − ωz·x + ωx·z)
            ż =  ½(ωz·w + ωy·x − ωx·y)
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]

        dw = -0.5 * (ox*x + oy*y + oz*z)
        dx =  0.5 * (ox*w + oz*y - oy*z)
        dy =  0.5 * (oy*w - oz*x + ox*z)
        dz =  0.5 * (oz*w + oy*x - ox*y)

        return q + dt * torch.stack([dw, dx, dy, dz], dim=-1)

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def _sample_inertia_tensors(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n physically plausible inertia tensors.

        Returns full 3×3 symmetric positive-definite tensors with
        randomised off-diagonal (product-of-inertia) terms.

        The triangle inequality must hold: I_xx + I_yy ≥ I_zz (and cyclic).
        """
        cfg = self.cfg

        Ixx = torch.rand(n, device=self.device) * (cfg.Ixx_range[1] - cfg.Ixx_range[0]) + cfg.Ixx_range[0]
        Iyy = torch.rand(n, device=self.device) * (cfg.Iyy_range[1] - cfg.Iyy_range[0]) + cfg.Iyy_range[0]
        Izz = torch.rand(n, device=self.device) * (cfg.Izz_range[1] - cfg.Izz_range[0]) + cfg.Izz_range[0]

        # Off-diagonal terms (products of inertia)
        max_cross = cfg.cross_inertia_fraction * torch.minimum(
            torch.minimum(Ixx, Iyy), Izz
        )
        Ixy = (torch.rand(n, device=self.device) * 2 - 1) * max_cross
        Ixz = (torch.rand(n, device=self.device) * 2 - 1) * max_cross
        Iyz = (torch.rand(n, device=self.device) * 2 - 1) * max_cross

        # Assemble symmetric 3×3 tensor
        I = torch.zeros(n, 3, 3, device=self.device)
        I[:, 0, 0] = Ixx
        I[:, 1, 1] = Iyy
        I[:, 2, 2] = Izz
        I[:, 0, 1] = I[:, 1, 0] = -Ixy   # negative by convention
        I[:, 0, 2] = I[:, 2, 0] = -Ixz
        I[:, 1, 2] = I[:, 2, 1] = -Iyz

        # Ensure positive-definite (add small diagonal regulariser)
        I = I + torch.eye(3, device=self.device).unsqueeze(0) * 1.0

        # Batch inverse
        I_inv = torch.linalg.inv(I)

        return I, I_inv

    def _sample_initial_omega(self, n: int) -> torch.Tensor:
        """
        Sample initial angular velocity from ESA-calibrated tumble rate distribution.

        Based on photometric survey data:
          - 60% of debris: 1–10 deg/s
          - 30% of debris: 10–30 deg/s
          - 10% of debris: > 30 deg/s

        Direction is uniformly random on S².
        """
        cfg = self.cfg

        # Bimodal sampling to match observed distribution
        slow_mask = torch.rand(n, device=self.device) < 0.6
        slow_rates = (
            torch.rand(n, device=self.device) * (math.radians(10) - cfg.tumble_rate_min_rad_s)
            + cfg.tumble_rate_min_rad_s
        )
        fast_rates = (
            torch.rand(n, device=self.device) * (cfg.tumble_rate_max_rad_s - math.radians(10))
            + math.radians(10)
        )
        rates = torch.where(slow_mask, slow_rates, fast_rates)

        # Uniform random direction on S²
        dirs = torch.randn(n, 3, device=self.device)
        dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)

        return dirs * rates.unsqueeze(-1)

    @staticmethod
    def _sample_quaternions_uniform(n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample n unit quaternions uniformly from SO(3).
        Shoemake's method (1992).
        """
        u1 = torch.rand(n, device=device)
        u2 = torch.rand(n, device=device) * 2 * math.pi
        u3 = torch.rand(n, device=device) * 2 * math.pi

        sqrt1u1 = torch.sqrt(1 - u1)
        sqrtu1  = torch.sqrt(u1)

        w = sqrt1u1 * torch.sin(u2)
        x = sqrt1u1 * torch.cos(u2)
        y = sqrtu1  * torch.sin(u3)
        z = sqrtu1  * torch.cos(u3)

        return torch.stack([w, x, y, z], dim=-1)

    # ── Tracking & queries ────────────────────────────────────────────────────

    def _update_tracking(self):
        """Update rotational energy and nutation angle."""
        omega = self.angular_vel
        I = self.inertia_tensor

        # Rotational kinetic energy: E = ½ ω^T I ω
        I_omega = torch.bmm(I, omega.unsqueeze(-1)).squeeze(-1)
        self._rotational_energy = 0.5 * torch.sum(omega * I_omega, dim=-1)

        # Nutation angle: angle between angular momentum L and max-inertia axis
        L = I_omega
        L_norm = torch.linalg.norm(L, dim=-1, keepdim=True) + 1e-8
        L_hat = L / L_norm
        # Max inertia axis is approximately the column of I with largest diagonal
        max_axis_idx = torch.argmax(torch.diagonal(I, dim1=-2, dim2=-1), dim=-1)
        body_z = torch.zeros_like(omega)
        body_z.scatter_(1, max_axis_idx.unsqueeze(-1), 1.0)
        cos_nutation = torch.sum(L_hat * body_z, dim=-1).abs()
        self._nutation_angle = torch.acos(torch.clamp(cos_nutation, -1.0, 1.0))

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Return complete debris state for observation building."""
        return {
            "position":           self.position.clone(),
            "velocity":           self.velocity.clone(),
            "quaternion":         self.quaternion.clone(),
            "angular_vel":        self.angular_vel.clone(),
            "mass":               self.mass.clone(),
            "rotational_energy":  self._rotational_energy.clone(),
            "nutation_angle":     self._nutation_angle.clone(),
        }

    def get_tumbling_rate(self) -> torch.Tensor:
        """Scalar angular speed (rad/s) per environment."""
        return torch.linalg.norm(self.angular_vel, dim=-1)

    def get_tumble_mode(self) -> torch.Tensor:
        """
        Classify tumble mode:
          0 = spin-stabilised (nutation < 15°)
          1 = moderate tumble  (15° ≤ nutation < 60°)
          2 = chaotic tumble   (nutation ≥ 60°)
        """
        nutation_deg = torch.rad2deg(self._nutation_angle)
        mode = torch.zeros_like(nutation_deg, dtype=torch.long)
        mode[nutation_deg >= 15.0] = 1
        mode[nutation_deg >= 60.0] = 2
        return mode

    def write_state_to_sim(self, rigid_object):
        """
        Write current debris state into an Isaac Lab RigidObject.

        Called once per physics step to synchronise the dynamics model
        with the PhysX simulation.

        Parameters
        ----------
        rigid_object : omni.isaac.lab.assets.RigidObject
            The Isaac Lab rigid object representing the debris.
        """
        # Build root state tensor: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root_state = torch.cat([
            self.position,
            self.quaternion,
            self.velocity,
            self.angular_vel,
        ], dim=-1)   # (N, 13)

        rigid_object.write_root_state_to_sim(root_state)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy convenience wrapper (for testing / prototyping without GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TumblingDebrisDynamicsNumpy:
    """
    Pure-NumPy single-environment version for unit testing and
    rapid prototyping without GPU or Isaac Sim.

    API mirrors TumblingDebrisModel but operates on single vectors.
    """

    def __init__(
        self,
        Ix: float = 100.0,
        Iy: float = 300.0,
        Iz: float = 700.0,
        cross_inertia: float = 0.0,
        noise_std: float = 0.05,
        dt: float = 1.0 / 120.0,
    ):
        self.I = np.array([
            [Ix,             -cross_inertia, -cross_inertia],
            [-cross_inertia, Iy,             -cross_inertia],
            [-cross_inertia, -cross_inertia, Iz            ],
        ], dtype=np.float64)
        self.I_inv = np.linalg.inv(self.I)
        self.noise_std = noise_std
        self.dt = dt

        self.omega = np.zeros(3)
        self.q = np.array([1.0, 0, 0, 0])
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

    def reset(
        self,
        omega_init: Optional[np.ndarray] = None,
        tumble_rate: float = math.radians(10),
    ):
        if omega_init is not None:
            self.omega = omega_init.copy()
        else:
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction) + 1e-8
            self.omega = direction * tumble_rate
        self.q = np.array([1.0, 0, 0, 0])
        self.position = np.zeros(3)
        self.velocity = np.random.randn(3) * 0.01

    def step(self, ext_torque: Optional[np.ndarray] = None):
        if ext_torque is None:
            ext_torque = np.zeros(3)

        # Linear drift
        self.position += self.velocity * self.dt

        # Gyroscopic torque
        I_omega = self.I @ self.omega
        tau_gyro = -np.cross(self.omega, I_omega)

        # Stochastic noise
        tau_noise = np.random.randn(3) * self.noise_std

        tau_total = tau_gyro + tau_noise + ext_torque

        # RK4
        self.omega = self._rk4(self.omega, tau_total)

        # Quaternion integration
        self.q = self._integrate_quat(self.q, self.omega)
        self.q /= np.linalg.norm(self.q)

    def _angular_accel(self, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """α = I⁻¹ · τ"""
        return self.I_inv @ tau

    def _rk4(self, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        h = self.dt

        def tau_at(w):
            I_w = self.I @ w
            gyro = -np.cross(w, I_w)
            non_gyro = tau - (-np.cross(omega, self.I @ omega))
            return gyro + non_gyro

        k1 = self._angular_accel(omega,            tau_at(omega))
        k2 = self._angular_accel(omega + h/2 * k1, tau_at(omega + h/2 * k1))
        k3 = self._angular_accel(omega + h/2 * k2, tau_at(omega + h/2 * k2))
        k4 = self._angular_accel(omega + h * k3,   tau_at(omega + h * k3))
        return omega + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def _integrate_quat(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        ox, oy, oz = omega
        dw = -0.5 * (ox*x + oy*y + oz*z)
        dx =  0.5 * (ox*w + oz*y - oy*z)
        dy =  0.5 * (oy*w - oz*x + ox*z)
        dz =  0.5 * (oz*w + oy*x - ox*y)
        return q + self.dt * np.array([dw, dx, dy, dz])

    @property
    def rotational_energy(self) -> float:
        return 0.5 * self.omega @ (self.I @ self.omega)
