"""
ppo_agent.py
============
Proximal Policy Optimisation (PPO) agent for the debris capture task.

Uses Stable-Baselines3's PPO implementation with a custom MLP policy
architecture tuned for the 36-D observation and 6-D continuous action space.

Network architecture:
  Actor  (policy): 36 → 256 → 256 → 128 → 6  (tanh output)
  Critic (value) : 36 → 256 → 256 → 128 → 1

Hyperparameters follow the recommendations in:
  Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
  Andrychowicz et al. (2021). "What Matters for On-Policy Deep Actor-Critic Methods."
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import gymnasium as gym
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Custom network architecture
# ─────────────────────────────────────────────────────────────────────────────

class DebrisCaptureNet(BaseFeaturesExtractor):
    """
    Custom feature extractor for the debris capture observation.

    Separates the observation into structured groups and processes them
    with shared trunk layers before the actor/critic heads.

    Observation layout (36-D):
      [0:3]   debris_rel_pos
      [3:7]   debris_quat
      [7:10]  debris_omega
      [10:17] EE pose (pos + quat)
      [17:31] joint states (pos + vel)
      [31:34] contact_force
      [34:36] time + curriculum

    Feature dim: 128 (compressed representation)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, features_dim),
            nn.ELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.trunk(observations)


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent with a custom network, VecNormalize, and checkpointing.

    Parameters
    ----------
    env_factory : callable
        Returns a fresh gymnasium environment instance (used for DummyVecEnv).
    n_envs : int
        Number of parallel environments.
    device : str
        Torch device ('cpu', 'cuda:0', etc.).
    log_dir : str
        Directory for TensorBoard logs and checkpoints.
    cfg : dict
        Override default hyperparameters (see DEFAULT_CFG).
    """

    DEFAULT_CFG: Dict[str, Any] = {
        "learning_rate":     3e-4,
        "n_steps":           2048,      # steps per env per update
        "batch_size":        256,
        "n_epochs":          10,
        "gamma":             0.99,
        "gae_lambda":        0.95,
        "clip_range":        0.2,
        "ent_coef":          0.005,
        "vf_coef":           0.5,
        "max_grad_norm":     0.5,
        "target_kl":         0.01,
        "net_arch":          [256, 256, 128],
    }

    def __init__(
        self,
        env_factory,
        n_envs: int = 8,
        device: str = "cuda:0",
        log_dir: str = "logs/ppo",
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir     = log_dir
        self.device      = device
        self.n_envs      = n_envs
        self.hparams     = {**self.DEFAULT_CFG, **(cfg or {})}

        os.makedirs(log_dir, exist_ok=True)

        # Build vectorised environment
        self.vec_env = self._make_vec_env(env_factory, n_envs)
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=self.hparams["gamma"],
        )

        policy_kwargs = {
            "features_extractor_class":  DebrisCaptureNet,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch":                  self.hparams["net_arch"],
            "activation_fn":             nn.ELU,
        }

        self.model = PPO(
            policy             = "MlpPolicy",
            env                = self.vec_env,
            learning_rate      = self.hparams["learning_rate"],
            n_steps            = self.hparams["n_steps"],
            batch_size         = self.hparams["batch_size"],
            n_epochs           = self.hparams["n_epochs"],
            gamma              = self.hparams["gamma"],
            gae_lambda         = self.hparams["gae_lambda"],
            clip_range         = self.hparams["clip_range"],
            ent_coef           = self.hparams["ent_coef"],
            vf_coef            = self.hparams["vf_coef"],
            max_grad_norm      = self.hparams["max_grad_norm"],
            target_kl          = self.hparams["target_kl"],
            policy_kwargs      = policy_kwargs,
            tensorboard_log    = log_dir,
            device             = device,
            verbose            = 1,
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 5_000_000,
        eval_env_factory = None,
        eval_freq: int = 50_000,
        save_freq: int = 100_000,
        tb_log_name: str = "ppo_debris",
    ) -> "PPOAgent":
        """
        Train the PPO agent.

        Parameters
        ----------
        total_timesteps  : int    — total environment steps
        eval_env_factory : callable | None — if provided, enables EvalCallback
        eval_freq        : int    — evaluate every N steps per env
        save_freq        : int    — checkpoint every N steps
        tb_log_name      : str    — TensorBoard run name

        Returns
        -------
        self (for method chaining)
        """
        callbacks = []

        # Checkpointing
        checkpoint_cb = CheckpointCallback(
            save_freq = save_freq,
            save_path = os.path.join(self.log_dir, "checkpoints"),
            name_prefix = "ppo_debris",
            save_vecnormalize = True,
        )
        callbacks.append(checkpoint_cb)

        # Evaluation
        if eval_env_factory is not None:
            eval_env = VecNormalize(
                DummyVecEnv([eval_env_factory]),
                training=False,
                norm_reward=False,
            )
            eval_cb = EvalCallback(
                eval_env,
                eval_freq          = eval_freq,
                n_eval_episodes    = 20,
                best_model_save_path = os.path.join(self.log_dir, "best_model"),
                log_path           = self.log_dir,
                deterministic      = True,
            )
            callbacks.append(eval_cb)

        self.model.learn(
            total_timesteps = total_timesteps,
            callback        = CallbackList(callbacks),
            tb_log_name     = tb_log_name,
            reset_num_timesteps = True,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action given observation.

        Parameters
        ----------
        obs : (OBS_DIM,) or (n_envs, OBS_DIM)
        deterministic : bool — use mean policy (no exploration noise)

        Returns
        -------
        action : np.ndarray
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model and VecNormalize statistics."""
        self.model.save(path)
        self.vec_env.save(path + "_vecnorm.pkl")

    @classmethod
    def load(
        cls,
        path: str,
        env_factory,
        n_envs: int = 1,
        device: str = "cpu",
    ) -> "PPOAgent":
        """Load a saved PPO agent."""
        agent = cls.__new__(cls)
        agent.log_dir = os.path.dirname(path)
        agent.device  = device
        agent.n_envs  = n_envs

        vec_env = cls._make_vec_env(env_factory, n_envs)
        vec_env = VecNormalize.load(path + "_vecnorm.pkl", vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        agent.vec_env = vec_env
        agent.model   = PPO.load(path, env=vec_env, device=device)
        return agent

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_vec_env(env_factory, n_envs: int):
        if n_envs == 1:
            return DummyVecEnv([env_factory])
        return SubprocVecEnv([env_factory] * n_envs)
