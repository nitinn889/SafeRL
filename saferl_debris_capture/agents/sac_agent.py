"""
sac_agent.py
============
Soft Actor-Critic (SAC) agent for the debris capture task.

SAC is an off-policy, maximum-entropy RL algorithm particularly well-suited
to continuous action spaces with complex reward landscapes. It maintains a
separate entropy temperature α that is automatically tuned to balance
exploration and exploitation.

Network architecture:
  Actor  : 36 → 256 → 256 → 128 → mean + log_std (6-D each)
  Critic : (36 + 6) → 256 → 256 → 128 → Q-value  (two critics for clipped Q)

References
----------
- Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy
  Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
- Haarnoja et al. (2019). "Soft Actor-Critic Algorithms and Applications."
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gymnasium as gym


# ─────────────────────────────────────────────────────────────────────────────
# Custom feature extractor (shared with SAC actor + critic)
# ─────────────────────────────────────────────────────────────────────────────

class DebrisSACNet(BaseFeaturesExtractor):
    """
    Encoder for the 36-D debris capture observation.

    Uses batch normalisation instead of layer normalisation because SAC
    operates on replay-buffer mini-batches (not full rollouts like PPO).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


# ─────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────────────────────────────────────

class SACAgent:
    """
    SAC agent with automatic entropy tuning and experience replay.

    SAC is preferred over PPO when:
      - Sample efficiency matters (off-policy reuse of past experience).
      - Exploration is critical (maximum-entropy objective).
      - The task has a smooth, continuous reward landscape.

    For the debris capture task, SAC is particularly effective at learning
    the fine-grained alignment manoeuvres required for grasping.

    Parameters
    ----------
    env_factory : callable
        Returns a fresh gymnasium environment instance.
    device : str
        Torch device.
    log_dir : str
        Directory for TensorBoard logs and checkpoints.
    cfg : dict
        Override default hyperparameters.
    """

    DEFAULT_CFG: Dict[str, Any] = {
        "learning_rate":       3e-4,
        "buffer_size":         1_000_000,
        "learning_starts":     10_000,   # steps before first gradient update
        "batch_size":          512,
        "tau":                 0.005,    # soft target update coefficient
        "gamma":               0.99,
        "train_freq":          1,        # update every N steps
        "gradient_steps":      1,
        "ent_coef":            "auto",   # automatic entropy tuning
        "target_entropy":      "auto",
        "use_sde":             False,    # state-dependent exploration
        "net_arch":            [256, 256, 128],
        "action_noise_std":    0.0,      # optional exploration noise
    }

    def __init__(
        self,
        env_factory,
        device: str = "cuda:0",
        log_dir: str = "logs/sac",
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir  = log_dir
        self.device   = device
        self.hparams  = {**self.DEFAULT_CFG, **(cfg or {})}

        os.makedirs(log_dir, exist_ok=True)

        # SAC requires a single-env interface (off-policy)
        self.env = VecNormalize(
            DummyVecEnv([env_factory]),
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=self.hparams["gamma"],
        )

        policy_kwargs = {
            "features_extractor_class":  DebrisSACNet,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch":                  self.hparams["net_arch"],
            "activation_fn":             nn.ReLU,
        }

        # Optional exploration noise
        action_noise = None
        if self.hparams["action_noise_std"] > 0:
            n_actions = self.env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean  = np.zeros(n_actions),
                sigma = self.hparams["action_noise_std"] * np.ones(n_actions),
            )

        self.model = SAC(
            policy           = "MlpPolicy",
            env              = self.env,
            learning_rate    = self.hparams["learning_rate"],
            buffer_size      = self.hparams["buffer_size"],
            learning_starts  = self.hparams["learning_starts"],
            batch_size       = self.hparams["batch_size"],
            tau              = self.hparams["tau"],
            gamma            = self.hparams["gamma"],
            train_freq       = self.hparams["train_freq"],
            gradient_steps   = self.hparams["gradient_steps"],
            ent_coef         = self.hparams["ent_coef"],
            target_entropy   = self.hparams["target_entropy"],
            use_sde          = self.hparams["use_sde"],
            action_noise     = action_noise,
            policy_kwargs    = policy_kwargs,
            tensorboard_log  = log_dir,
            device           = device,
            verbose          = 1,
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 3_000_000,
        eval_env_factory = None,
        eval_freq: int = 20_000,
        save_freq: int = 100_000,
        tb_log_name: str = "sac_debris",
    ) -> "SACAgent":
        """Train the SAC agent."""
        callbacks = []

        checkpoint_cb = CheckpointCallback(
            save_freq   = save_freq,
            save_path   = os.path.join(self.log_dir, "checkpoints"),
            name_prefix = "sac_debris",
            save_vecnormalize = True,
        )
        callbacks.append(checkpoint_cb)

        if eval_env_factory is not None:
            eval_env = VecNormalize(
                DummyVecEnv([eval_env_factory]),
                training=False,
                norm_reward=False,
            )
            eval_cb = EvalCallback(
                eval_env,
                eval_freq            = eval_freq,
                n_eval_episodes      = 20,
                best_model_save_path = os.path.join(self.log_dir, "best_model"),
                log_path             = self.log_dir,
                deterministic        = True,
            )
            callbacks.append(eval_cb)

        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = CallbackList(callbacks),
            tb_log_name         = tb_log_name,
            reset_num_timesteps = True,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        self.model.save(path)
        self.env.save(path + "_vecnorm.pkl")

    @classmethod
    def load(
        cls,
        path: str,
        env_factory,
        device: str = "cpu",
    ) -> "SACAgent":
        agent = cls.__new__(cls)
        agent.log_dir = os.path.dirname(path)
        agent.device  = device

        vec_env = VecNormalize.load(
            path + "_vecnorm.pkl",
            DummyVecEnv([env_factory])
        )
        vec_env.training    = False
        vec_env.norm_reward = False

        agent.env   = vec_env
        agent.model = SAC.load(path, env=vec_env, device=device)
        return agent
