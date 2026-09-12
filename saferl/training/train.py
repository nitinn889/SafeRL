"""Training entrypoint: python -m saferl.training.train [--config path]"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv
from saferl.training.metrics import MetricsCallback

sns.set_theme(style="darkgrid")


def make_env(cfg):
    """Factory function (not a lambda) so DummyVecEnv can create fresh envs."""
    def _factory():
        base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
        # The shield's motion model has to match the env it is guarding, so
        # derive its dynamics from the same config rather than trusting the
        # module defaults to stay in step with it.
        shield = SafetyShield.from_config(cfg)
        shielded = ShieldedEnv(base, shield)
        return Monitor(shielded, info_keywords=("cost",))
    return _factory


def plot_metrics(cb, output_path):
    if len(cb.episode_rewards) == 0:
        print("No episodes completed — try increasing timesteps.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SafeRL Training Metrics", fontsize=14, fontweight="bold")

    sns.lineplot(data=cb.episode_rewards, ax=axes[0], color="steelblue")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    sns.lineplot(data=np.cumsum(cb.episode_costs), ax=axes[1], color="crimson")
    axes[1].set_title("Cumulative Crashes (Cost)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Crashes")

    sns.lineplot(data=cb.episode_interventions, ax=axes[2], color="seagreen",
                 label="all interventions")
    if any(cb.episode_fallback_interventions):
        sns.lineplot(data=cb.episode_fallback_interventions, ax=axes[2],
                     color="darkorange", label="boxed-in fallbacks")
    axes[2].set_title("Shield Interventions per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Cumulative Interventions")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Graphs saved to {output_path}")


def train(config_path=None, model_out="saferl_model.zip"):
    cfg = load_config(config_path)
    # CPU by default and on purpose -- see the note in configs/default.yaml.
    device = cfg["training"].get("device", "cpu")
    print(f"Using device: {device}")

    v_env = DummyVecEnv([make_env(cfg)])
    model = PPO(cfg["training"]["policy"], v_env,
                verbose=cfg["training"]["verbose"], device=device)
    cb = MetricsCallback()

    timesteps = cfg["training"]["timesteps"]
    print(f"Training SafeRL agent for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=cb)
    print(f"Training done — {len(cb.episode_rewards)} episodes recorded.")

    inner = v_env.envs[0]
    while not isinstance(inner, ShieldedEnv) and hasattr(inner, "env"):
        inner = inner.env      # peel Monitor / any other wrapper
    if isinstance(inner, ShieldedEnv):
        sh = inner.shield
        print(f"Shield: {sh.n_checks} checks, {sh.n_triggered} triggered "
              f"({sh.n_substituted} substituted, {sh.n_fallback} boxed-in "
              f"fallbacks).")

    model.save(model_out)
    print(f"Model saved to {model_out}")

    plot_metrics(cb, cfg["training"]["output_plot"])
    v_env.close()
    return model, cb


def main():
    parser = argparse.ArgumentParser(description="Train the SafeRL PPO agent.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: saferl/configs/default.yaml)")
    parser.add_argument("--model-out", default="saferl_model.zip", help="Where to save the trained model")
    args = parser.parse_args()
    train(config_path=args.config, model_out=args.model_out)


if __name__ == "__main__":
    main()
