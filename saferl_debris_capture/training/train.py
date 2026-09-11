"""
train.py
========
Main training script for the SafeRL debris capture project.

Supported training modes:
  1. PPO (unshielded)         — baseline
  2. SAC (unshielded)         — baseline
  3. PPO + Shield             — SafeRL
  4. SAC + Shield             — SafeRL

Usage
-----
# Train PPO baseline
python training/train.py --algo ppo --total-steps 5000000

# Train SAC with shield
python training/train.py --algo sac --shield --threshold 0.05

# Resume from checkpoint
python training/train.py --algo ppo --resume logs/ppo/checkpoints/ppo_debris_1000000_steps

Configuration is loaded from training/config/{algo}_config.yaml, and can be
overridden with command-line flags.

Experiment tracking is handled by Weights & Biases (wandb) if installed.
Set WANDB_PROJECT and WANDB_ENTITY environment variables before running.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

# Ensure project root is on the path when running as a script
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.debris_capture_env import DebrisCaptureEnv, DebrisCaptureEnvCfg
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.shielded_agent import ShieldedAgent
from shield.abstraction import AbstractionConfig, StateAbstraction
from shield.shield_query import PRISMShieldQuery
from shield.shield_wrapper import ShieldWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_cfg: DebrisCaptureEnvCfg, shield: bool = False, threshold: float = 0.05):
    """
    Factory function compatible with stable-baselines3 VecEnv.

    If `shield=True`, wraps the base environment with ShieldWrapper for
    during-training safety enforcement.
    """
    def _factory():
        env = DebrisCaptureEnv(env_cfg, num_envs=1, device="cpu")

        if shield:
            # Use offline table if available, else create uniform-safe placeholder
            table_path = PROJECT_ROOT / "shield" / "prism_models" / "p_collision_table.npy"
            if table_path.exists():
                query = PRISMShieldQuery.from_table(table_path, safety_threshold=threshold)
            else:
                logger.warning(
                    "Probability table not found at %s. "
                    "Using all-safe placeholder. Generate with:\n"
                    "  python -m shield.shield_query generate-table",
                    table_path,
                )
                query = PRISMShieldQuery(mode="table", safety_threshold=threshold)
            env = ShieldWrapper(env, query, safety_threshold=threshold)

        return env

    return _factory


# ─────────────────────────────────────────────────────────────────────────────
# Training modes
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(args, cfg: dict):
    """Train a PPO agent."""
    env_cfg    = DebrisCaptureEnvCfg()
    env_factory = make_env(env_cfg, shield=args.shield, threshold=args.threshold)
    log_dir    = f"logs/ppo{'_shielded' if args.shield else ''}"

    agent = PPOAgent(
        env_factory  = env_factory,
        n_envs       = args.n_envs,
        device       = args.device,
        log_dir      = log_dir,
        cfg          = cfg.get("ppo", {}),
    )

    logger.info("Starting PPO training — total_timesteps=%d", args.total_steps)
    agent.train(
        total_timesteps  = args.total_steps,
        eval_env_factory = make_env(env_cfg),
        eval_freq        = cfg.get("eval_freq", 50_000),
        save_freq        = cfg.get("save_freq", 100_000),
        tb_log_name      = "ppo_debris",
    )
    agent.save(os.path.join(log_dir, "final_model"))
    logger.info("PPO training complete. Model saved to %s", log_dir)
    return agent


def train_sac(args, cfg: dict):
    """Train a SAC agent."""
    env_cfg    = DebrisCaptureEnvCfg()
    env_factory = make_env(env_cfg, shield=args.shield, threshold=args.threshold)
    log_dir    = f"logs/sac{'_shielded' if args.shield else ''}"

    agent = SACAgent(
        env_factory = env_factory,
        device      = args.device,
        log_dir     = log_dir,
        cfg         = cfg.get("sac", {}),
    )

    logger.info("Starting SAC training — total_timesteps=%d", args.total_steps)
    agent.train(
        total_timesteps  = args.total_steps,
        eval_env_factory = make_env(env_cfg),
        eval_freq        = cfg.get("eval_freq", 20_000),
        save_freq        = cfg.get("save_freq", 100_000),
        tb_log_name      = "sac_debris",
    )
    agent.save(os.path.join(log_dir, "final_model"))
    logger.info("SAC training complete. Model saved to %s", log_dir)
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SafeRL debris capture agent."
    )
    parser.add_argument(
        "--algo", choices=["ppo", "sac"], default="ppo",
        help="RL algorithm to use (default: ppo)."
    )
    parser.add_argument(
        "--shield", action="store_true",
        help="Enable probabilistic shield during training."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Safety threshold for shield intervention (default: 0.05)."
    )
    parser.add_argument(
        "--total-steps", type=int, default=5_000_000,
        help="Total environment steps (default: 5_000_000)."
    )
    parser.add_argument(
        "--n-envs", type=int, default=8,
        help="Number of parallel environments (default: 8, PPO only)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device (default: cuda:0)."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides default config)."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    """Load YAML config; merge with CLI overrides."""
    if args.config:
        config_path = args.config
    else:
        config_path = PROJECT_ROOT / "training" / "config" / f"{args.algo}_config.yaml"

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning("Config file not found at %s — using defaults.", config_path)
        cfg = {}

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args)

    logger.info("=== SafeRL Debris Capture Training ===")
    logger.info("Algorithm : %s", args.algo.upper())
    logger.info("Shield    : %s (threshold=%.3f)", args.shield, args.threshold)
    logger.info("Device    : %s", args.device)
    logger.info("Steps     : %d", args.total_steps)

    # Try W&B logging if available
    try:
        import wandb
        wandb.init(
            project = os.environ.get("WANDB_PROJECT", "saferl-debris-capture"),
            entity  = os.environ.get("WANDB_ENTITY", None),
            config  = {
                "algo":       args.algo,
                "shield":     args.shield,
                "threshold":  args.threshold,
                "total_steps": args.total_steps,
                **cfg,
            },
            name = f"{args.algo}{'_shielded' if args.shield else ''}",
        )
        logger.info("Weights & Biases logging enabled.")
    except ImportError:
        logger.info("wandb not installed — skipping W&B logging.")

    if args.algo == "ppo":
        train_ppo(args, cfg)
    elif args.algo == "sac":
        train_sac(args, cfg)


if __name__ == "__main__":
    main()
