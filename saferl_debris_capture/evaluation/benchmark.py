"""
benchmark.py
============
Evaluation benchmark suite for the SafeRL debris capture project.

Runs a standardised set of evaluation episodes across all four conditions:
  1. PPO (unshielded)
  2. SAC (unshielded)
  3. PPO + Shield (SafeRL)
  4. SAC + Shield (SafeRL)

For each condition, records:
  - Task success rate
  - Episode reward (mean ± std)
  - Cumulative safety cost (collision count)
  - Shield intervention rate (conditions 3 & 4 only)
  - Episode length
  - Capture time (steps to successful grasp)

Results are saved to evaluation/results/ as CSV and JSON.

Usage
-----
python evaluation/benchmark.py \
    --ppo-model logs/ppo/final_model \
    --sac-model logs/sac/final_model \
    --ppo-shielded-model logs/ppo_shielded/final_model \
    --sac-shielded-model logs/sac_shielded/final_model \
    --n-episodes 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.debris_capture_env import DebrisCaptureEnv, DebrisCaptureEnvCfg
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.shielded_agent import ShieldedAgent
from shield.shield_query import PRISMShieldQuery

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Episode result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Stores the outcome of a single evaluation episode."""
    condition:        str
    episode:          int
    success:          bool
    total_reward:     float
    total_cost:       float
    episode_length:   int
    capture_step:     Optional[int]    # None if not captured
    interventions:    int              # shield interventions this episode
    collision_count:  int


@dataclass
class BenchmarkResult:
    """Aggregated results across N evaluation episodes for one condition."""
    condition:             str
    n_episodes:            int
    success_rate:          float
    mean_reward:           float
    std_reward:            float
    mean_cost:             float
    std_cost:              float
    mean_episode_length:   float
    mean_capture_step:     float       # among successful episodes only
    intervention_rate:     float
    collision_rate:        float       # fraction of episodes with any collision
    episodes:              List[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("episodes")  # keep summary only for top-level dict
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """
    Runs standardised evaluation across multiple agent conditions.

    Parameters
    ----------
    env_cfg : DebrisCaptureEnvCfg
        Environment configuration (same for all conditions).
    n_episodes : int
        Number of evaluation episodes per condition.
    results_dir : str
        Output directory for CSV and JSON files.
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        env_cfg: DebrisCaptureEnvCfg = None,
        n_episodes: int = 200,
        results_dir: str = "evaluation/results",
        seed: int = 42,
    ):
        self.env_cfg     = env_cfg or DebrisCaptureEnvCfg()
        self.n_episodes  = n_episodes
        self.results_dir = results_dir
        self.seed        = seed
        os.makedirs(results_dir, exist_ok=True)

    # ── Main evaluation loop ─────────────────────────────────────────────────

    def evaluate_agent(
        self,
        agent,
        condition_name: str,
        is_shielded: bool = False,
    ) -> BenchmarkResult:
        """
        Run `n_episodes` evaluation episodes for one agent condition.

        Parameters
        ----------
        agent         : any agent with a .predict(obs) → action method
        condition_name: str — label for this condition (e.g., 'PPO+Shield')
        is_shielded   : bool — if True, record intervention count from agent

        Returns
        -------
        BenchmarkResult
        """
        env = DebrisCaptureEnv(self.env_cfg, num_envs=1, device="cpu")
        episodes: List[EpisodeResult] = []

        for ep in range(self.n_episodes):
            np.random.seed(self.seed + ep)
            obs, info = env.reset()
            obs_np = obs.cpu().numpy()[0] if hasattr(obs, "cpu") else obs

            total_reward    = 0.0
            total_cost      = 0.0
            ep_length       = 0
            capture_step    = None
            collision_count = 0
            interventions_before = (
                getattr(agent, "total_interventions", 0) if is_shielded else 0
            )

            done, truncated = False, False
            while not (done or truncated):
                action  = agent.predict(obs_np, deterministic=True)
                obs, reward, done, truncated, info = env.step(
                    action if not hasattr(action, "__len__") else
                    np.array(action).reshape(1, -1)
                )
                obs_np       = obs.cpu().numpy()[0] if hasattr(obs, "cpu") else obs
                reward_val   = float(reward.sum() if hasattr(reward, "sum") else reward)
                cost_val     = float(info.get("cost", 0))

                total_reward += reward_val
                total_cost   += cost_val
                ep_length    += 1

                if cost_val > 0:
                    collision_count += 1

                if info.get("is_success", False) and capture_step is None:
                    capture_step = ep_length

            interventions_after = (
                getattr(agent, "total_interventions", 0) if is_shielded else 0
            )

            episodes.append(EpisodeResult(
                condition       = condition_name,
                episode         = ep,
                success         = bool(capture_step is not None),
                total_reward    = total_reward,
                total_cost      = total_cost,
                episode_length  = ep_length,
                capture_step    = capture_step,
                interventions   = interventions_after - interventions_before,
                collision_count = collision_count,
            ))

        env.close()
        return self._aggregate(condition_name, episodes)

    def _aggregate(
        self,
        condition: str,
        episodes: List[EpisodeResult],
    ) -> BenchmarkResult:
        """Compute summary statistics from raw episode results."""
        rewards       = [e.total_reward   for e in episodes]
        costs         = [e.total_cost     for e in episodes]
        lengths       = [e.episode_length for e in episodes]
        success_eps   = [e for e in episodes if e.success]
        capture_steps = [e.capture_step   for e in success_eps if e.capture_step]
        intervs       = [e.interventions  for e in episodes]
        total_queries = sum(e.episode_length for e in episodes)

        return BenchmarkResult(
            condition           = condition,
            n_episodes          = len(episodes),
            success_rate        = len(success_eps) / len(episodes),
            mean_reward         = float(np.mean(rewards)),
            std_reward          = float(np.std(rewards)),
            mean_cost           = float(np.mean(costs)),
            std_cost            = float(np.std(costs)),
            mean_episode_length = float(np.mean(lengths)),
            mean_capture_step   = float(np.mean(capture_steps)) if capture_steps else float("nan"),
            intervention_rate   = sum(intervs) / total_queries if total_queries > 0 else 0.0,
            collision_rate      = sum(1 for e in episodes if e.collision_count > 0) / len(episodes),
            episodes            = episodes,
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def save_results(self, results: List[BenchmarkResult]):
        """Save all results to CSV and JSON."""
        # Summary table
        summary = pd.DataFrame([r.to_dict() for r in results])
        csv_path = os.path.join(self.results_dir, "benchmark_summary.csv")
        summary.to_csv(csv_path, index=False)
        logger.info("Summary CSV saved: %s", csv_path)

        # Full episode-level data
        all_episodes = []
        for r in results:
            all_episodes.extend([asdict(e) for e in r.episodes])
        ep_df = pd.DataFrame(all_episodes)
        ep_csv = os.path.join(self.results_dir, "benchmark_episodes.csv")
        ep_df.to_csv(ep_csv, index=False)

        # JSON dump
        json_path = os.path.join(self.results_dir, "benchmark_summary.json")
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("JSON summary saved: %s", json_path)

    def print_summary(self, results: List[BenchmarkResult]):
        """Print a formatted results table to stdout."""
        header = (
            f"{'Condition':<25} {'Success%':>10} {'Reward':>12} "
            f"{'Cost':>12} {'Interv%':>10} {'Collision%':>12}"
        )
        print("\n" + "=" * len(header))
        print("  BENCHMARK RESULTS — SafeRL Debris Capture")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.condition:<25} "
                f"{r.success_rate*100:>9.1f}% "
                f"{r.mean_reward:>12.1f} "
                f"{r.mean_cost:>12.3f} "
                f"{r.intervention_rate*100:>9.1f}% "
                f"{r.collision_rate*100:>11.1f}%"
            )
        print("=" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument("--ppo-model",          type=str, default=None)
    parser.add_argument("--sac-model",          type=str, default=None)
    parser.add_argument("--ppo-shielded-model", type=str, default=None)
    parser.add_argument("--sac-shielded-model", type=str, default=None)
    parser.add_argument("--n-episodes",         type=int, default=200)
    parser.add_argument("--device",             type=str, default="cpu")
    parser.add_argument("--threshold",          type=float, default=0.05)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args    = parse_args()
    suite   = BenchmarkSuite(n_episodes=args.n_episodes)
    results = []

    def env_factory():
        return DebrisCaptureEnv(DebrisCaptureEnvCfg(), num_envs=1, device="cpu")

    table_path = PROJECT_ROOT / "shield" / "prism_models" / "p_collision_table.npy"

    def get_shield_query():
        if table_path.exists():
            return PRISMShieldQuery.from_table(table_path, args.threshold)
        return PRISMShieldQuery(mode="table", safety_threshold=args.threshold)

    if args.ppo_model:
        agent = PPOAgent.load(args.ppo_model, env_factory, device=args.device)
        results.append(suite.evaluate_agent(agent, "PPO (unshielded)"))

    if args.sac_model:
        agent = SACAgent.load(args.sac_model, env_factory, device=args.device)
        results.append(suite.evaluate_agent(agent, "SAC (unshielded)"))

    if args.ppo_shielded_model:
        base  = PPOAgent.load(args.ppo_shielded_model, env_factory, device=args.device)
        agent = ShieldedAgent(base, get_shield_query(), args.threshold)
        results.append(suite.evaluate_agent(agent, "PPO + Shield", is_shielded=True))

    if args.sac_shielded_model:
        base  = SACAgent.load(args.sac_shielded_model, env_factory, device=args.device)
        agent = ShieldedAgent(base, get_shield_query(), args.threshold)
        results.append(suite.evaluate_agent(agent, "SAC + Shield", is_shielded=True))

    if results:
        suite.print_summary(results)
        suite.save_results(results)
    else:
        print("No agent models specified. Use --ppo-model / --sac-model etc.")


if __name__ == "__main__":
    main()
