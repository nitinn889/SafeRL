"""Standardized evaluation protocol for SafeRL checkpoints.

Usage:
    python -m saferl.eval.evaluate --checkpoint saferl/eval/phase6b/saferl_phase6b.zip
    python -m saferl.eval.evaluate --checkpoint saferl/eval/phase7/saferl_phase7.zip --sensor-range 6.0

Protocol (phases 8+):
    - 500 episodes (sufficient to bound a 10-point gap at 95% confidence)
    - Deterministic action selection (argmax of policy mean)
    - Fixed seed (42 by default) for reproducibility
    - Reports goal rate, intervention rate, collision count
    - Outputs per-episode CSV for downstream analysis
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv
from saferl.training.dual_critic import ConstrainedPPO, DualCriticPolicy


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(checkpoint_path: str, n_episodes: int = 500, seed: int = 42,
             deterministic: bool = True, sensor_range=None, config_path=None):
    cfg = load_config(config_path)
    if sensor_range is not None:
        cfg["env"]["sensor_range"] = sensor_range

    device = cfg["training"].get("device", "cpu")
    ckpt_hash = file_md5(checkpoint_path)

    np.random.seed(seed)
    torch.manual_seed(seed)

    base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
    shield = SafetyShield.from_config(cfg)
    env = ShieldedEnv(base, shield)

    model = ConstrainedPPO.load(
        checkpoint_path, env=None, device=device,
        custom_objects={"policy_class": DualCriticPolicy},
    )

    rows = []
    goals, collisions, total_interventions, total_steps = 0, 0, 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0.0
        ep_steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        reached_goal = int(ep_reward > 0)
        collided = int(info.get("cost", 0) > 0)
        ivs = env.interventions
        checks = env.shield.n_checks
        iv_rate = ivs / max(checks, 1)

        goals += reached_goal
        collisions += collided
        total_interventions += ivs
        total_steps += checks

        rows.append(dict(
            episode=ep + 1,
            steps=ep_steps,
            reward=round(ep_reward, 3),
            reached_goal=reached_goal,
            collided=collided,
            interventions=ivs,
            shield_checks=checks,
            intervention_rate=round(iv_rate, 5),
        ))

        env.interventions = 0
        env.shield.n_checks = 0

    env.close()

    goal_rate = goals / n_episodes
    iv_rate = total_interventions / max(total_steps, 1)

    result = dict(
        checkpoint=checkpoint_path,
        checkpoint_md5=ckpt_hash,
        n_episodes=n_episodes,
        seed=seed,
        deterministic=deterministic,
        sensor_range=sensor_range,
        goal_rate=round(goal_rate, 4),
        intervention_rate=round(iv_rate, 5),
        goals=goals,
        collisions=collisions,
        total_interventions=total_interventions,
        total_steps=total_steps,
    )
    return result, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stochastic", action="store_true",
                    help="Use stochastic actions instead of deterministic")
    ap.add_argument("--sensor-range", type=float, default=None,
                    help="Override sensor range (None = use config default)")
    ap.add_argument("--out-dir", default=None,
                    help="Save per-episode CSV here (default: no file output)")
    args = ap.parse_args()

    print(f"Evaluating: {args.checkpoint}")
    print(f"  episodes={args.episodes}  seed={args.seed}  "
          f"deterministic={not args.stochastic}  sensor_range={args.sensor_range}")

    result, rows = evaluate(
        args.checkpoint,
        n_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        sensor_range=args.sensor_range,
    )

    print(f"\nResults ({result['n_episodes']} episodes, "
          f"{'deterministic' if result['deterministic'] else 'stochastic'}, "
          f"seed={result['seed']}):")
    print(f"  checkpoint: {result['checkpoint']}")
    print(f"  md5:        {result['checkpoint_md5']}")
    print(f"  goal rate:         {result['goal_rate']*100:.1f}% "
          f"({result['goals']}/{result['n_episodes']})")
    print(f"  intervention rate: {result['intervention_rate']*100:.2f}% "
          f"({result['total_interventions']}/{result['total_steps']})")
    print(f"  collisions:        {result['collisions']}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "eval_episodes.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        summary_path = out / "eval_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result.keys()))
            w.writeheader()
            w.writerow(result)
        print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
