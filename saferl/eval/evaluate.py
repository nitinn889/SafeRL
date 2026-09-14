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


def evaluate_model(model, n_episodes: int = 500, seed: int = 42,
                   deterministic: bool = True, sensor_range=None,
                   config_path=None, preserve_rng: bool = False):
    """Run the protocol against an already-loaded model, for in-training probes.

    Shares the episode loop and every protocol parameter with `evaluate()`
    (deterministic actions, fixed seed, fixed episode count). It differs in
    one respect: the model already exists, so it is not loaded inside the
    seeded region. That shifts which random episodes get drawn, so a number
    from here is NOT bit-comparable with one from `evaluate()` and must not
    be reported as an authoritative figure. Use it for convergence
    monitoring, where a consistently-sampled window is what matters; use
    `evaluate()` for anything that goes in the README.

    preserve_rng: save/restore the global numpy+torch RNG state around the
    run, so seeding the probe does not perturb the training run's own
    exploration stream.
    """
    if preserve_rng:
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()

    try:
        cfg = load_config(config_path)
        # Evaluation always measures the real goal. A goal curriculum is a
        # training aid; letting it move the goal here would score a checkpoint
        # on easier tasks than the one being reported.
        cfg["env"]["goal_curriculum"] = False
        if sensor_range is not None:
            cfg["env"]["sensor_range"] = sensor_range

        np.random.seed(seed)
        torch.manual_seed(seed)

        base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
        shield = SafetyShield.from_config(cfg)
        env = ShieldedEnv(base, shield)

        return _run_episodes(model, env, n_episodes, seed, deterministic,
                             sensor_range)
    finally:
        if preserve_rng:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)


def evaluate(checkpoint_path: str, n_episodes: int = 500, seed: int = 42,
             deterministic: bool = True, sensor_range=None, config_path=None,
             curriculum=None):
    """Authoritative protocol entry point, for reporting a checkpoint's number.

    The statement order here (seed -> build env -> load model) is load-bearing
    and must not be rearranged. `ConstrainedPPO.load()` draws from the global
    RNG, so moving it relative to the seed shifts the env's state at the first
    reset and silently changes which episodes get sampled. This exact order is
    what phase 8's published figures were produced under; keeping it means a
    rerun reproduces them episode-for-episode.

    `curriculum` controls the difficulty the run is measured at, and it
    matters more than it looks. The config ships `curriculum: true` for
    training, and a *fresh* env restarts that ramp at episode 0 --
    `num_hazards = min(max_hazards, 1 + episode_count // 50)` -- so a default
    500-episode eval spends its first 200 episodes facing 1-4 hazards and
    only the last 300 at the full 5. Roughly 40% of the run is therefore
    easier than the scenario the demo actually shows. Leave this None to
    inherit the config (what every figure published before phase 10 used),
    or pass False to hold all `max_hazards` active from episode 0 and
    measure full fixed difficulty. Phase 10 reports both, labelled.
    """
    cfg = load_config(config_path)
    # Evaluation always measures the real goal. A goal curriculum is a
    # training aid; letting it move the goal here would score a checkpoint
    # on easier tasks than the one being reported.
    cfg["env"]["goal_curriculum"] = False
    if sensor_range is not None:
        cfg["env"]["sensor_range"] = sensor_range
    if curriculum is not None:
        cfg["env"]["curriculum"] = curriculum

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

    result, rows = _run_episodes(model, env, n_episodes, seed, deterministic,
                                 sensor_range)
    result["checkpoint"] = checkpoint_path
    result["checkpoint_md5"] = ckpt_hash
    # Stamp the difficulty condition into the result itself. Phase 10 found
    # two very different numbers for the same checkpoint depending on this
    # flag; recording it means a stored eval_summary.csv can never again be
    # read without knowing which condition produced it.
    result["curriculum"] = bool(cfg["env"].get("curriculum"))
    result["max_hazards"] = cfg["env"].get("max_hazards")
    result["dims"] = cfg["env"].get("dims", 2)
    return result, rows


def _run_episodes(model, env, n_episodes, seed, deterministic, sensor_range):
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
        checkpoint="",
        checkpoint_md5="",
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
    ap.add_argument("--config", default=None,
                    help="Config YAML (default: saferl/configs/default.yaml, the "
                         "planar env). Use saferl/configs/space3d.yaml for 3D.")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="Hold all max_hazards active from episode 0 (full "
                         "fixed difficulty). Without this the run inherits "
                         "the config's training curriculum, which restarts "
                         "its 1->5 hazard ramp at episode 0 and leaves ~40%% "
                         "of a 500-episode run below full difficulty.")
    args = ap.parse_args()

    curriculum = False if args.no_curriculum else None

    print(f"Evaluating: {args.checkpoint}")
    print(f"  episodes={args.episodes}  seed={args.seed}  "
          f"deterministic={not args.stochastic}  sensor_range={args.sensor_range}")
    print(f"  difficulty: {'full fixed (curriculum off)' if args.no_curriculum else 'config default (curriculum inherited)'}")

    result, rows = evaluate(
        args.checkpoint,
        n_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        sensor_range=args.sensor_range,
        config_path=args.config,
        curriculum=curriculum,
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
