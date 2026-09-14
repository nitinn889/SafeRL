"""Phase 6b: two-critic constrained PPO with curriculum and warm-start.

Run:  python -m saferl.training.train_constrained --timesteps 200000

Fixes the phase-6 collapse by combining three measures:
  1. Separate cost-value head (DualCriticPolicy) so the policy gradient can
     credit-assign task reward and intervention cost independently.
  2. Curriculum on the constraint target: starts loose (0.25, above the
     unconstrained policy's ~0.19 natural rate) and ramps linearly to the
     final target (0.05) over training.
  3. Warm-start from the 400k unconstrained checkpoint so the policy begins
     with a working navigation strategy instead of exploring from scratch
     under a constraint it can trivially satisfy by freezing.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv
from saferl.training.dual_critic import ConstrainedPPO, DualCriticPolicy


class Phase6bCallback(BaseCallback):
    """Per-episode metrics for the two-critic constrained training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rows: list[dict] = []
        self._reset_acc()

    def _reset_acc(self):
        self._reward = 0.0
        self._cost = 0.0
        self._fallback = 0.0
        self._collision = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = float(self.locals["rewards"][0])
        self._reward += reward
        self._cost += float(info.get("shield_cost", 0.0))
        self._fallback += float(info.get("shield_fallback", 0.0))
        self._collision += float(info.get("cost", 0.0))

        if "episode" in info:
            steps = int(info["episode"]["l"])
            lam = self.model.lam if hasattr(self.model, "lam") else 0.0
            target = (self.model.current_target_rate
                      if hasattr(self.model, "current_target_rate") else 0.0)
            self.rows.append(dict(
                timestep=int(self.num_timesteps),
                episode=len(self.rows) + 1,
                steps=steps,
                task_reward=round(self._reward, 3),
                interventions=int(self._cost),
                fallbacks=int(self._fallback),
                intervention_rate=round(self._cost / max(steps, 1), 5),
                collided=int(self._collision > 0),
                reached_goal=int(self._reward > 0),
                lam=round(float(lam), 4),
                target_rate=round(float(target), 4),
            ))
            self._reset_acc()
        return True


def build_env(cfg, seed):
    base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
    shielded = ShieldedEnv(base, SafetyShield.from_config(cfg))
    monitored = Monitor(shielded, info_keywords=("cost",))
    return DummyVecEnv([lambda: monitored])


def quick_eval(model, cfg, n_episodes=100):
    """Run the policy stochastically (matching training) and return goal rate."""
    base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
    shield = SafetyShield.from_config(cfg)
    env = ShieldedEnv(base, shield)
    goals = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        if ep_reward > 0:
            goals += 1
    env.close()
    return goals / n_episodes


def warm_start(model, checkpoint_path, device="cpu"):
    """Copy matching weights from a standard PPO checkpoint."""
    old = PPO.load(checkpoint_path, device=device)
    old_sd = old.policy.state_dict()
    new_sd = model.policy.state_dict()
    loaded = []
    for key in old_sd:
        if key in new_sd and old_sd[key].shape == new_sd[key].shape:
            new_sd[key] = old_sd[key]
            loaded.append(key)
    model.policy.load_state_dict(new_sd)
    skipped = [k for k in new_sd if k not in old_sd]
    print(f"Warm-start: loaded {len(loaded)}/{len(new_sd)} params, "
          f"fresh init: {skipped}")
    return loaded, skipped


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarise(name, rows, fraction=0.2):
    if not rows:
        return {"arm": name, "episodes": 0}
    k = max(1, int(len(rows) * fraction))
    head, tail = rows[:k], rows[-k:]

    def agg(chunk, key):
        return float(np.mean([r[key] for r in chunk]))

    return {
        "arm": name,
        "episodes": len(rows),
        "first_rate": agg(head, "intervention_rate"),
        "last_rate": agg(tail, "intervention_rate"),
        "first_task_reward": agg(head, "task_reward"),
        "last_task_reward": agg(tail, "task_reward"),
        "first_ivs_per_ep": agg(head, "interventions"),
        "last_ivs_per_ep": agg(tail, "interventions"),
        "first_steps": agg(head, "steps"),
        "last_steps": agg(tail, "steps"),
        "goals": sum(r["reached_goal"] for r in rows),
        "collisions": sum(r["collided"] for r in rows),
        "final_lambda": rows[-1]["lam"],
        "goal_rate_pct": round(100.0 * sum(r["reached_goal"] for r in rows) / len(rows), 1),
    }


def plot_phase6b(rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Phase 6b: two-critic constrained PPO (warm-started)",
                 fontsize=14, fontweight="bold")

    def smooth(v, w=31):
        if len(v) < w:
            return np.asarray(v, dtype=float)
        return np.convolve(v, np.ones(w) / w, mode="valid")

    ts = [r["timestep"] for r in rows]

    panels = [
        (axes[0, 0], "task_reward", "Task reward"),
        (axes[0, 1], "intervention_rate", "Intervention rate"),
        (axes[0, 2], "steps", "Episode length"),
        (axes[1, 0], "lam", "Lambda"),
        (axes[1, 1], "target_rate", "Curriculum target rate"),
    ]
    for ax, key, title in panels:
        y = smooth([r[key] for r in rows])
        ax.plot(ts[len(ts) - len(y):], y, color="crimson")
        ax.set_title(title)
        ax.set_xlabel("timestep")

    # Goal rate (cumulative)
    ax = axes[1, 2]
    cum_goals = np.cumsum([r["reached_goal"] for r in rows])
    cum_rate = cum_goals / np.arange(1, len(rows) + 1) * 100
    ax.plot(ts, cum_rate, color="seagreen")
    ax.set_title("Cumulative goal rate %")
    ax.set_xlabel("timestep")
    ax.set_ylabel("%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    print(f"Plot saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="saferl/eval/phase6b")
    ap.add_argument("--checkpoint",
                     default="saferl/eval/phase6/saferl_unconstrained.zip",
                     help="Checkpoint to warm-start from, or 'none' to train "
                          "from scratch")
    ap.add_argument("--config", default=None,
                     help="Config YAML (default: the planar default.yaml)")
    ap.add_argument("--lambda-max", type=float, default=None,
                     help="Override constrained.lambda_max. 0 pins lambda at 0, "
                          "which is unconstrained PPO through the same "
                          "two-critic plumbing -- how a from-scratch pretrain "
                          "is run without a separate trainer.")
    ap.add_argument("--model-name", default="saferl_phase6b",
                     help="Saved checkpoint file name (without .zip)")
    ap.add_argument("--target-start", type=float, default=0.25,
                     help="Curriculum start (loose budget)")
    ap.add_argument("--target-end", type=float, default=0.05,
                     help="Curriculum end (tight budget)")
    ap.add_argument("--eval-episodes", type=int, default=100,
                     help="Episodes for warm-start verification")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["training"].get("device", "cpu")
    ccfg = cfg.get("constrained", {})
    lambda_max = (args.lambda_max if args.lambda_max is not None
                  else ccfg.get("lambda_max", 5.0))

    print(f"config={args.config or 'default.yaml'}  dims={cfg['env'].get('dims', 2)}  "
          f"hazards={cfg['env']['max_hazards']}  lambda_max={lambda_max}")
    print(f"device={device}  timesteps={args.timesteps}  seed={args.seed}")
    print(f"curriculum: {args.target_start} -> {args.target_end}")
    print(f"checkpoint: {args.checkpoint}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    venv = build_env(cfg, args.seed)

    model = ConstrainedPPO(
        DualCriticPolicy, venv,
        target_rate_start=args.target_start,
        target_rate_end=args.target_end,
        constraint_lambda_lr=ccfg.get("lambda_lr", 0.1),
        constraint_lambda_max=lambda_max,
        constraint_ema_beta=ccfg.get("ema_beta", 0.9),
        verbose=1, device=device, seed=args.seed,
    )

    if args.checkpoint.lower() == "none":
        print("\nNo warm-start: training from scratch.")
    else:
        # --- Warm-start ---
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"ERROR: checkpoint not found: {checkpoint}")
            venv.close()
            return
        loaded, skipped = warm_start(model, str(checkpoint), device=device)

        # --- Verify warm-start ---
        print(f"\nVerifying warm-start ({args.eval_episodes} episodes, stochastic)...")
        goal_rate = quick_eval(model, cfg, n_episodes=args.eval_episodes)
        print(f"Warm-start goal rate: {goal_rate * 100:.1f}%")
        if goal_rate < 0.30:
            print("WARNING: warm-start goal rate < 30%, checkpoint may not have "
                  "loaded correctly. Proceeding but results may not be meaningful.")

    # --- Train ---
    print(f"\nTraining constrained policy for {args.timesteps} steps...")
    cb = Phase6bCallback()
    model.learn(total_timesteps=args.timesteps, callback=cb)

    # --- Save ---
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(cb.rows, out / "phase6b_episodes.csv")
    model.save(out / f"{args.model_name}.zip")

    s = summarise("phase6b", cb.rows)
    print(f"\n[phase6b] episodes={s['episodes']} goals={s['goals']} "
          f"({s['goal_rate_pct']}%) collisions={s['collisions']} "
          f"rate {s['first_rate']:.4f} -> {s['last_rate']:.4f} "
          f"task_reward {s['first_task_reward']:.1f} -> {s['last_task_reward']:.1f} "
          f"lambda={s['final_lambda']}")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        w.writeheader()
        w.writerow(s)

    plot_phase6b(cb.rows, out / "phase6b_constrained.png")
    venv.close()


if __name__ == "__main__":
    main()
