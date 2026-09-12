"""Constrained vs unconstrained PPO under the safety shield.

Run:  python -m saferl.training.train_constrained --timesteps 400000

Trains two arms that differ only in whether the Lagrangian penalty is live,
and writes a per-episode CSV for each so the *trend* in intervention rate is
inspectable, not just the final number. Phase 1's promised success signal was
"interventions per episode should trend toward zero as the policy internalises
the shield's boundary"; this is the apparatus that can show that or refute it.
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
from saferl.training.constrained import CostPenaltyWrapper


class ConstrainedMetricsCallback(BaseCallback):
    """Per-episode row: task reward, penalised reward, cost, lambda, outcome."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rows = []
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._task_r = 0.0
        self._penalty = 0.0
        self._cost = 0.0
        self._fallback = 0.0
        self._collision = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._task_r += float(info.get("task_reward", 0.0))
        self._penalty += float(info.get("penalty", 0.0))
        self._cost += float(info.get("shield_cost", 0.0))
        self._fallback += float(info.get("shield_fallback", 0.0))
        self._collision += float(info.get("cost", 0.0))

        if "episode" in info:
            steps = int(info["episode"]["l"])
            self.rows.append(dict(
                timestep=int(self.num_timesteps),
                episode=len(self.rows) + 1,
                steps=steps,
                task_reward=round(self._task_r, 3),
                penalised_reward=round(float(info["episode"]["r"]), 3),
                penalty=round(self._penalty, 3),
                interventions=int(self._cost),
                fallbacks=int(self._fallback),
                intervention_rate=round(self._cost / max(steps, 1), 5),
                collided=int(self._collision > 0),
                reached_goal=int(self._task_r > 0),
                lam=round(float(info.get("lambda", 0.0)), 4),
            ))
            self._reset_accumulators()
        return True


def build_env(cfg, constrained, seed):
    base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
    shielded = ShieldedEnv(base, SafetyShield.from_config(cfg))
    ccfg = cfg.get("constrained", {})
    penalised = CostPenaltyWrapper(
        shielded,
        target_rate=ccfg.get("target_rate", 0.002),
        lambda_lr=ccfg.get("lambda_lr", 2.0),
        lambda_init=ccfg.get("lambda_init", 0.0),
        lambda_max=ccfg.get("lambda_max", 50.0),
        ema_beta=ccfg.get("ema_beta", 0.9),
        enabled=constrained,
    )
    monitored = Monitor(penalised, info_keywords=("cost",))
    return penalised, DummyVecEnv([lambda: monitored])


def run_arm(cfg, constrained, timesteps, seed, device, out_dir):
    name = "constrained" if constrained else "unconstrained"
    np.random.seed(seed)
    torch.manual_seed(seed)
    wrapper, v_env = build_env(cfg, constrained, seed)
    model = PPO(cfg["training"]["policy"], v_env, verbose=0,
                device=device, seed=seed)
    cb = ConstrainedMetricsCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    out = Path(out_dir) / f"{name}_episodes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if cb.rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cb.rows[0].keys()))
            w.writeheader()
            w.writerows(cb.rows)

    shield = v_env.envs[0].env.env.shield
    model.save(Path(out_dir) / f"saferl_{name}.zip")
    v_env.close()
    return name, cb.rows, shield, wrapper


def summarise(name, rows, shield, wrapper, fraction=0.2):
    """Report first-vs-last slice of training, so a trend is visible."""
    if not rows:
        return {"arm": name, "episodes": 0}
    k = max(1, int(len(rows) * fraction))// 1
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
        "shield_triggered": shield.n_triggered,
        "shield_fallback": shield.n_fallback,
        "shield_checks": shield.n_checks,
    }


def plot(summaries, all_rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle("Phase 6: constrained vs unconstrained PPO under the shield",
                 fontsize=14, fontweight="bold")
    colors = {"unconstrained": "steelblue", "constrained": "crimson"}

    def smooth(v, w=51):
        if len(v) < w:
            return np.asarray(v, dtype=float)
        return np.convolve(v, np.ones(w) / w, mode="valid")

    for name, rows in all_rows.items():
        if not rows:
            continue
        c = colors.get(name, "grey")
        ts = [r["timestep"] for r in rows]
        for ax, key, title in (
            (axes[0], "task_reward", "Task reward (penalty excluded)"),
            (axes[1], "intervention_rate", "Shield interventions per step"),
            (axes[2], "penalised_reward", "Penalised reward (what PPO optimises)"),
        ):
            y = smooth([r[key] for r in rows])
            ax.plot(ts[len(ts) - len(y):], y, color=c, label=name)
            ax.set_title(title)
            ax.set_xlabel("timestep")
            ax.legend()
        if name == "constrained":
            axes[3].plot(ts, [r["lam"] for r in rows], color=c)
    axes[3].set_title("Lagrange multiplier $\\lambda$")
    axes[3].set_xlabel("timestep")

    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    print(f"Plot saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=None, help="override config training.device")
    ap.add_argument("--out-dir", default="saferl/eval/phase6")
    args = ap.parse_args()

    cfg = load_config()
    device = args.device or cfg["training"].get("device", "cpu")
    print(f"device={device}  timesteps={args.timesteps}  seed={args.seed}")

    summaries, all_rows = [], {}
    for constrained in (False, True):
        name, rows, shield, wrapper = run_arm(
            cfg, constrained, args.timesteps, args.seed, device, args.out_dir)
        s = summarise(name, rows, shield, wrapper)
        summaries.append(s)
        all_rows[name] = rows
        print(f"[{name}] episodes={s['episodes']} goals={s['goals']} "
              f"collisions={s['collisions']} "
              f"rate {s['first_rate']:.4f} -> {s['last_rate']:.4f} "
              f"task_reward {s['first_task_reward']:.1f} -> {s['last_task_reward']:.1f} "
              f"lambda={s['final_lambda']}")

    out = Path(args.out_dir)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    plot(summaries, all_rows, out / "phase6_constrained.png")


if __name__ == "__main__":
    main()
