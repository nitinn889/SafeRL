"""Phase 7: fine-tune the constrained policy under limited-range sensing.

Run:  python -m saferl.training.train_phase7 --timesteps 200000

The policy now observes only hazards within sensor_range (default 6.0).
Hazards beyond range appear as zeros — the same convention as unused
curriculum slots, so the observation space is unchanged. The shield
retains full privileged access and continues to intervene on threats
the policy cannot see.

This fine-tunes from the phase 6b checkpoint rather than training from
scratch, since the navigation skill transfers; only the hazard-avoidance
behaviour needs to adapt to partial observability.
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


class Phase7Callback(BaseCallback):
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


def quick_eval(model, cfg, n_episodes=100, label="eval"):
    base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
    shield = SafetyShield.from_config(cfg)
    env = ShieldedEnv(base, shield)
    goals, interventions, steps_total = 0, 0, 0
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
        interventions += env.interventions
        steps_total += env.shield.n_checks
        env.interventions = 0
        env.shield.n_checks = 0
    env.close()
    goal_rate = goals / n_episodes
    iv_rate = interventions / max(steps_total, 1)
    print(f"[{label}] goal_rate={goal_rate*100:.1f}%  "
          f"intervention_rate={iv_rate*100:.2f}%  "
          f"({n_episodes} episodes)")
    return goal_rate, iv_rate


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
    tail = rows[-k:]
    def agg(chunk, key):
        return float(np.mean([r[key] for r in chunk]))
    return {
        "arm": name,
        "episodes": len(rows),
        "last_rate": agg(tail, "intervention_rate"),
        "last_task_reward": agg(tail, "task_reward"),
        "last_ivs_per_ep": agg(tail, "interventions"),
        "last_steps": agg(tail, "steps"),
        "goals": sum(r["reached_goal"] for r in rows),
        "collisions": sum(r["collided"] for r in rows),
        "final_lambda": rows[-1]["lam"],
        "goal_rate_pct": round(100.0 * sum(r["reached_goal"] for r in rows) / len(rows), 1),
    }


def plot_phase7(rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Phase 7: limited-range sensing (fine-tuned from phase 6b)",
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
        ax.plot(ts[len(ts) - len(y):], y, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("timestep")

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
    ap.add_argument("--out-dir", default="saferl/eval/phase7")
    ap.add_argument("--checkpoint",
                     default="saferl/eval/phase6b/saferl_phase6b.zip",
                     help="Phase 6b checkpoint to fine-tune from")
    ap.add_argument("--target-rate", type=float, default=0.05,
                     help="Fixed constraint target (no curriculum)")
    ap.add_argument("--eval-episodes", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    device = args.device or cfg["training"].get("device", "cpu")
    ccfg = cfg.get("constrained", {})
    sensor_range = cfg["env"].get("sensor_range", 6.0)

    print(f"device={device}  timesteps={args.timesteps}  seed={args.seed}")
    print(f"sensor_range={sensor_range}  target_rate={args.target_rate}")
    print(f"checkpoint={args.checkpoint}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    venv = build_env(cfg, args.seed)

    # Load the phase 6b constrained model directly
    model = ConstrainedPPO.load(
        args.checkpoint, env=venv, device=device,
        custom_objects={"policy_class": DualCriticPolicy},
    )
    # Fix target rate at the tight budget — no curriculum needed since the
    # policy already learned constraint satisfaction in phase 6b
    model.target_rate_start = args.target_rate
    model.target_rate_end = args.target_rate
    model.constraint_lambda_lr = ccfg.get("lambda_lr", 0.1)
    model.constraint_lambda_max = ccfg.get("lambda_max", 5.0)

    print(f"\nPre-training eval under sensor_range={sensor_range}...")
    quick_eval(model, cfg, n_episodes=args.eval_episodes, label="pre-train")

    print(f"\nFine-tuning for {args.timesteps} steps under limited sensing...")
    cb = Phase7Callback()
    model.learn(total_timesteps=args.timesteps, callback=cb)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(cb.rows, out / "phase7_episodes.csv")
    model.save(out / "saferl_phase7.zip")

    print(f"\nPost-training eval under sensor_range={sensor_range}...")
    quick_eval(model, cfg, n_episodes=args.eval_episodes, label="post-train")

    s = summarise("phase7", cb.rows)
    print(f"\n[phase7] episodes={s['episodes']} goals={s['goals']} "
          f"({s['goal_rate_pct']}%) collisions={s['collisions']} "
          f"rate {s['last_rate']:.4f} "
          f"task_reward {s['last_task_reward']:.1f} "
          f"lambda={s['final_lambda']}")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        w.writeheader()
        w.writerow(s)

    plot_phase7(cb.rows, out / "phase7_limited_sensing.png")
    venv.close()


if __name__ == "__main__":
    main()
