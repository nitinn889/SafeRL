"""Phase 9: extended training from the phase 7 limited-sensing checkpoint,
with live TensorBoard metrics and convergence-based stopping.

Run:  python -m saferl.training.train_phase9

Watch it live:
    tensorboard --logdir saferl/eval/phase9/tb --port 6006
    # then open http://localhost:6006

Two things distinguish this from phase 7's script:

1. Step budget is decided by measured convergence, not a preset number. Every
   `--eval-every` steps the policy is probed with a deterministic held-out
   eval; training stops once goal rate and intervention rate have both gone
   flat across `--patience` consecutive probes. `--max-timesteps` is only a
   cost ceiling, not the intended stopping point.

2. Everything the project actually cares about is streamed to TensorBoard as
   scalars rather than printed. The existing MetricsCallback and the CSV +
   matplotlib summary are still produced -- TensorBoard is additive, for
   watching a run in flight, not a replacement for the static end-of-run
   artefacts.
"""
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv
from saferl.eval.evaluate import evaluate_model
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv
from saferl.training.dual_critic import ConstrainedPPO, DualCriticPolicy
from saferl.training.metrics import MetricsCallback


class Phase9Callback(BaseCallback):
    """Per-episode metrics: CSV rows (as in phase 6b/7) plus TensorBoard scalars.

    The CSV keeps the same columns phase 6b and 7 wrote, so the existing
    plotting and cross-phase comparisons keep working unchanged.
    """

    def __init__(self, rolling_window=100, verbose=0):
        super().__init__(verbose)
        self.rows: list[dict] = []
        self.rolling_window = rolling_window
        self._goals = deque(maxlen=rolling_window)
        self._rates = deque(maxlen=rolling_window)
        self._task_rewards = deque(maxlen=rolling_window)
        self._reset_acc()

    def _reset_acc(self):
        self._reward = 0.0
        self._cost = 0.0
        self._fallback = 0.0
        self._collision = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._reward += float(self.locals["rewards"][0])
        self._cost += float(info.get("shield_cost", 0.0))
        self._fallback += float(info.get("shield_fallback", 0.0))
        self._collision += float(info.get("cost", 0.0))

        if "episode" not in info:
            return True

        steps = int(info["episode"]["l"])
        lam = float(getattr(self.model, "lam", 0.0))
        target = float(getattr(self.model, "current_target_rate", 0.0))
        iv_rate = self._cost / max(steps, 1)
        reached_goal = int(self._reward > 0)

        # What the policy gradient is actually being pushed toward, as opposed
        # to the raw task return: task reward net of the Lagrangian price
        # currently charged for leaning on the shield. Watching the two
        # together is how you see whether lambda is biting.
        penalised_reward = self._reward - lam * self._cost

        self.rows.append(dict(
            timestep=int(self.num_timesteps),
            episode=len(self.rows) + 1,
            steps=steps,
            task_reward=round(self._reward, 3),
            interventions=int(self._cost),
            fallbacks=int(self._fallback),
            intervention_rate=round(iv_rate, 5),
            collided=int(self._collision > 0),
            reached_goal=reached_goal,
            lam=round(lam, 4),
            target_rate=round(target, 4),
            goal_fraction=round(float(info.get("goal_fraction", 1.0)), 3),
        ))

        self._goals.append(reached_goal)
        self._rates.append(iv_rate)
        self._task_rewards.append(self._reward)

        rec = self.logger.record
        rec("rollout/ep_task_reward", self._reward)
        rec("rollout/ep_penalised_reward", penalised_reward)
        rec("rollout/ep_intervention_rate", iv_rate)
        rec("rollout/ep_interventions", int(self._cost))
        rec("rollout/ep_fallback_interventions", int(self._fallback))
        rec("rollout/ep_length", steps)
        rec("rollout/goal_rate_rolling", float(np.mean(self._goals)))
        rec("rollout/intervention_rate_rolling", float(np.mean(self._rates)))
        rec("rollout/task_reward_rolling", float(np.mean(self._task_rewards)))

        self._reset_acc()
        return True


class ConvergenceCallback(BaseCallback):
    """Probe held-out performance periodically; stop once it stops improving.

    Two stopping conditions, either of which ends the run:

    1. Plateau -- across the last `patience` probes, goal rate spans no more
       than `goal_tol` and intervention rate spans no more than `iv_tol`.
       Both have to settle; a run whose goal rate is stable while lambda is
       still driving the intervention rate down has not finished.

    2. No improvement -- the best probe goal rate has not improved for
       `no_improve_patience` consecutive probes. Needed because condition 1
       cannot fire while the policy oscillates, and this policy does
       oscillate under a constraint target it cannot actually reach.

    The probe seed is deliberately NOT the seed the final authoritative eval
    uses. Probes both monitor convergence and select the best checkpoint, so
    scoring them on the same episodes the final number is reported on would
    be selecting on the test set and would inflate that number. Probes get
    their own episode set; the reported figure comes from the untouched one.
    """

    def __init__(self, eval_every, eval_episodes, eval_seed, patience,
                 goal_tol, iv_tol, out_dir, no_improve_patience=6,
                 best_ckpt_path=None, config_path=None, verbose=1):
        super().__init__(verbose)
        # Probes must build the same env the model trains on. Without this
        # they always loaded the planar default config, which is harmless for
        # a planar run and fatal for a 3D one (81-wide policy, 39-wide env).
        self.config_path = config_path
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.patience = patience
        self.goal_tol = goal_tol
        self.iv_tol = iv_tol
        self.no_improve_patience = no_improve_patience
        self.best_ckpt_path = best_ckpt_path
        self.out_dir = Path(out_dir)
        self._next_eval = eval_every
        self.probes: list[dict] = []
        self.converged_at = None
        self.stop_reason = None
        self.best_goal = -1.0
        self.best_probe = None
        self._since_improve = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_every

        result, _ = evaluate_model(
            self.model,
            n_episodes=self.eval_episodes,
            seed=self.eval_seed,
            deterministic=True,
            preserve_rng=True,
            config_path=self.config_path,
        )
        gr, ivr = result["goal_rate"], result["intervention_rate"]

        improved = gr > self.best_goal
        if improved:
            self.best_goal = gr
            self._since_improve = 0
            if self.best_ckpt_path is not None:
                self.model.save(self.best_ckpt_path)
            self.best_probe = dict(
                timestep=int(self.num_timesteps),
                goal_rate=gr, intervention_rate=ivr,
                collisions=result["collisions"],
            )
        else:
            self._since_improve += 1

        self.probes.append(dict(
            timestep=int(self.num_timesteps),
            goal_rate=gr,
            intervention_rate=ivr,
            collisions=result["collisions"],
            lam=round(float(getattr(self.model, "lam", 0.0)), 4),
            is_best=int(improved),
        ))

        self.logger.record("eval/goal_rate", gr)
        self.logger.record("eval/intervention_rate", ivr)
        self.logger.record("eval/collisions", result["collisions"])
        self.logger.record("eval/best_goal_rate", self.best_goal)
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(f"[probe @ {self.num_timesteps}] goal={gr*100:.1f}%  "
                  f"iv={ivr*100:.2f}%  collisions={result['collisions']}  "
                  f"lambda={getattr(self.model, 'lam', 0.0):.3f}"
                  f"{'  <- best' if improved else ''}")

        return self._check_converged()

    def _check_converged(self) -> bool:
        if len(self.probes) >= self.patience:
            window = self.probes[-self.patience:]
            goals = [p["goal_rate"] for p in window]
            ivs = [p["intervention_rate"] for p in window]
            goal_span = max(goals) - min(goals)
            iv_span = max(ivs) - min(ivs)

            if self.verbose:
                print(f"    plateau check (last {self.patience}): goal span "
                      f"{goal_span*100:.1f}pp (tol {self.goal_tol*100:.1f}), "
                      f"iv span {iv_span*100:.2f}pp (tol {self.iv_tol*100:.2f}); "
                      f"no-improve {self._since_improve}/{self.no_improve_patience}")

            if goal_span <= self.goal_tol and iv_span <= self.iv_tol:
                self.converged_at = int(self.num_timesteps)
                self.stop_reason = (
                    f"plateau: goal and intervention rate both flat across "
                    f"{self.patience} consecutive probes")
                print(f"\nSTOPPING at {self.converged_at} steps -- {self.stop_reason}")
                return False

        if self._since_improve >= self.no_improve_patience:
            self.converged_at = int(self.num_timesteps)
            self.stop_reason = (
                f"no improvement: best probe goal rate "
                f"({self.best_goal*100:.1f}%) not beaten for "
                f"{self._since_improve} consecutive probes")
            print(f"\nSTOPPING at {self.converged_at} steps -- {self.stop_reason}")
            return False

        return True

    def write_probes(self):
        if not self.probes:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "convergence_probes.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.probes[0].keys()))
            w.writeheader()
            w.writerows(self.probes)
        print(f"Wrote {path}")


def build_env(cfg):
    base = SafeNav3DEnv(render_mode="direct", **cfg["env"])
    shielded = ShieldedEnv(base, SafetyShield.from_config(cfg))
    monitored = Monitor(shielded, info_keywords=("cost",))
    return DummyVecEnv([lambda: monitored])


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
        "last_fallbacks_per_ep": agg(tail, "fallbacks"),
        "last_steps": agg(tail, "steps"),
        "goals": sum(r["reached_goal"] for r in rows),
        "collisions": sum(r["collided"] for r in rows),
        "final_lambda": rows[-1]["lam"],
        "goal_rate_pct": round(100.0 * sum(r["reached_goal"] for r in rows) / len(rows), 1),
    }


def plot_phase9(rows, probes, output_path,
                title="Phase 9: extended training from phase 7 (limited sensing)"):
    """Static end-of-run summary. Retained alongside TensorBoard, not replaced."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(title,
                 fontsize=14, fontweight="bold")

    def smooth(v, w=51):
        if len(v) < w:
            return np.asarray(v, dtype=float)
        return np.convolve(v, np.ones(w) / w, mode="valid")

    ts = [r["timestep"] for r in rows]
    panels = [
        (axes[0, 0], "task_reward", "Task reward"),
        (axes[0, 1], "intervention_rate", "Intervention rate"),
        (axes[0, 2], "steps", "Episode length"),
        (axes[1, 0], "lam", "Lambda"),
        (axes[1, 1], "fallbacks", "Fallback interventions / episode"),
    ]
    for ax, key, title in panels:
        y = smooth([r[key] for r in rows])
        ax.plot(ts[len(ts) - len(y):], y, color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("timestep")

    ax = axes[1, 2]
    if probes:
        pts = [p["timestep"] for p in probes]
        ax.plot(pts, [p["goal_rate"] * 100 for p in probes],
                "o-", color="seagreen", label="goal rate %")
        ax.plot(pts, [p["intervention_rate"] * 100 for p in probes],
                "s-", color="crimson", label="intervention rate %")
        ax.legend()
    ax.set_title("Held-out convergence probes")
    ax.set_xlabel("timestep")
    ax.set_ylabel("%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    print(f"Plot saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="saferl/eval/phase7/saferl_phase7.zip",
                    help="Limited-sensing dual-critic checkpoint to continue from")
    ap.add_argument("--max-timesteps", type=int, default=1_500_000,
                    help="Cost ceiling, not the intended stopping point")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="saferl/eval/phase9")
    ap.add_argument("--tb-dir", default="saferl/eval/phase9/tb")
    ap.add_argument("--target-rate", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=7,
                    help="Probe/selection seed. Deliberately NOT the final "
                         "eval's seed (42) -- probes pick the best checkpoint, "
                         "so scoring them on the reported episodes would be "
                         "selecting on the test set.")
    ap.add_argument("--patience", type=int, default=4,
                    help="Consecutive flat probes required to declare a plateau")
    ap.add_argument("--goal-tol", type=float, default=0.04,
                    help="Max goal-rate span across the window (fraction)")
    ap.add_argument("--iv-tol", type=float, default=0.02,
                    help="Max intervention-rate span across the window (fraction)")
    ap.add_argument("--no-improve-patience", type=int, default=6,
                    help="Stop if the best probe goal rate is not beaten for "
                         "this many consecutive probes")
    ap.add_argument("--config", default=None,
                    help="Config YAML (default: planar default.yaml). Used for "
                         "training AND for the convergence probes.")
    ap.add_argument("--run-name", default="phase9",
                    help="Prefix for saved checkpoints, CSVs and the "
                         "TensorBoard run (default keeps phase 9's names)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config={args.config or 'default.yaml'}  dims={cfg['env'].get('dims', 2)}  "
          f"hazards={cfg['env']['max_hazards']}  run_name={args.run_name}")
    device = args.device or cfg["training"].get("device", "cpu")
    ccfg = cfg.get("constrained", {})
    sensor_range = cfg["env"].get("sensor_range", 6.0)

    print(f"device={device}  max_timesteps={args.max_timesteps}  seed={args.seed}")
    print(f"sensor_range={sensor_range}  target_rate={args.target_rate}")
    print(f"checkpoint={args.checkpoint}")
    print(f"tensorboard_log={args.tb_dir}")
    print(f"convergence: {args.patience} consecutive probes of "
          f"{args.eval_episodes} episodes every {args.eval_every} steps, "
          f"goal span <= {args.goal_tol*100:.1f}pp and "
          f"iv span <= {args.iv_tol*100:.1f}pp")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    venv = build_env(cfg)

    model = ConstrainedPPO.load(
        args.checkpoint, env=venv, device=device,
        custom_objects={"policy_class": DualCriticPolicy},
        tensorboard_log=args.tb_dir,
    )
    model.target_rate_start = args.target_rate
    model.target_rate_end = args.target_rate
    model.constraint_lambda_lr = ccfg.get("lambda_lr", 0.1)
    model.constraint_lambda_max = ccfg.get("lambda_max", 5.0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ep_cb = Phase9Callback()
    metrics_cb = MetricsCallback()          # preserved from earlier phases
    conv_cb = ConvergenceCallback(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        patience=args.patience,
        goal_tol=args.goal_tol,
        iv_tol=args.iv_tol,
        no_improve_patience=args.no_improve_patience,
        best_ckpt_path=str(out / f"saferl_{args.run_name}_best.zip"),
        out_dir=out,
        config_path=args.config,
    )

    print(f"\nTraining (ceiling {args.max_timesteps} steps, "
          f"stopping on measured convergence)...")
    model.learn(
        total_timesteps=args.max_timesteps,
        callback=[ep_cb, metrics_cb, conv_cb],
        tb_log_name=args.run_name,
        reset_num_timesteps=True,
    )

    write_csv(ep_cb.rows, out / f"{args.run_name}_episodes.csv")
    conv_cb.write_probes()
    model.save(out / f"saferl_{args.run_name}.zip")

    s = summarise("phase9", ep_cb.rows)
    s["stopped_at"] = conv_cb.converged_at or "hit ceiling"
    s["stop_reason"] = conv_cb.stop_reason or "max-timesteps ceiling reached"
    s["best_probe_goal_rate"] = conv_cb.best_goal
    s["best_probe_timestep"] = (conv_cb.best_probe or {}).get("timestep", "n/a")
    print(f"\n[phase9] episodes={s['episodes']} goals={s['goals']} "
          f"({s['goal_rate_pct']}%) collisions={s['collisions']} "
          f"rate {s['last_rate']:.4f} "
          f"task_reward {s['last_task_reward']:.1f} "
          f"lambda={s['final_lambda']}")
    print(f"[phase9] stopped at {s['stopped_at']} -- {s['stop_reason']}")
    print(f"[phase9] best probe {conv_cb.best_goal*100:.1f}% goal at step "
          f"{s['best_probe_timestep']} -> saferl_{args.run_name}_best.zip")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        w.writeheader()
        w.writerow(s)

    plot_kw = {} if args.run_name == "phase9" else {
        "title": f"{args.run_name}: convergence-stopped run ({args.config or 'default.yaml'})"}
    plot_phase9(ep_cb.rows, conv_cb.probes, out / f"{args.run_name}_training.png", **plot_kw)
    venv.close()


if __name__ == "__main__":
    main()
