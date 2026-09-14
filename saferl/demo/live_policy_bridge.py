"""Drive a live trained-policy rollout and publish its state for UE to mirror.

Why this exists as a separate process rather than running inside UE: UE 5.8
embeds Python 3.11, this project's venv is Python 3.14, and torch/SB3 ship
compiled extensions that are ABI-locked to their interpreter. The policy
therefore cannot be imported into UE's Python at all. Rather than duplicate
the env's physics, shield and observation layout in UE Python (which would
drift from the real thing the moment either side changed), this runs the
*real* SafeNav3DEnv + SafetyShield + trained policy here and publishes the
resulting world state; `pie_session.py` in live-mirror mode just reads that
state and moves the UE actors to match. UE stays a renderer, which is the
role every phase since 2 has measured it into.

Usage:
    python -m saferl.demo.live_policy_bridge \
        --checkpoint saferl/eval/phase9/saferl_phase9.zip

Writes JSON world state to --state-file (default ue_spike/live_policy_state.json)
after every env step, and prints a one-line summary per finished episode.
"""
import argparse
import json
import os
import time

import numpy as np

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv
from saferl.training.dual_critic import ConstrainedPPO, DualCriticPolicy

DEFAULT_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ue_spike", "live_policy_state.json",
)


def _base_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _write_state(path, payload):
    """Atomic publish: UE polls this file every tick and must never read a
    half-written frame.

    The scratch name carries this process's PID. With a single fixed
    `.tmp` name, two bridges pointed at the same state file race: A writes
    tmp, B writes tmp, A's os.replace consumes it, and B's os.replace then
    dies with FileNotFoundError. That is not hypothetical -- it killed a
    phase 10 demo run when a previous bridge survived a pkill and the next
    run started alongside it.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="Config YAML (default: planar default.yaml; "
                         "saferl/configs/space3d.yaml for 3D free flight)")
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="env steps per second; the policy runs ~1000/s "
                         "unthrottled, which is unwatchable")
    ap.add_argument("--seed", type=int, default=7,
                    help="deliberately NOT 42 -- that seed is reserved for "
                         "held-out evaluation, and a demo should not be run "
                         "on the reported episodes")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--curriculum", action="store_true",
                    help="leave the env's hazard-count curriculum on. Off by "
                         "default here: a fresh env starts the curriculum at 1 "
                         "hazard and only reaches max_hazards after ~200 "
                         "episodes, which would show one rock moving while the "
                         "UE scene renders five. Off = full difficulty from "
                         "episode 0, matching what the UE scene depicts.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env_cfg = dict(cfg["env"])
    if not args.curriculum:
        env_cfg["curriculum"] = False

    np.random.seed(args.seed)

    base = SafeNav3DEnv(render_mode="direct", **env_cfg)
    shield = SafetyShield.from_config(cfg)
    env = ShieldedEnv(base, shield)
    raw = _base_env(env)

    model = ConstrainedPPO.load(
        args.checkpoint, env=None, device=cfg["training"].get("device", "cpu"),
        custom_objects={"policy_class": DualCriticPolicy},
    )

    print(f"[live] checkpoint: {args.checkpoint}")
    print(f"[live] hazards: {env_cfg['max_hazards']} "
          f"(curriculum={'on' if env_cfg.get('curriculum') else 'off'}), "
          f"sensor_range={env_cfg.get('sensor_range')}")
    print(f"[live] publishing state to {args.state_file} at ~{args.fps:.0f} steps/s")
    print(f"[live] deterministic={not args.stochastic}, seed={args.seed}")
    print("[live] Ctrl-C to stop.\n")

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    episode = 0
    totals = {"goals": 0, "collisions": 0, "timeouts": 0,
              "interventions": 0, "steps": 0}

    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            env.interventions = 0
            env.fallback_interventions = 0
            shield.n_checks = 0
            done = truncated = False
            ep_reward = 0.0
            ep_steps = 0
            collided = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                if info.get("cost", 0) > 0:
                    collided = 1

                _write_state(args.state_file, {
                    "t": time.time(),
                    "episode": episode,
                    "ep_step": ep_steps,
                    "agent": [float(x) for x in obs[0:3]],
                    "agent_vel": [float(x) for x in obs[3:6]],
                    "dims": int(raw.dims),
                    "goal": [float(x) for x in obs[6:9]],
                    "debris": [[float(c) for c in pos]
                               for pos in raw.hazard_positions],
                    "interventions": int(env.interventions),
                    "fallbacks": int(env.fallback_interventions),
                    "ep_reward": round(float(ep_reward), 2),
                    "totals": dict(totals),
                })

                if period:
                    time.sleep(period)

            reached_goal = int(ep_reward > 0)
            totals["steps"] += ep_steps
            totals["interventions"] += int(env.interventions)
            if reached_goal:
                totals["goals"] += 1
                result = "GOAL"
            elif collided:
                totals["collisions"] += 1
                result = "COLLISION"
            else:
                totals["timeouts"] += 1
                result = "timeout/oob"

            iv_rate = env.interventions / max(shield.n_checks, 1)
            goal_pct = 100.0 * totals["goals"] / episode
            print(f"[live] ep {episode:4d}  {result:11s} "
                  f"steps={ep_steps:4d}  reward={ep_reward:7.1f}  "
                  f"interventions={env.interventions:4d} ({iv_rate*100:5.2f}%)  "
                  f"| running goal rate {goal_pct:5.1f}% "
                  f"({totals['goals']}/{episode}), collisions {totals['collisions']}")
    except KeyboardInterrupt:
        print("\n[live] stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
