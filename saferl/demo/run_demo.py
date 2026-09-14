"""Fast headless demo: watch the trained policy fly without needing UE.

    python -m saferl.demo.run_demo                     # 10 episodes, headless
    python -m saferl.demo.run_demo --episodes 3 --gui  # PyBullet GUI window
    python -m saferl.demo.run_demo --curriculum        # config's ramped difficulty

This is the quick sanity path. For the full visual demo in Unreal, see
`demo/run_ue_demo.sh` (and the README's "Run the demo" section).

Rewritten in phase 10. The phase 2 version of this script ran the *bare*
env with no SafetyShield, so it exercised none of the safety layer this
project exists to build, defaulted to a `saferl_model.zip` that no longer
exists, opened a GUI window unconditionally (which hangs on this host's
Wayland session), and reported nothing but "Demo finished." It now runs the
same env + shield + policy stack the evaluation protocol uses, headless by
default, and prints the per-episode numbers that make it a useful check.
"""
import argparse
import time

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv
from saferl.training.dual_critic import ConstrainedPPO, DualCriticPolicy

DEFAULT_CHECKPOINT = "saferl/eval/phase9/saferl_phase9_best.zip"


def run_demo(model_path=DEFAULT_CHECKPOINT, config_path=None, episodes=10,
             gui=False, curriculum=False, deterministic=True, fps=None):
    cfg = load_config(config_path)
    env_cfg = dict(cfg["env"])
    env_cfg["curriculum"] = curriculum

    base = SafeNav3DEnv(render_mode="human" if gui else "direct", **env_cfg)
    shield = SafetyShield.from_config(cfg)
    env = ShieldedEnv(base, shield)

    model = ConstrainedPPO.load(
        model_path, env=None, device=cfg["training"].get("device", "cpu"),
        custom_objects={"policy_class": DualCriticPolicy},
    )

    print(f"checkpoint : {model_path}")
    print(f"difficulty : {env_cfg['max_hazards']} hazards, "
          f"curriculum={'on' if curriculum else 'off (full fixed)'}")
    print(f"sensing    : sensor_range={env_cfg.get('sensor_range')} "
          f"(shield sees true state regardless)")
    print()

    goals = collisions = total_iv = total_checks = 0
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        env.interventions = 0
        shield.n_checks = 0
        done = truncated = False
        ep_reward = 0.0
        steps = 0
        collided = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if info.get("cost", 0) > 0:
                collided = 1
            if fps:
                time.sleep(1.0 / fps)

        reached = int(ep_reward > 0)
        goals += reached
        collisions += collided
        total_iv += env.interventions
        total_checks += shield.n_checks
        iv_rate = env.interventions / max(shield.n_checks, 1)

        print(f"ep {ep:3d}  {'GOAL' if reached else ('COLLISION' if collided else 'no goal'):9s} "
              f"steps={steps:4d}  reward={ep_reward:7.1f}  "
              f"interventions={env.interventions:4d} ({iv_rate*100:5.2f}%)")

    env.close()
    print(f"\n{goals}/{episodes} goals ({goals/episodes*100:.1f}%), "
          f"{collisions} collisions, "
          f"intervention rate {total_iv/max(total_checks,1)*100:.2f}%")
    print("Note: a short run is a sanity check, not a measurement. The "
          "reported figures come from saferl.eval.evaluate over 500 episodes.")


def main():
    p = argparse.ArgumentParser(description="Fast headless demo of a trained SafeRL agent.")
    p.add_argument("--model", default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gui", action="store_true",
                   help="Open a PyBullet GUI window (headless otherwise)")
    p.add_argument("--curriculum", action="store_true",
                   help="Inherit the config's hazard-count ramp. Off by "
                        "default: a fresh env restarts that ramp at 1 hazard, "
                        "which is easier than the scenario the demo depicts.")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--fps", type=float, default=None,
                   help="Throttle to this many steps/sec (useful with --gui)")
    args = p.parse_args()

    run_demo(model_path=args.model, config_path=args.config,
             episodes=args.episodes, gui=args.gui, curriculum=args.curriculum,
             deterministic=not args.stochastic, fps=args.fps)


if __name__ == "__main__":
    main()
