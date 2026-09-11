"""Visual demo entrypoint: python -m saferl.demo.run_demo [--model path] [--config path]"""
import argparse
import time

from stable_baselines3 import PPO

from saferl.config import load_config
from saferl.env.base_env import SafeNav3DEnv


def run_demo(model_path="saferl_model.zip", config_path=None):
    cfg = load_config(config_path)
    model = PPO.load(model_path)

    demo_env = SafeNav3DEnv(
        size=cfg["env"]["size"],
        max_hazards=cfg["env"]["max_hazards"],
        curriculum=cfg["env"]["curriculum"],
        render_mode="human",
        force_mag=cfg["env"]["force_mag"],
        goal_threshold=cfg["env"]["goal_threshold"],
        hazard_threshold=cfg["env"]["hazard_threshold"],
    )
    obs, _ = demo_env.reset()
    fps = cfg["demo"]["fps"]
    for _ in range(cfg["demo"]["steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = demo_env.step(action)
        time.sleep(1.0 / fps)
        if done:
            obs, _ = demo_env.reset()

    demo_env.close()
    print("Demo finished.")


def main():
    parser = argparse.ArgumentParser(description="Run a visual demo of a trained SafeRL agent.")
    parser.add_argument("--model", default="saferl_model.zip", help="Path to a trained SB3 model")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: saferl/configs/default.yaml)")
    args = parser.parse_args()
    run_demo(model_path=args.model, config_path=args.config)


if __name__ == "__main__":
    main()
