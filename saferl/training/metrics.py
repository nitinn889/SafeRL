"""SB3 callback that records per-episode reward, cost, and shield interventions."""
from stable_baselines3.common.callbacks import BaseCallback

from saferl.shield.safety_shield import ShieldedEnv


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_interventions = []
        self._step_cost = 0  # accumulator between episode boundaries

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._step_cost += float(info.get("cost", 0))

        if "episode" in info:
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_costs.append(self._step_cost)
            self._step_cost = 0  # reset for next episode

            # Unwrap: DummyVecEnv -> Monitor -> ShieldedEnv
            env = self.training_env.envs[0]
            while hasattr(env, "env"):  # peel Monitor / other wrappers
                if isinstance(env, ShieldedEnv):
                    break
                env = env.env
            ivs = env.interventions if isinstance(env, ShieldedEnv) else 0
            self.episode_interventions.append(ivs)
        return True
