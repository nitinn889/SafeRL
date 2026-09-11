"""Analytic hard-constraint safety layer.

SafetyShield.check_and_fix currently just picks a random action when the
agent is within safe_dist of a hazard — a placeholder shield, not a real
constraint solver yet. That logic is out of scope for this phase (see
phase 3/5) and is preserved as-is.
"""
import numpy as np
import gymnasium as gym

from saferl.env.base_env import OBS_HEADER_LEN, OBS_PER_HAZARD


class SafetyShield:
    def __init__(self, safe_dist=2.2):
        self.safe_dist = safe_dist

    def check_and_fix(self, obs, action):
        pos = obs[0:3]
        # each hazard block is [pos(3), vel(3)]; this still only looks at
        # position -- using relative velocity is phase 5's redesign
        for i in range(OBS_HEADER_LEN, len(obs), OBS_PER_HAZARD):
            h_pos = obs[i:i + 3]
            if np.all(h_pos == 0):
                continue
            if np.linalg.norm(pos - h_pos) < self.safe_dist:
                return int(np.random.choice([0, 1, 2, 3])), True
        return action, False


class ShieldedEnv(gym.Wrapper):
    def __init__(self, env, shield):
        super().__init__(env)
        self.shield = shield
        self.interventions = 0
        self._last_obs = None  # obs tracked here, not via a private env method

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        safe_action, intervened = self.shield.check_and_fix(self._last_obs, action)
        if intervened:
            self.interventions += 1
        obs, reward, done, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        return obs, reward, done, truncated, info
