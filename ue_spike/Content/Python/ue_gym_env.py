"""gymnasium.Env wrapper around ue_bridge.UEBridge.

Same interface shape as saferl.env.base_env.SafeNav3DEnv (Discrete(4)
action space, Box observation space) so it can later be swapped in as
saferl/env/ue_env.py without touching the shield or training code --
but unvalidated: this has not been run against a live editor yet (see
ue_spike/README.md). No shield, no PPO here on purpose; this only proves
the loop shape, per the phase 2 spike scope.

Must be run from inside UE's embedded Python interpreter (the `unreal`
module is not importable from a normal external Python process).
"""
import numpy as np
import gymnasium as gym

from ue_bridge import UEBridge


class UESafeNavSpikeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.bridge = UEBridge()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-1e4, high=1e4, shape=(6,), dtype=np.float32
        )

    def _to_obs(self, obs_dict):
        return np.array(obs_dict["position"] + obs_dict["velocity"], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._to_obs(self.bridge.reset()), {}

    def step(self, action):
        obs_dict = self.bridge.step(int(action))
        obs = self._to_obs(obs_dict)
        reward = -0.1
        done = False
        return obs, reward, done, False, {}
