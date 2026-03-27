import time
import torch
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sns.set_theme(style="darkgrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 1. ENVIRONMENT
# FIX: Each instance creates its OWN PyBullet connection (p.DIRECT for training).
#      This prevents the 'Only one local connection' crash and sim-reset conflicts.
# ─────────────────────────────────────────────
class SafeNav3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "direct"]}

    def __init__(self, size=10, max_hazards=5, curriculum=False, render_mode="direct"):
        super().__init__()
        self.size = size
        self.max_hazards = max_hazards
        self.curriculum = curriculum
        self.render_mode = render_mode
        self.episode_count = 0
        self._client = -1                          # PyBullet client ID (set on first reset)
        self.hazard_positions = []

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-20, high=20,
            shape=(9 + 3 * max_hazards,),
            dtype=np.float32
        )
        self.goal_pos = np.array([size - 1, size - 1, 0.5], dtype=np.float32)

    # ------ PyBullet lifecycle ------
    def _connect(self):
        """Connect once; subsequent calls are no-ops."""
        if self._client >= 0:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

    def close(self):
        if self._client >= 0:
            try:
                p.disconnect(self._client)
            except Exception:
                pass
            self._client = -1

    # ------ gym API ------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        cid = self._client

        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)

        self.agent_id = p.loadURDF(
            "sphere2.urdf", [0, 0, 0.5], globalScaling=0.5,
            physicsClientId=cid
        )
        p.changeVisualShape(self.agent_id, -1, rgbaColor=[0, 0, 1, 1],
                            physicsClientId=cid)

        goal_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.6, rgbaColor=[0, 1, 0, 0.5],
            physicsClientId=cid
        )
        p.createMultiBody(
            baseVisualShapeIndex=goal_vis, basePosition=self.goal_pos.tolist(),
            physicsClientId=cid
        )

        self.episode_count += 1
        self._setup_hazards()
        return self._get_obs(), {}

    def _setup_hazards(self):
        cid = self._client
        if self.curriculum:
            num_hazards = min(self.max_hazards, 1 + self.episode_count // 50)
        else:
            num_hazards = self.max_hazards

        self.hazard_positions = []
        for _ in range(num_hazards):
            h_pos = [
                float(np.random.uniform(1, self.size - 2)),
                float(np.random.uniform(1, self.size - 2)),
                0.5
            ]
            p.loadURDF("r2d2.urdf", h_pos, globalScaling=0.6,
                       physicsClientId=cid)
            self.hazard_positions.append(h_pos)

    def _get_obs(self):
        cid = self._client
        pos, _ = p.getBasePositionAndOrientation(self.agent_id,
                                                  physicsClientId=cid)
        vel, _ = p.getBaseVelocity(self.agent_id, physicsClientId=cid)
        obs = list(pos) + list(vel) + list(self.goal_pos)
        for h in self.hazard_positions:
            obs.extend(h)
        target_len = self.observation_space.shape[0]
        # Pad with zeros if fewer hazards (curriculum) or slice to cap length
        while len(obs) < target_len:
            obs.extend([0.0, 0.0, 0.0])
        return np.array(obs[:target_len], dtype=np.float32)

    def step(self, action):
        cid = self._client
        force = [0.0, 0.0, 0.0]
        mag = 12
        if   action == 0: force[1] =  mag
        elif action == 1: force[1] = -mag
        elif action == 2: force[0] = -mag
        elif action == 3: force[0] =  mag

        p.applyExternalForce(
            self.agent_id, -1, force, [0, 0, 0],
            p.WORLD_FRAME, physicsClientId=cid
        )
        p.stepSimulation(physicsClientId=cid)

        obs    = self._get_obs()
        reward = -0.1
        cost   = 0
        done   = False

        if np.linalg.norm(obs[0:3] - self.goal_pos) < 1.0:
            reward += 100
            done = True

        if not done:
            for h_pos in self.hazard_positions:
                if np.linalg.norm(obs[0:3] - np.array(h_pos)) < 1.3:
                    cost   = 1
                    reward = -50
                    done   = True
                    break

        return obs, reward, done, False, {"cost": cost}


# ─────────────────────────────────────────────
# 2. SAFETY SHIELD
# FIX: ShieldedEnv.step() now uses self._last_obs (stored at reset/step)
#      instead of calling self.env._get_obs() which bypasses the wrapper stack.
# ─────────────────────────────────────────────
class SafetyShield:
    SAFE_DIST = 2.2

    def check_and_fix(self, obs, action):
        pos = obs[0:3]
        for i in range(9, len(obs), 3):
            h_pos = obs[i:i + 3]
            if np.all(h_pos == 0):
                continue
            if np.linalg.norm(pos - h_pos) < self.SAFE_DIST:
                return int(np.random.choice([0, 1, 2, 3])), True
        return action, False


class ShieldedEnv(gym.Wrapper):
    def __init__(self, env, shield):
        super().__init__(env)
        self.shield = shield
        self.interventions = 0
        self._last_obs = None          # FIX: track obs here, not via private method

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


# ─────────────────────────────────────────────
# 3. METRICS CALLBACK
# FIX: accumulate cost each step, reset at episode end.
#      Unwrap env stack properly to reach ShieldedEnv.interventions.
# ─────────────────────────────────────────────
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards      = []
        self.episode_costs        = []
        self.episode_interventions = []
        self._step_cost = 0            # accumulator between episode boundaries

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._step_cost += float(info.get("cost", 0))

        if "episode" in info:
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_costs.append(self._step_cost)
            self._step_cost = 0        # FIX: reset for next episode

            # Unwrap: DummyVecEnv → Monitor → ShieldedEnv
            env = self.training_env.envs[0]
            while hasattr(env, "env"):            # peel Monitor / other wrappers
                if isinstance(env, ShieldedEnv):
                    break
                env = env.env
            ivs = env.interventions if isinstance(env, ShieldedEnv) else 0
            self.episode_interventions.append(ivs)
        return True


# ─────────────────────────────────────────────
# 4. TRAINING
# FIX: factory function (not lambda) so DummyVecEnv can create fresh envs.
#      Monitor(info_keywords=("cost",)) so cost survives into infos dict.
# ─────────────────────────────────────────────
def make_env():
    base    = SafeNav3DEnv(curriculum=True, render_mode="direct")
    shielded = ShieldedEnv(base, SafetyShield())
    return Monitor(shielded, info_keywords=("cost",))


print("🚀 Training 3D SafeRL Agent...")
v_env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", v_env, verbose=1, device=device)
cb    = MetricsCallback()

TIMESTEPS = 30_000   # ↑ from 15k → more episodes → fuller graphs
model.learn(total_timesteps=TIMESTEPS, callback=cb)
print(f"✅ Training done — {len(cb.episode_rewards)} episodes recorded.")


# ─────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────
print("📊 Generating graphs...")
if len(cb.episode_rewards) == 0:
    print("❌ No episodes completed — try increasing TIMESTEPS.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SafeRL Training Metrics", fontsize=14, fontweight="bold")

    sns.lineplot(data=cb.episode_rewards, ax=axes[0], color="steelblue")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    sns.lineplot(data=np.cumsum(cb.episode_costs), ax=axes[1], color="crimson")
    axes[1].set_title("Cumulative Crashes (Cost)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Crashes")

    sns.lineplot(data=cb.episode_interventions, ax=axes[2], color="seagreen")
    axes[2].set_title("Shield Interventions per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Cumulative Interventions")

    plt.tight_layout()
    plt.savefig("saferl_metrics.png", dpi=150)
    plt.show()
    print("📈 Graphs saved to saferl_metrics.png")


# ─────────────────────────────────────────────
# 6. DEMO  (separate GUI env — training env stays DIRECT)
# ─────────────────────────────────────────────
print("📺 Running visual demo...")
demo_env = SafeNav3DEnv(render_mode="human")
obs, _   = demo_env.reset()
for _ in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = demo_env.step(action)
    time.sleep(1.0 / 60.0)
    if done:
        obs, _ = demo_env.reset()

demo_env.close()
v_env.close()
print("✅ All done!")
