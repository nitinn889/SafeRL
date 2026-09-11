"""PyBullet-backed 3D safe-navigation environment (SafeNav3DEnv).

Each instance owns its own PyBullet connection (p.DIRECT for training,
p.GUI for demos) so multiple envs never collide on a single shared
connection.
"""
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym


class SafeNav3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "direct"]}

    def __init__(self, size=10, max_hazards=5, curriculum=False, render_mode="direct",
                 force_mag=12.0, goal_threshold=1.0, hazard_threshold=1.3,
                 sim_substeps=10, agent_friction=0.0, max_episode_steps=1000):
        super().__init__()
        self.size = size
        self.max_hazards = max_hazards
        self.curriculum = curriculum
        self.render_mode = render_mode
        self.force_mag = force_mag
        self.goal_threshold = goal_threshold
        self.hazard_threshold = hazard_threshold
        self.sim_substeps = sim_substeps
        self.agent_friction = agent_friction
        self.max_episode_steps = max_episode_steps
        self.episode_count = 0
        self._step_count = 0
        self._client = -1  # PyBullet client ID (set on first reset)
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
        # sphere2.urdf is 10kg with lateralFriction 0.5; against gravity that
        # is ~49N of static friction, four times the thrust the action set can
        # produce, so the agent would weld itself to the plane and no episode
        # could ever terminate. A satellite has no ground to rub against.
        p.changeDynamics(self.agent_id, -1,
                         lateralFriction=self.agent_friction,
                         linearDamping=0.0, angularDamping=0.0,
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
        self._step_count = 0
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
        mag = self.force_mag
        if   action == 0: force[1] =  mag
        elif action == 1: force[1] = -mag
        elif action == 2: force[0] = -mag
        elif action == 3: force[0] =  mag

        # applyExternalForce only lasts a single substep, so one env step held
        # thrust for 1/240s -- 0.005 m/s of delta-v, far too little to cross the
        # field. Hold it across sim_substeps instead, decoupling the control
        # rate (~24Hz) from the physics rate (240Hz).
        for _ in range(self.sim_substeps):
            p.applyExternalForce(
                self.agent_id, -1, force, [0, 0, 0],
                p.WORLD_FRAME, physicsClientId=cid
            )
            p.stepSimulation(physicsClientId=cid)

        self._step_count += 1
        obs    = self._get_obs()
        reward = -0.1
        cost   = 0
        done   = False

        if np.linalg.norm(obs[0:3] - self.goal_pos) < self.goal_threshold:
            reward += 100
            done = True

        if not done:
            for h_pos in self.hazard_positions:
                if np.linalg.norm(obs[0:3] - np.array(h_pos)) < self.hazard_threshold:
                    cost   = 1
                    reward = -50
                    done   = True
                    break

        # truncation is a backstop for a wandering policy, not the termination
        # path: goal/collision above still end the episode on their own.
        truncated = (not done) and self._step_count >= self.max_episode_steps

        return obs, reward, done, truncated, {"cost": cost}
