"""PyBullet-backed 3D safe-navigation environment (SafeNav3DEnv).

Each instance owns its own PyBullet connection (p.DIRECT for training,
p.GUI for demos) so multiple envs never collide on a single shared
connection.
"""
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym

# Observation layout: [agent_pos(3), agent_vel(3), goal_pos(3)] then one
# [pos(3), vel(3)] block per hazard. The shield indexes hazards with these,
# so they must stay in sync with _get_obs().
OBS_HEADER_LEN = 9
OBS_PER_HAZARD = 6

PHYSICS_HZ = 240.0  # PyBullet's default simulation rate

# Discrete action -> unit thrust direction in the world frame. The shield has
# to predict where an action takes the agent, so this table is the single
# definition of what an action *means*; step() and SafetyShield both read it
# rather than each hardcoding their own copy.
ACTION_THRUST_DIRS = np.array([
    [0.0,  1.0, 0.0],   # 0: +Y
    [0.0, -1.0, 0.0],   # 1: -Y
    [-1.0, 0.0, 0.0],   # 2: -X
    [1.0,  0.0, 0.0],   # 3: +X
], dtype=np.float64)

# 3D free flight appends +Z/-Z after the planar four, so indices 0-3 mean
# exactly what they meant in 2D and scenarios and tests written against the
# planar table stay valid under either.
ACTION_THRUST_DIRS_3D = np.vstack([
    ACTION_THRUST_DIRS,
    [[0.0, 0.0,  1.0],   # 4: +Z
     [0.0, 0.0, -1.0]],  # 5: -Z
])


def thrust_dirs(dims):
    """The action table for a planar (2) or free-flight (3) env."""
    if dims == 2:
        return ACTION_THRUST_DIRS
    if dims == 3:
        return ACTION_THRUST_DIRS_3D
    raise ValueError(f"dims must be 2 or 3, got {dims!r}")

# sphere2.urdf's base mass. globalScaling resizes the geometry but does NOT
# rescale mass, so this holds regardless of the 0.5 scaling in reset().
# Thrust acceleration available to the agent is force_mag / AGENT_MASS.
AGENT_MASS = 10.0


class SafeNav3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "direct"]}

    def __init__(self, size=10, max_hazards=5, curriculum=False, render_mode="direct",
                 force_mag=48.0, goal_threshold=1.0, hazard_threshold=1.3,
                 sim_substeps=10, agent_friction=0.0, max_episode_steps=1000,
                 debris_min_speed=0.3, debris_max_speed=1.2,
                 debris_speed_ramp_episodes=200, bounds_margin=5.0,
                 out_of_bounds_penalty=-100.0, sensor_range=None, dims=2):
        super().__init__()
        # dims=2 is the planar env every result through phase 10 was produced
        # on: gravity, a ground plane, x/y thrust. dims=3 is free flight -- no
        # gravity, no plane, thrust and debris motion on all three axes.
        self.dims = int(dims)
        self.thrust_dirs = thrust_dirs(self.dims)
        self._axes = (0, 1, 2) if self.dims == 3 else (0, 1)
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
        self.bounds_margin = bounds_margin
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.debris_min_speed = debris_min_speed
        self.debris_max_speed = debris_max_speed
        self.debris_speed_ramp_episodes = debris_speed_ramp_episodes
        self.sensor_range = sensor_range
        self.episode_count = 0
        self._step_count = 0
        self._client = -1  # PyBullet client ID (set on first reset)
        self.hazard_positions = []
        self.hazard_velocities = []
        self._hazard_ids = []
        # one env step advances the sim by this much wall-clock, so debris
        # drift matches the agent's control timescale
        self.step_dt = sim_substeps / PHYSICS_HZ

        self.action_space = gym.spaces.Discrete(len(self.thrust_dirs))
        self.observation_space = gym.spaces.Box(
            low=-20, high=20,
            shape=(OBS_HEADER_LEN + OBS_PER_HAZARD * max_hazards,),
            dtype=np.float32
        )
        goal_z = size - 1 if self.dims == 3 else 0.5
        self.goal_pos = np.array([size - 1, size - 1, goal_z], dtype=np.float32)

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
        if self.dims == 3:
            # free flight: nothing to fall toward and nothing to rest on
            p.setGravity(0, 0, 0, physicsClientId=cid)
            start = [0.0, 0.0, 0.0]
        else:
            p.setGravity(0, 0, -9.81, physicsClientId=cid)
            p.loadURDF("plane.urdf", physicsClientId=cid)
            start = [0, 0, 0.5]

        self.agent_id = p.loadURDF(
            "sphere2.urdf", start, globalScaling=0.5,
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

    def _debris_speed_range(self):
        """Speed band to sample from. Under curriculum the upper bound ramps
        from min to max over debris_speed_ramp_episodes, so early episodes
        face near-static debris and later ones face the full speed."""
        if not self.curriculum or self.debris_speed_ramp_episodes <= 0:
            return self.debris_min_speed, self.debris_max_speed
        progress = min(1.0, self.episode_count / self.debris_speed_ramp_episodes)
        upper = self.debris_min_speed + progress * (self.debris_max_speed - self.debris_min_speed)
        return self.debris_min_speed, upper

    def _setup_hazards(self):
        cid = self._client
        if self.curriculum:
            num_hazards = min(self.max_hazards, 1 + self.episode_count // 50)
        else:
            num_hazards = self.max_hazards

        lo_speed, hi_speed = self._debris_speed_range()

        self.hazard_positions = []
        self.hazard_velocities = []
        self._hazard_ids = []
        for _ in range(num_hazards):
            h_pos = [
                float(np.random.uniform(1, self.size - 2)),
                float(np.random.uniform(1, self.size - 2)),
                # the planar branch draws nothing, so 2D consumes the global
                # RNG exactly as before and every published episode reproduces
                float(np.random.uniform(1, self.size - 2)) if self.dims == 3 else 0.5,
            ]
            body_id = p.loadURDF("r2d2.urdf", h_pos, globalScaling=0.6,
                                 physicsClientId=cid)
            # mass 0 makes debris kinematic: we drive their positions directly
            # each step instead of letting gravity drop them through the floor
            # or letting contact impulses shove them around.
            p.changeDynamics(body_id, -1, mass=0, physicsClientId=cid)

            if self.dims == 3:
                # uniform over the sphere: a normalised Gaussian draw, not a
                # uniform pair of angles, which would bunch headings at the poles
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u) or 1.0
                speed = float(np.random.uniform(lo_speed, hi_speed))
                h_vel = [speed * float(u[0]), speed * float(u[1]), speed * float(u[2])]
            else:
                heading = float(np.random.uniform(0, 2 * np.pi))
                speed = float(np.random.uniform(lo_speed, hi_speed))
                h_vel = [speed * float(np.cos(heading)), speed * float(np.sin(heading)), 0.0]

            self.hazard_positions.append(h_pos)
            self.hazard_velocities.append(h_vel)
            self._hazard_ids.append(body_id)

    def _advance_hazards(self):
        """Linear drift with reflection off the play-area boundary.

        Bouncing (rather than wrapping or respawning) keeps the debris count
        and the observation layout constant, and avoids an object teleporting
        across the field into the agent's path, which would be unavoidable by
        any policy and would poison the safety signal.
        """
        cid = self._client
        for i, (pos, vel) in enumerate(zip(self.hazard_positions, self.hazard_velocities)):
            for axis in self._axes:
                pos[axis] += vel[axis] * self.step_dt
                if pos[axis] < 0.0:
                    pos[axis] = -pos[axis]
                    vel[axis] = -vel[axis]
                elif pos[axis] > self.size:
                    pos[axis] = 2.0 * self.size - pos[axis]
                    vel[axis] = -vel[axis]
            p.resetBasePositionAndOrientation(
                self._hazard_ids[i], pos, [0, 0, 0, 1], physicsClientId=cid
            )

    def _build_obs(self, sensor_limited=True):
        cid = self._client
        pos, _ = p.getBasePositionAndOrientation(self.agent_id,
                                                  physicsClientId=cid)
        vel, _ = p.getBaseVelocity(self.agent_id, physicsClientId=cid)
        agent_pos = np.array(pos)
        obs = list(pos) + list(vel) + list(self.goal_pos)
        for h_pos, h_vel in zip(self.hazard_positions, self.hazard_velocities):
            if (sensor_limited and self.sensor_range is not None
                    and np.linalg.norm(np.array(h_pos) - agent_pos) > self.sensor_range):
                obs.extend([0.0] * OBS_PER_HAZARD)
            else:
                obs.extend(h_pos)
                obs.extend(h_vel)
        target_len = self.observation_space.shape[0]
        while len(obs) < target_len:
            obs.extend([0.0] * OBS_PER_HAZARD)
        return np.array(obs[:target_len], dtype=np.float32)

    def _get_obs(self):
        return self._build_obs(sensor_limited=True)

    def get_true_obs(self):
        """Full observation with all hazards regardless of sensor range.

        Used by the safety shield, which has privileged access to true
        hazard state — it is a dedicated safety system, not limited by
        the policy's sensor model (see phase 7 design decision).
        """
        return self._build_obs(sensor_limited=False)

    def _out_of_bounds(self, pos):
        """Has the agent drifted out of the play area by more than the margin?"""
        lo, hi = -self.bounds_margin, self.size + self.bounds_margin
        return not all(lo <= float(pos[ax]) <= hi for ax in self._axes)

    def step(self, action):
        cid = self._client
        force = (self.thrust_dirs[int(action)] * self.force_mag).tolist()

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

        # move debris before reading the observation and before the collision
        # check, so both see this step's positions rather than last step's
        self._advance_hazards()

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
        out_of_bounds = 0
        if not done and self._out_of_bounds(obs[0:3]):
            # Terminal task failure, not a time limit: the agent has left the
            # region the task is defined over and cannot come back on its own.
            reward += self.out_of_bounds_penalty
            done = True
            out_of_bounds = 1

        truncated = (not done) and self._step_count >= self.max_episode_steps

        return obs, reward, done, truncated, {"cost": cost, "out_of_bounds": out_of_bounds}
