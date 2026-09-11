# SafeRL-Guided Autonomous Space Debris Capture Using Probabilistic Shields

## A Career-Oriented Research & Development Project

---

> **Project Title:** Probabilistic Shield-Augmented Reinforcement Learning for Autonomous Capture of Tumbling Space Debris
>
> **Domain:** Space Robotics · Safe Reinforcement Learning · Autonomous Systems
>
> **Simulation Environment:** Isaac Lab (NVIDIA Isaac Sim) — Justification in accompanying chat
>
> **Estimated Timeline:** 9–12 Months (Academic / Research-Grade)

---

## Abstract

The proliferation of defunct satellites and debris in Low Earth Orbit (LEO) and Geostationary Orbit (GEO) poses a critical threat to future space operations. Active Debris Removal (ADR) is widely acknowledged as a necessary engineering frontier, yet the core challenge — autonomously attaching a robotic arm or capture mechanism to a tumbling, uncooperative object — remains largely unsolved in a reliable, verifiable, and safe manner.

This project addresses that gap by extending the Safe Reinforcement Learning (SafeRL) paradigm, specifically the **probabilistic shielding** methodology, to the domain of space robotic capture. Unlike traditional RL-based approaches, which optimize purely for task success, this system incorporates a **runtime safety monitor** derived from probabilistic model checking. The shield evaluates the probability that a proposed action will lead to a safety violation (collision, structural overload, or uncontrolled drift) and either permits or overrides the agent's decision — ensuring that the robot operates safely even under the extreme uncertainty introduced by the debris's stochastic tumbling dynamics.

The system is designed, trained, and validated inside **NVIDIA Isaac Lab**, a high-fidelity physics simulation environment that models rigid-body dynamics, contact forces, orbital microgravity, and 6-DOF manipulator kinematics with GPU-accelerated parallel rollouts. The project is structured into six clearly delineated phases, from mathematical formulation and environment design through to evaluation, documentation, and dissemination.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Core Technical Concepts](#2-core-technical-concepts)
3. [Simulation Software: NVIDIA Isaac Lab](#3-simulation-software-nvidia-isaac-lab)
4. [Phase 0: Foundation & Setup](#phase-0-foundation--setup)
5. [Phase 1: Problem Formulation](#phase-1-problem-formulation)
6. [Phase 2: Environment Design & Simulation](#phase-2-environment-design--simulation)
7. [Phase-3: Probabilistic Shield Design](#phase-3-probabilistic-shield-design)
8. [Phase 4: RL Agent Design & Integration](#phase-4-rl-agent-design--integration)
9. [Phase 5: Training, Evaluation & Benchmarking](#phase-5-training-evaluation--benchmarking)
10. [Phase 6: Documentation, Paper & Portfolio](#phase-6-documentation-paper--portfolio)
11. [Dependency Map & Technology Stack](#dependency-map--technology-stack)
12. [Expected Deliverables](#expected-deliverables)

---

## 1. Background & Motivation

### 1.1 The Space Debris Crisis

There are currently over **27,000 trackable debris objects** and millions of smaller fragments in Earth orbit. Even a single centimeter-sized fragment travelling at ~7.5 km/s in LEO carries the kinetic energy of a small grenade. As collisions generate more debris (the **Kessler Syndrome**), active removal becomes non-negotiable.

### 1.2 Why Capture Is Hard

A tumbling, defunct satellite presents several properties that make standard robotic capture planning unreliable:

- **Unknown inertia tensor:** The debris's mass distribution is not precisely known.
- **Stochastic angular velocity:** Tumbling rates may follow irregular, non-periodic patterns.
- **No cooperation:** No docking ports, no reflective markers, no communication.
- **Proximity hazard:** The capture robot itself risks collision if the approach trajectory is even slightly off.
- **Time-criticality:** Fuel constraints mean the approach window is finite.

### 1.3 Why Probabilistic Shields + RL?

- **Pure RL** optimizes task reward but cannot formally guarantee safety constraints will not be violated during exploration or deployment.
- **Rule-based systems** are brittle and cannot generalize to novel tumbling configurations.
- **Probabilistic Shields** provide a formal, runtime safety layer: they compute — in real time — the probability that an action leads to a safety-violating state, and intervene only when necessary. This keeps the RL agent's expressivity intact while providing a mathematically grounded safety guarantee.

This combination is the project's unique scientific and engineering contribution.

---

## 2. Core Technical Concepts

### 2.1 Markov Decision Process (MDP) Formulation

The debris-capture task is modeled as a **Constrained MDP (CMDP)**:

```
(S, A, T, R, C, γ, c_threshold)
```

- **S:** State space — 6-DOF pose of debris, 6-DOF pose of robot end-effector, angular velocity of debris, relative proximity, contact force readings
- **A:** Action space — delta joint velocities or Cartesian EE velocity commands
- **T:** Transition function — partially stochastic due to debris tumbling uncertainty
- **R:** Reward — shaped for approach, alignment, grasp success
- **C:** Cost function — penalizes near-collision events, force limit violation
- **γ:** Discount factor
- **c_threshold:** Maximum allowable cumulative safety cost

### 2.2 Probabilistic Shielding

A **probabilistic shield** is a formal safety enforcer derived from **Probabilistic Model Checking (PMC)**:

1. The environment dynamics are approximated as a **Discrete-Time Markov Chain (DTMC)** or **Markov Decision Process** over an abstracted state space.
2. A **safety specification** is written in **Probabilistic Computation Tree Logic (PCTL)**, e.g.:
   - `P≤0.05 [ F≤20 collision ]` — "The probability of collision within 20 steps must be ≤ 5%."
3. **PRISM** or **Storm** model checker computes, for each (state, action) pair, the probability of violating the specification.
4. At runtime, the shield:
   - Queries the RL agent for its preferred action.
   - Computes the safety probability for that action.
   - If safe → execute. If unsafe → replace with the least-restrictive safe action.

### 2.3 Tumbling Debris Dynamics

The debris is modeled as a rigid body with:

- Euler's rotation equations for 3D angular momentum
- Stochastic noise on angular velocity to simulate unknown mass distribution
- Probabilistic bounds on tumbling rate derived from observational data ranges (typically 1°/s to 30°/s for real debris)

---

## 3. Simulation Software: NVIDIA Isaac Lab

### 3.1 What Is Isaac Lab?

**Isaac Lab** (formerly Isaac Gym / OmniIsaac) is NVIDIA's open-source robot learning framework built on **Isaac Sim**, which uses **PhysX 5** for physics and **USD (Universal Scene Description)** for scene composition.

- GitHub: `https://github.com/isaac-sim/IsaacLab`
- Built on: Omniverse + PhysX 5 + Python
- Supports: GPU-parallelized RL training (thousands of simultaneous environments)
- Physics fidelity: Rigid body, articulated body, contact, soft body
- Integrates natively with: `rsl_rl`, `stable-baselines3`, `skrl`, `rl_games`

### 3.2 Why Isaac Lab for This Project

Isaac Lab is the most appropriate simulation environment for this project for the following reasons:

| Requirement | Isaac Lab Capability |
|---|---|
| Microgravity / zero-g simulation | PhysX gravity can be set to 0 per-scene; custom gravity vectors supported |
| Rigid body tumbling with noise | Full 6-DOF rigid body with custom torque injection scripts |
| 6-DOF robot arm | Native support for URDF/USD robot import (e.g., Franka, UR5, or custom) |
| Contact and force sensing | PhysX contact sensors, force/torque sensors on joints and EE |
| RL training at scale | GPU-parallel rollouts (4096+ envs simultaneously on a single GPU) |
| Custom reward/cost shaping | Python-based reward manager, fully scriptable |
| ROS 2 bridge | Available for future hardware-in-the-loop extensions |
| Open source & actively maintained | Apache 2.0 license, NVIDIA-backed |

No other simulator simultaneously offers **physics fidelity**, **RL-native GPU parallelism**, and **sensor realism** at the level Isaac Lab does. Alternatives like PyBullet lack GPU parallelism; MuJoCo lacks orbital/contact fidelity for this use case; Gazebo lacks native RL integration.

---

## Phase 0: Foundation & Setup

**Duration:** 2–3 Weeks
**Goal:** Get your full development environment operational and validate the simulation runs correctly.

---

### 0.1 Hardware & OS Requirements

- **OS:** Ubuntu 22.04 LTS (strongly recommended; Windows WSL2 also supported but slower)
- **GPU:** NVIDIA RTX 3080 / 4080 or better (VRAM ≥ 10 GB)
- **RAM:** 32 GB minimum
- **Storage:** 100 GB SSD (Isaac Sim + assets are large)
- **CUDA:** 12.1 or higher

---

### 0.2 Installing Isaac Sim & Isaac Lab

#### Step 1: Install NVIDIA Drivers & CUDA

```bash
# Check your driver version
nvidia-smi

# Install CUDA 12.1 toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get install cuda-12-1
```

#### Step 2: Install Isaac Sim via pip (Recommended for Isaac Lab 2.x)

```bash
# Create a conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Sim Python packages
pip install isaacsim==4.2.0.2 \
    isaacsim-rl isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    --extra-index-url https://pypi.nvidia.com
```

#### Step 3: Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install  # installs all dependencies including rsl_rl, skrl, sb3
```

#### Step 4: Validate Installation

```bash
# Run a basic test environment
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
# Should open the Omniverse viewport with an empty stage
```

#### Step 5: Install Additional Dependencies for This Project

```bash
pip install prism-api stable-baselines3 gymnasium torch \
    scipy numpy matplotlib pandas wandb
```

> **PRISM model checker** must also be installed separately:
> Download from `https://www.prismmodelchecker.org/download.php`
> Java 17+ required. Add PRISM to PATH.

---

### 0.3 Project Repository Structure

Set up your project repository with the following structure:

```
saferl_debris_capture/
├── envs/
│   ├── debris_capture_env.py       # Custom Isaac Lab environment
│   ├── debris_dynamics.py          # Tumbling debris model
│   └── reward_shaping.py           # Reward + cost functions
├── shield/
│   ├── prism_models/               # .pm PRISM model files
│   ├── shield_query.py             # Runtime shield interface
│   ├── shield_wrapper.py           # Wraps RL policy with shield
│   └── abstraction.py              # State space abstraction for PMC
├── agents/
│   ├── ppo_agent.py
│   ├── sac_agent.py
│   └── shielded_agent.py           # Agent + shield integration
├── training/
│   ├── train.py
│   └── config/
│       ├── ppo_config.yaml
│       └── sac_config.yaml
├── evaluation/
│   ├── benchmark.py
│   └── metrics.py
├── assets/
│   ├── robot/                      # URDF/USD robot files
│   └── debris/                     # USD debris model
├── notebooks/
│   └── analysis.ipynb
└── docs/
    └── paper_draft.md
```

---

## Phase 1: Problem Formulation

**Duration:** 2–3 Weeks
**Goal:** Formalize the entire problem mathematically before writing a single line of simulation code.

---

### 1.1 Define the State Space

Define exactly what the RL agent observes:

| State Variable | Dimension | Description |
|---|---|---|
| Debris position (relative) | 3 | x, y, z relative to robot base |
| Debris orientation (quaternion) | 4 | Tumbling orientation |
| Debris angular velocity | 3 | ωx, ωy, ωz |
| End-effector position | 3 | Cartesian EE position |
| End-effector orientation | 4 | EE quaternion |
| Joint positions | 6–7 | Robot joint angles |
| Joint velocities | 6–7 | Robot joint velocities |
| Contact forces at EE | 3 | Force vector at gripper |
| Time remaining | 1 | Normalized episode step count |

**Total observation dimension:** ~34–36

### 1.2 Define the Action Space

- **Option A (Joint Space):** Delta joint velocities — 6 or 7 continuous values ∈ [−1, 1], scaled to max joint velocity
- **Option B (Cartesian Space):** Delta EE position + delta EE orientation — 6 continuous values

Recommend **Option B** for interpretability and easier reward shaping. Use Isaac Lab's differential IK controller internally.

### 1.3 Define Reward Function

```
R(s, a) = w1 * r_approach + w2 * r_align + w3 * r_grasp - w4 * r_collision_penalty - w5 * r_effort
```

Where:
- `r_approach` = negative distance from EE to debris capture point (dense)
- `r_align` = cosine similarity between EE orientation and target docking orientation
- `r_grasp` = +1000 sparse reward for successful latch
- `r_collision_penalty` = force magnitude when unintended contact is detected
- `r_effort` = L2 norm of action (penalize jerky motion)

### 1.4 Define Safety Constraints

Formal safety constraints:

1. **Collision constraint:** Contact force at any non-capture surface < 5 N
2. **Proximity constraint:** Minimum safe standoff distance of 0.15 m from debris body
3. **Joint limit constraint:** All joint positions within 95% of URDF limits
4. **Angular velocity tracking:** EE angular velocity < 30°/s to avoid capture point overshoot

### 1.5 Write the PCTL Safety Specification

In PRISM syntax:

```prism
// Safety Property 1: Collision probability within 50 steps ≤ 5%
P<=0.05 [ F<=50 collision_state ]

// Safety Property 2: Probability of remaining in safe zone > 90%
P>=0.90 [ G<=100 safe_zone ]
```

Document these specifications in `shield/prism_models/safety_spec.pctl`.

### 1.6 Deliverables for Phase 1

- [ ] Written MDP formulation document (LaTeX or Markdown)
- [ ] Reward function specification with weight rationale
- [ ] Safety constraint document
- [ ] PCTL specification file
- [ ] State/action space diagram

---

## Phase 2: Environment Design & Simulation

**Duration:** 4–5 Weeks
**Goal:** Build a fully functional, realistic Isaac Lab environment for the debris capture task.

---

### 2.1 Design the Debris Asset

#### Option A: Use a Generic Rigid Body (Fast Start)

Create a simple box or cylinder USD asset in Isaac Lab:

```python
# envs/debris_dynamics.py
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

debris_cfg = RigidObjectCfg(
    prim_path="/World/Debris",
    spawn=sim_utils.CuboidCfg(
        size=(1.5, 0.8, 0.8),  # Approximate dead satellite dimensions
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=500.0),  # ~500 kg
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.7)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(3.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
```

#### Option B: Import a Real Satellite CAD Model (Recommended for Publications)

1. Download a free satellite mesh from NASA's 3D Resources: `https://nasa3d.arc.nasa.gov`
2. Convert to USD using NVIDIA Omniverse's CAD Importer
3. Assign physics properties in the USD stage

### 2.2 Implement Stochastic Tumbling

This is the central novelty of the environment. Implement tumbling as a controlled stochastic process:

```python
# envs/debris_dynamics.py
import torch

class TumblingDebrisModel:
    """
    Simulates unsteady debris tumbling with stochastic angular velocity injection.
    Based on Euler's rotation equations with noise perturbation.
    """

    def __init__(self, inertia_tensor: torch.Tensor, noise_std: float = 0.05):
        self.I = inertia_tensor  # 3x3 inertia tensor (kg.m^2)
        self.noise_std = noise_std

    def compute_torque(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Euler's equation: tau = I * alpha + omega x (I * omega)
        We apply a stochastic torque to simulate unknown mass distribution effects.
        """
        gyroscopic_torque = torch.linalg.cross(omega, self.I @ omega)
        stochastic_torque = torch.randn_like(omega) * self.noise_std
        return -gyroscopic_torque + stochastic_torque

    def sample_initial_omega(self) -> torch.Tensor:
        """
        Sample initial angular velocity uniformly from realistic debris tumble range.
        Based on ESA debris rotation rate surveys: 1–30 deg/s is common.
        """
        omega_magnitude = torch.FloatTensor(1).uniform_(0.017, 0.524)  # rad/s
        direction = torch.randn(3)
        direction /= direction.norm()
        return omega_magnitude * direction
```

Apply the computed torque via Isaac Lab's articulation API at each simulation step.

### 2.3 Build the Robot Environment

Use the **Franka Emika Panda** (7-DOF, freely available in Isaac Lab) or a custom 6-DOF arm:

```python
# envs/debris_capture_env.py
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

@configclass
class DebrisCaptureEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 15.0       # 15 second episodes
    decimation: int = 4                  # RL acts every 4 physics steps
    num_observations: int = 36
    num_actions: int = 6

    # Robot configuration
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"panda_joint.*": 0.0},
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=6.0,
        replicate_physics=True,
    )
```

### 2.4 Set Up Zero-Gravity Scene

Microgravity must be explicitly configured:

```python
# In your scene setup
sim_cfg = sim_utils.SimulationCfg(
    dt=1/120,           # 120 Hz physics
    render_interval=4,
    gravity=(0.0, 0.0, 0.0),  # Zero gravity for orbital scenario
)
```

Also add a slow drift velocity to the debris to simulate orbital relative motion:

```python
# Apply a small linear drift (0.01–0.05 m/s) to the debris root state
debris.write_root_velocity_to_sim(
    root_velocity=torch.tensor([0.02, 0.005, 0.0, *omega.tolist()])
)
```

### 2.5 Implement Observation, Reward & Cost Computation

```python
def _get_observations(self) -> dict:
    """Compute the full observation vector for the RL agent."""
    debris_pos = self.debris.data.root_pos_w - self.scene.env_origins
    debris_quat = self.debris.data.root_quat_w
    debris_ang_vel = self.debris.data.root_ang_vel_w
    ee_pos = self.robot.data.body_pos_w[:, self.ee_idx, :]
    ee_quat = self.robot.data.body_quat_w[:, self.ee_idx, :]
    joint_pos = self.robot.data.joint_pos
    joint_vel = self.robot.data.joint_vel
    contact_forces = self.contact_sensor.data.net_forces_w

    obs = torch.cat([
        debris_pos, debris_quat, debris_ang_vel,
        ee_pos, ee_quat, joint_pos, joint_vel,
        contact_forces.flatten(start_dim=1),
    ], dim=-1)

    return {"policy": obs}

def _get_rewards(self) -> torch.Tensor:
    """Compute shaped reward."""
    distance_to_capture = torch.norm(
        self.ee_pos - self.debris_capture_point, dim=-1
    )
    approach_reward = -distance_to_capture
    alignment_reward = self._compute_orientation_alignment()
    grasp_reward = (distance_to_capture < 0.05).float() * 1000.0
    effort_penalty = -0.001 * torch.norm(self.actions, dim=-1)
    return approach_reward + alignment_reward + grasp_reward + effort_penalty

def _get_safety_cost(self) -> torch.Tensor:
    """Compute safety cost (separate from reward — used by shield)."""
    collision_cost = (self.contact_forces.norm(dim=-1) > 5.0).float()
    proximity_cost = (self.distance_to_debris_body < 0.15).float()
    joint_limit_cost = self._check_joint_limits()
    return collision_cost + proximity_cost + joint_limit_cost
```

### 2.6 Deliverables for Phase 2

- [ ] Fully running Isaac Lab environment (zero gravity, tumbling debris, robot arm)
- [ ] Reward and cost functions validated with random policy
- [ ] Video recording of the environment with tumbling debris
- [ ] Environment unit tests confirming observation/action shapes

---

## Phase 3: Probabilistic Shield Design

**Duration:** 3–4 Weeks
**Goal:** Build the probabilistic shield that will enforce safety on the RL agent at runtime.

---

### 3.1 State Space Abstraction

The continuous state space must be abstracted into a finite-state representation for PRISM model checking. This is a critical and non-trivial step.

**Abstraction strategy — Tile Coding:**

Partition the relevant safety-critical subspace into discrete cells:

| Variable | Bins | Range |
|---|---|---|
| Distance to debris body | 10 | [0.0, 2.0] m |
| Relative approach velocity | 8 | [−0.5, 0.5] m/s |
| Debris angular velocity magnitude | 6 | [0, 0.6] rad/s |
| Contact force | 4 | [0, 20] N |

This yields a total abstract state space of ~1920 discrete states.

```python
# shield/abstraction.py

class StateAbstractor:
    """Maps continuous Isaac Lab observations to discrete PRISM states."""

    BINS = {
        "distance": torch.linspace(0.0, 2.0, 10),
        "approach_vel": torch.linspace(-0.5, 0.5, 8),
        "debris_omega_mag": torch.linspace(0.0, 0.6, 6),
        "contact_force": torch.linspace(0.0, 20.0, 4),
    }

    def abstract(self, obs: torch.Tensor) -> int:
        """Convert observation tensor to discrete PRISM state index."""
        distance = torch.norm(obs[0:3])
        approach_vel = obs[6]
        omega_mag = torch.norm(obs[7:10])
        force_mag = torch.norm(obs[-3:])

        d_bin = torch.bucketize(distance, self.BINS["distance"])
        v_bin = torch.bucketize(approach_vel, self.BINS["approach_vel"])
        w_bin = torch.bucketize(omega_mag, self.BINS["debris_omega_mag"])
        f_bin = torch.bucketize(force_mag, self.BINS["contact_force"])

        # Encode as single integer
        return (d_bin * 8 * 6 * 4 + v_bin * 6 * 4 + w_bin * 4 + f_bin).item()
```

### 3.2 Write the PRISM Probabilistic Model

Create `shield/prism_models/debris_capture.pm`:

```prism
dtmc

// State variables
module debris_capture

    distance   : [0..9] init 9;   // Discretized distance bin
    approach_v : [0..7] init 4;   // Approach velocity bin
    omega_mag  : [0..5] init 0;   // Debris angular velocity bin
    contact_f  : [0..3] init 0;   // Contact force bin

    // Transition probabilities estimated from environment rollouts
    // These are learned from Phase 2 data collection

    [] distance > 0 & contact_f = 0 ->
        0.85 : (distance' = distance - 1) & (contact_f' = 0)
        + 0.10 : (distance' = distance) & (contact_f' = 0)
        + 0.05 : (distance' = distance - 1) & (contact_f' = 1);

    [] distance = 0 ->
        0.70 : (contact_f' = 0)   // Successful grasp, no collision
        + 0.30 : (contact_f' = 2); // Collision on grasp attempt

endmodule

// Labels
label "collision" = contact_f >= 2;
label "safe" = contact_f = 0 & distance > 1;
label "captured" = distance = 0 & contact_f = 0;
```

> **Note:** Transition probabilities are **not hand-tuned**. They are estimated automatically from environment rollout statistics collected in Phase 2 (see Section 3.3).

### 3.3 Automated Transition Probability Estimation

Run a **data collection phase** using a random policy to estimate transition probabilities for the PRISM model:

```python
# shield/estimate_transitions.py

def collect_transition_statistics(env, num_episodes=2000):
    """
    Run random policy, record (s_abstract, a, s_abstract') transitions.
    Use these to estimate P(s' | s, a) for the PRISM model.
    """
    abstractor = StateAbstractor()
    transitions = defaultdict(lambda: defaultdict(int))

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs_prev = obs
            obs, reward, cost, done, info = env.step(action)

            s_prev = abstractor.abstract(obs_prev)
            s_next = abstractor.abstract(obs)
            transitions[(s_prev, action_bin)][s_next] += 1

    # Normalize to probabilities
    transition_probs = {}
    for (s, a), next_counts in transitions.items():
        total = sum(next_counts.values())
        transition_probs[(s, a)] = {
            s_prime: count / total
            for s_prime, count in next_counts.items()
        }

    return transition_probs
```

Export these probabilities to auto-generate the PRISM `.pm` file using a templating script.

### 3.4 Model Checking with PRISM

Run PRISM to compute safety probabilities for each (state, action) pair:

```bash
prism shield/prism_models/debris_capture.pm \
      shield/prism_models/safety_spec.pctl \
      -exportresults shield/safety_table.csv
```

This generates a CSV table: `(state, action) → P(violation)`.

### 3.5 Implement the Runtime Shield

```python
# shield/shield_wrapper.py
import pandas as pd

class ProbabilisticShield:
    """
    Runtime safety enforcer. Wraps the RL policy.
    Given a state, proposes action from policy, checks safety,
    and either permits or overrides the action.
    """

    def __init__(self, safety_table_path: str, threshold: float = 0.05):
        self.safety_table = pd.read_csv(safety_table_path, index_col=[0, 1])
        self.threshold = threshold
        self.abstractor = StateAbstractor()
        self.interventions = 0
        self.total_steps = 0

    def shield(self, obs: torch.Tensor, policy_action: torch.Tensor,
               candidate_actions: list) -> torch.Tensor:
        """
        Main shielding function.
        Returns: (final_action, was_shielded)
        """
        self.total_steps += 1
        abstract_state = self.abstractor.abstract(obs)
        action_bin = self._bin_action(policy_action)

        # Check if policy action is safe
        p_violation = self._query_safety(abstract_state, action_bin)

        if p_violation <= self.threshold:
            return policy_action, False  # Policy action is safe

        # Find safest alternative action
        self.interventions += 1
        safest_action = self._find_safest_action(abstract_state, candidate_actions)
        return safest_action, True

    def _query_safety(self, state: int, action_bin: int) -> float:
        try:
            return self.safety_table.loc[(state, action_bin), "p_violation"]
        except KeyError:
            return 0.0  # Unseen state: optimistically assume safe

    def intervention_rate(self) -> float:
        return self.interventions / max(1, self.total_steps)
```

### 3.6 Deliverables for Phase 3

- [ ] State abstraction module with unit tests
- [ ] Automated transition probability estimator
- [ ] PRISM model file (auto-generated)
- [ ] PRISM model checking run successful, safety table generated
- [ ] Shield wrapper module with intervention rate logging
- [ ] Shield validation: demonstrate it blocks known-unsafe actions

---

## Phase 4: RL Agent Design & Integration

**Duration:** 3–4 Weeks
**Goal:** Train a capable RL agent and integrate it with the probabilistic shield.

---

### 4.1 Algorithm Selection

Two algorithms are recommended — train both for comparison:

| Algorithm | Library | Why |
|---|---|---|
| **PPO** (Proximal Policy Optimization) | `rsl_rl` or `sb3` | Stable, GPU-native, standard baseline |
| **SAC** (Soft Actor-Critic) | `skrl` | Better sample efficiency, continuous action |

The shielded agent will ultimately use SAC due to its ability to maintain an explicit action distribution, making it easier to sample candidate alternative actions for the shield.

### 4.2 Policy Network Architecture

```python
# agents/ppo_agent.py
import torch.nn as nn

class CapturePolicy(nn.Module):
    """
    Actor-Critic network for debris capture.
    Input: observation (36-dim)
    Output: action mean + log_std (6-dim each)
    """

    def __init__(self, obs_dim=36, act_dim=6, hidden=[256, 256, 128]):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ELU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.Linear(hidden[2], act_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.Linear(hidden[2], 1)
        )
```

### 4.3 Shielded Agent Wrapper

```python
# agents/shielded_agent.py

class ShieldedAgent:
    """
    Wraps any RL policy with the probabilistic shield.
    During training: shield is active but recorded (not blocking training gradients).
    During evaluation: shield is active and enforcing.
    """

    def __init__(self, policy, shield, num_candidates=10):
        self.policy = policy
        self.shield = shield
        self.num_candidates = num_candidates

    def act(self, obs: torch.Tensor, deterministic=False):
        with torch.no_grad():
            policy_action = self.policy.act(obs, deterministic)

            # Generate candidate alternative actions (Gaussian samples)
            candidates = [
                self.policy.act(obs, deterministic=False)
                for _ in range(self.num_candidates)
            ]

        final_action, was_shielded = self.shield.shield(
            obs, policy_action, candidates
        )
        return final_action, was_shielded
```

### 4.4 Training Configuration

`training/config/ppo_config.yaml`:

```yaml
env:
  num_envs: 4096
  episode_length: 450   # steps (at 30 Hz decision rate = 15 sec)
  obs_dim: 36
  act_dim: 6

algorithm: PPO
ppo:
  learning_rate: 3.0e-4
  n_steps: 24
  batch_size: 4096
  n_epochs: 5
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.005
  vf_coef: 1.0
  max_grad_norm: 1.0

training:
  total_timesteps: 50_000_000
  save_interval: 500
  log_interval: 10
  checkpoint_dir: "checkpoints/ppo"

logging:
  wandb_project: "saferl_debris_capture"
  tags: ["ppo", "shielded", "phase4"]
```

### 4.5 Training Launch

```bash
# Train unshielded baseline (PPO)
./isaaclab.sh -p training/train.py \
    --task DebrisCapture \
    --algorithm ppo \
    --shield none \
    --run_name baseline_ppo

# Train shielded agent (PPO + Shield)
./isaaclab.sh -p training/train.py \
    --task DebrisCapture \
    --algorithm ppo \
    --shield probabilistic \
    --shield_threshold 0.05 \
    --run_name shielded_ppo
```

Track all runs with **Weights & Biases (wandb)**: `wandb login` before training.

### 4.6 Deliverables for Phase 4

- [ ] PPO baseline trained to convergence (~50M steps)
- [ ] SAC baseline trained to convergence
- [ ] Shielded PPO trained
- [ ] Shielded SAC trained
- [ ] Learning curves for all 4 agents logged to wandb
- [ ] Checkpoint selection (best policy by episode return)

---

## Phase 5: Training, Evaluation & Benchmarking

**Duration:** 3–4 Weeks
**Goal:** Rigorously evaluate all agents, prove shield's effectiveness, generate publication-quality results.

---

### 5.1 Evaluation Metrics

Design metrics across three axes: **Performance**, **Safety**, and **Shield Behavior**:

| Metric | Description | Unit |
|---|---|---|
| **Success Rate** | % of episodes ending in successful grasp | % |
| **Episode Return** | Mean cumulative reward per episode | scalar |
| **Collision Rate** | % of episodes with at least one collision event | % |
| **Cumulative Safety Cost** | Mean cost per episode | scalar |
| **Time to Capture** | Steps to successful grasp (successful episodes only) | steps |
| **Shield Intervention Rate** | % of steps where shield overrode policy action | % |
| **Shield Intervention Trend** | Does intervention rate decrease as training progresses? | over time |
| **Policy Performance Under Shield** | Return loss compared to unshielded | % |

### 5.2 Evaluation Protocol

Evaluate each agent over **1,000 episodes** (unseen random debris tumbling configs):

```python
# evaluation/benchmark.py

def evaluate_agent(agent, env, num_episodes=1000, shield=None):
    metrics = {
        "success_rate": [],
        "episode_return": [],
        "collision_rate": [],
        "cumulative_cost": [],
        "time_to_capture": [],
        "shield_interventions": [],
    }

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return, ep_cost, steps = 0, 0, 0
        collided = False

        while not done:
            if shield:
                action, shielded = shield.act(obs)
                metrics["shield_interventions"].append(int(shielded))
            else:
                action = agent.act(obs, deterministic=True)

            obs, reward, cost, done, info = env.step(action)
            ep_return += reward
            ep_cost += cost
            steps += 1
            if cost > 0:
                collided = True

        metrics["success_rate"].append(info.get("success", False))
        metrics["episode_return"].append(ep_return)
        metrics["collision_rate"].append(collided)
        metrics["cumulative_cost"].append(ep_cost)
        if info.get("success"):
            metrics["time_to_capture"].append(steps)

    return {k: np.mean(v) for k, v in metrics.items()}
```

### 5.3 Ablation Studies

Run the following ablation experiments to understand each component's contribution:

1. **No Shield + No Safety Cost:** Pure RL (baseline)
2. **No Shield + Safety Cost in Reward:** Constrained RL without shield
3. **Shield (threshold=0.10):** Looser shield
4. **Shield (threshold=0.05):** Target configuration
5. **Shield (threshold=0.01):** Very tight shield
6. **Shield with perfect transition model:** Upper bound
7. **Shield with 50% noisier debris dynamics:** Robustness test

### 5.4 Generate Publication-Quality Figures

Required figures for paper/portfolio:

1. **Learning curves:** Return and collision cost vs. timesteps (all agents overlaid)
2. **Shield intervention rate over training:** Shows agent learning to comply with shield
3. **Success rate vs. debris tumbling speed:** Performance degradation analysis
4. **Safety cost distribution:** Histogram for shielded vs. unshielded
5. **Capture trajectory visualization:** 3D plot of EE path to debris
6. **State visitation heatmap:** Where in state space did the agent spend time?

```python
# evaluation/metrics.py — Example: Plot intervention rate over training
import matplotlib.pyplot as plt

def plot_intervention_rate(log_dir):
    data = pd.read_csv(f"{log_dir}/shield_log.csv")
    smoothed = data["intervention_rate"].rolling(window=100).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(data["step"], smoothed, label="Shield Intervention Rate")
    plt.xlabel("Training Step")
    plt.ylabel("Intervention Rate")
    plt.title("Probabilistic Shield Intervention Rate Over Training")
    plt.legend()
    plt.savefig("figures/intervention_rate.png", dpi=300, bbox_inches="tight")
```

### 5.5 Deliverables for Phase 5

- [ ] Evaluation results table (all 4 agents × all metrics)
- [ ] Ablation study results
- [ ] All 6 publication figures generated at 300 DPI
- [ ] Statistical significance tests (Wilcoxon rank-sum or t-test) between shielded and unshielded
- [ ] Video demos of: (a) successful capture, (b) shield intervention during approach

---

## Phase 6: Documentation, Paper & Portfolio

**Duration:** 2–3 Weeks
**Goal:** Package the project for career impact — a preprint, GitHub repository, and portfolio entry.

---

### 6.1 Write the Research Paper

Structure the paper following ICRA / IROS / NeurIPS-SafeML format:

```
1. Introduction (1 page)
   - Space debris problem motivation
   - Limitations of existing ADR approaches
   - Contribution: SafeRL + probabilistic shields for capture

2. Related Work (1 page)
   - SafeRL methods: Lagrangian, CBF, shielding
   - Space robotics and ADR: ESA ClearSpace, JAXA HTV
   - Tumbling debris dynamics literature

3. Problem Formulation (1 page)
   - CMDP formulation
   - Safety specification in PCTL

4. Methodology (2 pages)
   - Environment description
   - Tumbling debris model
   - Shield design and state abstraction
   - RL agent architecture

5. Experiments (2 pages)
   - Setup
   - Results table
   - Ablation analysis
   - Discussion

6. Conclusion & Future Work (0.5 page)
   - Summary of contributions
   - Limitations
   - Future: Real hardware transfer, ROS 2 integration, multi-debris scenarios
```

Target venues: **IEEE ICRA 2026**, **IROS 2026**, **NeurIPS Workshop on Safe RL**, **arXiv (cs.RO + cs.LG)**

### 6.2 Prepare the GitHub Repository

The repository should be clean, documented, and immediately reproducible:

```
README.md
├── Project overview + GIF of simulation
├── Installation instructions (exact commands)
├── Quick-start (reproduce main results in 1 command)
├── Results table
└── Citation block (BibTeX)

CONTRIBUTING.md
LICENSE (MIT or Apache 2.0)
requirements.txt / environment.yaml
```

### 6.3 Portfolio Entry Checklist

- [ ] **Project page** (GitHub Pages or Notion): title, abstract, method figure, results, video
- [ ] **Demo video** (90 seconds): problem → method → result → safety demo
- [ ] **LinkedIn post** with key results image and link to GitHub
- [ ] **arXiv preprint** submitted before conference submission
- [ ] **Slide deck** (10 slides) for research presentations and interviews

---

## Dependency Map & Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT STACK                             │
├──────────────────┬──────────────────────────────────────────────┤
│ Simulation       │ NVIDIA Isaac Lab 2.x (Isaac Sim 4.x, PhysX5) │
│ Physics          │ PhysX 5 (rigid body, contact, zero-gravity)   │
│ Robot Model      │ Franka Panda (URDF → USD)                     │
│ RL Framework     │ rsl_rl (PPO), skrl (SAC)                      │
│ Safety Layer     │ PRISM Model Checker 4.x                       │
│ Deep Learning    │ PyTorch 2.x + CUDA 12.1                       │
│ Experiment Track │ Weights & Biases (wandb)                      │
│ Numerics         │ NumPy, SciPy, pandas                          │
│ Visualization    │ Matplotlib, Plotly                            │
│ Language         │ Python 3.10                                   │
│ Paper            │ LaTeX (Overleaf recommended)                  │
│ Version Control  │ Git + GitHub                                  │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## Expected Deliverables

| Phase | Primary Deliverable | Career Value |
|---|---|---|
| 0 | Working Isaac Lab install | Shows systems competency |
| 1 | MDP formulation document | Demonstrates rigorous research thinking |
| 2 | Custom Isaac Lab environment | Shows robotics/simulation engineering |
| 3 | Probabilistic shield module | Core novel contribution |
| 4 | 4 trained agents + learning curves | Shows ML engineering |
| 5 | Results table + figures + video | Publication-ready output |
| 6 | GitHub repo + paper draft + portfolio | Career-facing impact |

---

## Timeline Summary

| Week | Phase | Key Milestone |
|---|---|---|
| 1–3 | Phase 0 | Isaac Lab environment verified, repo initialized |
| 4–6 | Phase 1 | Full MDP formulation written and reviewed |
| 7–11 | Phase 2 | Custom environment running with tumbling debris |
| 12–15 | Phase 3 | Shield implemented, PRISM model verified |
| 16–19 | Phase 4 | All 4 agents trained and checkpointed |
| 20–23 | Phase 5 | All results, figures, and videos produced |
| 24–26 | Phase 6 | Paper draft, GitHub published, portfolio live |

---

*Document prepared for SafeRL Space Debris Capture Project — v1.0*
*This is a living document; update each phase section with results as you progress.*
