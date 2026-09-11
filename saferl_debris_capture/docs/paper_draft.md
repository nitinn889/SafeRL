# Paper Draft — Probabilistic Shield-Augmented RL for Space Debris Capture

**Working Title:** Probabilistic Shield-Augmented Reinforcement Learning for Autonomous Capture of Tumbling Space Debris in NVIDIA Isaac Lab

**Target Venues:** ICRA 2027 / IROS 2027 / IEEE Robotics and Automation Letters (RA-L)

---

## 1. Abstract (Draft)

[150–200 words]

The proliferation of defunct satellites in Low Earth Orbit poses an existential threat to future space operations. Active Debris Removal (ADR) via autonomous robotic capture is technically non-trivial: the target debris tumbles unpredictably with an unknown inertia tensor and presents no cooperative docking interface. Pure reinforcement learning methods optimise for task success but provide no formal safety guarantees during exploration or deployment. We present a probabilistic shield-augmented RL framework that enforces runtime safety constraints derived from probabilistic model checking (PMC). The shield is built from a PRISM-verified Markov chain abstraction of the debris-capture environment and intervenes — replacing the RL agent's proposed action with the least-restrictive safe alternative — whenever the predicted collision probability exceeds a user-defined threshold. We train PPO and SAC agents inside NVIDIA Isaac Lab, a GPU-parallelised physics simulation environment, with and without the shield. Experimental results across 200 evaluation episodes demonstrate that our shielded agents achieve [X]% fewer collision events and a [Y]% higher successful capture rate compared to unshielded baselines, with an average shield intervention rate of [Z]%.

---

## 2. Introduction

### 2.1 Motivation
- Kessler syndrome risk
- Active Debris Removal state of the art
- Why pure RL is insufficient
- Our contribution: runtime safety via probabilistic shielding

### 2.2 Contributions
1. First application of PRISM-based probabilistic shielding to 6-DOF space debris capture
2. GPU-vectorised debris dynamics model with stochastic tumbling noise
3. Isaac Lab custom environment with full contact/force sensing
4. Comprehensive benchmarking: PPO vs SAC, shielded vs unshielded

---

## 3. Related Work

### 3.1 Safe Reinforcement Learning
- Constrained MDPs (Altman 1999)
- CPO (Achiam et al., 2017)
- Safe exploration (Berkenkamp et al., 2017 — Lyapunov)
- Shielding (Jansen et al., 2020 CONCUR)

### 3.2 Space Robotics and Debris Capture
- JAXA HTV-X capture arm
- ESA e.Deorbit study
- DLR CAESAR
- RL approaches: (Hovell & Ulrich, 2021), (Capuano et al., 2020)

### 3.3 Probabilistic Model Checking
- PRISM tool (Kwiatkowska et al., 2011)
- Runtime shields in autonomous systems

---

## 4. Problem Formulation

### 4.1 Constrained MDP
Define the CMDP (S, A, T, R, C, γ, c_threshold).

### 4.2 State and Action Spaces
- Table: 36-D observation layout
- 6-D Cartesian EE action space

### 4.3 Reward and Cost Functions
- Equation: R(s,a) = w1*r_app + w2*r_align + w3*r_grasp − w4*r_col − w5*r_eff
- Cost: C(s,a) = 1 if F_contact > F_threshold

---

## 5. Simulation Environment

### 5.1 NVIDIA Isaac Lab
- Physics: PhysX 5, contact sensing, 6-DOF articulation
- GPU parallelism: up to 4096 envs

### 5.2 Tumbling Debris Model
- Euler's rotation equations
- Stochastic angular velocity noise
- Inertia tensor randomisation

### 5.3 Robot Model
- Franka Panda 7-DOF arm
- Differential IK controller
- Contact force sensing at EE

---

## 6. Probabilistic Shield

### 6.1 State Space Abstraction
- (d, f, t) abstraction: 45 abstract states
- Conservative binning

### 6.2 PRISM Model (DTMC)
- Transition probabilities (table from offline rollouts)
- PCTL safety specifications

### 6.3 Runtime Shield
- Online query pipeline
- Least-restrictive action selection
- Intervention rate analysis

---

## 7. Agents

### 7.1 PPO Baseline and PPO+Shield
### 7.2 SAC Baseline and SAC+Shield
### 7.3 Curriculum Training

---

## 8. Experiments

### 8.1 Setup
- Hardware: RTX 4090, 32GB RAM
- 200 evaluation episodes per condition
- Seeds: 42, 43, 44 (3 independent runs)

### 8.2 Metrics
- Success rate
- Mean episode reward (± std)
- Cumulative safety cost
- Collision rate
- Shield intervention rate
- Capture time

### 8.3 Results (placeholder tables)

| Condition | Success% | Reward | Cost | Intervention% |
|---|---|---|---|---|
| PPO | | | | N/A |
| SAC | | | | N/A |
| PPO+Shield | | | | |
| SAC+Shield | | | | |

### 8.4 Ablation Studies
- Shield threshold sensitivity (0.01, 0.05, 0.10, 0.20)
- Abstraction granularity (coarser/finer state space)
- PRISM table vs online mode

---

## 9. Discussion

### 9.1 Shield Conservatism vs Task Performance
### 9.2 Failure Modes
### 9.3 Limitations
- 3-action abstract policy may miss nuanced manoeuvres
- PRISM table calibrated from simulation — sim-to-real gap

---

## 10. Conclusion and Future Work

- Real hardware validation (ESA testbed)
- Learned shield (neural network approximation of PRISM table)
- Multi-arm capture scenarios
- Sim-to-real transfer

---

## Appendix A: PRISM Model Listing

[Insert debris_capture.pm]

## Appendix B: Hyperparameter Tables

[Insert ppo_config.yaml and sac_config.yaml]

## Appendix C: Supplementary Figures

[Learning curves, intervention heatmaps, success rate vs threshold plots]
