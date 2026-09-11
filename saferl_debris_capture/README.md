# SafeRL Debris Capture — Probabilistic Shield-Augmented RL

**Project:** Probabilistic Shield-Augmented Reinforcement Learning for Autonomous Capture of Tumbling Space Debris  
**Simulation:** NVIDIA Isaac Lab (Isaac Sim + PhysX 5)  
**Safety:** PRISM Probabilistic Model Checking

---

## Repository Structure

```
saferl_debris_capture/
├── envs/
│   ├── debris_capture_env.py       # Custom Isaac Lab environment (36-D obs, 6-D action)
│   ├── debris_dynamics.py          # Euler equation tumbling model (GPU-vectorised + NumPy)
│   └── reward_shaping.py           # Approach/align/grasp reward + collision cost
├── shield/
│   ├── prism_models/
│   │   └── debris_capture.pm       # PRISM DTMC model (45 abstract states, 3 actions)
│   ├── abstraction.py              # Continuous obs → (d, f, t) abstract state
│   ├── shield_query.py             # PRISM query interface (table + online modes)
│   └── shield_wrapper.py           # Gymnasium env wrapper that enforces shield
├── agents/
│   ├── ppo_agent.py                # PPO with custom ELU network + VecNormalize
│   ├── sac_agent.py                # SAC with automatic entropy tuning
│   └── shielded_agent.py           # RL agent + shield composite
├── training/
│   ├── train.py                    # Main training script (CLI)
│   └── config/
│       ├── ppo_config.yaml         # PPO hyperparameters + curriculum schedule
│       └── sac_config.yaml         # SAC hyperparameters
├── evaluation/
│   ├── benchmark.py                # 4-condition benchmark suite (200 eps each)
│   └── metrics.py                  # SafeRL metrics + statistical testing
├── assets/
│   ├── robot/                      # URDF/USD robot files (Franka Panda)
│   └── debris/                     # USD debris satellite model
├── notebooks/
│   └── analysis.ipynb              # Results visualisation and paper figures
└── docs/
    └── paper_draft.md              # Paper outline and draft
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment (Python 3.10 required for Isaac Lab)
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Sim
pip install isaacsim==4.2.0.2 \
    isaacsim-rl isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install && cd ..

# Install project dependencies
pip install stable-baselines3 gymnasium torch scipy numpy matplotlib pandas wandb pyyaml
```

### 2. Install PRISM Model Checker

Download PRISM from https://www.prismmodelchecker.org/download.php (Java 17+ required).

```bash
# Add to PATH (adjust path as needed)
export PATH=$PATH:/opt/prism/bin
prism -version   # verify installation
```

### 3. Generate Probability Table (offline, one-time)

```bash
python -c "
from shield.shield_query import PRISMShieldQuery
query = PRISMShieldQuery.from_prism(
    prism_executable='prism',
    model_path='shield/prism_models/debris_capture.pm',
)
query.generate_table('shield/prism_models/p_collision_table.npy')
"
```

### 4. Train

```bash
# PPO baseline (unshielded)
python training/train.py --algo ppo --total-steps 5000000

# SAC + Shield (SafeRL)
python training/train.py --algo sac --shield --threshold 0.05

# Monitor with TensorBoard
tensorboard --logdir logs/
```

### 5. Benchmark

```bash
python evaluation/benchmark.py \
    --ppo-model logs/ppo/final_model \
    --ppo-shielded-model logs/ppo_shielded/final_model \
    --sac-model logs/sac/final_model \
    --sac-shielded-model logs/sac_shielded/final_model \
    --n-episodes 200
```

---

## Technical Architecture

### Safety Pipeline

```
Observation (36-D)
      │
      ▼
StateAbstraction                     → (d, f, t)  [45 abstract states]
      │
      ▼
PRISMShieldQuery.collision_prob()    → P(collision | state, action)
      │
      ├─ P ≤ 0.05 → ALLOW  (RL agent's action passed through)
      │
      └─ P > 0.05 → INTERVENE (substitute least-restrictive safe action)
```

### PRISM Abstract State Space

| Variable | Values | Description |
|---|---|---|
| `d` | 0–4 | EE-to-debris distance bucket (0=contact, 4=far) |
| `f` | 0–2 | Contact force bucket (0=safe, 1=warning, 2=overload) |
| `t` | 0–2 | Tumbling rate bucket (0=slow, 1=medium, 2=fast) |

Total: 5 × 3 × 3 = **45 abstract states**, 3 abstract actions.

### Reward Function

```
R(s,a) = 1.0 × r_approach    (dense: −‖EE − debris‖)
        + 0.5 × r_align       (cosine similarity of orientations)
        + 1000 × r_grasp      (sparse: +1 if dist < 5cm)
        − 50  × r_collision   (contact force excess)
        − 0.01 × r_effort     (L2 action norm)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{saferl_debris_2026,
  title  = {Probabilistic Shield-Augmented RL for Autonomous Space Debris Capture},
  author = {[Author Name]},
  year   = {2026},
  url    = {https://github.com/[username]/saferl_debris_capture}
}
```

---

## License

MIT License. See [LICENSE](../LICENSE) for details.
