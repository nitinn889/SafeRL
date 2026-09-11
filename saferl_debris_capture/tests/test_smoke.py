"""
Smoke test — runs without Isaac Lab or PRISM installed.
Tests the pure-Python components that are always importable.

Validates:
  1. Debris dynamics (NumPy single-env): Euler equations, RK4, quaternion
  2. Debris dynamics (Torch batch): full 3x3 inertia, gyroscopic torque, OU noise
  3. State abstraction: continuous → discrete binning
  4. PRISM shield query: table mode
  5. Reward shaping: potential-based shaping
  6. SafeRL metrics: aggregation + bootstrap CI

Usage:  python3 tests/test_smoke.py
"""

import sys
import math
import numpy as np
sys.path.insert(0, ".")  # run from saferl_debris_capture/

print("=" * 60)
print("  SafeRL Debris Capture — Smoke Tests")
print("=" * 60)

# ── 1. NumPy dynamics ────────────────────────────────────────────────────────
print("\n1. Testing debris dynamics (NumPy single-env)...")
from envs.debris_dynamics import TumblingDebrisDynamicsNumpy
dyn = TumblingDebrisDynamicsNumpy(Ix=100, Iy=300, Iz=700, cross_inertia=10.0, noise_std=0.02)
dyn.reset(tumble_rate=math.radians(10))

initial_energy = dyn.rotational_energy
for _ in range(200):
    dyn.step()
speed = np.linalg.norm(dyn.omega)
final_energy = dyn.rotational_energy

assert 0.0 < speed < 10.0, f"Unexpected tumble speed: {speed}"
print(f"   ✓ Tumble speed after 200 steps: {math.degrees(speed):.2f} deg/s")
print(f"   ✓ Rotational energy: {initial_energy:.1f} → {final_energy:.1f} J")
print(f"   ✓ Quaternion norm: {np.linalg.norm(dyn.q):.6f} (should be ~1.0)")
assert abs(np.linalg.norm(dyn.q) - 1.0) < 0.01, "Quaternion drift!"

# ── 2. Torch dynamics (batch) ────────────────────────────────────────────────
print("\n2. Testing debris dynamics (Torch batch, 64 envs)...")
import torch
from envs.debris_dynamics import TumblingDebrisModel, DebrisDynamicsConfig

cfg = DebrisDynamicsConfig(
    tumble_rate_min_rad_s=math.radians(1),
    tumble_rate_max_rad_s=math.radians(30),
    ou_sigma=0.02,
    dt=1.0/120.0,
)
model = TumblingDebrisModel(cfg=cfg, device="cpu")
model.reset(num_envs=64)

assert model.inertia_tensor.shape == (64, 3, 3), f"Bad inertia shape: {model.inertia_tensor.shape}"
assert model.angular_vel.shape == (64, 3), f"Bad omega shape: {model.angular_vel.shape}"

# Check inertia is positive definite (all eigenvalues > 0)
eigvals = torch.linalg.eigvalsh(model.inertia_tensor)
assert (eigvals > 0).all(), "Inertia tensor has non-positive eigenvalues!"
print(f"   ✓ Inertia eigenvalue range: [{eigvals.min():.1f}, {eigvals.max():.1f}]")

# Step 100 times
for _ in range(100):
    model.step()

rates = model.get_tumbling_rate()
modes = model.get_tumble_mode()
state = model.get_state()
print(f"   ✓ Tumble rates: min={math.degrees(rates.min()):.1f}°/s, max={math.degrees(rates.max()):.1f}°/s")
print(f"   ✓ Tumble modes: {modes.unique().tolist()}")
print(f"   ✓ Rotational energy: {state['rotational_energy'].mean():.1f} J (mean)")

# Check quaternion normalisation
q_norms = torch.linalg.norm(model.quaternion, dim=-1)
assert (q_norms - 1.0).abs().max() < 0.01, "Quaternion normalisation drifted!"
print(f"   ✓ Quaternion norm error: {(q_norms - 1.0).abs().max():.6f}")

# ── 3. Partial reset test ────────────────────────────────────────────────────
print("\n3. Testing partial reset...")
reset_ids = torch.tensor([0, 5, 10, 63])
old_omega = model.angular_vel.clone()
model.reset(64, env_ids=reset_ids)
# Reset envs should have new values, others unchanged
unchanged_ids = torch.tensor([1, 2, 3, 4, 6])
assert torch.allclose(model.angular_vel[unchanged_ids], old_omega[unchanged_ids]), \
    "Partial reset modified environments that should be untouched!"
print(f"   ✓ Partial reset preserved non-reset environments")

# ── 4. State abstraction ─────────────────────────────────────────────────────
print("\n4. Testing state abstraction...")
from shield.abstraction import StateAbstraction, AbstractionConfig

abstraction = StateAbstraction()

# Test various observation scenarios
test_cases = [
    # (name, obs modifications, expected d, expected f, expected t)
    ("far + safe + slow", {0: 3.0, 10: 0.0, 7: 0.01, 31: 1.0}, 4, 0, 0),
    ("close + warning + fast", {0: 0.1, 10: 0.0, 7: 0.3, 31: 12.0}, 1, 1, 2),
]

for name, mods, exp_d, exp_f, exp_t in test_cases:
    obs = np.zeros(36, dtype=np.float32)
    for idx, val in mods.items():
        obs[idx] = val
    state = abstraction.abstract(obs)
    print(f"   ✓ {name}: d={state.d}, f={state.f}, t={state.t}")
    assert state.d == exp_d, f"Expected d={exp_d}, got {state.d}"
    assert state.f == exp_f, f"Expected f={exp_f}, got {state.f}"
    assert state.t == exp_t, f"Expected t={exp_t}, got {state.t}"

# Test state-index round-trip
for idx in range(abstraction.num_abstract_states):
    s = abstraction.index_to_state(idx)
    assert abstraction.state_to_index(s) == idx
print(f"   ✓ state↔index round-trip: all {abstraction.num_abstract_states} states OK")

# ── 5. Shield query ──────────────────────────────────────────────────────────
print("\n5. Testing PRISM shield query (table mode)...")
from shield.shield_query import PRISMShieldQuery, ProbabilityTable
from shield.abstraction import AbstractState

query = PRISMShieldQuery(mode="table", safety_threshold=0.05)
test_state = AbstractState(d=2, f=0, t=1)
p = query.collision_prob(test_state, action=0)
safe_set = query.safe_action_set(test_state)
safest = query.safest_action(test_state)
print(f"   ✓ P(collision | approach) = {p:.3f}")
print(f"   ✓ Safe action set: {safe_set}")
print(f"   ✓ Safest action: {safest}")

# ── 6. Reward shaping ────────────────────────────────────────────────────────
print("\n6. Testing reward shaping...")
from envs.reward_shaping import potential_based_shaping

obs_prev = np.zeros((2, 36), dtype=np.float32)
obs_curr = np.zeros((2, 36), dtype=np.float32)

# Env 0: moved closer to debris (positive shaping)
obs_prev[0, 10] = 2.0   # EE far
obs_curr[0, 10] = 1.0   # EE closer

# Env 1: moved away (negative shaping)
obs_prev[1, 10] = 1.0
obs_curr[1, 10] = 2.0

shaping = potential_based_shaping(obs_prev, obs_curr)
assert shaping[0] > 0, f"Expected positive shaping for approach, got {shaping[0]}"
assert shaping[1] < 0, f"Expected negative shaping for retreat, got {shaping[1]}"
print(f"   ✓ Approach shaping: +{float(shaping[0]):.4f}")
print(f"   ✓ Retreat shaping:  {float(shaping[1]):.4f}")

# ── 7. Metrics ────────────────────────────────────────────────────────────────
print("\n7. Testing metrics module...")
from evaluation.metrics import SafeRLMetrics, EpisodeMetrics

metrics = SafeRLMetrics(n_bootstrap=1000)
episodes = [
    EpisodeMetrics(reward=r, cost=c, success=s, episode_length=100,
                   capture_step=50 if s else None, interventions=i)
    for r, c, s, i in [
        (150.0, 0.0, True,  5),
        (-50.0, 2.0, False, 2),
        (120.0, 0.0, True,  8),
        (-30.0, 1.0, False, 0),
        (200.0, 0.0, True,  3),
        (-80.0, 3.0, False, 1),
    ]
]

agg = metrics.compute(episodes)
print(f"   ✓ Success rate:      {agg.success_rate:.2f}")
print(f"   ✓ Mean reward:       {agg.mean_reward:.1f}")
print(f"   ✓ Collision rate:    {agg.collision_rate:.2f}")
print(f"   ✓ Reward 95% CI:    ({agg.reward_95ci[0]:.1f}, {agg.reward_95ci[1]:.1f})")
assert 0.0 <= agg.success_rate <= 1.0
assert agg.reward_95ci[0] <= agg.mean_reward <= agg.reward_95ci[1]

# ── 8. Comparison test ───────────────────────────────────────────────────────
print("\n8. Testing statistical comparison...")
episodes_b = [
    EpisodeMetrics(reward=r, cost=c, success=s, episode_length=80,
                   capture_step=40 if s else None, interventions=i)
    for r, c, s, i in [
        (180.0, 0.0, True,  1),
        (160.0, 0.0, True,  2),
        (190.0, 0.0, True,  1),
        (50.0,  0.0, False, 0),
        (170.0, 0.0, True,  1),
        (-10.0, 1.0, False, 0),
    ]
]

comparison = metrics.compare(episodes, episodes_b, "PPO", "PPO+Shield")
key = "PPO_vs_PPO+Shield"
print(f"   ✓ Reward p-value:   {comparison[key]['reward_pvalue']:.4f}")
print(f"   ✓ Cost p-value:     {comparison[key]['cost_pvalue']:.4f}")
print(f"   ✓ Cohen's d:        {comparison[key]['reward_cohen_d']:.3f}")
print(f"   ✓ Success Δ:        {comparison[key]['success_delta']:+.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
