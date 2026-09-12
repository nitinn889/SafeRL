"""Tests for the safety shield (phase 5: least-restrictive safe action).

Lives apart from tests/test_env.py so shield behaviour and environment
dynamics can be read and changed independently. The three shield tests that
were in test_env.py were moved here.
"""
import numpy as np
import pytest

from saferl.env.base_env import (
    ACTION_THRUST_DIRS, AGENT_MASS, OBS_HEADER_LEN, OBS_PER_HAZARD,
    PHYSICS_HZ, SafeNav3DEnv,
)
from saferl.config import load_config
from saferl.shield.safety_shield import (
    RandomReplacementShield, SafetyShield, ShieldedEnv,
)

# Read from configs/default.yaml rather than restated. Phase 6 raised
# force_mag 12 -> 48 and halved the horizon; a fixture that hardcoded
# lookahead_steps=40 combined the new thrust with the old horizon and gave
# the shield a 6.67-unit reachable set instead of 1.67, which broke geometry
# these tests depend on. The defaults now follow the config by construction.
CFG = load_config()
SAFE_DIST = CFG["shield"]["safe_dist"]
LOOKAHEAD_STEPS = CFG["shield"]["lookahead_steps"]
AGENT_Z = 0.25       # where the agent rests on the plane
HAZARD_Z = 0.5       # env default debris height


def make_obs(agent_pos, agent_vel=(0.0, 0.0, 0.0), hazards=(), max_hazards=5):
    """Build a well-formed observation in the live 39-wide layout.

    Uses OBS_HEADER_LEN / OBS_PER_HAZARD rather than hardcoded offsets, so a
    future layout change breaks the env tests rather than silently producing
    a shield test that checks nothing.
    """
    obs = np.zeros(OBS_HEADER_LEN + OBS_PER_HAZARD * max_hazards, dtype=np.float32)
    obs[0:3] = agent_pos
    obs[3:6] = agent_vel
    obs[6:9] = (9.0, 9.0, 0.5)          # goal
    for i, (h_pos, h_vel) in enumerate(hazards):
        base = OBS_HEADER_LEN + i * OBS_PER_HAZARD
        obs[base:base + 3] = h_pos
        obs[base + 3:base + 6] = h_vel
    return obs


@pytest.fixture
def shield():
    return SafetyShield.from_config(CFG)


# ── Goal C case 1: the trigger must be driven by relative velocity ──────────


def test_triggers_on_a_hazard_that_is_far_but_closing(shield):
    """5 units away -- well outside safe_dist -- but inbound at 3 units/sec.

    This is the case a distance-only trigger cannot see. The hazard covers
    the gap inside the lookahead window, so the shield must act now.
    """
    obs = make_obs((5.0, 5.0, AGENT_Z),
                   hazards=[((5.0, 10.0, HAZARD_Z), (0.0, -3.0, 0.0))])

    assert np.linalg.norm(obs[0:3] - obs[9:12]) > SAFE_DIST  # not close yet
    action, intervened = shield.check_and_fix(obs, action=0)
    assert intervened is True
    assert action in (0, 1, 2, 3)


def test_does_not_trigger_on_the_same_hazard_receding(shield):
    """Identical geometry, velocity reversed. Position alone cannot tell these
    two cases apart, so this is the paired half of the test above."""
    obs = make_obs((5.0, 5.0, AGENT_Z),
                   hazards=[((5.0, 10.0, HAZARD_Z), (0.0, +3.0, 0.0))])

    action, intervened = shield.check_and_fix(obs, action=0)
    assert intervened is False
    assert action == 0


def test_old_distance_only_shield_misses_the_closing_hazard():
    """Pins down the gap being closed: on the exact observation the new shield
    intervenes on, the phase-1 placeholder does nothing at all."""
    obs = make_obs((5.0, 5.0, AGENT_Z),
                   hazards=[((5.0, 10.0, HAZARD_Z), (0.0, -3.0, 0.0))])

    _, old_intervened = RandomReplacementShield(safe_dist=SAFE_DIST).check_and_fix(obs, 0)
    _, new_intervened = SafetyShield.from_config(CFG).check_and_fix(obs, 0)

    assert old_intervened is False
    assert new_intervened is True


# ── Goal C case 2: least-restrictive selection among several safe actions ───


def test_prefers_a_sidestep_over_a_reversal(shield):
    """+X is blocked; +Y, -Y and -X are all safe. The shield must take a
    sidestep (deviation 1.2*sqrt(2)) rather than the reversal (2.4), and must
    do so deterministically -- the whole point of replacing the random draw."""
    obs = make_obs((5.0, 5.0, AGENT_Z),
                   hazards=[((8.0, 5.0, HAZARD_Z), (0.0, 0.0, 0.0))])

    assert shield.safe_actions(obs) == [0, 1, 2]   # everything but +X
    chosen = {shield.check_and_fix(obs, action=3)[0] for _ in range(20)}

    assert chosen <= {0, 1}, "picked a reversal (or a blocked action) over a sidestep"
    assert len(chosen) == 1, "selection must be deterministic, not random"
    assert shield.last_decision.kind == "substituted"
    assert shield.last_decision.deviation < shield._deviation(2, 3)


def test_breaks_a_deviation_tie_toward_the_larger_clearance(shield):
    """+Y and -Y are equally far from the intended +X in thrust terms, so the
    tie is broken on predicted clearance: the shield must dodge away from the
    second hazard, not into it."""
    blocker = ((8.0, 5.0, HAZARD_Z), (0.0, 0.0, 0.0))   # blocks +X

    below = make_obs((5.0, 5.0, AGENT_Z),
                     hazards=[blocker, ((5.0, 1.0, HAZARD_Z), (0.0, 0.0, 0.0))])
    above = make_obs((5.0, 5.0, AGENT_Z),
                     hazards=[blocker, ((5.0, 9.0, HAZARD_Z), (0.0, 0.0, 0.0))])

    assert {0, 1} <= set(shield.safe_actions(below))    # both sidesteps legal
    assert {0, 1} <= set(shield.safe_actions(above))

    assert shield.check_and_fix(below, action=3)[0] == 0   # hazard below -> go +Y
    assert shield.check_and_fix(above, action=3)[0] == 1   # hazard above -> go -Y


def test_does_not_substitute_when_the_intended_action_is_already_safe(shield):
    """A safe action must be passed straight through even with the hazard
    nearby-ish; least-restrictive means not restricting when unnecessary."""
    obs = make_obs((5.0, 5.0, AGENT_Z),
                   hazards=[((8.0, 5.0, HAZARD_Z), (0.0, 0.0, 0.0))])

    action, intervened = shield.check_and_fix(obs, action=2)   # -X, away from it
    assert intervened is False
    assert action == 2
    assert shield.n_triggered == 0


# ── Goal C case 3: the boxed-in fallback ───────────────────────────────────


def test_boxed_in_falls_back_to_the_action_that_buys_the_most_time(shield):
    """Hazards closing from +Y fast and -Y slowly, with no clear flank: no
    action holds safe_dist, so the shield must maximise time-to-violation.

    Under the phase-6 physics (a = 4.8 m/s^2, H = 20 steps) fleeing -Y delays
    the breach to 0.417s, against 0.333s for either sideways dodge and 0.292s
    for the proposed +Y -- so -Y is the unique best. The shield must pick it
    and label the decision a fallback.
    """
    obs = make_obs((5.0, 5.0, AGENT_Z), hazards=[
        ((5.0, 8.0, HAZARD_Z), (0.0, -2.5, 0.0)),
        ((5.0, 2.0, HAZARD_Z), (0.0, +1.0, 0.0)),
    ])

    assert shield.safe_actions(obs) == []          # genuinely boxed in
    action, intervened = shield.check_and_fix(obs, action=0)

    assert intervened is True
    assert action == 1, "should flee the faster hazard"
    assert shield.last_decision.kind == "fallback"
    assert shield.last_decision.n_safe_actions == 0
    assert shield.n_fallback == 1
    assert shield.n_substituted == 0


def test_fallback_and_substitution_are_counted_separately(shield):
    """The two cases mean different things, so they must not share a counter."""
    substitutable = make_obs((5.0, 5.0, AGENT_Z),
                             hazards=[((8.0, 5.0, HAZARD_Z), (0.0, 0.0, 0.0))])
    boxed_in = make_obs((5.0, 5.0, AGENT_Z), hazards=[
        ((5.0 + d[0] * 2.0, 5.0 + d[1] * 2.0, HAZARD_Z), (0.0, 0.0, 0.0))
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
    ])

    shield.check_and_fix(substitutable, action=3)
    shield.check_and_fix(boxed_in, action=3)

    assert shield.n_triggered == 2
    assert shield.n_substituted == 1
    assert shield.n_fallback == 1


# ── Goal C case 4: no over-triggering ──────────────────────────────────────


def test_clear_field_never_intervenes(shield):
    obs = make_obs((1.0, 1.0, AGENT_Z),
                   hazards=[((15.0, 15.0, HAZARD_Z), (0.0, 0.0, 0.0))])

    action, intervened = shield.check_and_fix(obs, action=2)
    assert intervened is False
    assert action == 2


def test_hazard_moving_parallel_does_not_trigger(shield):
    """Close-ish but not closing: a hazard tracking alongside the agent at the
    same velocity never breaches, so intervening would be pure over-triggering."""
    obs = make_obs((5.0, 5.0, AGENT_Z), agent_vel=(1.0, 0.0, 0.0),
                   hazards=[((5.0, 8.0, HAZARD_Z), (1.0, 0.0, 0.0))])

    action, intervened = shield.check_and_fix(obs, action=3)
    assert intervened is False
    assert action == 3


def test_no_hazards_in_view_passes_through(shield):
    """A zero-padded observation (curriculum with few live hazards) must not
    be read as a hazard sitting at the origin."""
    obs = make_obs((5.0, 5.0, AGENT_Z), hazards=[])

    action, intervened = shield.check_and_fix(obs, action=1)
    assert intervened is False
    assert action == 1


# ── moved from tests/test_env.py ───────────────────────────────────────────


def test_shield_intervenes_near_hazard():
    obs = make_obs((1.0, 1.0, AGENT_Z),
                   hazards=[((1.5, 1.0, HAZARD_Z), (0.0, 0.0, 0.0))])

    action, intervened = SafetyShield.from_config(CFG).check_and_fix(obs, action=0)
    assert intervened is True
    assert action in (0, 1, 2, 3)


def test_shield_does_not_intervene_when_clear():
    obs = make_obs((1.0, 1.0, AGENT_Z),
                   hazards=[((15.0, 15.0, HAZARD_Z), (0.0, 0.0, 0.0))])

    action, intervened = SafetyShield.from_config(CFG).check_and_fix(obs, action=2)
    assert intervened is False
    assert action == 2


def test_shielded_env_tracks_last_obs_without_reaching_into_env():
    base = SafeNav3DEnv(size=10, max_hazards=5, curriculum=False, render_mode="direct")
    shielded = ShieldedEnv(base, SafetyShield.from_config(CFG))
    obs, _ = shielded.reset()
    assert shielded._last_obs is not None
    np.testing.assert_array_equal(obs, shielded._last_obs)
    shielded.step(0)
    assert shielded._last_obs is not None
    shielded.close()


# ── the shield's motion model must match the env it is guarding ────────────


def test_shield_action_model_matches_the_env_dynamics():
    """Every action must accelerate the agent the way ACTION_THRUST_DIRS says.

    The shield predicts trajectories from this table; if step() ever stopped
    agreeing with it, the shield would be confidently guarding the wrong
    world and nothing else in the suite would notice.
    """
    import pybullet as p

    env = SafeNav3DEnv(size=10, max_hazards=1, curriculum=False, render_mode="direct")
    env.reset()
    expected_dv = (env.force_mag / AGENT_MASS) * env.step_dt

    for action in range(env.action_space.n):
        p.resetBaseVelocity(env.agent_id, linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0], physicsClientId=env._client)
        obs, *_ = env.step(action)
        dv = np.array(obs[3:6], dtype=np.float64)
        dv[2] = 0.0     # gravity/contact settling, not thrust
        np.testing.assert_allclose(dv, ACTION_THRUST_DIRS[action] * expected_dv,
                                   rtol=1e-3, atol=1e-6)
    env.close()


def test_agent_mass_constant_matches_the_urdf():
    """AGENT_MASS feeds the shield's acceleration; a URDF swap must not make
    it silently wrong."""
    import pybullet as p

    env = SafeNav3DEnv(size=10, max_hazards=1, curriculum=False, render_mode="direct")
    env.reset()
    mass = p.getDynamicsInfo(env.agent_id, -1, physicsClientId=env._client)[0]
    assert mass == pytest.approx(AGENT_MASS)
    env.close()


def test_default_step_dt_matches_the_env():
    env = SafeNav3DEnv(render_mode="direct")
    assert SafetyShield().step_dt == pytest.approx(env.sim_substeps / PHYSICS_HZ)
    env.close()


# ── wrapper bookkeeping ────────────────────────────────────────────────────


def test_shielded_env_counts_fallbacks_separately():
    """ShieldedEnv must read the kind of intervention, not just that one
    happened -- otherwise phase 9 cannot tell the two cases apart."""
    base = SafeNav3DEnv(size=10, max_hazards=4, curriculum=False, render_mode="direct")
    shield = SafetyShield.from_config(CFG)
    shielded = ShieldedEnv(base, shield)
    shielded.reset()

    # ring the agent so that nothing is safe, then step once
    agent = shielded._last_obs[0:3]
    for i in range(len(base.hazard_positions)):
        d = ((1, 0), (-1, 0), (0, 1), (0, -1))[i % 4]
        base.hazard_positions[i] = [float(agent[0] + d[0] * 2.0),
                                    float(agent[1] + d[1] * 2.0), HAZARD_Z]
        base.hazard_velocities[i] = [0.0, 0.0, 0.0]
    shielded._last_obs = base._get_obs()

    shielded.step(3)
    assert shielded.interventions == 1
    assert shielded.fallback_interventions == 1

    # a clear field must move neither counter
    for i in range(len(base.hazard_positions)):
        base.hazard_positions[i] = [50.0, 50.0, HAZARD_Z]
    shielded._last_obs = base._get_obs()
    shielded.step(3)
    assert shielded.interventions == 1
    assert shielded.fallback_interventions == 1
    shielded.close()
