"""Tests for the shield in 3D free flight (saferl/configs/space3d.yaml).

tests/test_shield.py keeps covering the planar shield unchanged. What 3D adds
is the vertical axis: ±Z actions to substitute, and agent vertical velocity
that the planar model deliberately ignores but free flight must propagate.
"""
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from saferl.config import load_config
from saferl.env.base_env import (
    ACTION_THRUST_DIRS, OBS_HEADER_LEN, OBS_PER_HAZARD, SafeNav3DEnv,
)
from saferl.shield.safety_shield import (
    RandomReplacementShield, SafetyShield, ShieldedEnv,
)

SPACE3D = Path(__file__).resolve().parents[1] / "saferl" / "configs" / "space3d.yaml"
CFG = load_config(SPACE3D)
N_HAZARDS = CFG["env"]["max_hazards"]
PLUS_Y, MINUS_Y, MINUS_X, PLUS_X, PLUS_Z, MINUS_Z = range(6)


def make_obs(agent_pos, agent_vel=(0.0, 0.0, 0.0), hazards=(), max_hazards=N_HAZARDS):
    obs = np.zeros(OBS_HEADER_LEN + OBS_PER_HAZARD * max_hazards, dtype=np.float32)
    obs[0:3] = agent_pos
    obs[3:6] = agent_vel
    obs[6:9] = (9.0, 9.0, 9.0)
    for i, (h_pos, h_vel) in enumerate(hazards):
        base = OBS_HEADER_LEN + i * OBS_PER_HAZARD
        obs[base:base + 3] = h_pos
        obs[base + 3:base + 6] = h_vel
    return obs


@pytest.fixture
def shield():
    return SafetyShield.from_config(CFG)


def test_from_config_builds_the_shield_over_the_env_action_table(shield):
    assert shield._acc.shape == (6, 3)
    assert shield._has_z_thrust is True
    planar = SafetyShield.from_config(load_config())
    assert planar._acc.shape == (4, 3)
    assert planar._has_z_thrust is False


def test_agent_climbing_into_a_hazard_triggers_and_dives_away(shield):
    """The agent is already climbing fast toward a parked hazard overhead.

    Only diving (-Z) can shed that vertical velocity in time: sideways thrust
    at 4.8 m/s^2 doesn't displace far enough before the climb closes the gap.
    A shield that zeroed the agent's vertical velocity, as the planar model
    does, would see a static 4-unit gap and wave the collision through.
    """
    obs = make_obs((5.0, 5.0, 5.0), agent_vel=(0.0, 0.0, 4.0),
                   hazards=[((5.0, 5.0, 9.0), (0.0, 0.0, 0.0))])

    action, intervened = shield.check_and_fix(obs, action=PLUS_X)
    assert intervened is True
    assert shield.safe_actions(obs) == [MINUS_Z]
    assert action == MINUS_Z
    assert shield.last_decision.kind == "substituted"

    planar_model = SafetyShield(safe_dist=shield.safe_dist,
                                lookahead_steps=shield.lookahead_steps,
                                accel=shield.accel, step_dt=shield.step_dt,
                                thrust_dirs=ACTION_THRUST_DIRS)
    _, planar_intervened = planar_model.check_and_fix(obs, action=PLUS_X)
    assert planar_intervened is False, "the planar model should miss this"


def test_hazard_receding_vertically_does_not_trigger(shield):
    obs = make_obs((5.0, 5.0, 5.0),
                   hazards=[((5.0, 5.0, 7.5), (0.0, 0.0, 2.0))])
    action, intervened = shield.check_and_fix(obs, action=PLUS_X)
    assert intervened is False
    assert action == PLUS_X


def test_prefers_a_sidestep_including_vertical_over_a_reversal(shield):
    obs = make_obs((5.0, 5.0, 5.0),
                   hazards=[((8.0, 5.0, 5.0), (0.0, 0.0, 0.0))])     # blocks +X

    assert shield.safe_actions(obs) == [PLUS_Y, MINUS_Y, MINUS_X, PLUS_Z, MINUS_Z]
    chosen = {shield.check_and_fix(obs, action=PLUS_X)[0] for _ in range(20)}

    assert chosen <= {PLUS_Y, MINUS_Y, PLUS_Z, MINUS_Z}, "picked the reversal"
    assert len(chosen) == 1, "selection must be deterministic"
    assert shield.last_decision.deviation < shield._deviation(MINUS_X, PLUS_X)


def test_boxed_in_on_all_six_sides_falls_back(shield):
    ring = [((5.0 + dx * 2.0, 5.0 + dy * 2.0, 5.0 + dz * 2.0), (0.0, 0.0, 0.0))
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    obs = make_obs((5.0, 5.0, 5.0), hazards=ring)

    assert shield.safe_actions(obs) == []
    action, intervened = shield.check_and_fix(obs, action=PLUS_X)
    assert intervened is True
    assert shield.last_decision.kind == "fallback"
    assert shield.n_fallback == 1


def test_random_replacement_baseline_draws_from_every_3d_action():
    obs = make_obs((5.0, 5.0, 5.0), hazards=[((5.5, 5.0, 5.0), (0.0, 0.0, 0.0))])

    six = RandomReplacementShield(n_actions=6, rng=np.random.RandomState(0))
    assert {six.check_and_fix(obs, 0)[0] for _ in range(300)} == set(range(6))

    four = RandomReplacementShield(rng=np.random.RandomState(0))
    assert {four.check_and_fix(obs, 0)[0] for _ in range(300)} <= set(range(4))


def test_privileged_view_holds_for_a_hazard_out_of_sensor_range_overhead():
    """The policy can't see a hazard 4.5 units overhead (sensor range 3.0) but
    the shield must still stop the agent thrusting up into it."""
    env = SafeNav3DEnv(**{**CFG["env"], "max_hazards": 1, "curriculum": False,
                          "render_mode": "direct", "sensor_range": 3.0})
    shielded = ShieldedEnv(env, SafetyShield.from_config(CFG))
    shielded.reset()

    p.resetBasePositionAndOrientation(env.agent_id, [5.0, 5.0, 5.0], [0, 0, 0, 1],
                                      physicsClientId=env._client)
    env.hazard_positions[0] = [5.0, 5.0, 9.5]
    env.hazard_velocities[0] = [0.0, 0.0, -2.0]
    shielded._last_obs = env._get_obs()
    shielded._last_true_obs = env.get_true_obs()

    block = slice(OBS_HEADER_LEN, OBS_HEADER_LEN + OBS_PER_HAZARD)
    assert np.allclose(shielded._last_obs[block], 0.0)
    assert not np.allclose(shielded._last_true_obs[block], 0.0)

    shielded.step(PLUS_Z)
    assert shielded.interventions >= 1
    shielded.close()
