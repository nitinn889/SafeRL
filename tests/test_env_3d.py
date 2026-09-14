"""Tests for the 3D free-flight env (dims=3, saferl/configs/space3d.yaml).

tests/test_env.py keeps covering the planar env on default.yaml, unchanged.
This file covers what dims=3 adds: no gravity or floor, vertical thrust, and
debris spawning, moving, reflecting and bounding on all three axes.
"""
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from saferl.config import load_config
from saferl.env.base_env import (
    ACTION_THRUST_DIRS, ACTION_THRUST_DIRS_3D, AGENT_MASS, OBS_HEADER_LEN,
    OBS_PER_HAZARD, SafeNav3DEnv, thrust_dirs,
)
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv

SPACE3D = Path(__file__).resolve().parents[1] / "saferl" / "configs" / "space3d.yaml"
CFG = load_config(SPACE3D)


def make_env(**overrides):
    kwargs = dict(CFG["env"])
    kwargs.update(curriculum=False, render_mode="direct")
    kwargs.update(overrides)
    return SafeNav3DEnv(**kwargs)


def park_single_hazard(env, pos=(8.0, 8.0, 8.0)):
    """Put the only hazard somewhere out of the way and stop it drifting."""
    env.hazard_positions[0] = list(pos)
    env.hazard_velocities[0] = [0.0, 0.0, 0.0]


@pytest.fixture
def env():
    e = make_env()
    yield e
    e.close()


# ── config ─────────────────────────────────────────────────────────────────


def test_space3d_config_inherits_default_and_overrides_only_what_it_states():
    base = load_config()
    assert base["env"]["dims"] == 2
    assert CFG["env"]["dims"] == 3
    assert CFG["env"]["max_hazards"] == 12
    for key in ("force_mag", "sim_substeps", "sensor_range", "bounds_margin",
                "out_of_bounds_penalty", "hazard_threshold", "goal_threshold",
                "max_episode_steps"):
        assert CFG["env"][key] == base["env"][key], key
    assert CFG["shield"] == base["shield"]
    assert CFG["constrained"] == base["constrained"]


def test_planar_env_is_unchanged_by_default():
    """dims defaults to 2: Discrete(4) and a ground plane, as before phase 11."""
    e = SafeNav3DEnv(max_hazards=5, curriculum=False, render_mode="direct")
    e.reset()
    assert e.dims == 2
    assert e.action_space.n == 4
    # plane + agent + goal marker + hazards
    assert p.getNumBodies(physicsClientId=e._client) == 3 + len(e.hazard_positions)
    e.close()


# ── spaces and the action table ────────────────────────────────────────────


def test_3d_action_table_keeps_the_planar_indices():
    np.testing.assert_array_equal(ACTION_THRUST_DIRS_3D[:4], ACTION_THRUST_DIRS)
    np.testing.assert_array_equal(ACTION_THRUST_DIRS_3D[4], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(ACTION_THRUST_DIRS_3D[5], [0.0, 0.0, -1.0])
    assert thrust_dirs(2) is ACTION_THRUST_DIRS
    assert thrust_dirs(3) is ACTION_THRUST_DIRS_3D
    with pytest.raises(ValueError):
        thrust_dirs(4)


def test_spaces_are_six_actions_and_an_81_wide_observation(env):
    assert env.action_space.n == 6
    expected = OBS_HEADER_LEN + OBS_PER_HAZARD * 12
    assert expected == 81
    assert env.observation_space.shape == (expected,)
    obs, _ = env.reset()
    assert obs.shape == (expected,)


def test_goal_is_the_far_corner_of_the_cube(env):
    np.testing.assert_array_equal(env.goal_pos, [env.size - 1] * 3)
    env.reset()
    p.resetBasePositionAndOrientation(env.agent_id, env.goal_pos.tolist(),
                                      [0, 0, 0, 1], physicsClientId=env._client)
    obs, reward, done, truncated, info = env.step(0)
    assert done is True
    assert reward > 0


# ── free flight ────────────────────────────────────────────────────────────


def test_no_ground_plane_is_loaded(env):
    env.reset()
    # agent + goal marker + hazards; no plane
    assert p.getNumBodies(physicsClientId=env._client) == 2 + len(env.hazard_positions)


def test_no_gravity_agent_keeps_altitude_under_horizontal_thrust():
    e = make_env(max_hazards=1)
    e.reset()
    park_single_hazard(e)
    for _ in range(40):
        obs, *_ = e.step(3)                  # +X only
    assert abs(float(obs[2])) < 1e-6         # would be falling under gravity
    assert abs(float(obs[5])) < 1e-6
    assert obs[0] > 1.0                      # and it did actually move
    e.close()


def test_vertical_thrust_moves_the_agent_up_and_down():
    e = make_env(max_hazards=1)
    for action, sign in ((4, 1.0), (5, -1.0)):
        e.reset()
        park_single_hazard(e)
        for _ in range(30):
            obs, *_ = e.step(action)
        assert sign * float(obs[2]) > 0.5, f"action {action} did not move z"
    e.close()


def test_every_action_accelerates_the_agent_as_the_3d_table_says():
    """The shield predicts trajectories from ACTION_THRUST_DIRS_3D; if step()
    disagreed on any axis -- including z, which the planar test ignores -- the
    shield would be guarding the wrong world."""
    e = make_env(max_hazards=1)
    e.reset()
    park_single_hazard(e)
    expected_dv = (e.force_mag / AGENT_MASS) * e.step_dt
    for action in range(e.action_space.n):
        p.resetBaseVelocity(e.agent_id, linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0], physicsClientId=e._client)
        obs, *_ = e.step(action)
        np.testing.assert_allclose(obs[3:6], ACTION_THRUST_DIRS_3D[action] * expected_dv,
                                   rtol=1e-3, atol=1e-6)
    e.close()


# ── debris ─────────────────────────────────────────────────────────────────


def test_debris_spawn_and_drift_through_the_whole_volume(env):
    zs, vzs = [], []
    for _ in range(5):
        env.reset()
        for pos, vel in zip(env.hazard_positions, env.hazard_velocities):
            assert all(1.0 <= c <= env.size - 2 for c in pos)
            speed = float(np.linalg.norm(vel))
            assert env.debris_min_speed - 1e-9 <= speed <= env.debris_max_speed + 1e-9
            zs.append(pos[2])
            vzs.append(vel[2])
    assert np.ptp(zs) > 2.0, "debris are not spread vertically"
    assert np.mean(np.abs(vzs)) > 0.05, "debris are not drifting vertically"


def test_debris_reflect_off_the_z_boundaries_and_stay_in_the_cube(env):
    env.reset()
    env.hazard_positions[0][2] = env.size - 0.01
    env.hazard_velocities[0] = [0.0, 0.0, 50.0]

    env.step(0)
    assert env.hazard_velocities[0][2] < 0, "z velocity should reverse at the boundary"

    for _ in range(200):
        env.step(0)
        for pos in env.hazard_positions:
            for c in pos:
                assert -1e-6 <= c <= env.size + 1e-6


@pytest.mark.parametrize("z_sign", [1.0, -1.0])
def test_leaving_the_cube_vertically_terminates_with_the_penalty(env, z_sign):
    env.reset()
    far = env.size + env.bounds_margin + 1.0 if z_sign > 0 else -(env.bounds_margin + 1.0)
    p.resetBasePositionAndOrientation(env.agent_id, [0.5, 0.5, far], [0, 0, 0, 1],
                                      physicsClientId=env._client)
    obs, reward, done, truncated, info = env.step(0)
    assert done is True
    assert truncated is False
    assert info["out_of_bounds"] == 1
    assert info["cost"] == 0


# ── the full stack ─────────────────────────────────────────────────────────


def test_ppo_accepts_the_3d_spaces():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    base = make_env()
    shielded = ShieldedEnv(base, SafetyShield.from_config(CFG))
    model = PPO("MlpPolicy", Monitor(shielded, info_keywords=("cost",)),
                n_steps=64, batch_size=32, verbose=0, device="cpu")
    model.learn(total_timesteps=64)

    obs, _ = shielded.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert base.action_space.contains(int(action))
    shielded.close()


# ── goal curriculum ────────────────────────────────────────────────────────

def _reach_goal(e):
    """Put the agent on the goal and take one step: a genuine goal hit."""
    p.resetBasePositionAndOrientation(e.agent_id, e.goal_pos.tolist(),
                                      [0, 0, 0, 1], physicsClientId=e._client)
    _, reward, done, _, info = e.step(0)
    assert done and reward > 0
    return info


def test_goal_curriculum_needs_curriculum_on_too():
    e = make_env(goal_curriculum=True)          # make_env sets curriculum=False
    e.reset()
    np.testing.assert_array_equal(e.goal_pos, [9, 9, 9])
    assert e.goal_fraction == 1.0
    e.close()


def test_goal_curriculum_starts_near_and_promotes_on_success():
    env_cfg = CFG["env"]
    frac, step = env_cfg["goal_start_fraction"], env_cfg["goal_fraction_step"]
    e = make_env(curriculum=True, goal_curriculum=True)
    e.reset()
    np.testing.assert_allclose(e.goal_pos, np.full(3, 9.0 * frac), atol=1e-5)
    for _ in range(env_cfg["goal_promote_window"]):
        assert _reach_goal(e)["goal_fraction"] == pytest.approx(frac)
        e.reset()
    assert e.goal_fraction == pytest.approx(frac + step)
    np.testing.assert_allclose(e.goal_pos, np.full(3, 9.0 * (frac + step)), atol=1e-5)
    e.close()


def test_goal_curriculum_holds_while_the_policy_fails():
    e = make_env(curriculum=True, goal_curriculum=True, max_episode_steps=2)
    e.reset()
    for _ in range(2 * CFG["env"]["goal_promote_window"]):
        done = truncated = False
        while not (done or truncated):
            _, _, done, truncated, _ = e.step(0)
        e.reset()
    assert e.goal_fraction == pytest.approx(CFG["env"]["goal_start_fraction"])
    e.close()
