import numpy as np
import pytest

from saferl.env.base_env import SafeNav3DEnv, OBS_HEADER_LEN, OBS_PER_HAZARD
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv

# Shield behaviour is tested in tests/test_shield.py; this file covers the
# environment. SafetyShield/ShieldedEnv are still imported because the PPO
# integration test below exercises the full wrapped stack.


@pytest.fixture
def env():
    e = SafeNav3DEnv(size=10, max_hazards=5, curriculum=False, render_mode="direct")
    yield e
    e.close()


def test_reset_returns_correctly_shaped_obs(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_does_not_crash(env):
    env.reset()
    for action in range(env.action_space.n):
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "cost" in info
        if done:
            env.reset()


def test_goal_reached_terminates(env):
    """Teleport the agent onto the goal; the next step must end the episode.

    Regression test for the phase 2 bug where no episode ever terminated:
    sphere2.urdf's 10kg mass and 0.5 lateral friction produced ~49N of static
    friction against 12N of thrust, welding the agent to the plane so neither
    the goal check nor the collision check could ever fire.
    """
    import pybullet as p

    env.reset()
    p.resetBasePositionAndOrientation(
        env.agent_id, env.goal_pos.tolist(), [0, 0, 0, 1],
        physicsClientId=env._client,
    )
    obs, reward, done, truncated, info = env.step(0)

    assert done is True
    assert reward > 0          # +100 goal bonus dominates the -0.1 step cost
    assert info["cost"] == 0


def test_hazard_collision_terminates(env):
    """Teleport the agent onto a hazard; the next step must end with cost=1."""
    import pybullet as p

    env.reset()
    hazard = env.hazard_positions[0]
    p.resetBasePositionAndOrientation(
        env.agent_id, hazard, [0, 0, 0, 1], physicsClientId=env._client,
    )
    obs, reward, done, truncated, info = env.step(0)

    assert done is True
    assert info["cost"] == 1
    assert reward == -50


def test_agent_actually_moves(env):
    """Sustained thrust must displace the agent -- the friction-lock guard."""
    env.reset()
    start, _ = env._get_obs()[0:3], None
    for _ in range(50):
        env.step(3)  # +X thrust
    end = env._get_obs()[0:3]

    assert np.linalg.norm(end[0:2] - start[0:2]) > 0.5


def test_truncation_fires_without_terminating(env):
    """A wandering policy should truncate, not terminate, at the step cap."""
    env.max_episode_steps = 5
    env.reset()
    for _ in range(4):
        obs, reward, done, truncated, info = env.step(0)
        assert truncated is False
    obs, reward, done, truncated, info = env.step(0)
    assert truncated is True
    assert done is False


# ── phase 4: dynamic debris field ────────────────────────────────────────


def test_debris_actually_move_between_steps(env):
    """Velocity must be applied, not just stored (the obvious silent bug)."""
    env.reset()
    before = [list(h) for h in env.hazard_positions]
    env.step(0)
    after = [list(h) for h in env.hazard_positions]

    assert len(before) == len(after) > 0
    moved = [np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(after, before)]
    assert all(d > 0.0 for d in moved), f"some debris did not move: {moved}"


def test_debris_positions_in_obs_track_the_live_positions(env):
    """The obs must carry this step's debris state, not a reset snapshot."""
    obs, _ = env.reset()
    obs, *_ = env.step(0)

    for i, (h_pos, h_vel) in enumerate(zip(env.hazard_positions, env.hazard_velocities)):
        base = OBS_HEADER_LEN + i * OBS_PER_HAZARD
        np.testing.assert_allclose(obs[base:base + 3], h_pos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(obs[base + 3:base + 6], h_vel, rtol=1e-5, atol=1e-5)


def test_collision_check_uses_current_hazard_position(env):
    """Park a hazard on the agent mid-episode; the very next step must end it.

    Guards against the collision check reading a stale reset-time snapshot.
    """
    import pybullet as p

    env.reset()
    agent_pos, _ = p.getBasePositionAndOrientation(env.agent_id, physicsClientId=env._client)
    # drop the hazard right on top of the agent, and stop it drifting away
    env.hazard_positions[0][0] = agent_pos[0]
    env.hazard_positions[0][1] = agent_pos[1]
    env.hazard_positions[0][2] = agent_pos[2]
    env.hazard_velocities[0] = [0.0, 0.0, 0.0]

    obs, reward, done, truncated, info = env.step(0)
    assert done is True
    assert info["cost"] == 1


def test_debris_bounce_off_boundary_and_stay_in_play_area(env):
    """Boundary behaviour is reflection: debris stay inside and reverse."""
    env.reset()
    # aim one hazard straight at the upper x boundary, fast
    env.hazard_positions[0][0] = env.size - 0.01
    env.hazard_velocities[0] = [50.0, 0.0, 0.0]

    env.step(0)
    assert env.hazard_velocities[0][0] < 0, "velocity should reverse at the boundary"

    for _ in range(200):
        env.step(0)
        for pos in env.hazard_positions:
            assert -1e-6 <= pos[0] <= env.size + 1e-6
            assert -1e-6 <= pos[1] <= env.size + 1e-6


def test_obs_shape_accounts_for_debris_velocity(env):
    expected = OBS_HEADER_LEN + OBS_PER_HAZARD * env.max_hazards
    assert env.observation_space.shape == (expected,)
    obs, _ = env.reset()
    assert obs.shape == (expected,)


def test_ppo_accepts_the_new_observation_space():
    """A PPO policy must build and predict against the widened obs."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    base = SafeNav3DEnv(size=10, max_hazards=5, curriculum=False, render_mode="direct")
    shielded = ShieldedEnv(base, SafetyShield(safe_dist=2.2))
    model = PPO("MlpPolicy", Monitor(shielded, info_keywords=("cost",)),
                n_steps=64, batch_size=32, verbose=0, device="cpu")
    model.learn(total_timesteps=64)

    obs, _ = shielded.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert base.action_space.contains(int(action))
    shielded.close()


def test_debris_speed_curriculum_ramps_with_episode_count():
    env = SafeNav3DEnv(size=10, max_hazards=5, curriculum=True, render_mode="direct",
                       debris_min_speed=0.3, debris_max_speed=1.2,
                       debris_speed_ramp_episodes=200)
    env.episode_count = 0
    assert env._debris_speed_range() == (0.3, 0.3)

    env.episode_count = 200
    assert env._debris_speed_range() == (0.3, 1.2)

    env.episode_count = 100
    lo, hi = env._debris_speed_range()
    assert lo == 0.3 and 0.3 < hi < 1.2
    env.close()


def test_no_curriculum_uses_full_speed_band():
    env = SafeNav3DEnv(curriculum=False, debris_min_speed=0.3, debris_max_speed=1.2,
                       render_mode="direct")
    assert env._debris_speed_range() == (0.3, 1.2)
    env.close()


# ── phase 6: out-of-bounds backstop ──────────────────────────────────────


def test_out_of_bounds_truncates_without_terminating(env):
    """Drifting out of the play area ends the episode as a truncation.

    Nothing walls the agent in and there is no drag, so at the phase-6 thrust
    a wandering policy reaches |xy| ~ 158 -- outside the observation space's
    own bounds and far from anything the task is about.
    """
    import pybullet as p

    env.reset()
    far = env.size + env.bounds_margin + 1.0
    p.resetBasePositionAndOrientation(
        env.agent_id, [far, 0.5, 0.25], [0, 0, 0, 1], physicsClientId=env._client,
    )
    obs, reward, done, truncated, info = env.step(0)

    assert truncated is True
    assert done is False
    assert info["cost"] == 0


def test_inside_the_margin_does_not_truncate(env):
    """The margin is slack, not a wall: just outside the field is still fine."""
    import pybullet as p

    env.reset()
    just_outside = env.size + env.bounds_margin - 1.0
    p.resetBasePositionAndOrientation(
        env.agent_id, [just_outside, 0.5, 0.25], [0, 0, 0, 1],
        physicsClientId=env._client,
    )
    obs, reward, done, truncated, info = env.step(0)

    assert truncated is False


def test_goal_termination_beats_the_bounds_check(env):
    """An in-bounds goal must still terminate rather than being masked."""
    import pybullet as p

    env.reset()
    p.resetBasePositionAndOrientation(
        env.agent_id, env.goal_pos.tolist(), [0, 0, 0, 1],
        physicsClientId=env._client,
    )
    obs, reward, done, truncated, info = env.step(0)

    assert done is True
    assert truncated is False
