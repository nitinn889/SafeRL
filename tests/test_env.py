import numpy as np
import pytest

from saferl.env.base_env import SafeNav3DEnv
from saferl.shield.safety_shield import SafetyShield, ShieldedEnv


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


def test_shield_intervenes_near_hazard():
    shield = SafetyShield(safe_dist=2.2)
    obs = np.zeros(9 + 3 * 5, dtype=np.float32)
    obs[0:3] = [1.0, 1.0, 0.5]  # agent position
    obs[9:12] = [1.5, 1.0, 0.5]  # hazard within safe_dist

    action, intervened = shield.check_and_fix(obs, action=0)
    assert intervened is True
    assert action in (0, 1, 2, 3)


def test_shield_does_not_intervene_when_clear():
    shield = SafetyShield(safe_dist=2.2)
    obs = np.zeros(9 + 3 * 5, dtype=np.float32)
    obs[0:3] = [1.0, 1.0, 0.5]
    obs[9:12] = [15.0, 15.0, 0.5]  # far away hazard

    action, intervened = shield.check_and_fix(obs, action=2)
    assert intervened is False
    assert action == 2


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


def test_shielded_env_tracks_last_obs_without_reaching_into_env():
    base = SafeNav3DEnv(size=10, max_hazards=5, curriculum=False, render_mode="direct")
    shielded = ShieldedEnv(base, SafetyShield(safe_dist=2.2))
    obs, _ = shielded.reset()
    assert shielded._last_obs is not None
    np.testing.assert_array_equal(obs, shielded._last_obs)
    shielded.step(0)
    assert shielded._last_obs is not None
    shielded.close()
