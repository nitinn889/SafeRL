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


def test_shielded_env_tracks_last_obs_without_reaching_into_env():
    base = SafeNav3DEnv(size=10, max_hazards=5, curriculum=False, render_mode="direct")
    shielded = ShieldedEnv(base, SafetyShield(safe_dist=2.2))
    obs, _ = shielded.reset()
    assert shielded._last_obs is not None
    np.testing.assert_array_equal(obs, shielded._last_obs)
    shielded.step(0)
    assert shielded._last_obs is not None
    shielded.close()
