"""Invariant tests for the wind-turbine environment and config.

Run: pytest -q
These guard the properties that everything downstream relies on: bounded
observations, correct Gymnasium contract, reproducible seeding, action
clipping, and a sane power curve.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import CONFIG, get_paths, resolve  # noqa: E402
from env import WindTurbineEnv  # noqa: E402


def make_env(randomize=False):
    return WindTurbineEnv(CONFIG, randomize=randomize)


def test_reset_returns_obs_in_space():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_obs_stays_in_space_under_random_rollout():
    env = make_env()
    obs, _ = env.reset(seed=1)
    for _ in range(500):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), f"obs left space: {obs}"
        if terminated or truncated:
            obs, _ = env.reset(seed=1)


def test_step_contract():
    env = make_env()
    env.reset(seed=2)
    out = env.step(env.action_space.sample())
    assert len(out) == 5  # obs, reward, terminated, truncated, info
    _, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    for key in ("power", "theoretical_power", "rotor_speed", "pitch",
                "overspeed", "pitch_delta"):
        assert key in info


def test_deterministic_seeding():
    """Same seed + same actions -> identical trajectory (no global RNG leakage)."""
    actions = [np.array([0.3], dtype=np.float32) for _ in range(50)]
    traj = []
    for _ in range(2):
        env = make_env()
        obs, _ = env.reset(seed=123)
        seq = [obs.copy()]
        for a in actions:
            obs, *_ = env.step(a)
            seq.append(obs.copy())
        traj.append(np.array(seq))
    assert np.allclose(traj[0], traj[1])


def test_action_clipping_keeps_pitch_in_bounds():
    env = make_env()
    env.reset(seed=3)
    for _ in range(200):
        _, _, term, trunc, _ = env.step(np.array([10.0], dtype=np.float32))  # past [-1, 1]
        assert env.min_pitch <= env.current_pitch <= env.max_pitch
        if term or trunc:
            env.reset(seed=3)


def test_rotor_speed_within_physical_clamp():
    env = make_env()
    env.reset(seed=4)
    for _ in range(300):
        _, _, term, trunc, _ = env.step(np.array([-1.0], dtype=np.float32))  # max capture
        assert 0.0 <= env.rotor_speed <= env.hard_rotor_speed
        if term or trunc:
            env.reset(seed=4)


def test_theoretical_power_curve():
    env = make_env()
    assert env._theoretical_power(env.cut_in_wind - 0.1) == 0.0
    assert env._theoretical_power(env.rated_wind + 5) == env.rated_power
    mid_lo = env._theoretical_power(6.0)
    mid_hi = env._theoretical_power(11.0)
    assert 0.0 < mid_lo < mid_hi < env.rated_power  # monotonic increasing region


def test_eval_env_has_no_randomization():
    env = make_env(randomize=False)
    for s in range(8):
        env.reset(seed=s)
        assert env.wind_scale == 1.0


def test_config_paths_resolve():
    data_file, models_dir, logs_dir, images_dir, results_dir = get_paths()
    for d in (models_dir, logs_dir, images_dir, results_dir):
        assert os.path.isabs(d) and os.path.isdir(d)
    assert os.path.isabs(resolve("data/foo.csv"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
