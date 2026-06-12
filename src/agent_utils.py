"""Shared helpers for evaluation: loading agents, the PID baseline, and a
deterministic rollout that records the multi-objective metrics.

Keeping this in one place guarantees the RL agent and the PID baseline are
scored with *identical* accounting on the *same* wind sequence — which is the
whole point of a fair comparison.
"""
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from env import WindTurbineEnv

# Each data row is a 10-minute SCADA interval.
HOURS_PER_STEP = 10.0 / 60.0


def load_rl_predict(seed, models_dir, cfg=CONFIG):
    """Return a predict_fn(raw_obs)->action for the trained agent of `seed`,
    applying the saved VecNormalize observation statistics. Returns None if
    the model is missing."""
    model_path = os.path.join(models_dir, f"ppo_wind_turbine_seed{seed}.zip")
    vecnorm_path = os.path.join(models_dir, f"vecnormalize_seed{seed}.pkl")
    if not os.path.exists(model_path):
        return None

    model = PPO.load(model_path)
    normalizer = None
    if os.path.exists(vecnorm_path):
        normalizer = VecNormalize.load(
            vecnorm_path, DummyVecEnv([lambda: WindTurbineEnv(cfg)])
        )
        normalizer.training = False
        normalizer.norm_reward = False

    def predict(raw_obs):
        obs = raw_obs
        if normalizer is not None:
            obs = normalizer.normalize_obs(raw_obs.reshape(1, -1))[0]
        action, _ = model.predict(obs, deterministic=True)
        return action

    return predict


class PIDController:
    """Standard PID on rotor-speed error -> pitch increment."""

    def __init__(self, kp, ki, kd, target):
        self.kp, self.ki, self.kd, self.target = kp, ki, kd, target
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, measurement):
        error = measurement - self.target
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def make_pid_predict(kp, ki, kd, target):
    """Return a predict_fn(raw_obs, env)->action. The PID reads the true rotor
    speed from the env (not the normalised obs)."""
    pid = PIDController(kp, ki, kd, target)
    pid.reset()

    def predict(raw_obs, env):
        out = pid.compute(env.rotor_speed)
        return np.array([np.clip(out, -1.0, 1.0)], dtype=np.float32)

    return predict


def run_episode(env, predict_fn, n_steps, start_idx, needs_env=False):
    """Roll out a controller for `n_steps` from a fixed start index and return
    the multi-objective metrics dict.

    Metrics:
      energy_mwh         - total energy produced (MWh)
      violation_rate     - % of steps with rotor speed over the soft limit
      avg_pitch_travel   - mean |delta pitch| per step (actuator wear)
      capture_efficiency - generated / theoretical power (%)
      mean_reward        - average per-step reward
    """
    env.reset()
    env.current_idx = start_idx
    env.current_pitch = 0.0
    env.rotor_speed = env.initial_rotor_speed
    env.steps_taken = 0
    obs = env._get_obs()

    total_gen = 0.0
    total_theo = 0.0
    violations = 0
    travel = []
    rewards = []
    hist = {"wind": [], "power": [], "pitch": [], "rotor": []}

    for _ in range(n_steps):
        action = predict_fn(obs, env) if needs_env else predict_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_gen += info["power"]
        total_theo += info["theoretical_power"]
        violations += int(info["overspeed"])
        travel.append(info["pitch_delta"])
        rewards.append(reward)
        hist["wind"].append(obs[0] * env.wind_norm)
        hist["power"].append(info["power"])
        hist["pitch"].append(info["pitch"])
        hist["rotor"].append(info["rotor_speed"])

        if terminated or truncated:
            break

    steps = len(rewards)
    return {
        "energy_mwh": total_gen * HOURS_PER_STEP / 1000.0,
        "violation_rate": 100.0 * violations / max(1, steps),
        "violations": violations,
        "avg_pitch_travel": float(np.mean(travel)) if travel else 0.0,
        "capture_efficiency": 100.0 * total_gen / total_theo if total_theo > 0 else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "steps": steps,
        "history": hist,
    }


def fixed_start_index(cfg=CONFIG):
    """Deterministically pick the eval start index from the eval seed so every
    controller sees the identical wind sequence."""
    env = WindTurbineEnv(cfg)
    env.reset(seed=cfg["eval"]["eval_seed"])
    return env.current_idx


def eval_window_starts(cfg=CONFIG):
    """N evenly-spaced start indices spanning the dataset, so every controller is
    scored across many weather conditions rather than one slice."""
    env = WindTurbineEnv(cfg)
    n = cfg["eval"]["n_windows"]
    steps = cfg["eval"]["eval_steps"]
    hi = max(1, env.max_idx - steps)
    return [int(round(x)) for x in np.linspace(0, hi, n)]


# Metrics that are averaged across windows (everything except bookkeeping).
_AVG_KEYS = (
    "energy_mwh",
    "violation_rate",
    "violations",
    "avg_pitch_travel",
    "capture_efficiency",
    "mean_reward",
)


def run_over_windows(env, predict_fn, starts, n_steps, needs_env=False):
    """Run a controller across several windows and return the metrics averaged
    over windows (plus the per-window energy/violation arrays for plotting)."""
    per = [run_episode(env, predict_fn, n_steps, s, needs_env) for s in starts]
    out = {k: float(np.mean([m[k] for m in per])) for k in _AVG_KEYS}
    out["energy_per_window"] = [m["energy_mwh"] for m in per]
    out["viol_per_window"] = [m["violation_rate"] for m in per]
    return out
