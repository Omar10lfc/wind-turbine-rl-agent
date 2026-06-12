"""Regenerate the README figures (learning curve + synthetic storm test).

Uses the trained seed-0 agent with its saved VecNormalize statistics.
Run after training:  python src/figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, get_paths
from env import WindTurbineEnv
from agent_utils import load_rl_predict


def plot_learning_curve(cfg, logs_dir, images_dir, seed=0):
    seed_log = os.path.join(logs_dir, f"seed{seed}")
    if not os.path.exists(seed_log):
        print("No logs found; train first.")
        return
    try:
        x, y = ts2xy(load_results(seed_log), "timesteps")
    except Exception:
        print("No monitor.csv in logs.")
        return
    if len(x) < 2:
        return
    window = max(10, int(len(x) * 0.1))
    y_smooth = pd.Series(y).rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, alpha=0.2, color="#999999", label="Raw episode reward")
    plt.plot(x, y_smooth, color="#0066cc", linewidth=2.5, label="Smoothed trend")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title(f"Learning Curve (seed {seed})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(images_dir, "agent_learning_curve.png")
    plt.savefig(out, dpi=120)
    print(f"Saved -> {out}")


def run_storm_test(cfg, models_dir, images_dir, seed=0):
    predict = load_rl_predict(seed, models_dir, cfg)
    if predict is None:
        print("Model not found; train first.")
        return

    env = WindTurbineEnv(cfg)
    duration = 100
    winds_ramp = np.concatenate(
        [np.linspace(5, 25, 40), np.full(20, 25.0), np.linspace(25, 5, 40)]
    )

    def theoretical(w):
        if w < 3.5:
            return 0.0
        if w >= 13.0:
            return cfg["env"]["rated_power"]
        return cfg["env"]["rated_power"] * ((w - 3.5) / (13.0 - 3.5)) ** 3

    env.data = pd.DataFrame(
        {
            "wind_speed": winds_ramp,
            "wind_sin": np.zeros(duration),
            "wind_cos": np.zeros(duration),
            "theoretical_power": [theoretical(w) for w in winds_ramp],
        }
    )
    env.max_idx = duration - 1
    env.episode_max_steps = duration
    env.reset()
    env.current_idx = 0
    obs = env._get_obs()

    winds, powers, pitches, rotors = [], [], [], []
    for _ in range(duration - 1):
        action = predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        winds.append(obs[0] * env.wind_norm)
        pitches.append(info["pitch"])
        rotors.append(info["rotor_speed"])
        powers.append(info["power"])
        if terminated or truncated:
            break

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax1.plot(winds, color="blue", label="Wind Speed", linestyle="--")
    ax1.set_ylabel("Wind (m/s)", color="blue")
    ax1t = ax1.twinx()
    ax1t.plot(powers, color="green", linewidth=2, label="Power")
    ax1t.set_ylabel("Power (kW)", color="green")
    ax1t.axhline(cfg["env"]["rated_power"], color="red", linestyle=":", label="Rated")
    ax1.set_title("Wind vs Power")
    ax1.grid(True, alpha=0.3)

    ax2.plot(pitches, color="orange", linewidth=2)
    ax2.set_ylabel("Pitch Angle (°)")
    ax2.set_title("Agent Action: Pitch Control")
    ax2.grid(True, alpha=0.3)

    ax3.plot(rotors, color="purple", linewidth=2)
    ax3.axhline(cfg["env"]["max_rotor_speed"], color="red", linestyle="--",
                label="Soft limit")
    ax3.set_ylabel("Rotor RPM")
    ax3.set_title("Safety Check: Rotor Speed")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(images_dir, "storm_test.png")
    plt.savefig(out, dpi=120)
    print(f"Saved -> {out}")


def main():
    cfg = CONFIG
    _, models_dir, logs_dir, images_dir, _ = get_paths(cfg)
    seed = cfg["deploy"]["seed"]
    plot_learning_curve(cfg, logs_dir, images_dir, seed=seed)
    run_storm_test(cfg, models_dir, images_dir, seed=seed)
    print("Figures generated.")


if __name__ == "__main__":
    main()
