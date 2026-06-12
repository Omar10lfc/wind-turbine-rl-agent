"""Rigorous multi-objective evaluation.

Produces, on an identical wind sequence:
  1. N-seed RL agent metrics with mean +/- std (energy, safety, actuator wear).
  2. A *tuned* PID baseline found by sweeping a gain grid (not a strawman).
  3. The energy-vs-safety Pareto picture for the PID family + the RL agents.

Outputs go to results/: per-seed CSV, PID sweep CSV, a Markdown summary table
(pasted into the README), and a Pareto scatter plot.
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, get_paths
from env import WindTurbineEnv
from agent_utils import (
    load_rl_predict,
    make_pid_predict,
    run_over_windows,
    eval_window_starts,
)


def evaluate_rl(cfg, models_dir, starts, n_steps):
    """Run every trained seed across all windows; per-seed metrics are averaged
    over windows. Returns a DataFrame (one row per seed)."""
    rows = []
    for seed in cfg["train"]["seeds"]:
        predict = load_rl_predict(seed, models_dir, cfg)
        if predict is None:
            print(f"  [skip] no model for seed {seed}")
            continue
        env = WindTurbineEnv(cfg)
        m = run_over_windows(env, predict, starts, n_steps)
        m["seed"] = seed
        rows.append(m)
        print(
            f"  seed {seed}: energy={m['energy_mwh']:.2f} MWh  "
            f"viol={m['violation_rate']:.2f}%  travel={m['avg_pitch_travel']:.3f} deg/step"
        )
    return pd.DataFrame(rows)


def sweep_pid(cfg, starts, n_steps):
    """Grid-sweep PID gains; each config averaged over the same windows."""
    e = cfg["eval"]
    rows = []
    for kp in e["pid_kp_grid"]:
        for kd in e["pid_kd_grid"]:
            env = WindTurbineEnv(cfg)
            predict = make_pid_predict(kp, 0.0, kd, e["pid_target_rpm"])
            m = run_over_windows(env, predict, starts, n_steps, needs_env=True)
            m.update({"kp": kp, "kd": kd})
            rows.append(m)
    df = pd.DataFrame(rows)
    print(f"  swept {len(df)} PID configurations")
    return df


def pick_tuned_pid(pid_df):
    """Fair baseline = the PID config that maximises energy subject to the
    fewest safety violations (lexicographic: safety first, then energy)."""
    min_viol = pid_df["violations"].min()
    safe = pid_df[pid_df["violations"] == min_viol]
    return safe.loc[safe["energy_mwh"].idxmax()]


def agg(df, col):
    return df[col].mean(), df[col].std(ddof=0)


def write_summary(rl_df, tuned, results_dir, n_windows):
    """Write a Markdown summary table (mean +/- std) for the README."""
    lines = []
    lines.append(
        "## Multi-Objective Evaluation (N=%d seeds, averaged over %d wind windows)\n"
        % (len(rl_df), n_windows)
    )
    lines.append("| Metric | PPO Agent (mean ± std) | Tuned PID Baseline | Notes |")
    lines.append("| :--- | :--- | :--- | :--- |")

    e_m, e_s = agg(rl_df, "energy_mwh")
    v_m, v_s = agg(rl_df, "violation_rate")
    t_m, t_s = agg(rl_df, "avg_pitch_travel")
    c_m, c_s = agg(rl_df, "capture_efficiency")

    lines.append(
        f"| Energy produced (MWh) | {e_m:.2f} ± {e_s:.2f} | {tuned['energy_mwh']:.2f} | "
        "Higher = more capture |"
    )
    lines.append(
        f"| Safety violation rate (%) | {v_m:.2f} ± {v_s:.2f} | {tuned['violation_rate']:.2f} | "
        "Lower = safer (rotor over soft limit) |"
    )
    lines.append(
        f"| Actuator travel (°/step) | {t_m:.3f} ± {t_s:.3f} | {tuned['avg_pitch_travel']:.3f} | "
        "Lower = less pitch-bearing wear |"
    )
    lines.append(
        f"| Power capture efficiency (%) | {c_m:.2f} ± {c_s:.2f} | {tuned['capture_efficiency']:.2f} | "
        "Generated / theoretical |"
    )
    lines.append(
        f"\n*Tuned PID:* kp={tuned['kp']}, ki=0.0, kd={tuned['kd']}, "
        f"target={CONFIG['eval']['pid_target_rpm']} RPM "
        "(best energy at the lowest violation count from the gain sweep).\n"
    )
    text = "\n".join(lines)
    out = os.path.join(results_dir, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved -> {out}")


def plot_pareto(rl_df, pid_df, tuned, images_dir):
    """Energy vs safety-violation trade-off; marker shade = actuator wear."""
    # Shared colour scale (actuator travel) across PID + PPO so the third
    # dimension is comparable for both controller families.
    travel_all = list(pid_df["avg_pitch_travel"]) + list(rl_df["avg_pitch_travel"])
    vmin, vmax = min(travel_all), max(travel_all)

    plt.figure(figsize=(9, 6))
    sc = plt.scatter(
        pid_df["violation_rate"],
        pid_df["energy_mwh"],
        c=pid_df["avg_pitch_travel"],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=80,
        edgecolor="k",
        label="PID sweep",
    )
    plt.colorbar(sc, label="Actuator travel (°/step)  —  lower is better")
    plt.scatter(
        rl_df["violation_rate"],
        rl_df["energy_mwh"],
        c=rl_df["avg_pitch_travel"],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        marker="*",
        s=320,
        edgecolor="red",
        linewidths=1.8,
        label="PPO agents (per seed)",
        zorder=5,
    )
    plt.scatter(
        [tuned["violation_rate"]],
        [tuned["energy_mwh"]],
        facecolors="none",
        edgecolors="blue",
        s=320,
        linewidths=2.5,
        label="Tuned PID (chosen baseline)",
        zorder=6,
    )
    plt.xlabel("Safety violation rate (%)  →  worse")
    plt.ylabel("Energy produced (MWh)  →  better")
    plt.title("Energy vs. Safety Trade-off: PPO vs. PID family")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(images_dir, "pareto_tradeoff.png")
    plt.savefig(out, dpi=120)
    print(f"Saved -> {out}")


def main():
    cfg = CONFIG
    _, models_dir, _, images_dir, results_dir = get_paths(cfg)
    n_steps = cfg["eval"]["eval_steps"]
    starts = eval_window_starts(cfg)
    print(f"Evaluating across {len(starts)} wind windows of {n_steps} steps "
          f"(start indices {starts})\n")

    print("Evaluating RL agents (averaged over windows)...")
    rl_df = evaluate_rl(cfg, models_dir, starts, n_steps)
    if rl_df.empty:
        print("No trained models found. Run: python src/train.py --all-seeds")
        return

    print("\nSweeping PID baseline (averaged over windows)...")
    pid_df = sweep_pid(cfg, starts, n_steps)
    tuned = pick_tuned_pid(pid_df)

    drop = ["energy_per_window", "viol_per_window"]
    rl_df.drop(columns=drop).to_csv(
        os.path.join(results_dir, "rl_seed_metrics.csv"), index=False
    )
    pid_df.drop(columns=drop).to_csv(
        os.path.join(results_dir, "pid_sweep.csv"), index=False
    )

    write_summary(rl_df, tuned, results_dir, len(starts))
    plot_pareto(rl_df, pid_df, tuned, images_dir)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
