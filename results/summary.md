## Multi-Objective Evaluation (N=5 seeds, averaged over 8 wind windows)

| Metric | PPO Agent (mean ± std) | Tuned PID Baseline | Notes |
| :--- | :--- | :--- | :--- |
| Energy produced (MWh) | 217.28 ± 13.32 | 210.62 | Higher = more capture |
| Safety violation rate (%) | 7.08 ± 3.41 | 3.30 | Lower = safer (rotor over soft limit) |
| Actuator travel (°/step) | 0.493 ± 0.335 | 1.925 | Lower = less pitch-bearing wear |
| Power capture efficiency (%) | 93.31 ± 4.17 | 92.33 | Generated / theoretical |

*Tuned PID:* kp=1.0, ki=0.0, kd=1.0, target=13.0 RPM (best energy at the lowest violation count from the gain sweep).
