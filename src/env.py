"""Simplified wind-turbine pitch-control environment.

IMPORTANT — modelling honesty
-----------------------------
This is a *control-oriented* simulation, not a validated aeroelastic model.
Power capture is modelled as ``theoretical_power * cos(pitch)`` (a feathering
proxy) and the rotor uses a lumped force balance with dimensionless gains.
It reproduces the correct *qualitative* control trade-offs (feathering sheds
power and brakes the rotor; gusts spin it up) but the absolute kW / RPM values
are NOT calibrated to a real 3.6 MW machine. For physically faithful results
the environment should be backed by BEM/OpenFAST. See README "Methods &
Limitations".

All constants come from ``config.yaml`` so experiments are reproducible.
"""
import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, resolve


class WindTurbineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, randomize=False):
        super().__init__()
        cfg = config or CONFIG
        e = cfg["env"]
        r = cfg["reward"]
        # Domain randomisation is opt-in (training only); eval keeps scale=1.0.
        self.randomize = randomize and e.get("domain_randomization", False)

        # --- Load wind data ------------------------------------------------
        data_path = resolve(cfg["paths"]["data_file"])
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
        else:
            print(f"Warning: data file not found at {data_path}. Using dummy data.")
            self.data = pd.DataFrame(
                {
                    "wind_speed": [10.0] * 100,
                    "wind_sin": [0.0] * 100,
                    "wind_cos": [0.0] * 100,
                    "theoretical_power": [3000.0] * 100,
                }
            )
        self.max_idx = len(self.data) - 1

        # --- Physics constants (from config) -------------------------------
        self.min_pitch = e["min_pitch"]
        self.max_pitch = e["max_pitch"]
        self.max_rotor_speed = e["max_rotor_speed"]     # soft safety limit
        self.hard_rotor_speed = e["hard_rotor_speed"]   # physical clamp
        self.rated_power = e["rated_power"]
        self.initial_rotor_speed = e["initial_rotor_speed"]
        self.episode_max_steps = e["episode_max_steps"]
        self.pitch_rate = e["pitch_rate"]
        self.wind_norm = e["wind_norm"]

        # Domain-randomisation params + theoretical power-curve breakpoints.
        self.gust_prob = e.get("gust_prob", 0.0)
        self.wind_scale_range = e.get("wind_scale_range", [1.0, 1.0])
        self.wind_clip = e.get("wind_clip", 30.0)
        self.rated_wind = e.get("rated_wind", 13.0)
        self.cut_in_wind = e.get("cut_in_wind", 3.5)
        self.wind_scale = 1.0

        self.input_force_gain = e["input_force_gain"]
        self.input_force_ref_wind = e["input_force_ref_wind"]
        self.load_gain = e["load_gain"]

        # --- Reward weights ------------------------------------------------
        self.power_weight = r["power_weight"]
        self.safety_penalty = r["safety_penalty"]
        self.smoothness_penalty = r["smoothness_penalty"]
        self.terminate_on_overspeed = r["terminate_on_overspeed"]

        # --- Spaces --------------------------------------------------------
        # Action: incremental pitch change in degrees, clipped to [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: [wind/25, sin, cos, rotor/max_rotor, pitch/90].
        # Bounds are widened past 1.0 because rotor speed can transiently
        # exceed the soft limit (up to hard_rotor/max_rotor ~= 1.43).
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

        self.current_idx = 0
        self.current_pitch = 0.0
        self.rotor_speed = self.initial_rotor_speed
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        # Seeds the Gymnasium RNG (self.np_random) — the correct, reproducible
        # way (no global np.random.seed side effects).
        super().reset(seed=seed)

        self.current_idx = int(
            self.np_random.integers(0, max(1, self.max_idx - self.episode_max_steps))
        )
        self.current_pitch = 0.0
        self.rotor_speed = self.initial_rotor_speed
        self.steps_taken = 0

        # Domain randomisation: occasionally scale this episode's wind up to a
        # storm regime (training only). Eval/dashboard keep scale = 1.0.
        if self.randomize and self.np_random.random() < self.gust_prob:
            lo, hi = self.wind_scale_range
            self.wind_scale = float(self.np_random.uniform(lo, hi))
        else:
            self.wind_scale = 1.0

        return self._get_obs(), {}

    def _theoretical_power(self, wind):
        """Manufacturer-style power curve (cut-in -> cubic -> rated). Used when
        wind has been scaled away from the dataset values."""
        if wind < self.cut_in_wind:
            return 0.0
        if wind >= self.rated_wind:
            return self.rated_power
        frac = (wind - self.cut_in_wind) / (self.rated_wind - self.cut_in_wind)
        return self.rated_power * frac ** 3

    def _row(self):
        """Current data row, with the index clamped to the last valid row.
        Defensive: a correctly-used env resets after truncation, but clamping
        guarantees we never index past the end (e.g. on tiny dummy datasets)."""
        return self.data.iloc[min(self.current_idx, self.max_idx)]

    def _wind_and_power(self):
        """Effective wind and theoretical power for the current row, applying any
        domain-randomisation wind scaling consistently."""
        row = self._row()
        if self.wind_scale == 1.0:
            return row["wind_speed"], row["theoretical_power"]
        wind = min(row["wind_speed"] * self.wind_scale, self.wind_clip)
        return wind, self._theoretical_power(wind)

    def _get_obs(self):
        row = self._row()
        wind, _ = self._wind_and_power()
        return np.array(
            [
                wind / self.wind_norm,
                row["wind_sin"],
                row["wind_cos"],
                self.rotor_speed / self.max_rotor_speed,
                self.current_pitch / self.max_pitch,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        wind_speed, theoretical_max = self._wind_and_power()

        # Action: incremental pitch change (scaled by the actuator pitch rate).
        pitch_delta = float(np.clip(action[0], -1.0, 1.0)) * self.pitch_rate
        self.current_pitch = float(
            np.clip(self.current_pitch + pitch_delta, self.min_pitch, self.max_pitch)
        )

        # Feathering proxy: capture falls with cos(pitch).
        efficiency = max(0.0, np.cos(np.radians(self.current_pitch)))
        power_output = 0.0 if self.rotor_speed < 1.0 else theoretical_max * efficiency

        # Damped first-order rotor model with a STABLE equilibrium (qualitative,
        # not BEM). Aerodynamic torque drives the rotor up; a generator/friction
        # load term (proportional to rotor speed) pulls it back, so the rotor
        # settles at rotor* = aero_torque / load_gain. Feathering (higher pitch)
        # lowers aero_torque via cos(pitch) and hence the equilibrium speed.
        aero_torque = (
            (wind_speed / self.input_force_ref_wind) ** 2
            * efficiency
            * self.input_force_gain
        )
        load_torque = self.load_gain * self.rotor_speed
        speed_change = aero_torque - load_torque
        self.rotor_speed = float(
            np.clip(self.rotor_speed + speed_change, 0.0, self.hard_rotor_speed)
        )

        # --- Multi-objective reward ---------------------------------------
        reward = self.power_weight * (power_output / self.rated_power)
        overspeed = self.rotor_speed > self.max_rotor_speed
        if overspeed:
            reward -= self.safety_penalty
        reward -= self.smoothness_penalty * abs(pitch_delta)

        # Termination vs truncation (proper Gymnasium semantics).
        terminated = bool(overspeed and self.terminate_on_overspeed)

        self.current_idx += 1
        self.steps_taken += 1
        truncated = bool(
            self.current_idx >= self.max_idx
            or self.steps_taken >= self.episode_max_steps
        )

        info = {
            "power": power_output,
            "theoretical_power": theoretical_max,
            "rotor_speed": self.rotor_speed,
            "pitch": self.current_pitch,
            "overspeed": overspeed,
            "pitch_delta": abs(pitch_delta),
        }
        return self._get_obs(), reward, terminated, truncated, info
