import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import os

class WindTurbineEnv(gym.Env):
    def __init__(self):
        super(WindTurbineEnv, self).__init__()
        
        # --- Load Data (Smart Path Finding) ---
        # This checks multiple locations to find the data regardless of where you run main.py from
        possible_paths = [
            'data/sim_weather_clean.csv',           # Running from root
            '../data/sim_weather_clean.csv',        # Running from src/
            r'C:\Users\omarm\OneDrive\Desktop\wind-turbine-rl-Project\data\sim_weather_clean.csv' # Fallback to absolute
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"Data found at: {file_path}")
                break
        
        if file_path is None:
            print("Warning: Data file not found! Using Dummy Data.")
            d = {'wind_speed': [10]*100, 'wind_sin': [0]*100, 'wind_cos': [0]*100, 'theoretical_power': [3000]*100}
            self.data = pd.DataFrame(d)
        else:
            self.data = pd.read_csv(file_path)
            
        self.max_idx = len(self.data) - 1

        # --- Constants ---
        self.min_pitch = 0.0
        self.max_pitch = 90.0
        self.max_rotor_speed = 14.0  # RPM Safety Limit
        self.rated_power = 3600.0    # kW
        
        # --- Action & Observation ---
        # Action: Pitch change [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [Wind Speed, Sin, Cos, Rotor Speed, Pitch]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random start
        self.current_idx = np.random.randint(0, max(1, self.max_idx - 1000))
        
        # Initial State
        self.current_pitch = 0.0
        self.rotor_speed = 10.0 
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_idx]
        
        # Normalize inputs for the AI
        return np.array([
            row['wind_speed'] / 25.0, 
            row['wind_sin'], 
            row['wind_cos'], 
            self.rotor_speed / self.max_rotor_speed, 
            self.current_pitch / 90.0
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_idx]
        wind_speed = row['wind_speed']
        theoretical_max = row['theoretical_power']
        
        # 1. Action: Pitch Change
        pitch_delta = np.clip(action[0], -1.0, 1.0)
        self.current_pitch = np.clip(self.current_pitch + pitch_delta, self.min_pitch, self.max_pitch)
        
        # 2. Physics: Efficiency (Feathering)
        # Efficiency drops as pitch increases
        efficiency = max(0, np.cos(np.radians(self.current_pitch)))
        
        # 3. Physics: Power Generation
        if self.rotor_speed < 1.0:
            power_output = 0.0
        else:
            power_output = theoretical_max * efficiency

        # 4. Physics: Rotor Dynamics (The Tuned Version)
        # Input Force: Wind pushing the blades
        input_force = (wind_speed / 12.0)**2 * efficiency * 5.0 
        
        # Resistive Force: Generator load
        resistive_force = (power_output / 3600.0) * 5.0
        
        # Drag Force: Pitching up creates drag (braking)
        drag_force = (self.current_pitch / 90.0) * 2.0 
        
        # Net Acceleration
        speed_change = input_force - resistive_force - drag_force
        self.rotor_speed = np.clip(self.rotor_speed + speed_change, 0.0, 20.0)

        # 5. Reward Calculation
        reward = 0
        
        # Goal A: Generate Power
        reward += (power_output / self.rated_power)
        
        terminated = False
        # Goal B: Safety Penalty
        if self.rotor_speed > self.max_rotor_speed:
            reward -= 20.0 # Huge penalty for overspeeding
            terminated = True 
        
        # Goal C: Smoothness Penalty
        reward -= 0.01 * np.abs(pitch_delta)

        # 6. Next Step
        self.current_idx += 1
        if self.current_idx >= self.max_idx:
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {"power": power_output,"theoretical_power": theoretical_max}