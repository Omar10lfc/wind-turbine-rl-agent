from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import WindTurbineEnv
import matplotlib.pyplot as plt

log_dir = "logs"
models_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 2. Create Environment with "Monitor" (Crucial for plotting graphs later)
env = WindTurbineEnv()
env = Monitor(env, log_dir)

# 3. Define the PPO Model
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
    ent_coef=0.01, # Encourages exploration so it doesn't get stuck
    tensorboard_log=log_dir,
    seed=42
)

# Train
# 100,000 steps is a good start.
print("Training started...")
model.learn(total_timesteps=100000)

# Save the final brain
model.save(f"{models_dir}/ppo_wind_turbine_final")
print("Training Complete! Model saved.")