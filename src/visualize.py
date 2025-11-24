import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import WindTurbineEnv

def create_dashboard_gif():
    # 1. Setup Paths Automatically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    images_dir = os.path.join(project_root, 'images')
    
    model_path = os.path.join(models_dir, "ppo_wind_turbine_final.zip")
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return

    print("Loading Model and Environment...")
    model = PPO.load(model_path)
    env = WindTurbineEnv()
    obs, _ = env.reset()

    # 2. Generate Storm Data
    storm_duration = 100
    winds_ramp = np.concatenate([
        np.linspace(5, 25, 40), 
        np.linspace(25, 25, 20), 
        np.linspace(25, 5, 40)
    ])
    
    def get_theoretical_power(w):
        if w < 3.5: return 0
        if w >= 13.0: return 3600.0
        return 3600 * ((w - 3.5) / (13.0 - 3.5))**3

    theoretical_curve = [get_theoretical_power(w) for w in winds_ramp]
    
    # Override Env Data
    env.data = pd.DataFrame({
        'wind_speed': winds_ramp, 
        'wind_sin': [0]*100, 
        'wind_cos': [0]*100, 
        'theoretical_power': theoretical_curve
    })
    env.max_idx = len(env.data) - 1
    env.current_idx = 0 

    # 3. Run Simulation & Capture Frames
    print("🎬 Running Simulation to generate Animation Data...")
    history = {'wind': [], 'power': [], 'pitch': [], 'rotor': []}
    
    for _ in range(storm_duration - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        history['wind'].append(obs[0] * 25.0)
        history['power'].append(info['power'])
        history['pitch'].append(obs[4] * 90.0)
        history['rotor'].append(obs[3] * 14.0)

    # 4. Create the Dashboard Animation
    print("🎨 Drawing Dashboard...")
    fig = plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
    gs = fig.add_gridspec(2, 3)

    # A. Wind Gauge
    ax_wind = fig.add_subplot(gs[0, 0])
    ax_wind.set_facecolor('#1e1e1e')
    ax_wind.set_title("Wind Speed (m/s)", color='white')
    bar_wind = ax_wind.bar(0, 0, color='cyan', width=0.5)
    ax_wind.set_ylim(0, 30)
    ax_wind.set_xticks([])
    ax_wind.tick_params(colors='white')

    # B. Pitch Gauge
    ax_pitch = fig.add_subplot(gs[0, 2])
    ax_pitch.set_facecolor('#1e1e1e')
    ax_pitch.set_title("Blade Pitch (°)", color='white')
    bar_pitch = ax_pitch.bar(0, 0, color='orange', width=0.5)
    ax_pitch.set_ylim(0, 90)
    ax_pitch.set_xticks([])
    ax_pitch.tick_params(colors='white')

    # C. Turbine Visual
    ax_turb = fig.add_subplot(gs[:, 1], polar=True)
    ax_turb.set_facecolor('#1e1e1e')
    ax_turb.set_xticks([])
    ax_turb.set_yticks([])
    ax_turb.grid(False)
    ax_turb.set_title("Turbine RPM", color='white')
    
    # Draw blades as simple lines
    lines = [ax_turb.plot([], [], color='white', linewidth=3)[0] for _ in range(3)]

    # D. Power Chart
    ax_power = fig.add_subplot(gs[1, 0::2]) 
    ax_power.set_facecolor('#1e1e1e')
    ax_power.set_title("Power Output (kW)", color='white')
    line_power, = ax_power.plot([], [], color='#00ff00', linewidth=2)
    ax_power.set_xlim(0, 100)
    ax_power.set_ylim(0, 4000)
    ax_power.tick_params(colors='white')
    ax_power.grid(True, alpha=0.2)

    # Update Function
    def update(frame):
        # Update Bars
        w = history['wind'][frame]
        p = history['pitch'][frame]
        rot = history['rotor'][frame]

        bar_wind[0].set_height(w)
        bar_pitch[0].set_height(p)
        bar_wind[0].set_color('red' if w > 13 else 'cyan')
        
        # Update Power Line
        line_power.set_data(range(frame), history['power'][:frame])
        
        # Rotate Blades (Simulated)
        speed_factor = rot / 2.0 # Visual speed scaling
        base_angle = frame * speed_factor
        
        for i, line in enumerate(lines):
            # Draw 3 blades at 0, 120, 240 degrees offset
            angle_rad = np.deg2rad([base_angle + i*120, base_angle + i*120])
            line.set_data(angle_rad, [0, 1])

        return bar_wind, bar_pitch, line_power

    print("Saving GIF... (This takes about 20 seconds)")
    
    # Save to images folder
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    save_path = os.path.join(images_dir, 'dashboard.gif')
    
    # Using Pillow writer (standard in Python)
    ani = animation.FuncAnimation(fig, update, frames=len(history['wind']), interval=50, blit=False)
    ani.save(save_path, writer='pillow', fps=20)
    
    print(f"Dashboard saved successfully to: {save_path}")

if __name__ == "__main__":
    create_dashboard_gif()