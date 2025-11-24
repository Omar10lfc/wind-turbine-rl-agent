import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Add current folder to path so we can import env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import WindTurbineEnv

def get_paths():
    """
    Returns the absolute paths for logs, models, and images folders.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    log_dir = os.path.join(project_root, 'logs')
    models_dir = os.path.join(project_root, 'models')
    images_dir = os.path.join(project_root, 'images')
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    return log_dir, models_dir, images_dir

# --- TEST 1: Learning Curve ---
def plot_learning_curve():
    log_dir, _, images_dir = get_paths()
    print(f"Generating Learning Curve from: {log_dir}")
    
    if not os.path.exists(log_dir):
        print("Logs not found.")
        return

    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
    except Exception:
        print("No monitor.csv found in logs.")
        return

    if len(x) < 2: return

    # Dynamic Smoothing
    window_size = max(10, int(len(x) * 0.1)) 
    y_smoothed = pd.Series(y).rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, alpha=0.2, color='#999999', label='Raw Reward')
    plt.plot(x, y_smoothed, color='#0066cc', linewidth=2.5, label='Trend')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Agent Learning Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'agent_learning_curve.png'))
    print("Saved 'agent_learning_curve.png'")

# --- TEST 2: Real Variable Wind ---
def run_variable_wind_test():
    _, models_dir, images_dir = get_paths()
    model_path = os.path.join(models_dir, "ppo_wind_turbine_final.zip")

    if not os.path.exists(model_path):
        print("Model not found! Train first.")
        return

    print("Running Variable Wind Simulation (Real Data)...")
    model = PPO.load(model_path)
    test_env = WindTurbineEnv()
    obs, _ = test_env.reset()

    winds, powers = [], []

    # Run 200 steps (Real Kaggle Data sequence)
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        
        winds.append(obs[0] * 25.0) 
        powers.append(info['power'])
        
        if done: obs, _ = test_env.reset()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Time Step (10 mins)')
    ax1.set_ylabel('Wind Speed (m/s)', color='tab:blue')
    ax1.plot(winds, color='tab:blue', label='Wind Speed', alpha=0.6, linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Power Generated (kW)', color='tab:green')
    ax2.plot(powers, color='tab:green', label='Power Output', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title("RL Agent Performance: Real Wind Data")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'variable_wind_test.png'))
    print("Saved 'variable_wind_test.png'")

# --- TEST 3: Synthetic Storm (Safety Test) ---
def run_storm_test():
    _, models_dir, images_dir = get_paths()
    model_path = os.path.join(models_dir, "ppo_wind_turbine_final.zip")
    
    if not os.path.exists(model_path): return

    print("Running Synthetic Storm Simulation...")
    model = PPO.load(model_path)
    env = WindTurbineEnv()
    obs, _ = env.reset()

    # 1. Define the Storm Profile (Wind Ramp)
    storm_duration = 100
    winds_ramp = np.concatenate([
        np.linspace(5, 25, 40),   # Ramp up
        np.linspace(25, 25, 20),  # Hold
        np.linspace(25, 5, 40)    # Ramp down
    ])

    # 2. Calculate Theoretical Power
    def get_theoretical_power(w):
        if w < 3.5: return 0
        if w >= 13.0: return 3600.0
        return 3600 * ((w - 3.5) / (13.0 - 3.5))**3

    theoretical_curve = [get_theoretical_power(w) for w in winds_ramp]

    # 3. Override Environment Data
    storm_data = pd.DataFrame({
        'wind_speed': winds_ramp,
        'wind_sin': np.zeros(storm_duration), 
        'wind_cos': np.zeros(storm_duration),
        'theoretical_power': theoretical_curve 
    })
    env.data = storm_data
    env.max_idx = len(storm_data) - 1
    env.current_idx = 0 

    # 4. Run Simulation
    winds, powers, pitches, rotors = [], [], [], []

    for _ in range(storm_duration - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        winds.append(obs[0] * 25.0) 
        pitches.append(obs[4] * 90.0)
        rotors.append(obs[3] * 14.0) 
        powers.append(info['power'])

    # 5. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Wind vs Power
    ax1.plot(winds, color='blue', label='Wind Speed', linestyle='--')
    ax1.set_ylabel('Wind (m/s)', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(powers, color='green', label='Power', linewidth=2)
    ax1_twin.set_ylabel('Power (kW)', color='green')
    ax1_twin.axhline(3600, color='red', linestyle=':', label='Max Limit')
    ax1.set_title("Wind vs Power (Corrected Physics)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pitch Angle
    ax2.plot(pitches, color='orange', linewidth=2)
    ax2.set_ylabel('Pitch Angle (°)')
    ax2.set_title("Agent Action: Pitch Control")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rotor Speed
    ax3.plot(rotors, color='purple', linewidth=2)
    ax3.axhline(14.0, color='red', linestyle='--', label='Safety Limit')
    ax3.set_ylabel('Rotor RPM')
    ax3.set_title("Safety Check: Rotor Speed")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'storm_test.png'))
    print("Saved 'storm_test.png'")
    
    
    
# --- TEST 4: AI vs PID Benchmark ---

class PIDController:
    """ Simple PID implementation for comparison """
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.prev_error = 0
        self.integral = 0

    def compute(self, measurement):
        error = measurement - self.target
        self.integral += error
        derivative = error - self.prev_error
        
        # Calculate output and update state
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

def run_comparison_test():
    print("\n⚔️  Running Benchmark: AI Agent vs. Standard PID Controller...")
    
    # Setup
    _, models_dir, _ = get_paths()
    model_path = os.path.join(models_dir, "ppo_wind_turbine_final.zip")
    if not os.path.exists(model_path): return

    model = PPO.load(model_path)
    env = WindTurbineEnv()
    
    # Configuration
    steps = 1000 # Longer test
    # A standard PID tuning
    pid = PIDController(kp=0.5, ki=0.0, kd=0.1, target=13.0) 
    
    # --- ROUND 1: THE AI AGENT ---
    obs, _ = env.reset(seed=42) 
    ai_power_total = 0
    ai_safety_violations = 0
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        ai_power_total += info['power']
        
        # Check Safety
        if env.rotor_speed > env.max_rotor_speed:
            ai_safety_violations += 1
            
        if done: obs, _ = env.reset()

    # --- ROUND 2: THE PID CONTROLLER ---
    obs, _ = env.reset(seed=42) 
    pid_power_total = 0
    pid_safety_violations = 0
    
    for _ in range(steps):
        current_rpm = env.rotor_speed
        
        # PID Control Logic
        pitch_change = pid.compute(current_rpm)
        action = [np.clip(pitch_change, -1.0, 1.0)]
        
        obs, _, done, _, info = env.step(action)
        pid_power_total += info['power']
        
        # Check Safety
        if env.rotor_speed > env.max_rotor_speed:
            pid_safety_violations += 1
            
        if done: obs, _ = env.reset()

    # --- RESULTS ---
    print(f"\n{'='*40}")
    print(f"       BENCHMARK RESULTS       ")
    print(f"{'='*40}")
    
    print(f"AI Energy:   {ai_power_total:,.0f} kW")
    print(f"AI Violations: {ai_safety_violations} steps")
    print(f"-"*20)
    print(f"PID Energy:  {pid_power_total:,.0f} kW")
    print(f"PID Violations: {pid_safety_violations} steps")
    print(f"{'='*40}")

    # The Final Verdict Logic
    if pid_safety_violations > ai_safety_violations:
        print("VERDICT: AI WINS (SAFETY).")
        print("The PID generated more power but was reckless.")
        print("The AI learned to prioritize the survival of the machine.")
    elif ai_power_total > pid_power_total:
        print("VERDICT: AI WINS (EFFICIENCY).")
    else:
        print("VERDICT: PID is still superior. Consider training for 500k steps.")
        
# --- TEST 5: Quantitative Report Card ---
def run_metrics_test():
    print("\nGenerating Engineering Report Card...")
    
    # 1. Setup
    _, models_dir, _ = get_paths()
    model_path = os.path.join(models_dir, "ppo_wind_turbine_final.zip")
    
    if not os.path.exists(model_path):
        print("Model not found! Train first.")
        return

    model = PPO.load(model_path)
    env = WindTurbineEnv()
    obs, _ = env.reset()

    # 2. Run Simulation
    steps = 1000
    
    total_generated_power = 0
    total_theoretical_power = 0
    safety_violations = 0
    pitch_movements = []
    last_pitch = 0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        
        # A. Power Data (THE FIX)
        # Now we get both values directly from the environment
        generated = info['power']
        theoretical = info['theoretical_power']
        
        total_generated_power += generated
        total_theoretical_power += theoretical

        # B. Safety Data
        rotor_speed = obs[3] * 14.0 # Un-normalize
        if rotor_speed > 14.0:
            safety_violations += 1

        # C. Stability Data
        current_pitch = obs[4] * 90.0
        pitch_movements.append(abs(current_pitch - last_pitch))
        last_pitch = current_pitch

        if done: obs, _ = env.reset()

    # 3. Calculate Final Metrics
    pce = (total_generated_power / total_theoretical_power) * 100 if total_theoretical_power > 0 else 0
    violation_rate = (safety_violations / steps) * 100
    avg_pitch_travel = np.mean(pitch_movements)

    # 4. Print Report Card
    print("\n" + "="*40)
    print("       AI AGENT REPORT CARD       ")
    print("="*40)
    print(f"1. Power Efficiency (PCE):   {pce:.2f}%")
    # Strict grading: 95-101% is great. Over 101% implies a physics bug (which we just fixed).
    print(f"   -> Verdict: {'GREAT' if 90 <= pce <= 100 else 'GOOD' if pce > 80 else 'SUSPICIOUS'}")
    
    print(f"\n2. Safety Violation Rate:    {violation_rate:.2f}%")
    print(f"   -> Verdict: {'GREAT' if violation_rate == 0 else 'ACCEPTABLE' if violation_rate < 2 else 'DANGEROUS'}")
    
    print(f"\n3. Average Pitch Movement:   {avg_pitch_travel:.2f} degrees/step")
    print(f"   -> Verdict: {'SMOOTH' if avg_pitch_travel < 1.0 else 'JITTERY '}")
    print("="*40)

if __name__ == "__main__":
    plot_learning_curve()
    run_variable_wind_test()
    run_storm_test()
    run_comparison_test()
    run_metrics_test()
    print("\n All Evaluation Graphs and Tests Generated!")