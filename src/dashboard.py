import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# --- PATH SETUP ---
# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env import WindTurbineEnv
except ImportError:
    from src.env import WindTurbineEnv

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Wind Turbine AI Controller",
    page_icon="🌬️",
    layout="wide"
)

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_resources():
    # Find model path automatically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'models', 'ppo_wind_turbine_final.zip')
    
    if not os.path.exists(model_path):
        return None, None
    
    model = PPO.load(model_path)
    env = WindTurbineEnv()
    return model, env

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = {
        'wind': [], 'power': [], 'pitch': [], 'rotor': [], 'limit': []
    }
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🎮 Control Panel")
simulation_mode = st.sidebar.radio("Mode", ["Manual Control", "Storm Scenario"])

manual_wind = 0.0
if simulation_mode == "Manual Control":
    manual_wind = st.sidebar.slider("Set Wind Speed (m/s)", 0.0, 30.0, 10.0, 0.5)
    st.sidebar.caption("💡 Try moving this past 13 m/s to see the AI pitch up!")

run_sim = st.sidebar.button("Step Simulation")
auto_run = st.sidebar.checkbox("Auto-Run (Live)")
reset_btn = st.sidebar.button("Reset System")

if reset_btn:
    st.session_state.history = {'wind': [], 'power': [], 'pitch': [], 'rotor': [], 'limit': []}
    st.session_state.step_count = 0

# --- MAIN LOGIC ---
st.title("🌬️ Autonomous Wind Turbine Dashboard")
st.markdown("""
**Real-time Inference:** The PPO Agent observes the wind and adjusts blade pitch to maximize power while ensuring safety.
""")

model, env = load_resources()

if model is None:
    st.error("❌ Model not found! Please run 'python src/train.py' first.")
    st.stop()

# Function to run one timestep
def step_environment():
    current_step = st.session_state.step_count
    
    # 1. Determine Wind Speed based on mode
    if simulation_mode == "Manual Control":
        w = manual_wind
    else:
        # Synthetic Storm Profile
        if current_step < 30: w = 5 + (current_step * 0.5) # Ramp Up
        elif current_step < 60: w = 20.0 # Hold High
        else: w = 20 - ((current_step - 60) * 0.5) # Ramp Down
        if w < 0: w = 0
    
    # 2. Calculate Theoretical Power for this wind speed
    # Used for accurate internal calculations
    if w < 3.5: t_pow = 0
    elif w >= 13.0: t_pow = 3600
    else: t_pow = 3600 * ((w - 3.5) / (13.0 - 3.5))**3
    
    # --- THE FIX: Create a 100-row buffer ---
    # We create 100 identical rows. This ensures that when the environment
    # tries to read "Index + 1", it always finds valid data.
    env.data = pd.DataFrame({
        'wind_speed': [w] * 100, 
        'wind_sin': [0] * 100, 
        'wind_cos': [0] * 100, 
        'theoretical_power': [t_pow] * 100
    })
    
    # Reset internal index to 0 so it reads the first of these rows
    env.max_idx = 99
    env.current_idx = 0
    
    # 3. Handle First Step Initialization
    # If it's the very first step, we need to initialize the physics variables
    if current_step == 0:
        env.rotor_speed = 10.0
        env.current_pitch = 0.0

    # 4. Construct Observation Manually
    # We do this to ensure the Agent sees exactly what the Slider is set to
    # Structure: [Wind/25, Sin, Cos, Rotor/14, Pitch/90]
    obs = np.array([
        w / 25.0, 
        0.0, 
        0.0, 
        env.rotor_speed / 14.0, 
        env.current_pitch / 90.0
    ], dtype=np.float32)

    # 5. Predict Action & Step Environment
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, info = env.step(action)
    
    # 6. Save History for Plotting
    st.session_state.history['wind'].append(w)
    st.session_state.history['power'].append(info['power'])
    st.session_state.history['pitch'].append(env.current_pitch)
    st.session_state.history['rotor'].append(env.rotor_speed)
    st.session_state.history['limit'].append(3600)
    
    st.session_state.step_count += 1

# Run Logic
if run_sim or auto_run:
    step_environment()
    if auto_run:
        time.sleep(0.1)
        st.rerun()

# --- VISUALIZATION ---

# 1. KPI Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

if len(st.session_state.history['wind']) > 0:
    curr_wind = st.session_state.history['wind'][-1]
    curr_power = st.session_state.history['power'][-1]
    curr_pitch = st.session_state.history['pitch'][-1]
    curr_rotor = st.session_state.history['rotor'][-1]
else:
    curr_wind, curr_power, curr_pitch, curr_rotor = 0, 0, 0, 0

kpi1.metric("💨 Wind Speed", f"{curr_wind:.1f} m/s", delta_color="inverse")
kpi2.metric("⚡ Power Output", f"{curr_power:.0f} kW")
kpi3.metric("📐 Blade Pitch", f"{curr_pitch:.1f} °", delta_color="off")

# Safety Logic for Rotor Color
rotor_label = "⚙️ Rotor RPM"
if curr_rotor > 14.0:
    rotor_label = "⚠️ OVER-SPEED!"
kpi4.metric(rotor_label, f"{curr_rotor:.1f} RPM", delta_color="off")


# 2. Charts
st.markdown("### 📈 Live Telemetry")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.caption("Environment (Wind) & Response (Pitch)")
    chart_data_1 = pd.DataFrame({
        'Wind (m/s)': st.session_state.history['wind'],
        'Pitch Angle (°)': st.session_state.history['pitch']
    })
    st.line_chart(chart_data_1, height=250, color=["#0000FF", "#FFA500"])

with chart_col2:
    st.caption("Power Generation Performance")
    chart_data_2 = pd.DataFrame({
        'Power (kW)': st.session_state.history['power'],
        'Limit (kW)': st.session_state.history['limit']
    })
    st.line_chart(chart_data_2, height=250, color=["#00FF00", "#FF0000"])

# 3. Decision Logic Explanation
if curr_wind > 13.0:
    st.warning(f"⚠️ **High Wind Detected ({curr_wind:.1f} m/s)!** The Agent is feathering the blades (Pitch: {curr_pitch:.1f}°) to prevent mechanical damage.")
elif curr_wind < 13.0 and curr_pitch > 10.0:
    st.warning(f"⚠️ **Recovery Mode:** Wind is safe ({curr_wind:.1f} m/s), but Agent is stabilizing rotor momentum (Pitch: {curr_pitch:.1f}°).")
elif curr_wind < 3.5:
    st.info("💤 **Low Wind.** Turbine is in standby to conserve energy.")
else:
    st.success("✅ **Optimal Conditions.** Agent is tracking maximum power point.")