from src.env import WindTurbineEnv
env = WindTurbineEnv()
obs, _ = env.reset()

print("Initial Observation:", obs)

# Run 10 random steps
for i in range(20):
    action = env.action_space.sample() 
    obs, reward, done, _, info = env.step(action)
    
    print(f"Step {i}: Power={info['power']:.2f} kW, Reward={reward:.4f}")
    
    if done:
        print("Turbine exploded (or episode finished)!")
        break