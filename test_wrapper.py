from env_wrapper import OvercookedGymEnv

env = OvercookedGymEnv("cramped_room")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

for _ in range(10):
    obs, r, done, trunc, info = env.step(env.action_space.sample())

print("Wrapper works.")