import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env_wrapper import OvercookedGymEnv

parser = argparse.ArgumentParser()
parser.add_argument("--layout", default="cramped_room")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=200_000)
args = parser.parse_args()

def make_env():
    return Monitor(OvercookedGymEnv(layout_name=args.layout, horizon=400))

vec_env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, ent_coef=0.01, verbose=1, seed=args.seed,
)

model.learn(total_timesteps=args.timesteps)

model_name = f"ppo_no_comm__{args.layout}__seed{args.seed}"
model.save(model_name)
print(f"\n Training complete. Saved as: {model_name}.zip")