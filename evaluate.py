import argparse
from stable_baselines3 import PPO
from env_wrapper import OvercookedGymEnv

parser = argparse.ArgumentParser()
parser.add_argument("--layout", default="cramped_room")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episodes", type=int, default=20)
args = parser.parse_args()

env = OvercookedGymEnv(layout_name=args.layout, horizon=400)
model_name = f"ppo_no_comm__{args.layout}__seed{args.seed}"
model = PPO.load(model_name, env=env)

scores_sparse = []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=1000 + ep)
    done = trunc = False
    sparse_only = 0.0
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=False)
        obs, r, done, trunc, info = env.step(int(action))
        sparse_only += info["sparse_r"]
    scores_sparse.append(sparse_only)

mean_soups = sum(scores_sparse) / (20 * args.episodes)
print(f"\n=== {args.layout} (seed {args.seed}) ===")
print(f"Mean soup deliveries per episode: {mean_soups:.2f}")
print(f"Per-episode sparse rewards: {[int(s) for s in scores_sparse]}")