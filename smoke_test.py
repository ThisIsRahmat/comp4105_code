from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
import numpy as np

mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=400)
env.reset()

total_reward = 0
deliveries = 0
for t in range(400):
    a0 = Action.INDEX_TO_ACTION[np.random.choice(6)]
    a1 = Action.INDEX_TO_ACTION[np.random.choice(6)]
    next_state, reward, done, info = env.step((a0, a1))
    total_reward += reward
    if reward > 0:
        deliveries += 1
    if done:
        break

print(f"Random episode complete. Reward: {total_reward}. Deliveries: {deliveries}")