import sys, csv, os, random
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)


# changable environment variables 
layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
seeds = [0, 1, 2, 3, 4]



# set agents algo
agent_0 = GreedyHumanModel(mlam)
agent_1 = GreedyHumanModel(mlam)
agent_pair = AgentPair(agent_0, agent_1)

# to compute results 
os.makedirs("results", exist_ok=True)
rows = []

for layout in layouts: 
    print(f"Running in {layout}")
    mdp = OvercookedGridworld.from_layout_name(layout)
    print("Computing MLAM...")
    env = OvercookedEnv.from_mdp(mdp, horizon=400)


    mlam = MediumLevelActionManager.from_pickle_or_compute(
    mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=True,)





env = OvercookedEnv.from_mdp(mdp, horizon=400)











# mdp = OvercookedGridworld.from_layout_name("cramped_room")
# env = OvercookedEnv.from_mdp(mdp, horizon=400)

# print("Computing MLAM...")
# mlam = MediumLevelActionManager.from_pickle_or_compute(
#     mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=True,
# )



print("Running episode...")
traj_array, ep_length, sparse_r, shaped_r = env.run_agents(agent_pair, display=False)

print(f"\nEpisode length: {ep_length}")
print(f"Sparse reward:  {sparse_r}")
print(f"Soups delivered: {sparse_r // 20}")

# per-agent breakdown
final_info = traj_array[-1][4]
ep_stats = final_info["episode"]["ep_game_stats"]
deliveries_per_agent = [len(d) for d in ep_stats["soup_delivery"]]
print(f"Deliveries per agent: {deliveries_per_agent}")