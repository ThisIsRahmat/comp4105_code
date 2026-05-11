
import sys, csv, os, random
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)

# experiment parameters 
condition = "independent"
layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
seeds     = [0, 1, 2, 3, 4]
horizon   = 400
csv_path  = os.path.join("results", f"{condition}.csv")
csv_cols  = [
    "condition", "layout", "seed", "ep_length", "sparse_reward",
    "deliveries", "deliveries_a0", "deliveries_a1",
    "idle_a0", "idle_a1", "collisions",
]

# helpers - defines the no_movement action - move zero in x and y axis 
# excused to calculate the collision metrics - basically checks if agent hasn't moved 
stay = (0, 0)


# calculates the metrics for csv results later in main 

def compute_metrics_inline(traj, sparse_reward, ep_stats):
    idle_a0 = idle_a1 = collisions = 0

    for t in range(len(traj) - 1):
        s_t  = traj[t][0]
        s_tp = traj[t + 1][0]
        ja   = traj[t][1]  # joint action tuple

# counts idleness of the agents
        # idle = position unchanged
        if s_t.players[0].position == s_tp.players[0].position:
            idle_a0 += 1
        if s_t.players[1].position == s_tp.players[1].position:
            idle_a1 += 1

        # counts collisions - basically when the agents move into the same cell or swap positions
        p0, p1     = s_t.players[0].position,  s_t.players[1].position
        p0n, p1n   = s_tp.players[0].position, s_tp.players[1].position
        swapped    = (p0n == p1 and p1n == p0)
        same_cell  = (p0n == p1n and ja[0] != stay and ja[1] != stay)
        if swapped or same_cell:
            collisions += 1


    # counts deliveries - the number of soups delivered by agents 
    deliveries    = sparse_reward // 20
    deliveries_a0 = len(ep_stats["soup_delivery"][0])
    deliveries_a1 = len(ep_stats["soup_delivery"][1])
    return deliveries, deliveries_a0, deliveries_a1, idle_a0, idle_a1, collisions


# main 
def main():
    os.makedirs("results", exist_ok=True)

    # write CSV header
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_cols)

    for layout in layouts:
        print("="*60)
        print(f"Layout: {layout}")
        print("="*60)

        mdp = OvercookedGridworld.from_layout_name(
            layout,
            start_all_orders=[{"ingredients": ["onion", "onion", "onion"]}],
            start_bonus_orders=[],
        )
        print("  Computing MLAM …")
        mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=True,
        )

        for seed in seeds:
            # seed RNGs
            random.seed(seed)
            np.random.seed(seed)

            env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

            agent_0 = GreedyHumanModel(mlam)
            agent_1 = GreedyHumanModel(mlam)
            agent_pair = AgentPair(agent_0, agent_1)

            print(f"  Seed {seed} … ", end="", flush=True)
            traj, ep_length, sparse_r, shaped_r = env.run_agents(
                agent_pair, display=False
            )

            final_info = traj[-1][4]
            ep_stats   = final_info["episode"]["ep_game_stats"]

            deliveries, d_a0, d_a1, idle0, idle1, colls = compute_metrics_inline(
                traj, sparse_r, ep_stats
            )

            row = [
                condition, layout, seed, ep_length, sparse_r,
                deliveries, d_a0, d_a1, idle0, idle1, colls,
            ]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print(f"reward={sparse_r}  deliveries={deliveries}  "
                  f"idle=({idle0},{idle1})  collisions={colls}")

    print(f"\n Results at {csv_path}")


if __name__ == "__main__":
    main()
