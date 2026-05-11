
import sys, csv, os, random
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)

# experiment parameters 
condition = "message_passing"
layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
seeds     = [0, 1, 2, 3, 4]
horizon   = 400
csv_path  = os.path.join("results", f"{condition}.csv")
csv_cols  = [
    "condition", "layout", "seed", "ep_length", "sparse_reward",
    "deliveries", "deliveries_a0", "deliveries_a1",
    "idle_a0", "idle_a1", "collisions",
]

STAY_ACTION = Action.STAY   # (0, 0)

# direction vectors for the five possible actions
ACTION_DELTA = {
    (0, -1): (0, -1),   # north
    (0,  1): (0,  1),   # south
    (-1, 0): (-1, 0),   # west
    (1,  0): (1,  0),   # east
    (0,  0): (0,  0),   # stay
    "interact": (0, 0),  # interact doesn't move
}


# Blackboard 
# simple shared memory: maps target positions to claiming agent index.
class Blackboard:
    def __init__(self):
        self.claims = {}          # pos → agent_index

    def clear(self):
        self.claims.clear()


# attempt to claim a position on the blackboard
    def try_claim(self, pos, agent_index):

        # if pos is already claimed by self then return True
        if pos in self.claims:
            return self.claims[pos] == agent_index   # already claimed by self
        # otherwise, claim the position for this agent
        self.claims[pos] = agent_index
        return True


# target-position helper 
def target_pos(player, action):
    # where the player would end up if the action succeeds (ignoring walls / collisions)
    if action == "interact":
        return player.position
    dx, dy = action
    px, py = player.position
    return (px + dx, py + dy)


# inline metrics 
STAY_TUPLE = (0, 0)

def compute_metrics_inline(states, actions, sparse_reward, ep_stats):
    # compute metrics from parallel lists of states and joint-actions
    idle_a0 = idle_a1 = collisions = 0

    for t in range(len(states) - 1):
        s_t  = states[t]
        s_tp = states[t + 1]
        ja   = actions[t]

        if s_t.players[0].position == s_tp.players[0].position:
            idle_a0 += 1
        if s_t.players[1].position == s_tp.players[1].position:
            idle_a1 += 1

        p0, p1   = s_t.players[0].position,  s_t.players[1].position
        p0n, p1n = s_tp.players[0].position, s_tp.players[1].position
        swapped  = (p0n == p1 and p1n == p0)
        same     = (p0n == p1n and ja[0] != STAY_TUPLE and ja[1] != STAY_TUPLE)
        if swapped or same:
            collisions += 1

    deliveries    = sparse_reward // 20
    deliveries_a0 = len(ep_stats["soup_delivery"][0])
    deliveries_a1 = len(ep_stats["soup_delivery"][1])
    return deliveries, deliveries_a0, deliveries_a1, idle_a0, idle_a1, collisions


# run one episode with message-passing 
def run_episode_mp(env, agents, mlam):
    # manual step loop with Blackboard-based conflict resolution
    bb = Blackboard()

    # set agent indices (normally done by AgentPair)
    for idx, ag in enumerate(agents):
        ag.set_agent_index(idx)
        ag.set_mdp(env.mdp)

    states  = []
    actions = []
    total_sparse = 0

    while not env.is_done():
        state = env.state
        states.append(state)

        # each agent picks an action via GreedyHumanModel
        intended = []
        for ag in agents:
            a, info = ag.action(state)
            intended.append(a)

        # compute target positions + resolve via blackboard
        bb.clear()
        final_actions = list(intended)

        # agent 0 always gets priority
        tgt0 = target_pos(state.players[0], intended[0])
        bb.try_claim(tgt0, 0)

        tgt1 = target_pos(state.players[1], intended[1])
        if not bb.try_claim(tgt1, 1):
            # if there is conflict then agent 1 is forced to stay
            final_actions[1] = STAY_ACTION

        joint_action = tuple(final_actions)
        actions.append(joint_action)

        # step environment
        next_state, r_t, done, info = env.step(joint_action)
        total_sparse += r_t

    # final state for metric computation
    states.append(env.state)

    ep_stats = info["episode"]["ep_game_stats"]
    ep_length = env.state.timestep

    return states, actions, ep_length, total_sparse, ep_stats


#  main 
def main():
    os.makedirs("results", exist_ok=True)

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
            random.seed(seed)
            np.random.seed(seed)

            env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

            agents = [GreedyHumanModel(mlam), GreedyHumanModel(mlam)]

            print(f"  Seed {seed} … ", end="", flush=True)
            states, acts, ep_length, sparse_r, ep_stats = run_episode_mp(
                env, agents, mlam
            )

            deliveries, d_a0, d_a1, idle0, idle1, colls = compute_metrics_inline(
                states, acts, sparse_r, ep_stats
            )

            row = [
                condition, layout, seed, ep_length, sparse_r,
                deliveries, d_a0, d_a1, idle0, idle1, colls,
            ]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print(f"reward={sparse_r}  deliveries={deliveries}  "
                  f"idle=({idle0},{idle1})  collisions={colls}")

    print(f"\n Results: {csv_path}")


if __name__ == "__main__":
    main()
