# import sys, csv, os, random
# import numpy as np
# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# from overcooked_ai_py.mdp.actions import Action
# from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
# from overcooked_ai_py.planning.planners import (
#     MediumLevelActionManager,
#     NO_COUNTERS_PARAMS,
# )

# # experiment parameters
# condition = "role_based"
# layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
# seeds     = [0, 1, 2, 3, 4]
# horizon   = 400
# csv_path  = os.path.join("results", f"{condition}.csv")
# csv_cols  = [
#     "condition", "layout", "seed", "ep_length", "sparse_reward",
#     "deliveries", "deliveries_a0", "deliveries_a1",
#     "idle_a0", "idle_a1", "collisions",
# ]

# STAY_ACTION = Action.STAY          # (0, 0)

# # role-restricted agent wrapper
# class RoleBasedAgent(Agent):
#     # """
#     # Role-restricted greedy agent.
#     # Overrides medium-level goal selection so the agent never plans to pick up 
#     # the wrong type of object, rather than freezing after the fact.
#     # """

#     def __init__(self, role, mlam):
#         assert role in ("fetcher", "deliverer")
#         self.role = role
#         self.mlam = mlam
#         self.inner = GreedyHumanModel(mlam)
#         super().__init__()

#     def set_agent_index(self, agent_index):
#         super().set_agent_index(agent_index)
#         self.inner.set_agent_index(agent_index)

#     def set_mdp(self, mdp):
#         super().set_mdp(mdp)
#         self.inner.set_mdp(mdp)

#     def reset(self):
#         super().reset()
#         self.inner.reset()

#     # """Filter the inner agent's motion goals by role."""
#     def ml_action(self, state):
#         """Filter the inner agent's motion goals by role."""
#         motion_goals = self.inner.ml_action(state) or []
#         me = state.players[self.agent_index]
#         mdp = self.mlam.mdp
    
#         counter_objects = mdp.get_counter_objects_dict(state)
#         pot_states_dict = mdp.get_pot_states(state)

#         allowed = []
#         if self.role == "fetcher":
#             if not me.has_object():
#                 allowed = self.mlam.pickup_onion_actions(counter_objects) or []
#             elif me.get_object().name == "onion":
#                 allowed = self.mlam.put_onion_in_pot_actions(pot_states_dict) or []
#             else:
#                 allowed = self.mlam.place_obj_on_counter_actions(state) or []
#         else:  # deliverer
#             if not me.has_object():
#                 allowed = self.mlam.pickup_dish_actions(counter_objects) or []
#             elif me.get_object().name == "dish":
#                 allowed = self.mlam.pickup_soup_with_dish_actions(pot_states_dict) or []
#             elif me.get_object().name == "soup":
#                 allowed = self.mlam.deliver_soup_actions() or []
#             else:
#                 allowed = self.mlam.place_obj_on_counter_actions(state) or []

#         # if role-allowed goals are empty, fall back to unfiltered goals
#         if not allowed:
#             return motion_goals if motion_goals else None
#         return allowed
    
#     # """Override action() to use our filtered ml_action, then follow same low-level logic 
#     #     as GreedyHumanModel."""
#         def action(self, state):
#             possible_motion_goals = self.ml_action(state)
        
#         # if no goals available at all, return STAY
#             if not possible_motion_goals:
#                 return STAY_ACTION, {"action_probs": self.a_probs_from_action(STAY_ACTION)}
        
#             start_pos_and_or = state.players[self.agent_index].pos_and_or
#             chosen_goal, chosen_action, action_probs = self.inner.choose_motion_goal(
#                 start_pos_and_or, possible_motion_goals
#             )
#             return chosen_action, {"action_probs": action_probs}
# # inline metrics
# STAY_TUPLE = (0, 0)

# def compute_metrics_inline(traj, sparse_reward, ep_stats):
#     idle_a0 = idle_a1 = collisions = 0

#     for t in range(len(traj) - 1):
#         s_t  = traj[t][0]
#         s_tp = traj[t + 1][0]
#         ja   = traj[t][1]

#         if s_t.players[0].position == s_tp.players[0].position:
#             idle_a0 += 1
#         if s_t.players[1].position == s_tp.players[1].position:
#             idle_a1 += 1

#         p0, p1   = s_t.players[0].position,  s_t.players[1].position
#         p0n, p1n = s_tp.players[0].position, s_tp.players[1].position
#         swapped  = (p0n == p1 and p1n == p0)
#         same     = (p0n == p1n and ja[0] != STAY_TUPLE and ja[1] != STAY_TUPLE)
#         if swapped or same:
#             collisions += 1

#     deliveries    = sparse_reward // 20
#     deliveries_a0 = len(ep_stats["soup_delivery"][0])
#     deliveries_a1 = len(ep_stats["soup_delivery"][1])
#     return deliveries, deliveries_a0, deliveries_a1, idle_a0, idle_a1, collisions


# # main
# def main():
#     os.makedirs("results", exist_ok=True)

#     with open(csv_path, "w", newline="") as f:
#         csv.writer(f).writerow(csv_cols)

#     for layout in layouts:
#         print(f"\n{'='*60}")
#         print(f"Layout: {layout}")
#         print(f"{'='*60}")

#         mdp = OvercookedGridworld.from_layout_name(
#             layout,
#             start_all_orders=[{"ingredients": ["onion", "onion", "onion"]}],
#             start_bonus_orders=[],
#         )
#         print("  Computing MLAM …")
#         mlam = MediumLevelActionManager.from_pickle_or_compute(
#             mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=True,
#         )

#         for seed in seeds:
#             random.seed(seed)
#             np.random.seed(seed)

#             env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

#             agent_0 = RoleBasedAgent("fetcher", mlam)
#             agent_1 = RoleBasedAgent("deliverer", mlam)
#             agent_pair = AgentPair(agent_0, agent_1)

#             print(f"  Seed {seed} … ", end="", flush=True)
#             traj, ep_length, sparse_r, shaped_r = env.run_agents(
#                 agent_pair, display=False
#             )

#             final_info = traj[-1][4]
#             ep_stats   = final_info["episode"]["ep_game_stats"]

#             deliveries, d_a0, d_a1, idle0, idle1, colls = compute_metrics_inline(
#                 traj, sparse_r, ep_stats
#             )

#             row = [
#                 condition, layout, seed, ep_length, sparse_r,
#                 deliveries, d_a0, d_a1, idle0, idle1, colls,
#             ]
#             with open(csv_path, "a", newline="") as f:
#                 csv.writer(f).writerow(row)

#             print(f"reward={sparse_r}  deliveries={deliveries}  "
#                   f"idle=({idle0},{idle1})  collisions={colls}")

#     print(f"\nResults: {csv_path}")


# if __name__ == "__main__":
#     main()
