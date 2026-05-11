from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

mdp = OvercookedGridworld.from_layout_name("cramped_room")

print([m for m in dir(mdp) if not m.startswith("_")])  # list mdp's public attributes
# print([m for m in dir(state) if not m.startswith("_")])  # same for state
# help(OvercookedGridworld.from_layout_name)  # docstrings


# The kitchen blueprint
print("=== terrain_mtx ===")
for row in mdp.terrain_mtx:
    print(row)

# Where are key things?
print("\n=== locations ===")
print("Pots:           ", mdp.get_pot_locations())
print("Onion dispenser:", mdp.get_onion_dispenser_locations())
print("Dish dispenser: ", mdp.get_dish_dispenser_locations())
print("Serving:        ", mdp.get_serving_locations())
print("Player starts:  ", mdp.start_player_positions)

#  The starting state
state = mdp.get_standard_start_state()

print("\n=== state ASCII rendering ===")
print(mdp.state_string(state))

print("\n=== state players ===")
for i, player in enumerate(state.players):
    print(f"Player {i}: pos={player.position}, orient={player.orientation}, holds={player.held_object}")

print("\n=== state objects ===")
print(state.objects)
print("\n=== timestep ===")
print(state.timestep)