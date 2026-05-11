from openai.types.realtime import conversation_item_input_audio_transcription_segment
import sys, csv, os, random, json, traceback
from dotenv import load_dotenv
from anthropic import Anthropic
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)


# load .env

load_dotenv()


# experiment parameters 
condition       = "llm_negotiation"
layouts         = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
seeds           = [0, 1, 2, 3, 4]
horizon         = 400
csv_path        = os.path.join("results", "llm_negotiation.csv")
csv_cols        = [
    "condition", "layout", "seed", "ep_length", "sparse_reward",
    "deliveries", "deliveries_a0", "deliveries_a1",
    "idle_a0", "idle_a1", "collisions",
]

# consecutive idle ticks before LLM is invoked
idle_threshold  = 3    
stuck_threshold = 10

# hard cap per episode
max_llm_calls   = 3    

llm_model       = "claude-haiku-4-5-20251001"

stay_action = Action.STAY   # (0, 0)


# Blackboard (same as message-passing) 
class Blackboard:
    def __init__(self):
        self.claims = {}

    def clear(self):
        self.claims.clear()

    def try_claim(self, pos, agent_index):
        if pos in self.claims:
            return self.claims[pos] == agent_index
        self.claims[pos] = agent_index
        return True


# Role-based wrapper (reused for role enforcement after LLM) 
class RoleWrapper:

    def __init__(self, inner: GreedyHumanModel, role=None):
        self.inner = inner
        self.role = role

    @property
    def agent_index(self):
        return self.inner.agent_index

    def set_agent_index(self, idx):
        self.inner.set_agent_index(idx)

    def set_mdp(self, mdp):
        self.inner.set_mdp(mdp)

    def reset(self):
        self.inner.reset()

    # role is logged but not enforced — LLM negotiation is purely advisory
    def action(self, state):
        # LLM negotiation is observational — roles logged but not enforced
        return self.inner.action(state)

# helpers 
def target_pos(player, action):
    if action == "interact":
        return player.position
    dx, dy = action
    px, py = player.position
    return (px + dx, py + dy)


def build_llm_prompt(state, agents, mdp, total_sparse, llm_call_number, ticks_remaining):
    """
    Build a negotiation-style prompt with rich state context.
    The LLM is framed as a coordinator deciding the best role assignment 
    given the current kitchen state, not just a deadlock-breaker.
    """
    pot_states = mdp.get_pot_states(state)
    deliveries_so_far = total_sparse // 20

    # describe each agent in detail
    agent_descriptions = []
    for i, player in enumerate(state.players):
        held = player.get_object().name if player.has_object() else "nothing"
        role = agents[i].role if agents[i].role else "unassigned"
        orient = player.orientation
        agent_descriptions.append(
            f"  Agent {i}:\n"
            f"    position = {player.position}\n"
            f"    facing = {orient}\n"
            f"    holding = {held}\n"
            f"    current role = {role}"
        )

    # describe kitchen state
    pots_ready    = pot_states.get('ready', [])
    pots_cooking  = pot_states.get('cooking', [])
    pots_empty    = pot_states.get('empty', [])
    pots_partial  = pot_states.get('partially_full', [])

    # key locations from the layout
    pot_locs    = mdp.get_pot_locations()
    onion_locs  = mdp.get_onion_dispenser_locations()
    dish_locs   = mdp.get_dish_dispenser_locations()
    serve_locs  = mdp.get_serving_locations()

    lines = [
        "You are a coordinator negotiating role assignments between two agents in an Overcooked kitchen.",
        "Your goal: maximise the number of soups delivered in the remaining time.",
        "",
        "GAME RULES:",
        "  - A soup requires 3 onions placed into a pot, which then cooks automatically.",
        "  - Once cooked, a dish must be used to collect the soup from the pot.",
        "  - The soup must then be carried to a serving location to count as a delivery.",
        "  - Each delivery scores 20 reward.",
        "",
        "ROLE DEFINITIONS:",
        "  - 'fetcher': picks up onions from dispensers and places them in pots.",
        "  - 'deliverer': picks up dishes, collects cooked soup from pots, delivers to serving locations.",
        "  - Roles are not strict — agents will still navigate freely, but role enforcement prevents them from picking up the wrong object type.",
        "",
        "CURRENT KITCHEN STATE:",
        f"  Onion dispensers at: {onion_locs}",
        f"  Dish dispensers at:  {dish_locs}",
        f"  Pots at:             {pot_locs}",
        f"  Serving locations:   {serve_locs}",
        "",
        f"  Pots ready to serve:   {pots_ready}",
        f"  Pots cooking:          {pots_cooking}",
        f"  Pots partially full:   {pots_partial}",
        f"  Pots empty:            {pots_empty}",
        "",
        "AGENTS:",
        *agent_descriptions,
        "",
        f"PROGRESS: {deliveries_so_far} soups delivered so far. {ticks_remaining} ticks remaining.",
        f"This is negotiation call #{llm_call_number} for this episode.",
        "",
        "DECISION:",
        "Decide the best role assignment given the current state.",
        "Consider: which agent is closer to the most useful next task? Is a pot ready to be served (favouring a deliverer)? Are pots empty and needing onions (favouring a fetcher)? Are the current roles already optimal?",
        "",
        "If the current role assignment is already optimal for the state, KEEP it the same.",
        "Only change roles if the current assignment is clearly mismatched with what the kitchen needs.",
        "",
        'Respond with ONLY a JSON object, no markdown, no explanation:',
        '{"agent_0": "fetcher" or "deliverer", "agent_1": "fetcher" or "deliverer", "reasoning": "one short sentence"}',
    ]

    return "\n".join(lines)

# Call Claude
def call_llm(prompt, client):
    response = client.messages.create(
        model=llm_model,
        max_tokens=200,  # increased from 100 to allow reasoning
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


# inline metrics 
STAY_TUPLE = (0, 0)

def compute_metrics_inline(states, actions, sparse_reward, ep_stats):
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


# run one episode 
def run_episode_llm(env, agents, mlam, mdp, client):

    bb = Blackboard()

    for idx, ag in enumerate(agents):
        ag.set_agent_index(idx)
        ag.set_mdp(mdp)

    states       = []
    actions_list = []
    total_sparse = 0


    # llm reward trigger tracker 

    last_reward_tick = 0
    reward_timeout = 40
    llm_cooldown = 30
    last_llm_tick = -llm_cooldown



    # idle tracking 
    idle_streak  = [0, 0]
    prev_pos     = [None, None]
    llm_calls    = 0

    while not env.is_done():
        state = env.state
        states.append(state)

        # update idle streaks 
        for i in range(2):
            cur_pos = state.players[i].position
            if prev_pos[i] is not None and cur_pos == prev_pos[i]:
                idle_streak[i] += 1
            else:
                idle_streak[i] = 0
            prev_pos[i] = cur_pos

        # LLM deadlock detection 
# Logic for triggering the LLM
        curr_tick = len(states)
        
        # Condition A: Physical Deadlock (someone isn't moving)
        physically_stuck = (idle_streak[0] >= 10 or idle_streak[1] >= 10) or \
                           (idle_streak[0] >= 3 and idle_streak[1] >= 3)
        
        # Condition B: Dynamic Deadlock / Livelock (moving but no rewards)
        no_progress = (curr_tick - last_reward_tick >= reward_timeout)
        
        # Condition C: Guardrails (max calls and cooldown)
        off_cooldown = (curr_tick - last_llm_tick >= llm_cooldown)

        if (physically_stuck or no_progress) and llm_calls < max_llm_calls and off_cooldown:
            try:
                ticks_remaining = horizon - curr_tick
                prompt = build_llm_prompt(
                    state, agents, mdp,
                    total_sparse=total_sparse,
                    llm_call_number=llm_calls + 1,
                    ticks_remaining=ticks_remaining,
                )
                
                result = call_llm(prompt, client)
                agents[0].role = result.get("agent_0", agents[0].role)
                agents[1].role = result.get("agent_1", agents[1].role)
                reasoning = result.get("reasoning", "")
                
                llm_calls += 1
                last_llm_tick = curr_tick
                idle_streak = [0, 0] # Reset streaks after intervention
                
                trigger_type = "PHYSICAL" if physically_stuck else "THROUGHPUT"
                print(f"    [LLM #{llm_calls} @ {curr_tick}t] Type: {trigger_type}")
                print(f"    Roles: a0={agents[0].role}, a1={agents[1].role} | {reasoning}")
                
            except Exception as e:
                print(f"    [LLM call failed: {e}]")

        # agent actions 
        intended = []
        for ag in agents:
            a, info = ag.action(state)
            intended.append(a)

        # blackboard conflict resolution 
        # --- METHODOLOGY: PROACTIVE EVASION ---
        bb.clear()
        final_actions = list(intended)
        
        # 1. Agent 0 (Leader) claims its target first
        tgt0 = target_pos(state.players[0], intended[0])
        bb.try_claim(tgt0, 0)

        # 2. Agent 1 (Follower) checks if it is physically blocking the Leader's path
        # If Agent 0 wants to move into the tile where Agent 1 is currently standing:
        if tgt0 == state.players[1].position:
            # Agent 1 MUST move to a random empty adjacent tile to let Agent 0 pass
            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(possible_moves)
            evaded = False
            for move in possible_moves:
                # Check if this escape tile is free
                if bb.try_claim(target_pos(state.players[1], move), 1):
                    final_actions[1] = move
                    evaded = True
                    break
            if not evaded: 
                final_actions[1] = Action.STAY
        else:
            # Normal conflict resolution if not directly blocking
            tgt1 = target_pos(state.players[1], intended[1])
            if not bb.try_claim(tgt1, 1):
                final_actions[1] = Action.STAY

        joint_action = tuple(final_actions)
        actions_list.append(joint_action)

        next_state, r_t, done, info = env.step(joint_action)
        total_sparse += r_t

        next_state, r_t, done, info = env.step(joint_action)
        
        # Methodology: Reset the 'no_progress' clock when a delivery is made
        if r_t > 0:
            total_sparse += r_t
            last_reward_tick = len(states)

    states.append(env.state)
    ep_stats  = info["episode"]["ep_game_stats"]
    ep_length = env.state.timestep

    return states, actions_list, ep_length, total_sparse, ep_stats, llm_calls


# main 
def main():
    os.makedirs("results", exist_ok=True)

    # set up anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_cols)

    for layout in layouts:
        print(f"\n{'='*60}")
        print(f"Layout: {layout}")
        print(f"{'='*60}")

        mdp = OvercookedGridworld.from_layout_name(layout)
        print("  Computing MLAM …")
        mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, mlam_params=NO_COUNTERS_PARAMS, force_compute=True,
        )

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)

            env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

            agents = [
                RoleWrapper(GreedyHumanModel(mlam), role=None),
                RoleWrapper(GreedyHumanModel(mlam), role=None),
            ]

            print(f"  Seed {seed} … ", end="", flush=True)
            states, acts, ep_length, sparse_r, ep_stats, n_llm = run_episode_llm(
                env, agents, mlam, mdp, client
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
                  f"idle=({idle0},{idle1})  collisions={colls}  "
                  f"llm_calls={n_llm}")

    print(f"\n Results written to {csv_path}")


if __name__ == "__main__":
    main() 