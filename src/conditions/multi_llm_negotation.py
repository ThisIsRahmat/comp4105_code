import sys, csv, os, random, json
import numpy as np
from anthropic import Anthropic
import openai
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)

# --- LLM client imports (only loaded if needed) ---

# from openai import OpenAI  # uncomment when adding GPT
# import google.generativeai as genai  # uncomment when adding Gemini


# === experiment parameters ===
condition       = "llm_negotiation"
layouts         = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
seeds           = [0, 1, 2, 3, 4]
horizon         = 400
idle_threshold  = 8
max_llm_calls   = 3

# multi-LLM config: set to one model now, expand later
MODELS = [
    "claude-haiku-4-5-20251001",
    # "gpt-4o-mini",       # uncomment when ready
    # "gemini-1.5-flash",  # uncomment when ready
]

stay_action = Action.STAY


# === Blackboard (per-tick conflict resolution) ===
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


# === Role-based wrapper (enforces LLM-assigned roles) ===
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

    def action(self, state):
        if self.role is not None:
            player = state.players[self.agent_index]
            if player.has_object():
                obj_name = player.get_object().name
                if self.role == "fetcher" and obj_name != "onion":
                    return stay_action, {"action_probs": self.inner.a_probs_from_action(stay_action)}
                if self.role == "deliverer" and obj_name == "onion":
                    return stay_action, {"action_probs": self.inner.a_probs_from_action(stay_action)}
        return self.inner.action(state)


# === LLM client setup and dispatch ===
def make_clients():
    """Initialise all LLM clients; only sets up the ones we have keys for."""
    clients = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        clients["anthropic"] = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # if os.environ.get("OPENAI_API_KEY"):
    #     clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # if os.environ.get("GEMINI_API_KEY"):
    #     genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    #     clients["gemini"] = genai
    return clients


# Map model name to provider family.
def model_family(model_name):
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gpt"):
        return "openai"
    if model_name.startswith("gemini"):
        return "gemini"
    raise ValueError(f"Unknown model: {model_name}")


def call_llm(prompt, model_name, clients):
    """Send prompt to the appropriate LLM and parse JSON role assignment."""
    family = model_family(model_name)

    if family == "anthropic":
        client = clients["anthropic"]
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

    elif family == "openai":
        client = clients["openai"]
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()

    elif family == "gemini":
        genai = clients["gemini"]
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text.strip()

    # strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


# === prompt builder ===
def build_llm_prompt(state, agents, mdp):
    pot_states = mdp.get_pot_states(state)
    lines = [
        "You are coordinating two agents in an Overcooked kitchen.",
        "Both agents have been stuck (idle) for at least 8 ticks.",
        "Reassign their roles to break the deadlock.",
        "",
        "Current state:",
    ]
    for i, player in enumerate(state.players):
        held = player.get_object().name if player.has_object() else "nothing"
        role = agents[i].role if agents[i].role else "none"
        lines.append(f"  Agent {i}: pos={player.position}, holding={held}, current_role={role}")
    lines.append(f"  Pots ready: {pot_states.get('ready', [])}")
    lines.append(f"  Pots cooking: {pot_states.get('cooking', [])}")
    lines.append(f"  Pots empty: {pot_states.get('empty', [])}")
    lines.append("")
    lines.append("Respond with ONLY a JSON object (no markdown, no explanation):")
    lines.append('{"agent_0": "fetcher" or "deliverer", "agent_1": "fetcher" or "deliverer"}')
    lines.append("")
    lines.append("Rules:")
    lines.append('  - "fetcher" handles onions only')
    lines.append('  - "deliverer" handles dishes and soup only')
    lines.append("  - Assign different roles to each agent")
    return "\n".join(lines)


# === metrics ===
STAY_TUPLE = (0, 0)

def compute_metrics_inline(states, actions, sparse_reward, ep_stats):
    idle_a0 = idle_a1 = collisions = 0
    for t in range(len(states) - 1):
        s_t, s_tp = states[t], states[t + 1]
        ja = actions[t]
        if s_t.players[0].position == s_tp.players[0].position:
            idle_a0 += 1
        if s_t.players[1].position == s_tp.players[1].position:
            idle_a1 += 1
        p0, p1   = s_t.players[0].position,  s_t.players[1].position
        p0n, p1n = s_tp.players[0].position, s_tp.players[1].position
        if (p0n == p1 and p1n == p0) or (p0n == p1n and ja[0] != STAY_TUPLE and ja[1] != STAY_TUPLE):
            collisions += 1
    deliveries    = sparse_reward // 20
    deliveries_a0 = len(ep_stats["soup_delivery"][0])
    deliveries_a1 = len(ep_stats["soup_delivery"][1])
    return deliveries, deliveries_a0, deliveries_a1, idle_a0, idle_a1, collisions


# === run one episode ===
def target_pos(player, action):
    if action == "interact":
        return player.position
    dx, dy = action
    px, py = player.position
    return (px + dx, py + dy)


def run_episode_llm(env, agents, mlam, mdp, clients, model_name):
    bb = Blackboard()
    for idx, ag in enumerate(agents):
        ag.set_agent_index(idx)
        ag.set_mdp(mdp)

    states, actions_list = [], []
    total_sparse = 0
    idle_streak = [0, 0]
    prev_pos    = [None, None]
    llm_calls   = 0

    while not env.is_done():
        state = env.state
        states.append(state)

        # update idle streaks
        for i in range(2):
            cur = state.players[i].position
            if prev_pos[i] is not None and cur == prev_pos[i]:
                idle_streak[i] += 1
            else:
                idle_streak[i] = 0
            prev_pos[i] = cur

        # LLM trigger
        if (idle_streak[0] >= idle_threshold
                and idle_streak[1] >= idle_threshold
                and llm_calls < max_llm_calls):
            try:
                prompt = build_llm_prompt(state, agents, mdp)
                result = call_llm(prompt, model_name, clients)
                agents[0].role = result.get("agent_0", agents[0].role)
                agents[1].role = result.get("agent_1", agents[1].role)
                llm_calls += 1
                idle_streak = [0, 0]
                print(f"    [LLM call #{llm_calls}] → a0={agents[0].role}, a1={agents[1].role}")
            except Exception as e:
                print(f"    [LLM call failed: {e}]")

        # agent actions + blackboard conflict resolution
        intended = [ag.action(state)[0] for ag in agents]
        bb.clear()
        final_actions = list(intended)
        bb.try_claim(target_pos(state.players[0], intended[0]), 0)
        if not bb.try_claim(target_pos(state.players[1], intended[1]), 1):
            final_actions[1] = stay_action

        joint_action = tuple(final_actions)
        actions_list.append(joint_action)
        next_state, r_t, done, info = env.step(joint_action)
        total_sparse += r_t

    states.append(env.state)
    ep_stats  = info["episode"]["ep_game_stats"]
    ep_length = env.state.timestep
    return states, actions_list, ep_length, total_sparse, ep_stats, llm_calls



def main():
    os.makedirs("results", exist_ok=True)
    clients = make_clients()

    csv_cols = [
        "condition", "model", "layout", "seed", "ep_length", "sparse_reward",
        "deliveries", "deliveries_a0", "deliveries_a1",
        "idle_a0", "idle_a1", "collisions", "llm_calls",
    ]

    for model_name in MODELS:
        # safety check: skip if no client for this provider
        try:
            family = model_family(model_name)
            if family not in clients:
                print(f"Skipping {model_name}: no API key for {family}")
                continue
        except ValueError as e:
            print(e); continue

        # one CSV per model — easier to compare later
        csv_path = os.path.join("results", f"llm_negotiation_{family}.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_cols)

        print(f"\n{'#'*60}")
        print(f"# Model: {model_name}")
        print(f"{'#'*60}")

        for layout in layouts:
            print(f"\nLayout: {layout}")
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
                agents = [
                    RoleWrapper(GreedyHumanModel(mlam), role=None),
                    RoleWrapper(GreedyHumanModel(mlam), role=None),
                ]
                print(f"  Seed {seed} … ", end="", flush=True)
                states, acts, ep_length, sparse_r, ep_stats, n_llm = run_episode_llm(
                    env, agents, mlam, mdp, clients, model_name
                )
                d, d0, d1, idle0, idle1, colls = compute_metrics_inline(
                    states, acts, sparse_r, ep_stats
                )
                row = [
                    condition, model_name, layout, seed, ep_length, sparse_r,
                    d, d0, d1, idle0, idle1, colls, n_llm,
                ]
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(f"reward={sparse_r}  deliveries={d}  idle=({idle0},{idle1})  "
                      f"collisions={colls}  llm_calls={n_llm}")

        print(f"\n  Results written to {csv_path}")


if __name__ == "__main__":
    main()