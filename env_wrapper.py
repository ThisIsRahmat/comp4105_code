import gymnasium as gym
from gymnasium import spaces
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action


class OvercookedGymEnv(gym.Env):
    """Treats both Overcooked agents as a single joint policy.
    Joint action: 6 × 6 = 36 discrete actions."""

    def __init__(self, layout_name="cramped_room", horizon=400):
        super().__init__()
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.horizon = horizon
        self.action_space = spaces.Discrete(36)

        sample = self._featurize(self.base_env.state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample.shape, dtype=np.float32
        )
        self._t = 0

    def _featurize(self, state):
        feats = self.mdp.lossless_state_encoding(state)
        return np.concatenate([f.flatten() for f in feats]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset()
        self._t = 0 
        return self._featurize(self.base_env.state), {}

    def step(self, action):
        a0 = Action.INDEX_TO_ACTION[action // 6]
        a1 = Action.INDEX_TO_ACTION[action % 6]
        next_state, sparse_reward, done, info = self.base_env.step((a0, a1))
        self._t += 1
        terminated = bool(done)
        truncated = self._t >= self.horizon
        obs = self._featurize(next_state)

        # Shaped reward: dense per-step rewards Overcooked computes for sub-goals
        # (picking up onion, putting in pot, etc.). We sum across both agents.
        shaped_reward = sum(info.get("shaped_r_by_agent", [0.0, 0.0]))

        # During training we use shaped + sparse to give a learning signal
        # at every step. We log them separately in info for evaluation.
        total_reward = sparse_reward + shaped_reward

        return obs, float(total_reward), terminated, truncated, {
            "sparse_r": sparse_reward,
            "shaped_r": shaped_reward,
        }