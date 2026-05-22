"""
Single-agent gymnasium.Env wrapper around BomberEnv.

We always control agent_id=0.  The other three slots are filled
by opponent agents sampled from a LeaguePool on each reset().

Usage:
    pool = LeaguePool(kit_dir)
    env  = BomberGymEnv(pool)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import sys
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import importlib.util as _ilu

KIT_DIR = Path(__file__).resolve().parent.parent / "kit"
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from engine.game import BomberEnv

# Load kit agent classes under a distinct module name so ppo_agent/agent.py
# is never shadowed in sys.modules["agent"].
def _load_kit_agent_pkg():
    kit_agent_init = KIT_DIR / "agent" / "__init__.py"
    spec = _ilu.spec_from_file_location(
        "_kit_agents",
        str(kit_agent_init),
        submodule_search_locations=[str(KIT_DIR / "agent")],
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules["_kit_agents"] = mod
    spec.loader.exec_module(mod)
    return mod

_kit_agents = _load_kit_agent_pkg()
RandomAgent      = _kit_agents.RandomAgent
SimpleRuleAgent  = _kit_agents.SimpleRuleAgent
SmarterRuleAgent = _kit_agents.SmarterRuleAgent
GeniusRuleAgent  = _kit_agents.GeniusRuleAgent
BoxFarmerAgent   = _kit_agents.BoxFarmerAgent
TacticalRuleAgent = _kit_agents.TacticalRuleAgent

from encode_obs import encode_obs, N_CHANNELS, AUX_DIM, GRID_H, GRID_W
from reward import compute_reward


# ── Checkpoint agent wrapper ──────────────────────────────────────────────────

class CheckpointAgent:
    """Loads a PPO checkpoint and wraps it as an act(obs) agent."""

    def __init__(self, checkpoint_path: str, agent_id: int):
        import torch
        from model import BomberExtractor  # local import avoids circular deps

        self.agent_id = agent_id
        checkpoint_path = str(checkpoint_path)

        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        from model import BomberNet
        self.net = BomberNet()
        self.net.load_state_dict(data["policy_state_dict"])
        self.net.eval()

    def act(self, obs) -> int:
        import torch
        spatial, aux = encode_obs(obs, self.agent_id)
        with torch.no_grad():
            s_t = torch.from_numpy(spatial).unsqueeze(0)
            a_t = torch.from_numpy(aux).unsqueeze(0)
            logits = self.net.actor_forward(s_t, a_t)
            return int(logits.argmax(dim=1).item())


# ── League pool ───────────────────────────────────────────────────────────────

_BASELINE_FACTORIES = {
    "random":    RandomAgent,
    "simple":    SimpleRuleAgent,
    "smarter":   SmarterRuleAgent,
    "genius":    GeniusRuleAgent,
    "box_farmer": BoxFarmerAgent,
    "tactical":  TacticalRuleAgent,
}


class LeaguePool:
    """
    Pool of opponent factories.  On each sample(), returns 3 agent instances.

    Baselines are always present.  Checkpoint paths can be added with
    add_checkpoint(); the pool will load them on the next sample().
    """

    def __init__(self, initial_baselines: Optional[List[str]] = None):
        if initial_baselines is None:
            initial_baselines = list(_BASELINE_FACTORIES.keys())
        self._baselines = [b for b in initial_baselines if b in _BASELINE_FACTORIES]
        self._checkpoints: List[str] = []  # mutable — updated during training

    def add_checkpoint(self, path: str):
        self._checkpoints.append(str(path))

    def _make_baseline(self, name: str, agent_id: int):
        return _BASELINE_FACTORIES[name](agent_id)

    def _make_checkpoint(self, path: str, agent_id: int):
        try:
            return CheckpointAgent(path, agent_id)
        except Exception as e:
            # Fall back to random if checkpoint fails to load
            print(f"[LeaguePool] Failed to load checkpoint {path}: {e}; using RandomAgent")
            return RandomAgent(agent_id)

    def sample(self, n: int = 3) -> list:
        """Return n fresh agent instances with agent_ids 1..n."""
        all_options = list(self._baselines)
        if self._checkpoints:
            all_options += [("ckpt", p) for p in self._checkpoints]

        agents = []
        for slot in range(n):
            agent_id = slot + 1  # we occupy slot 0
            choice = random.choice(all_options)
            if isinstance(choice, tuple):  # checkpoint
                agents.append(self._make_checkpoint(choice[1], agent_id))
            else:
                agents.append(self._make_baseline(choice, agent_id))
        return agents


# ── Gym environment ───────────────────────────────────────────────────────────

class BomberGymEnv(gym.Env):
    """
    Single-agent wrapper.  Observation space is Dict:
        "map": Box(float32, shape=(11, 13, 13))
        "aux": Box(float32, shape=(3,))
    Action space: Discrete(6)
    """

    metadata = {"render_modes": []}

    def __init__(self, league_pool: LeaguePool, max_steps: int = 500):
        super().__init__()
        self.league_pool = league_pool
        self.max_steps = max_steps
        self._env = BomberEnv(max_steps=max_steps)
        self._opponents: list = []
        self._prev_obs = None

        self.observation_space = spaces.Dict({
            "map": spaces.Box(0.0, 1.0, shape=(N_CHANNELS, GRID_H, GRID_W), dtype=np.float32),
            "aux": spaces.Box(0.0, 2.0, shape=(AUX_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(6)

    # Expose opponents so train.py can inspect which baselines were used
    @property
    def current_opponents(self):
        return self._opponents

    def reset(self, seed=None, options=None):
        self._opponents = self.league_pool.sample(3)
        raw = self._env.reset(seed=seed)
        self._prev_obs = raw
        spatial, aux = encode_obs(raw, agent_id=0)
        return {"map": spatial, "aux": aux}, {}

    def step(self, action: int):
        raw = self._prev_obs
        actions = [int(action), 0, 0, 0]
        for i, opp in enumerate(self._opponents):
            try:
                actions[i + 1] = int(opp.act(raw))
            except Exception:
                actions[i + 1] = 0

        next_raw, game_terminated, game_truncated = self._env.step(actions)
        reward = compute_reward(raw, next_raw, agent_id=0)

        # Episode ends when WE die OR the game naturally ends
        our_alive = int(next_raw["players"][0][2])
        terminated = game_terminated or (our_alive == 0)
        truncated = game_truncated and not terminated

        self._prev_obs = next_raw
        spatial, aux = encode_obs(next_raw, agent_id=0)
        obs_out = {"map": spatial, "aux": aux}

        info = {}
        if terminated or truncated:
            info["win"] = int(our_alive == 1 and int(sum(p[2] for p in next_raw["players"])) <= 1)
            info["alive"] = our_alive

        return obs_out, reward, terminated, truncated, info
