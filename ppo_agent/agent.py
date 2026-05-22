"""
Submission-ready Agent class.

File layout for the ZIP:
    agent.py          ← this file
    encode_obs.py     ← copied from ppo_agent/
    model.py          ← copied from ppo_agent/
    model.pth         ← saved with train.py (_save_actor)

The Agent class must match the competition interface exactly:
    class Agent:
        def __init__(self, agent_id: int): ...
        def act(self, obs: dict) -> int: ...
"""

from pathlib import Path
import numpy as np
import torch

# ── local imports (available inside the ZIP) ──────────────────────────────────
from encode_obs import encode_obs
from model import BomberNet


class Agent:
    team_id = "BomberPPO"

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self._net = BomberNet()
        self._net.eval()

        ckpt_path = Path(__file__).parent / "model.pth"
        if ckpt_path.exists():
            data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            self._net.load_state_dict(data["policy_state_dict"])
        else:
            print(f"[Agent] WARNING: model.pth not found at {ckpt_path}; using random weights")

    def act(self, obs: dict) -> int:
        spatial, aux = encode_obs(obs, self.agent_id)
        s_t = torch.from_numpy(spatial).unsqueeze(0)
        a_t = torch.from_numpy(aux).unsqueeze(0)
        with torch.no_grad():
            logits = self._net.actor_forward(s_t, a_t)
            return int(logits.argmax(dim=1).item())
