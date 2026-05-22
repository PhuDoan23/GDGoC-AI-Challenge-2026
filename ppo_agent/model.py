"""
BomberNet – custom CNN + MLP feature extractor for SB3 PPO.

Also exports BomberNet (stand-alone actor) used by CheckpointAgent and
the submission Agent.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from gymnasium import spaces

KIT_DIR = Path(__file__).resolve().parent.parent / "kit"
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from encode_obs import N_CHANNELS, AUX_DIM

# ── Shared CNN backbone ───────────────────────────────────────────────────────

class _ConvBackbone(nn.Module):
    def __init__(self, in_channels: int = N_CHANNELS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layers(x)


# ── SB3 feature extractor ─────────────────────────────────────────────────────

_GRID = 13  # grid is always 13×13
_CNN_FLAT = 64 * _GRID * _GRID   # 10816 after three 3×3 same-padding convs
_FEATURES_DIM = 256 + 32          # cnn_proj + aux_mlp outputs

if _SB3_AVAILABLE:
    class BomberExtractor(BaseFeaturesExtractor):
        """
        Custom feature extractor for Dict obs space {"map": (11,13,13), "aux": (3,)}.
        Used with SB3's MultiInputPolicy.
        """

        def __init__(self, observation_space: spaces.Dict):
            # super().__init__() must be called before any self.xxx = nn.Module(...)
            super().__init__(observation_space, features_dim=_FEATURES_DIM)

            c       = observation_space["map"].shape[0]   # 11
            aux_dim = observation_space["aux"].shape[0]   # 3

            self.cnn      = _ConvBackbone(c)
            self.cnn_proj = nn.Sequential(nn.Linear(_CNN_FLAT, 256), nn.ReLU())
            self.aux_mlp  = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())

        def forward(self, obs: dict) -> torch.Tensor:
            cnn_feat = self.cnn_proj(self.cnn(obs["map"]))
            aux_feat = self.aux_mlp(obs["aux"])
            return torch.cat([cnn_feat, aux_feat], dim=1)


# ── Standalone actor network (used for checkpointing & inference) ─────────────

class BomberNet(nn.Module):
    """
    Lightweight actor-only network.  Shares the same architecture as the
    PPO policy head so checkpoints stay compatible.

    Saved as {"policy_state_dict": ...} by train.py.
    Loaded by CheckpointAgent and submission Agent.
    """

    N_ACTIONS = 6

    def __init__(self, in_channels: int = N_CHANNELS, aux_dim: int = AUX_DIM):
        super().__init__()
        self.cnn = _ConvBackbone(in_channels)
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, in_channels, 13, 13)).shape[1]

        self.cnn_proj = nn.Sequential(nn.Linear(cnn_out, 256), nn.ReLU())
        self.aux_mlp  = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.actor    = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ReLU(),
            nn.Linear(128, self.N_ACTIONS),
        )

    def actor_forward(self, spatial: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.cnn_proj(self.cnn(spatial)), self.aux_mlp(aux)], dim=1)
        return self.actor(feat)

    def forward(self, spatial: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.actor_forward(spatial, aux)


# ── PPO factory ───────────────────────────────────────────────────────────────

def _best_device() -> str:
    if torch.cuda.is_available():
        # P100 and older cards are sm_60; PyTorch 2.1+ requires sm_70+
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 7:
            return "cuda"
        print(f"[device] GPU sm_{cap[0]}{cap[1]} not supported by this PyTorch build — falling back to CPU")
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_ppo(vec_env, *, device="auto", log_dir="./logs/ppo", **ppo_kwargs):
    """Build a configured SB3 PPO with BomberExtractor."""
    if not _SB3_AVAILABLE:
        raise ImportError("stable_baselines3 is required for training")

    from stable_baselines3 import PPO

    policy_kwargs = {
        "features_extractor_class": BomberExtractor,
        "net_arch": dict(pi=[128], vf=[128]),
    }

    if device == "auto":
        device = _best_device()

    defaults = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=device,
    )
    defaults.update(ppo_kwargs)

    return PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        **defaults,
    )
