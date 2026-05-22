"""
PPO training script for Bomberland.

Phase 1 (warm-start): train against weak baselines.
Phase 2 (league):     mix baselines + frozen self snapshots.

Usage:
    # Phase 1 – 5M steps vs simple baselines
    python train.py --phase warmup --total_steps 5_000_000 --n_envs 8

    # Phase 2 – continue with league
    python train.py --phase league --total_steps 20_000_000 --n_envs 8 \
                    --load_checkpoint ckpts/warmup_final.pth

    # Quick smoke test
    python train.py --phase warmup --total_steps 50000 --n_envs 2
"""

import sys
import os
import argparse
import random
import time
from pathlib import Path

import torch
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = ROOT / "kit"
PPO_DIR = ROOT / "ppo_agent"
for d in (str(ROOT), str(KIT_DIR), str(PPO_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from gym_wrapper import BomberGymEnv, LeaguePool
from model import make_ppo, BomberNet
from encode_obs import encode_obs


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _save_actor(ppo_model, path: str):
    """Extract and save the actor weights from the PPO model."""
    net = BomberNet()
    sb3_extractor = ppo_model.policy.features_extractor
    sb3_actor_layers = ppo_model.policy.mlp_extractor.policy_net
    sb3_action_net   = ppo_model.policy.action_net

    # Copy CNN + aux weights from the shared extractor
    net.cnn.layers.load_state_dict(sb3_extractor.cnn.layers.state_dict())
    net.cnn_proj.load_state_dict(sb3_extractor.cnn_proj.state_dict())
    net.aux_mlp.load_state_dict(sb3_extractor.aux_mlp.state_dict())

    # Copy actor head (pi layers + action_net)
    pi_weights = list(sb3_actor_layers.children())
    actor_head_weights = list(net.actor.children())
    # [Linear(288→128), ReLU, Linear(128→6)]
    actor_head_weights[0].weight.data.copy_(pi_weights[0].weight.data)
    actor_head_weights[0].bias.data.copy_(pi_weights[0].bias.data)
    actor_head_weights[2].weight.data.copy_(sb3_action_net.weight.data)
    actor_head_weights[2].bias.data.copy_(sb3_action_net.bias.data)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state_dict": net.state_dict()}, path)
    print(f"[Checkpoint] Actor saved → {path}")


def _save_sb3(ppo_model, path: str):
    """Save the full SB3 PPO model (for resuming training)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ppo_model.save(path)
    print(f"[Checkpoint] SB3 model saved → {path}")


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_vs_baseline(ppo_model, baseline_name: str, n_games: int = 50, seed: int = 0) -> dict:
    """Run n_games against all three slots filled with baseline_name.  Returns stats."""
    from gym_wrapper import _BASELINE_FACTORIES

    factory = _BASELINE_FACTORIES[baseline_name]
    wins = draws = losses = 0

    for ep in range(n_games):
        from engine.game import BomberEnv
        env = BomberEnv(max_steps=500)
        obs = env.reset(seed=seed + ep)
        opponents = [factory(i + 1) for i in range(3)]
        done = False

        while not done:
            spatial, aux = encode_obs(obs, agent_id=0)
            obs_dict = {
                "map": spatial[None],
                "aux": aux[None],
            }
            action, _ = ppo_model.predict(obs_dict, deterministic=True)
            action = int(action[0])
            actions = [action] + [int(opp.act(obs)) for opp in opponents]
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

        final_players = obs["players"]
        our_alive = int(final_players[0][2])
        enemies_alive = sum(int(final_players[pid][2]) for pid in range(1, 4))

        if our_alive == 1 and enemies_alive == 0:
            wins += 1
        elif our_alive == 1:
            draws += 1
        else:
            losses += 1

    total = wins + draws + losses
    return {
        "baseline": baseline_name,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
    }


# ── callbacks ────────────────────────────────────────────────────────────────

class LeagueCallback(BaseCallback):
    """
    Every `checkpoint_freq` steps:
      1. Save an actor checkpoint.
      2. Add it to the league pool (so envs start facing old selves).
      3. Log win-rate vs tactical_rule_agent.
    """

    def __init__(
        self,
        league_pool: LeaguePool,
        checkpoint_dir: str,
        checkpoint_freq: int = 500_000,
        eval_freq: int = 500_000,
        n_eval_games: int = 30,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.pool = league_pool
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.n_eval = n_eval_games
        self._next_ckpt = checkpoint_freq
        self._next_eval = eval_freq

    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step >= self._next_ckpt:
            path = str(self.ckpt_dir / f"actor_{step}.pth")
            _save_actor(self.model, path)
            self.pool.add_checkpoint(path)
            self._next_ckpt += self.ckpt_freq

        if step >= self._next_eval:
            stats = eval_vs_baseline(self.model, "tactical", n_games=self.n_eval)
            wr = stats["win_rate"]
            print(f"\n[Eval @ {step:,}] vs tactical: {wr:.1%}  "
                  f"(W={stats['wins']} D={stats['draws']} L={stats['losses']})")
            if self.logger:
                self.logger.record("eval/win_rate_vs_tactical", wr)
                self.logger.dump(step)
            self._next_eval += self.eval_freq

        return True


class WarmupCallback(BaseCallback):
    """Logs win-rate vs simple/smarter baselines during warm-up."""

    def __init__(self, eval_freq: int = 200_000, n_eval_games: int = 20):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval = n_eval_games
        self._next = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            for baseline in ("simple", "smarter"):
                stats = eval_vs_baseline(self.model, baseline, n_games=self.n_eval)
                wr = stats["win_rate"]
                print(f"  [Eval @ {self.num_timesteps:,}] vs {baseline}: {wr:.1%}")
            self._next += self.eval_freq
        return True


# ── env factory ───────────────────────────────────────────────────────────────

def make_env_factory(league_pool: LeaguePool, max_steps: int = 500):
    def _factory():
        return BomberGymEnv(league_pool, max_steps=max_steps)
    return _factory


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["warmup", "league"], default="warmup")
    p.add_argument("--total_steps", type=int, default=5_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="Path to SB3 .zip checkpoint to resume from")
    p.add_argument("--ckpt_dir", type=str, default="ckpts")
    p.add_argument("--log_dir", type=str, default="logs/ppo")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_epochs", type=int, default=10,
                   help="PPO epochs per rollout (reduce to 4 for faster training)")
    p.add_argument("--n_steps", type=int, default=2048,
                   help="PPO rollout steps per env per update")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = Path(ROOT) / args.ckpt_dir
    log_dir  = Path(ROOT) / args.log_dir

    # ── Phase-specific league setup ──────────────────────────────────────────
    if args.phase == "warmup":
        initial_baselines = ["random", "simple", "smarter"]
    else:
        initial_baselines = ["random", "simple", "smarter", "genius", "box_farmer", "tactical"]

    pool = LeaguePool(initial_baselines)

    # ── Build vectorised environments ────────────────────────────────────────
    env_factory = make_env_factory(pool)
    vec_env = DummyVecEnv([env_factory for _ in range(args.n_envs)])

    # ── Load or create PPO model ─────────────────────────────────────────────
    if args.load_checkpoint:
        print(f"Resuming from {args.load_checkpoint}")
        model = PPO.load(args.load_checkpoint, env=vec_env, device=args.device)
    else:
        model = make_ppo(
            vec_env,
            device=args.device,
            log_dir=str(log_dir),
            n_epochs=args.n_epochs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {n_params:,}")

    # ── Callbacks ────────────────────────────────────────────────────────────
    if args.phase == "warmup":
        callbacks = [WarmupCallback(eval_freq=200_000, n_eval_games=20)]
    else:
        callbacks = [
            LeagueCallback(
                league_pool=pool,
                checkpoint_dir=str(ckpt_dir / "league"),
                checkpoint_freq=500_000,
                eval_freq=500_000,
                n_eval_games=30,
            )
        ]

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting {args.phase} training for {args.total_steps:,} steps "
          f"with {args.n_envs} envs…")
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=callbacks, reset_num_timesteps=False)
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed/60:.1f} min")

    # ── Save final checkpoints ───────────────────────────────────────────────
    phase_tag = args.phase
    _save_sb3(model, str(ckpt_dir / f"{phase_tag}_final"))
    _save_actor(model, str(ckpt_dir / f"{phase_tag}_actor_final.pth"))

    # ── Final eval ───────────────────────────────────────────────────────────
    print("\n=== Final evaluation ===")
    for baseline in ("random", "simple", "smarter", "genius", "tactical"):
        stats = eval_vs_baseline(model, baseline, n_games=50)
        wr = stats["win_rate"]
        print(f"  vs {baseline:12s}: {wr:5.1%}  "
              f"(W={stats['wins']} D={stats['draws']} L={stats['losses']})")


if __name__ == "__main__":
    main()
