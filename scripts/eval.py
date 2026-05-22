"""
Local evaluation gate — run before every submission.

Usage:
    python scripts/eval.py --agent ppo_agent/agent.py --n_games 50
    python scripts/eval.py --agent ppo_agent/agent.py --baselines tactical genius smarter --n_games 50
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = ROOT / "kit"
for d in (str(ROOT), str(KIT_DIR), str(ROOT / "ppo_agent")):
    if d not in sys.path:
        sys.path.insert(0, d)

from engine.game import BomberEnv
from competition.evaluation.runtime_guard import load_agent_instance
import importlib.util as _ilu, importlib as _il

def _load_kit_agents():
    kit_init = KIT_DIR / "agent" / "__init__.py"
    spec = _ilu.spec_from_file_location(
        "_kit_agents_eval", str(kit_init),
        submodule_search_locations=[str(KIT_DIR / "agent")]
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules["_kit_agents_eval"] = mod
    spec.loader.exec_module(mod)
    return mod

_kit = _load_kit_agents()
RandomAgent    = _kit.RandomAgent
SimpleRuleAgent = _kit.SimpleRuleAgent
SmarterRuleAgent = _kit.SmarterRuleAgent
GeniusRuleAgent  = _kit.GeniusRuleAgent
BoxFarmerAgent   = _kit.BoxFarmerAgent
TacticalRuleAgent = _kit.TacticalRuleAgent

BASELINE_MAP = {
    "random":    RandomAgent,
    "simple":    SimpleRuleAgent,
    "smarter":   SmarterRuleAgent,
    "genius":    GeniusRuleAgent,
    "box_farmer": BoxFarmerAgent,
    "tactical":  TacticalRuleAgent,
}


def run_eval(agent_path: str, baseline_name: str, n_games: int, seed: int) -> dict:
    agent_file = Path(agent_path)
    if agent_file.is_dir():
        agent_file = agent_file / "agent.py"

    factory = BASELINE_MAP[baseline_name]
    wins = draws = losses = 0

    for ep in range(n_games):
        our_agent = load_agent_instance(str(agent_file), 0)
        opponents = [factory(i + 1) for i in range(3)]
        env = BomberEnv(max_steps=500)
        obs = env.reset(seed=seed + ep)
        done = False

        while not done:
            try:
                our_action = int(our_agent.act(obs))
            except Exception:
                our_action = 0
            actions = [our_action] + [int(opp.act(obs)) for opp in opponents]
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
        "survive_rate": (wins + draws) / total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=str, required=True, help="Path to agent.py or its folder")
    p.add_argument("--baselines", nargs="+",
                   default=["random", "simple", "smarter", "genius", "tactical"])
    p.add_argument("--n_games", type=int, default=50)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    print(f"Evaluating: {args.agent}")
    print(f"Games per baseline: {args.n_games}\n")

    for baseline in args.baselines:
        stats = run_eval(args.agent, baseline, args.n_games, args.seed)
        wr = stats["win_rate"]
        sr = stats["survive_rate"]
        print(f"  vs {baseline:12s}: win={wr:5.1%}  survive={sr:5.1%}  "
              f"(W={stats['wins']} D={stats['draws']} L={stats['losses']})")


if __name__ == "__main__":
    main()
