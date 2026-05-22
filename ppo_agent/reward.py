"""
Per-step reward for the 4-player Bomberland FFA.

Extends the kit's reward.py with:
- Stronger win/death signals
- Kill reward per enemy eliminated
- Explicit 4-player enemy counting (not just 1v1)

compute_reward(prev_obs, curr_obs, agent_id) -> float
"""

import sys
from pathlib import Path
import numpy as np

KIT_DIR = Path(__file__).resolve().parent.parent / "kit"
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from engine.map import Map

BOMB_MAX_TIMER = 7

REWARD = {
    "win":              5.0,   # last player alive
    "kill":             1.5,   # each enemy killed this step
    "survive_step":     0.02,  # alive reward per step
    "item":             0.3,   # collect radius or capacity item
    "box_bomb":         0.1,   # place bomb adjacent to a box
    "die":             -3.0,   # we die
    "in_blast":        -0.5,   # standing in current blast zone
    "predicted_blast": -0.2,   # standing in a bomb's future blast (timer ≤ 4)
    "approach_enemy":   0.02,  # per unit closer to nearest enemy
    "time":            -0.005, # small step penalty
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_bombs(bombs):
    arr = np.asarray(bombs)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float64)


def _bomb_radius(players, owner_id):
    return 1 + int(players[int(owner_id)][4])


def _blast_tiles(grid, bx, by, radius):
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == Map.WALL:
                break
            tiles.add((tx, ty))
            if cell == Map.BOX:
                break
    return tiles


def _in_blast(obs, x, y):
    """True if (x, y) is inside any current bomb's blast zone."""
    bombs = _parse_bombs(obs["bombs"])
    grid = obs["map"]
    players = obs["players"]
    ix, iy = int(x), int(y)
    for b in bombs:
        bx, by = int(b[0]), int(b[1])
        r = _bomb_radius(players, int(b[3]))
        if (ix, iy) in _blast_tiles(grid, bx, by, r):
            return True
    return False


def _in_predicted_blast(obs, x, y, horizon=4):
    """True if (x,y) is in a bomb blast zone AND that bomb has timer ≤ horizon."""
    bombs = _parse_bombs(obs["bombs"])
    grid = obs["map"]
    players = obs["players"]
    ix, iy = int(x), int(y)
    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        if timer > horizon:
            continue
        r = _bomb_radius(players, int(b[3]))
        if (ix, iy) in _blast_tiles(grid, bx, by, r):
            return True
    return False


def _alive_count(players, agent_id):
    return sum(1 for pid in range(len(players)) if pid != agent_id and int(players[pid][2]) == 1)


def _nearest_enemy_dist(players, agent_id, x, y):
    best = None
    for pid in range(len(players)):
        if pid == agent_id or int(players[pid][2]) == 0:
            continue
        d = abs(int(players[pid][0]) - int(x)) + abs(int(players[pid][1]) - int(y))
        best = d if best is None else min(best, d)
    return best


# ── main function ─────────────────────────────────────────────────────────────

def compute_reward(prev_obs, curr_obs, agent_id: int) -> float:
    if prev_obs is None:
        return 0.0

    aid = int(agent_id)
    prev_p = prev_obs["players"]
    curr_p = curr_obs["players"]

    prev_alive = int(prev_p[aid][2])
    curr_alive = int(curr_p[aid][2])

    # ── terminal conditions ──────────────────────────────────────
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARD["die"])

    reward = float(REWARD["survive_step"])
    reward += float(REWARD["time"])

    # ── kill reward ──────────────────────────────────────────────
    prev_enemies = _alive_count(prev_p, aid)
    curr_enemies = _alive_count(curr_p, aid)
    kills = prev_enemies - curr_enemies
    if kills > 0:
        reward += kills * REWARD["kill"]
    if curr_enemies == 0 and prev_enemies > 0:
        reward += REWARD["win"]

    # ── position ─────────────────────────────────────────────────
    cx, cy = int(curr_p[aid][0]), int(curr_p[aid][1])
    px, py = int(prev_p[aid][0]), int(prev_p[aid][1])

    # blast zone penalties
    if _in_blast(curr_obs, cx, cy):
        reward += REWARD["in_blast"]
    elif _in_predicted_blast(curr_obs, cx, cy, horizon=4):
        reward += REWARD["predicted_blast"]

    # ── approach enemy ───────────────────────────────────────────
    if curr_enemies > 0:
        prev_d = _nearest_enemy_dist(prev_p, aid, px, py)
        curr_d = _nearest_enemy_dist(curr_p, aid, cx, cy)
        if prev_d is not None and curr_d is not None:
            reward += REWARD["approach_enemy"] * (prev_d - curr_d)

    # ── item collection ──────────────────────────────────────────
    prev_radius = int(prev_p[aid][4])
    prev_bombs_left = int(prev_p[aid][3])
    curr_radius = int(curr_p[aid][4])
    curr_bombs_left = int(curr_p[aid][3])

    stepped_tile = prev_obs["map"][cx, cy]
    if stepped_tile in (Map.ITEM_RADIUS, Map.ITEM_CAPACITY):
        reward += REWARD["item"]
    elif curr_radius > prev_radius:
        reward += REWARD["item"]

    # ── bomb near box ────────────────────────────────────────────
    if curr_bombs_left < prev_bombs_left:
        adj = [
            prev_obs["map"][max(0, cx - 1), cy],
            prev_obs["map"][min(12, cx + 1), cy],
            prev_obs["map"][cx, max(0, cy - 1)],
            prev_obs["map"][cx, min(12, cy + 1)],
        ]
        if Map.BOX in adj:
            reward += REWARD["box_bomb"]

    return float(reward)
