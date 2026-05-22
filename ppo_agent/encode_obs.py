"""
11-channel spatial + 3-scalar auxiliary observation encoder.

Channel layout (axis 0 of the (11, 13, 13) tensor):
  0  grass
  1  wall
  2  box
  3  item_radius
  4  item_capacity
  5  our position
  6  all live enemy positions (combined)
  7  bomb timer map  (high = bomb about to explode)
  8  our own bomb markers
  9  danger map      (tiles in any bomb's blast radius, high = sooner explosion)
  10 BFS reachability from our position (normalized, 0 = unreachable)

Auxiliary vector (3,):
  0  bombs_left / 5
  1  bomb_radius_bonus / 4   (max bonus = 4 → radius 5)
  2  alive_enemies / 3
"""

import numpy as np
from collections import deque

GRID_H = 13
GRID_W = 13
N_CHANNELS = 11
AUX_DIM = 3

BOMB_MAX_TIMER = 7

# Map cell constants (match engine/map.py)
GRASS = 0
WALL = 1
BOX = 2
ITEM_RADIUS = 3
ITEM_CAPACITY = 4


def _safe_bombs(bombs):
    """Return bombs as a 2-D array (N, 4), or empty (0, 4) if no bombs."""
    arr = np.asarray(bombs)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32)


def _bomb_radius(players, owner_id):
    """Approximate blast radius using owner's current radius bonus."""
    return 1 + int(players[int(owner_id)][4])


def _blast_tiles(grid, bx, by, radius):
    """Cross-shaped blast, stops at wall, stops-and-includes at box."""
    h, w = grid.shape
    tiles = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            tiles.append((tx, ty))
            if cell == BOX:
                break
    return tiles


def compute_danger_map(grid, bombs, players):
    """
    Float map where value at (r, c) = urgency of the most dangerous bomb
    that will hit that tile.  urgency = (BOMB_MAX_TIMER - timer + 1) / BOMB_MAX_TIMER
    so a bomb with timer=1 (about to explode) gives value 1.0.
    """
    h, w = grid.shape
    danger = np.zeros((h, w), dtype=np.float32)
    bombs_arr = _safe_bombs(bombs)
    for b in bombs_arr:
        bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        urgency = (BOMB_MAX_TIMER - timer + 1) / BOMB_MAX_TIMER
        radius = _bomb_radius(players, owner_id)
        for tx, ty in _blast_tiles(grid, bx, by, radius):
            danger[tx, ty] = max(danger[tx, ty], urgency)
    return danger


def compute_reachability(grid, start_x, start_y, bombs):
    """
    BFS distance from (start_x, start_y), normalized to [0, 1].
    Bomb cells block movement.  Returns 0 for unreachable/unvisited cells
    and 1 for the farthest reachable cell.
    """
    h, w = grid.shape
    dist_map = np.zeros((h, w), dtype=np.float32)
    if not (0 <= start_x < h and 0 <= start_y < w):
        return dist_map

    bombs_arr = _safe_bombs(bombs)
    bomb_cells = set()
    for b in bombs_arr:
        bomb_cells.add((int(b[0]), int(b[1])))

    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    q.append((start_x, start_y, 0))
    visited[start_x, start_y] = True
    max_d = 0

    while q:
        x, y, d = q.popleft()
        dist_map[x, y] = float(d)
        max_d = max(max_d, d)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < h and 0 <= ny < w):
                continue
            if visited[nx, ny]:
                continue
            cell = int(grid[nx, ny])
            if cell in (WALL, BOX) or (nx, ny) in bomb_cells:
                continue
            visited[nx, ny] = True
            q.append((nx, ny, d + 1))

    if max_d > 0:
        dist_map /= max_d
    return dist_map


def encode_obs(obs, agent_id: int):
    """
    Encode a game observation into (spatial, aux) tensors.

    Args:
        obs:      dict with keys 'map' (13,13), 'players' (4,5), 'bombs' (N,4)
        agent_id: integer 0-3 identifying which player we are

    Returns:
        spatial: np.float32 (N_CHANNELS, GRID_H, GRID_W)
        aux:     np.float32 (AUX_DIM,)
    """
    grid = obs["map"]        # (13, 13)  int
    players = obs["players"] # (4, 5)    int
    bombs = obs["bombs"]     # (N, 4) or empty

    H, W = grid.shape
    aid = int(agent_id)

    channels = []

    # ch 0-4: one-hot map cell type
    for v in (GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY):
        channels.append((grid == v).astype(np.float32))

    # ch 5: our position
    my_row = int(players[aid][0])
    my_col = int(players[aid][1])
    my_alive = int(players[aid][2])
    my_pos = np.zeros((H, W), dtype=np.float32)
    if my_alive:
        my_pos[my_row, my_col] = 1.0
    channels.append(my_pos)

    # ch 6: all live enemy positions (combined into one channel)
    enemy_pos = np.zeros((H, W), dtype=np.float32)
    for pid in range(len(players)):
        if pid == aid:
            continue
        pr, pc, palive = int(players[pid][0]), int(players[pid][1]), int(players[pid][2])
        if palive:
            enemy_pos[pr, pc] = 1.0
    channels.append(enemy_pos)

    # ch 7: bomb timer map (high value = about to explode)
    bomb_timer_map = np.zeros((H, W), dtype=np.float32)
    bombs_arr = _safe_bombs(bombs)
    for b in bombs_arr:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        urgency = (BOMB_MAX_TIMER - timer + 1) / BOMB_MAX_TIMER
        bomb_timer_map[bx, by] = max(bomb_timer_map[bx, by], urgency)
    channels.append(bomb_timer_map)

    # ch 8: our own bomb markers
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs_arr:
        bx, by, _, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if owner_id == aid:
            bomb_owned[bx, by] = 1.0
    channels.append(bomb_owned)

    # ch 9: danger map (precomputed blast zones)
    danger = compute_danger_map(grid, bombs_arr, players)
    channels.append(danger)

    # ch 10: BFS reachability from our position
    reach = compute_reachability(grid, my_row, my_col, bombs_arr)
    channels.append(reach)

    spatial = np.stack(channels, axis=0)  # (11, 13, 13) float32

    # Auxiliary scalars
    my_bombs_left = int(players[aid][3])
    my_radius_bonus = int(players[aid][4])
    alive_enemies = sum(
        1 for pid in range(len(players))
        if pid != aid and int(players[pid][2]) == 1
    )
    aux = np.array([
        my_bombs_left / 5.0,
        my_radius_bonus / 4.0,
        alive_enemies / 3.0,
    ], dtype=np.float32)

    return spatial, aux
