"""Board-based enclosure and capture detection

This implementation uses the common rule model in which an enclosure counts
only when it contains at least one opponent dot, and the enclosed region then
becomes unavailable for further play

Connectivity model
------------------
* Player dots are 8-connected -> you can connect dot-dot in 8 directions

    ↖  ↑  ↗
    ←  X  →
    ↙  ↓  ↘

* Background flood fill uses 4-connectivity -> you can fill areas in 4 directions

       ↑
    ←  X  →
       ↓

  Flood fill can move only horizontally and vertically, not diagonally

* Previously captured territory remains traversable for enclosure geometry
  The current player's dots on ``board`` are the only walls
"""

from dataclasses import dataclass

import numpy as np

try:  # Support package and direct imports
    from .board import (
        EMPTY,
        ORTHOGONAL,
        PLAYERS,
        get_neighbors,
        opponent_of,
    )
    from .groups import rebuild_groups
except ImportError:  # pragma: no cover - exercised by the direct test runner
    from board import EMPTY, ORTHOGONAL, PLAYERS, get_neighbors, opponent_of
    from groups import rebuild_groups


# --------------------------------------------------------------------------
# Enclosure detection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureInfo:
    """Result of one capture check"""

    captured_dots: tuple
    captured_regions: tuple

    @property
    def happened(self):
        return bool(self.captured_dots)


def find_candidate_regions(board, territory, last_move, player):
    """Return local flood-fill seeds around the newly placed dot

    ``territory`` is retained for API compatibility
    Only the current player's
    dots on ``board`` affect enclosure geometry
    """
    row, col = last_move
    seeds = []

    # Only orthogonal neighbors can belong to a flood-filled region for which
    # last_move is part of the geometric boundary
    # Diagonal neighbors still matter to the dot graph
    # but they are not flood-fill entry points
    for neighbor in get_neighbors(row, col, board.shape, include_diagonals=False):
        # Captured territory is game-state information only
        # It is deliberately not filtered here because flood fill must pass through it

        # Dots of the current player form the actual wall
        if board[neighbor] == player:
            continue

        seeds.append(neighbor)

    return seeds


def flood_fill_region(board, seed, player, visited):
    """Flood-fill one candidate region through non-player cells

    The current player's dots are walls
    Previously captured territory is
    traversable and does not affect enclosure geometry

    Returns
    -------
    region : list[tuple[int, int]]
        Cells belonging to the connected background region
    reaches_edge : bool
        True when the region can reach the outside of the board
    opponent_cells : list[tuple[int, int]]
        Opponent dots found inside this region
    """
    rows, cols = board.shape
    opponent = opponent_of(player)
    region = []
    opponent_cells = []
    reaches_edge = False

    if visited[seed]:
        return region, reaches_edge, opponent_cells

    if board[seed] == player:
        return region, reaches_edge, opponent_cells

    stack = [seed]
    visited[seed] = True

    while stack:
        # Remove last element and assign it to row, col
        row, col = stack.pop()
        region.append((row, col))

        if board[row, col] == opponent:
            opponent_cells.append((row, col))

        # Verify if the edge of game is hit
        if row == 0 or col == 0 or row == rows - 1 or col == cols - 1:
            reaches_edge = True

        for dr, dc in ORTHOGONAL:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[neighbor]:
                continue

            # No territory check here because territory is not a geometric wall
            if board[neighbor] == player:
                continue

            visited[neighbor] = True
            stack.append(neighbor)

    return region, reaches_edge, opponent_cells


def find_enclosed_regions(board, territory, last_move, player):
    """Return geometrically enclosed regions adjacent to ``last_move``

    An empty loop is not a capture in Dots, therefore only enclosed regions
    containing at least one opponent dot are returned
    """
    visited = np.zeros(board.shape, dtype=bool)
    enclosed = []

    # Candidate regions around the newly placed dot:
    #
    #          seed
    #            ↓
    #    seed -> ● <- seed
    #            ↑
    #          seed
    #
    #            ● = last_move
    #
    # Player dots form the wall
    # The orthogonally surrounding cells become flood-fill seeds
    # used to check whether they belong to an enclosed region
    for seed in find_candidate_regions(board, territory, last_move, player):
        # At the beginning all of the seeds are not visited (False)
        if visited[seed]:
            continue

        # Main Dots logic ->
        # Find areas around the new dot that could potentially be enclosed
        region, reaches_edge, opponent_cells = flood_fill_region(
            board,
            seed,
            player,
            visited,
        )

        if region and not reaches_edge and opponent_cells:
            enclosed.append((region, opponent_cells))

    return enclosed


def _could_have_closed_loop(board, territory, last_move, player, groups=None):
    """Return whether adding ``last_move`` creates a player-dot graph cycle

    ``groups`` should represent connectivity before ``last_move`` was added
    Standalone callers may omit it; in that case the pre-move graph is rebuilt
    from active player dots using ``board`` and ``territory``
    """
    row, col = last_move

    # Player dots use 8-connectivity, so every same-player neighbor may provide
    # one side of a path through last_move
    same_player_neighbors = [
        neighbor
        for neighbor in get_neighbors(
            row,
            col,
            board.shape,
            include_diagonals=True,
        )
        if board[neighbor] == player and territory[neighbor] == EMPTY
    ]

    # With fewer than two neighbors, last_move can only extend a line or tree
    if len(same_player_neighbors) < 2:
        return False

    # DotsGame supplies its authoritative pre-move active connectivity
    # Standalone callers rebuild the same state from board and territory
    if groups is None:
        pre_move_board = board.copy()
        pre_move_board[last_move] = EMPTY
        groups = rebuild_groups(pre_move_board, territory)

    # If two neighbors already had the same root before the move, an existing
    # path connected them
    # Adding neighbor -> last_move -> neighbor creates a second path
    # and therefore closes a graph cycle
    roots = set()
    for neighbor in same_player_neighbors:
        root = groups.find(neighbor)
        if root in roots:
            return True
        roots.add(root)

    return False


# New move
#     ↓
# _could_have_closed_loop()
#     ↓
# Could this move have closed a loop?
#     ↓
# NO  -> stop, nothing to check
# YES
#     ↓
# find_candidate_regions()
#     ↓
# Find areas around the new dot
# that could potentially be enclosed
#     ↓
# Flood fill
#     ↓
# Is the area geometrically enclosed?
#     ↓
# Does the inside contain an opponent dot?
def detect_capture_info(board, last_move, player, territory=None, groups=None):
    """Return detailed capture information caused by ``last_move``

    ``board`` must already contain the newly placed dot
    When supplied,
    ``groups`` must describe connectivity before that dot was added

    Return
        -> CaptureInfo(captured_dots=(), captured_regions=()) for no capture
        -> or populated CaptureInfo for a capture
    """
    if territory is None:
        territory = np.zeros(board.shape, dtype=int)

    row, col = last_move
    if not (0 <= row < board.shape[0] and 0 <= col < board.shape[1]):
        raise ValueError("last_move is outside the board")
    if player not in PLAYERS:
        raise ValueError("player must be PLAYER_1 or PLAYER_2")
    if board[last_move] != player:
        raise ValueError("last_move cell must contain the moving player's dot")
    if territory[last_move] != EMPTY:
        raise ValueError("last_move cannot be inside captured territory")

    # _could_have_closed_loop() == False
    # -> return empty CaptureInfo immediately
    # -> no flood fill is needed
    #
    # _could_have_closed_loop() == True
    # -> do NOT return here
    # -> continue with flood fill to verify whether a real capture happened
    if not _could_have_closed_loop(
        board,
        territory,
        last_move,
        player,
        groups,
    ):
        return CaptureInfo(captured_dots=(), captured_regions=())

    captured_dots = []
    captured_regions = []

    for region, opponent_cells in find_enclosed_regions(
        board,
        territory,
        last_move,
        player,
    ):
        # Territory is game state and not geometry
        # Filter opponent dots after flood fill so a scored dot is not captured twice
        new_opponent_cells = [
            cell
            for cell in opponent_cells
            if territory[cell] == EMPTY
        ]
        if not new_opponent_cells:
            continue

        captured_regions.append(tuple(region))
        captured_dots.extend(new_opponent_cells)

    return CaptureInfo(
        captured_dots=tuple(sorted(set(captured_dots))),
        captured_regions=tuple(captured_regions),
    )


def detect_capture(board, last_move, player, territory=None, groups=None):
    """Return opponent coordinates captured by ``last_move``

    This wrapper keeps the simple API of the original implementation
    """
    info = detect_capture_info(
        board,
        last_move,
        player,
        territory=territory,
        groups=groups,
    )
    return list(info.captured_dots)
