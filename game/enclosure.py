"""Capture / enclosure detection for the game of Dots (Kropki)

Internal board representation
-----------------------------
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1

The board stores dots only. Captured territory is stored separately in
``DotsGame.territory`` so captured dots do not have to be deleted from the
board and captured empty intersections cannot become playable again

This implementation uses the common rule model in which an enclosure counts
only when it contains at least one opponent dot, and the enclosed region then
becomes unavailable for further play

Connectivity model
------------------
* Active dots of one player are 8-connected -> you can connect dot-dot in 8 directions
* Background flood fill uses 4-connectivity -> you can fill areas in 4 directions
    -> 8-connected neighborhood used for player dots

    ↖  ↑  ↗
    ←  X  →
    ↙  ↓  ↘

    A dot is connected to all 8 surrounding cells,
    including diagonal neighbors


    -> 4-connected neighborhood used for flood fill

        ↑
    ←  X  →
        ↓

    Flood fill can move only horizontally and vertically, not diagonally

* Previously captured territory is not traversable and cannot participate in
  later groups or captures

The Union-Find structure is used only as a cheap cycle trigger. Flood fill is
the geometric verification step that decides whether a candidate region is
actually closed off from the board edge
"""

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Board representation
# --------------------------------------------------------------------------
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1

PLAYERS = (PLAYER_1, PLAYER_2)


# --------------------------------------------------------------------------
# Presentation layer
# --------------------------------------------------------------------------
SYMBOLS = {
    EMPTY: "·",
    PLAYER_1: "●",
    PLAYER_2: "○",
}

COLORS = {
    PLAYER_1: "31",  # red ANSI code
    PLAYER_2: "34",  # blue ANSI code
}

BLOCKED_SYMBOL = "×"


# --------------------------------------------------------------------------
# Neighborhoods
# --------------------------------------------------------------------------
# Vlues set as -> (row, col)
_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_NEIGHBORS_8 = _ORTHOGONAL + _DIAGONAL

# _NEIGHBORS_8 = (
#     (-1, -1), (-1, 0), (-1, 1),
#     ( 0, -1),          ( 0, 1),
#     ( 1, -1), ( 1, 0), ( 1, 1)
# )


def opponent_of(player):
    """Return the opponent of ``player`."""
    if player not in PLAYERS:
        raise ValueError("player must be PLAYER_1 or PLAYER_2")
    return -player


def get_neighbors(row, col, shape, include_diagonals=True):
    """
    Return neighbors of ``(row, col) coorinates``
    Takes predefined array of positions and substracts it from row and col position
    """
    rows, cols = shape
    deltas = _NEIGHBORS_8 if include_diagonals else _ORTHOGONAL

    neighbors = []
    # dr = delta row    -> change in the row index
    # dc = delta column -> change in the column index
    #
    # nr = new row      -> current row + dr
    # nc = new column   -> current column + dc
    #
    # Example:
    # dr = -1, dc = 0  -> move one cell up
    # dr =  1, dc = 1  -> move one cell down-right
    for dr, dc in deltas:
        nr, nc = row + dr, col + dc
        # Check whether the new row and column are still inside the board
        # Valid indexes are:
        # 0 <= row < number of rows
        # 0 <= col < number of columns
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))

    return neighbors


# --------------------------------------------------------------------------
# Connected groups
# --------------------------------------------------------------------------
class UnionFind:
    """Disjoint-set structure over active dot coordinates."""

    def __init__(self):
        self._parent = {}
        self._rank = {}

    def __contains__(self, cell):
        return cell in self._parent

    def add(self, cell):
        if cell not in self._parent:
            self._parent[cell] = cell
            self._rank[cell] = 0

    def find(self, cell):
        if cell not in self._parent:
            raise KeyError(f"cell {cell} is not registered in UnionFind")

        root = cell
        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression
        while self._parent[cell] != cell:
            parent = self._parent[cell]
            self._parent[cell] = root
            cell = parent

        return root

    # union() assumes that a and b are valid dots that should be connected
    # It does NOT check whether they are neighbors on the board
    # Neighbor validation is done earlier in place_dot() using get_neighbors()
    # and checking whether both dots belong to the same player
    def union(self, a, b):
        # 
        # Find root A
        # Find root B
        #     ↓
        # Are they the same root?
        #     ↓ no
        # Attach root B under root A
        #     ↓
        # If both trees had the same rank,
        # increase the rank of the new root

        # Find root of the dot a and b
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        # Union by rank
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a

        self._parent[root_b] = root_a

        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1

        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)


def find_group(groups, cell):
    """Return the canonical root of the group containing ``cell``."""
    return groups.find(cell)


def merge_groups(groups, a, b):
    """Merge the groups containing ``a`` and ``b``."""
    return groups.union(a, b)


def rebuild_groups(board, territory=None):
    """Build Union-Find from all active dots currently on the board."""
    if territory is None:
        territory = np.zeros(board.shape, dtype=int)

    groups = UnionFind()
    rows, cols = board.shape

    # First register every active dot
    for row in range(rows):
        for col in range(cols):
            if board[row, col] in PLAYERS and territory[row, col] == EMPTY:
                groups.add((row, col))

    # Then connect same-player active neighbors
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)

            if cell not in groups:
                continue

            player = board[cell]

            for neighbor in get_neighbors(row, col, board.shape, include_diagonals=True):
                if neighbor not in groups:
                    continue
                if board[neighbor] == player:
                    groups.union(cell, neighbor)

    return groups


# --------------------------------------------------------------------------
# Enclosure detection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureInfo:
    """Result of one capture check."""

    captured_dots: tuple
    captured_regions: tuple

    @property
    def happened(self):
        return bool(self.captured_dots)


def find_candidate_regions(board, territory, last_move, player):
    """Return local flood-fill seeds around the newly placed dot."""
    row, col = last_move
    seeds = []

    for neighbor in get_neighbors(row, col, board.shape, include_diagonals=True):
        # Captured territory is dead and must never be searched again
        if territory[neighbor] != EMPTY:
            continue

        # Active dots of the current player form the wall
        if board[neighbor] == player:
            continue

        seeds.append(neighbor)

    return seeds


def flood_fill_region(board, territory, start, player, visited):
    """Flood-fill one candidate region through active non-player cells

    The current player's active dots are walls
    Previously captured territory is also a wall

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

    if visited[start]:
        return [], False, []

    if territory[start] != EMPTY:
        return [], False, []

    if board[start] == player:
        return [], False, []

    stack = [start]
    visited[start] = True

    region = []
    opponent_cells = []
    reaches_edge = False

    while stack:
        row, col = stack.pop()
        region.append((row, col))

        if board[row, col] == opponent:
            opponent_cells.append((row, col))

        if row == 0 or col == 0 or row == rows - 1 or col == cols - 1:
            reaches_edge = True

        for dr, dc in _ORTHOGONAL:
            nr, nc = row + dr, col + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc]:
                continue
            if territory[nr, nc] != EMPTY:
                continue
            if board[nr, nc] == player:
                continue

            visited[nr, nc] = True
            stack.append((nr, nc))

    return region, reaches_edge, opponent_cells


def find_enclosed_regions(board, territory, last_move, player):
    """Return newly enclosed regions adjacent to ``last_move``

    An empty loop is not a capture in Dots, therefore only enclosed regions
    containing at least one opponent dot are returned
    """
    visited = np.zeros(board.shape, dtype=bool)
    enclosed = []

    for seed in find_candidate_regions(board, territory, last_move, player):
        if visited[seed]:
            continue

        region, reaches_edge, opponent_cells = flood_fill_region(
            board,
            territory,
            seed,
            player,
            visited,
        )

        if region and not reaches_edge and opponent_cells:
            enclosed.append((region, opponent_cells))

    return enclosed


def _could_have_closed_loop(board, territory, last_move, player, groups=None):
    """Cheaply decide whether ``last_move`` could have closed a cycle."""
    row, col = last_move

    same_player_neighbors = [
        neighbor
        for neighbor in get_neighbors(row, col, board.shape, include_diagonals=True)
        if board[neighbor] == player and territory[neighbor] == EMPTY
    ]

    if len(same_player_neighbors) < 2:
        return False

    # Without a pre-move DSU we use a weaker safe trigger
    if groups is None:
        return True

    # A new cycle is possible when at least two neighbors already belonged
    # to the same connected component before the new dot was added
    roots = [groups.find(neighbor) for neighbor in same_player_neighbors]
    return len(set(roots)) < len(roots)


def detect_capture_info(board, last_move, player, territory=None, groups=None):
    """Return detailed capture information caused by ``last_move``

    ``board`` must already contain the newly placed dot
    ``groups`` should describe active connectivity before the new move
    """
    if territory is None:
        territory = np.zeros(board.shape, dtype=int)

    row, col = last_move

    if not (0 <= row < board.shape[0] and 0 <= col < board.shape[1]):
        raise ValueError("last_move is outside the board")

    if player not in PLAYERS:
        raise ValueError("player must be PLAYER_1 or PLAYER_2")

    if board[row, col] != player:
        raise ValueError("last_move cell must contain the moving player's dot")

    if territory[row, col] != EMPTY:
        raise ValueError("last_move cannot be inside captured territory")

    if not _could_have_closed_loop(board, territory, last_move, player, groups):
        return CaptureInfo(captured_dots=(), captured_regions=())

    captured_dots = []
    captured_regions = []

    for region, opponent_cells in find_enclosed_regions(
        board,
        territory,
        last_move,
        player,
    ):
        captured_regions.append(tuple(region))
        captured_dots.extend(opponent_cells)

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


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_board(board, territory=None, colorize=False):
    """Render the board without changing its internal representation."""
    if territory is None:
        territory = np.zeros(board.shape, dtype=int)

    lines = []

    for row in range(board.shape[0]):
        cells = []

        for col in range(board.shape[1]):
            value = board[row, col]

            if value == EMPTY and territory[row, col] != EMPTY:
                symbol = BLOCKED_SYMBOL
            else:
                symbol = SYMBOLS[value]

            if colorize and value in PLAYERS:
                symbol = f"\033[{COLORS[value]}m{symbol}\033[0m"

            cells.append(symbol)

        lines.append(" ".join(cells))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------
class DotsGame:
    """Dots board with incremental group tracking and capture state."""

    def __init__(self, rows, cols):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        self.board = np.zeros((rows, cols), dtype=int)

        # 0 means active/playable area
        # 1 or -1 means territory captured by that player
        self.territory = np.zeros((rows, cols), dtype=int)

        self.groups = UnionFind()
        self.score = {
            PLAYER_1: 0,
            PLAYER_2: 0,
        }

    def is_legal_move(self, row, col):
        """Return True when a dot may be placed at ``(row, col)``."""
        if not (0 <= row < self.board.shape[0] and 0 <= col < self.board.shape[1]):
            return False

        return (
            self.board[row, col] == EMPTY
            and self.territory[row, col] == EMPTY
        )

    def legal_moves(self):
        """Return all currently legal moves."""
        legal = np.argwhere(
            (self.board == EMPTY)
            & (self.territory == EMPTY)
        )
        return [tuple(cell) for cell in legal]

    # Update connected groups after placing a new dot
    #
    # Flow:
    # place_dot()
    #     ↓
    # get_neighbors()
    #     ↓
    # check whether the neighbor belongs to the same player
    #     ↓ yes
    # merge_groups(...)
    #     ↓
    # groups.union(a, b)
    #
    # UnionFind itself does not check board geometry
    # get_neighbors() ensures that union() is called only for actual neighboring dots
    # belonging to the same player
    def place_dot(self, row, col, player):
        """Place one dot and return the opponent dots captured by this move."""
        if player not in PLAYERS:
            raise ValueError("player must be PLAYER_1 or PLAYER_2")

        if not self.is_legal_move(row, col):
            raise ValueError("cell is occupied, blocked, or outside the board")

        last_move = (row, col)

        # Place the new dot first so geometric verification sees the new wall
        self.board[last_move] = player

        # groups still describes the position before this move
        capture = detect_capture_info(
            self.board,
            last_move,
            player,
            territory=self.territory,
            groups=self.groups,
        )

        # Register the new active dot and merge it with active same-player dots
        self.groups.add(last_move)

        # Iterate over all possible neighbors of the last_move and verify
        # -> neighbor is not empty
        # -> neighbor does not belong to opposite player (verifes PLAYER_1 = 1 and PLAYER_2 = -1)
        # -> neighbor not added to the gropus - skip it
        # get_neighbors() ensures that only actual neighboring cells are checked
        # union() is called only when the neighboring dot belongs to the same player
        for neighbor in get_neighbors(row, col, self.board.shape, include_diagonals=True):
            if self.territory[neighbor] != EMPTY:
                continue
            if self.board[neighbor] != player:
                continue
            if neighbor not in self.groups:
                continue

            self.groups.union(last_move, neighbor)

        if capture.happened:
            # Keep captured dots on the board and mark the whole captured region
            # as unavailable for later moves
            for region in capture.captured_regions:
                for cell in region:
                    self.territory[cell] = player

            self.score[player] += len(capture.captured_dots)

            # Captured dots must not participate in later connected groups
            # Rebuilding after a capture avoids stale DSU connections
            self.groups = rebuild_groups(self.board, self.territory)

        return list(capture.captured_dots)

    def render(self, colorize=False):
        return render_board(
            self.board,
            territory=self.territory,
            colorize=colorize,
        )
