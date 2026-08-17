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

        # Above the same as:
        # parent = self._parent[root]
        # while parent != root:
        #     root = parent
        #     parent = self._parent[root]

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

        # If already connected, return False
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
    """Return local flood-fill seeds (starting points) around the newly placed dot"""
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


def flood_fill_region(board, territory, seed, player, visited):
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

    region = []
    opponent_cells = []
    reaches_edge = False

    if visited[seed]:
        return region, reaches_edge, opponent_cells

    if territory[seed] != EMPTY:
        return region, reaches_edge, opponent_cells

    if board[seed] == player:
        return region, reaches_edge, opponent_cells

    stack = [seed]
    visited[seed] = True

    while stack:
        # Remove last element and assing it to row, col
        row, col = stack.pop()
        region.append((row, col))

        if board[row, col] == opponent:
            opponent_cells.append((row, col))

        # Verify if edge of gameplay is hitted
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

    # Candidate regions around the newly placed dot:
    #
    #    seed seed seed
    #      ↓    ↓    ↓
    #      .    .    .
    #      ●    ●    ●
    #           ↑
    #       last_move
    #      .    .    .
    #      ↑    ↑    ↑
    #    seed seed seed
    #
    # Player dots form the wall
    # The surrounding cells become flood-fill seeds
    # used to check whether they belong to an enclosed region
    for seed in find_candidate_regions(board, territory, last_move, player):
        # At the begining all of the seeds are not visited (False)
        if visited[seed]:
            continue
        
        # Main dots logic -> 
        # Find areas around the new dot that could potentially be enclosed
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
    """Cheaply decide whether ``last_move`` could have closed a cycle"""
    row, col = last_move

    # Same player naighbor is the area with own dots and teritory not catched
    # TODO -> this might be changed (depending on dots rules)
    same_player_neighbors = [
        neighbor
        for neighbor in get_neighbors(row, col, board.shape, include_diagonals=True)
        if board[neighbor] == player and territory[neighbor] == EMPTY
    ]

    # If 1 max dot of the same player return False
    if len(same_player_neighbors) < 2:
        return False

    # Without a pre-move DSU we use a weaker safe trigger
    # groups = UnionFind method, if is not passed, maybe something is already connected?
    if groups is None:
        return True

    # A new cycle is possible when at least two neighbors already belonged
    # to the same connected component before the new dot was added
    roots = [groups.find(neighbor) for neighbor in same_player_neighbors]
    # Example 1:
    # roots = [(1, 1), (1, 1)]
    #
    # len(roots) = 2
    # len(set(roots)) = 1
    #
    # 1 < 2 -> True
    # At least two neighboring dots have the same root,
    # so they already belong to the same connected group


    # Example 2:
    # roots = [(1, 1), (3, 4)]
    #
    # len(roots) = 2
    # len(set(roots)) = 2
    #
    # 2 < 2 -> False
    # The neighboring dots have different roots,
    # so they belong to different connected groups
    return len(set(roots)) < len(roots)


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
# Check whether any of these areas
# is actually enclosed
def detect_capture_info(board, last_move, player, territory=None, groups=None):
    """Return detailed capture information caused by ``last_move``

    ``board`` must already contain the newly placed dot
    ``groups`` should describe active connectivity before the new move

    Return
        -> CaptureInfo(captured_dots=(), captured_regions=() -> Empty CaptureInfo class
        -> or 
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

    # _could_have_closed_loop() == False
    # -> return empty CaptureInfo immediately
    # -> no flood fill is needed
    #
    # _could_have_closed_loop() == True
    # -> do NOT return here
    # -> continue with flood fill to verify whether a real capture happened
    if not _could_have_closed_loop(board, territory, last_move, player, groups):\
        # If _could_have_closed_loop(...) == False
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

        # board stores the dots currently placed on the board
        #  0 -> empty cell
        #  1 -> PLAYER_1 dot
        # -1 -> PLAYER_2 dot
        self.board = np.zeros((rows, cols), dtype=int)

        # Rerritory stores areas that have already been captured by some player
        # and can not be captured anymore
        #  0 -> active/playable area
        #  1 -> area captured by PLAYER_1
        # -1 -> area captured by PLAYER_2
        #
        # A territory cell does not have to contain a dot
        # It simply means that this area was already enclosed and is no longer playable
        self.territory = np.zeros((rows, cols), dtype=int)

        self.groups = UnionFind()
        self.score = {
            PLAYER_1: 0,
            PLAYER_2: 0,
        }

    def is_legal_move(self, row, col):
        """Return True when a dot may be placed at ``(row, col)``"""
        # Verify if place row, col is smaller than the play area
        # If so, retunr False, move not allowed
        if not (0 <= row < self.board.shape[0] and 0 <= col < self.board.shape[1]):
            return False

        # Return True if place you want to put dot is
        # -> still empty
        # -> and "teritory" is not taken by any player
        return (
            self.board[row, col] == EMPTY
            and self.territory[row, col] == EMPTY
        )

    def legal_moves(self):
        """Return all currently legal moves coordinates"""
        legal = np.argwhere(
            (self.board == EMPTY)
            & (self.territory == EMPTY)
        )
        return [tuple(cell) for cell in legal]

    # Place a new dot and update capture / connectivity state
    #
    # Flow:
    # place_dot()
    #     ↓
    # validate the move
    #     ↓
    # add the new dot to board
    #     ↓
    # detect_capture_info()
    #     ├── board represents the state AFTER the move
    #     └── groups still represents connectivity BEFORE the move
    #     ↓
    # add the new dot to UnionFind
    #     ↓
    # get_neighbors()
    #     ↓
    # keep only active neighboring dots belonging to the same player
    #     ↓
    # groups.union(last_move, neighbor)
    #     ↓
    # if a capture happened:
    #     ├── mark captured regions in territory
    #     ├── update score
    #     └── rebuild UnionFind
    def place_dot(self, row, col, player):
        """Place one dot and return the opponent dots captured by this move."""
        if player not in PLAYERS:
            raise ValueError("player must be PLAYER_1 or PLAYER_2")

        if not self.is_legal_move(row, col):
            raise ValueError("cell is occupied, blocked, or outside the board")

        last_move = (row, col)

        # ---------------------------------------------------------------
        # CAPTURE DETECTION
        #
        # 1. Update board first
        #
        #    board = state AFTER the move
        #
        #    Flood fill (explores all reachable connected cells in a region) 
        #    -> must see the newly placed dot because this dot
        #    may be the one that closes the boundary around an opponent
        #
        #    Example:
        #
        #    Before:
        #
        #    ● ● ●
        #    ● ○ .
        #    ● ● ●
        #
        #    After:
        #
        #    ● ● ●
        #    ● ○ ●   <- last_move closes the boundary
        #    ● ● ●
        #
        self.board[last_move] = player

        # 2. Detect capture using the OLD UnionFind state
        #
        #    board  = state AFTER the move
        #    groups = state BEFORE the move
        #    -> Clue: Use groups to quickly check whether the newly placed dot
        #       could have closed a loop
        #
        #       If yes, run the more expensive flood fill verification
        #
        #    The new dot is already visible on board, but it has NOT been
        #    added to UnionFind yet
        #    # TODO -> verify this with clear mind
        #    This lets us check whether the new dot connects neighboring
        #    dots that were already part of the same connected group
        #   
        #    If they were already connected before this move, the new dot
        #    may have closed a loop
        capture = detect_capture_info(
            self.board,
            last_move,
            player,
            territory=self.territory,
            groups=self.groups,
        )

        # 3. Update UnionFind AFTER capture detection
        #
        #    -> add the new dot as a new one-element group
        #    -> find its neighboring dots
        #    -> connect it with active neighboring dots of the same player
        #
        #    UnionFind itself does not check board geometry
        #    get_neighbors() ensures that only real neighboring cells
        #    are considered
        self.groups.add(last_move)

        for neighbor in get_neighbors(
            row,
            col,
            self.board.shape,
            include_diagonals=True,
        ):
            # Skip previously captured territory
            if self.territory[neighbor] != EMPTY:
                continue

            # Skip empty cells and opponent dots
            if self.board[neighbor] != player:
                continue

            # Skip dots that are not active UnionFind nodes
            if neighbor not in self.groups:
                continue

            # Both dots are active neighbors belonging to the same player
            self.groups.union(last_move, neighbor)

        # ---------------------------------------------------------------
        # APPLY CAPTURE
        # ---------------------------------------------------------------

        if capture.happened:
            # Mark the whole captured region as unavailable territory
            for region in capture.captured_regions:
                for cell in region:
                    self.territory[cell] = player

            # Add captured opponent dots to the player's score
            self.score[player] += len(capture.captured_dots)

            # Rebuild UnionFind because captured dots must no longer
            # participate in active connected groups
            self.groups = rebuild_groups(
                self.board,
                self.territory,
            )

        return list(capture.captured_dots)

    def render(self, colorize=False):
        return render_board(
            self.board,
            territory=self.territory,
            colorize=colorize,
    )
