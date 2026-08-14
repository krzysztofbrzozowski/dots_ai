"""Capture / enclosure detection for the game of Dots (Kropki).

The engine works purely on integers::

    EMPTY    = 0
    PLAYER_1 = 1
    PLAYER_2 = -1

Rendering (symbols / colors) lives in a separate layer and never influences
the detection logic (requirement: capture code must not depend on colors or
display symbols).

Connectivity model
------------------
* Dots of the same player are **8-connected** (orthogonal + diagonal).  This
  is the connectivity used to decide which dots form a single "wall"/group.
* Empty cells (and opponent cells) are **4-connected** for the flood-fill.

8-connected foreground + 4-connected background is the standard digital
topology pairing that keeps the Jordan curve theorem valid: a pair of
diagonally touching dots seals the corner between them, so the background can
never "leak" diagonally through a corner that is geometrically closed.

Why cycle detection alone is not enough
---------------------------------------
A graph cycle (detected cheaply with union-find) is only a *trigger*.  Adding
a dot below the middle of a 3-dot row creates a cycle in the group graph yet
encloses nothing.  The flood-fill is the ground truth: a region is considered
enclosed **only if it cannot reach the board edge**.
"""

import numpy as np

# --------------------------------------------------------------------------
# Board representation (game logic)
# --------------------------------------------------------------------------
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1

# --------------------------------------------------------------------------
# Presentation layer (used ONLY by render_board, never by capture logic)
# --------------------------------------------------------------------------
SYMBOLS = {
    EMPTY: "·",
    PLAYER_1: "●",
    PLAYER_2: "○",
}

COLORS = {
    PLAYER_1: "red",
    PLAYER_2: "blue",
}

_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_NEIGHBORS_8 = _ORTHOGONAL + _DIAGONAL


def opponent_of(player):
    """The opponent of a player, derived from the numeric representation."""
    return -player


def get_neighbors(row, col, shape, include_diagonals=True):
    """Return in-bounds neighbor coordinates of ``(row, col)``.

    ``include_diagonals=True`` yields the 8-neighborhood (required for dot
    connectivity); ``False`` yields only the 4-neighborhood (used for the
    background flood-fill).
    """
    rows, cols = shape
    deltas = _NEIGHBORS_8 if include_diagonals else _ORTHOGONAL
    result = []
    for dr, dc in deltas:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


class UnionFind:
    """Disjoint-set / union-find over ``(row, col)`` cells.

    Tracks the connected groups of a player's dots.  A newly placed dot may
    create a new group, join one, merge several, or connect two parts of the
    same group (the latter is exactly what closes a loop).
    """

    def __init__(self):
        self._parent = {}
        self._rank = {}

    def add(self, cell):
        if cell not in self._parent:
            self._parent[cell] = cell
            self._rank[cell] = 0

    def find(self, cell):
        if cell not in self._parent:
            self.add(cell)
        root = cell
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: flatten the chain to keep find() near O(1).
        while self._parent[cell] != cell:
            self._parent[cell], cell = root, self._parent[cell]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # already in the same group
        # Union by rank.
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)


def find_group(groups, cell):
    """Return the canonical root of the group containing ``cell``."""
    return groups.find(cell)


def merge_groups(groups, a, b):
    """Merge the groups containing ``a`` and ``b``; True if they were distinct."""
    return groups.union(a, b)


def find_candidate_regions(board, last_move, player):
    """Return cells next to ``last_move`` that might have just been enclosed.

    Only cells adjacent to the newly placed dot can belong to a region whose
    boundary was just closed, so these are the only flood-fill seeds we ever
    need to inspect (keeps the check local instead of scanning the board).
    """
    r, c = last_move
    seeds = []
    for n in get_neighbors(r, c, board.shape, include_diagonals=True):
        if board[n] != player:
            seeds.append(n)
    return seeds


def flood_fill_region(board, start, player, visited):
    """Flood-fill ``start`` through EMPTY + opponent cells (4-connectivity).

    The current player's dots act as walls.  Returns a tuple::

        region        -- list of (row, col) cells in the region
        reaches_edge  -- True if the region touches the board boundary
        opponent_cells-- opponent dots found inside the region

    ``reaches_edge`` is the definitive enclosure test: a region that can reach
    the boundary is connected to the outside and therefore not enclosed.
    """
    rows, cols = board.shape
    opponent = opponent_of(player)
    if visited[start] or board[start] == player:
        return [], False, []

    stack = [start]
    visited[start] = True
    region = []
    opponent_cells = []
    reaches_edge = False

    while stack:
        r, c = stack.pop()
        region.append((r, c))
        if board[r, c] == opponent:
            opponent_cells.append((r, c))
        if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
            reaches_edge = True
        for dr, dc in _ORTHOGONAL:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and not visited[nr, nc]
                    and board[nr, nc] != player):
                visited[nr, nc] = True
                stack.append((nr, nc))

    return region, reaches_edge, opponent_cells


def find_enclosed_regions(board, last_move, player):
    """Return every region (list of cells) enclosed by ``player`` after the move.

    Shares a single ``visited`` mask across all seeds so each cell is touched
    at most once per call.
    """
    visited = np.zeros(board.shape, dtype=bool)
    enclosed = []
    for seed in find_candidate_regions(board, last_move, player):
        if visited[seed]:
            continue
        region, reaches_edge, _ = flood_fill_region(board, seed, player, visited)
        if region and not reaches_edge:
            enclosed.append(region)
    return enclosed


def detect_capture(board, last_move, player, groups=None):
    """Return opponent dots captured by the move at ``last_move`` (or ``[]``).

    Parameters
    ----------
    board : np.ndarray
        2D array of EMPTY / PLAYER_1 / PLAYER_2.  Must already contain the dot
        at ``last_move``.
    last_move : tuple(int, int)
        The (row, col) just played.
    player : int
        PLAYER_1 or PLAYER_2.
    groups : UnionFind, optional
        Must describe dot connectivity *before* ``last_move`` was placed.  It
        is used only to cheaply decide whether a loop could have closed; the
        actual capture is always verified geometrically by the flood-fill.
    """
    r, c = last_move
    if board[r, c] != player:
        raise ValueError("last_move cell must contain the moving player's dot")

    same_player_neighbors = [
        n for n in get_neighbors(r, c, board.shape, include_diagonals=True)
        if board[n] == player
    ]

    # --- cycle trigger (cheap) -----------------------------------------
    # Closing a loop requires the new dot to connect to at least two dots that
    # were already part of the same group.  Without a DSU we fall back to the
    # weaker (but safe) condition of having >= 2 same-player neighbors.
    if groups is not None:
        roots = {groups.find(n) for n in same_player_neighbors}
        closed_loop = len(roots) < len(same_player_neighbors)
    else:
        closed_loop = len(same_player_neighbors) >= 2

    if not closed_loop:
        return []

    # --- geometric verification (ground truth) -------------------------
    captured = []
    for region in find_enclosed_regions(board, last_move, player):
        for cell in region:
            if board[cell] == opponent_of(player):
                captured.append(cell)
    return sorted(captured)


def render_board(board, colorize=False):
    """Translate the internal representation into display symbols.

    This is the only place that depends on SYMBOLS/COLORS; it never mutates
    ``board`` and is never consulted by the capture logic.
    """
    lines = []
    for r in range(board.shape[0]):
        cells = []
        for c in range(board.shape[1]):
            value = board[r, c]
            symbol = SYMBOLS[value]
            if colorize and value in COLORS:
                code = "31" if COLORS[value] == "red" else "34"
                symbol = f"\033[{code}m{symbol}\033[0m"
            cells.append(symbol)
        lines.append(" ".join(cells))
    return "\n".join(lines)


class DotsGame:
    """Incremental Dots engine: board + union-find groups.

    Capture detection never scans the whole board; it only inspects the
    neighborhood of the last move and the regions it may have just closed.
    """

    def __init__(self, rows, cols):
        self.board = np.zeros((rows, cols), dtype=int)
        self.groups = UnionFind()

    def place_dot(self, row, col, player):
        """Play a dot and return the list of captured opponent cells."""
        if player not in (PLAYER_1, PLAYER_2):
            raise ValueError("player must be PLAYER_1 or PLAYER_2")
        if self.board[row, col] != EMPTY:
            raise ValueError("cell is already occupied")

        # 1. Place the dot first so the flood-fill treats it as a boundary.
        self.board[row, col] = player

        # 2. Detect captures.  `self.groups` does NOT contain the new dot yet,
        #    so cycle detection observes the pre-move connectivity (correct).
        captured = detect_capture(self.board, (row, col), player, groups=self.groups)

        # 3. Register the new dot and merge it with any same-player neighbors.
        self.groups.add((row, col))
        for n in get_neighbors(row, col, self.board.shape, include_diagonals=True):
            if self.board[n] == player:
                merge_groups(self.groups, (row, col), n)

        # 4. Remove captured opponent dots from the board.
        for (cr, cc) in captured:
            self.board[cr, cc] = EMPTY

        return captured

    def render(self, colorize=False):
        return render_board(self.board, colorize=colorize)
