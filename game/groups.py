"""Connected groups of active dots"""

import numpy as np

try:  # Support package and direct imports from the game folder
    from .board import EMPTY, PLAYERS, get_neighbors
except ImportError:  # pragma: no cover - exercised by the direct test runner
    from board import EMPTY, PLAYERS, get_neighbors


# --------------------------------------------------------------------------
# Connected groups
# --------------------------------------------------------------------------
class UnionFind:
    """Disjoint-set structure over active dot coordinates"""

    def __init__(self):
        self._parent = {}
        self._rank = {}

    def __contains__(self, cell):
        return cell in self._parent

    def copy(self):
        """Return an independent copy of this connectivity structure."""
        copied = UnionFind()
        copied._parent = self._parent.copy()
        copied._rank = self._rank.copy()
        return copied

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
        # parent = _parent[root]
        # while parent != root:
        #     root = parent
        #     parent = _parent[root]

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
    """Return the canonical root of the group containing ``cell``"""
    return groups.find(cell)


def merge_groups(groups, a, b):
    """Merge the groups containing ``a`` and ``b``"""
    return groups.union(a, b)


def rebuild_groups(board, territory=None):
    """Build Union-Find from active dots currently on the board"""
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

            for neighbor in get_neighbors(
                row,
                col,
                board.shape,
                include_diagonals=True,
            ):
                if neighbor not in groups:
                    continue
                if board[neighbor] == player:
                    groups.union(cell, neighbor)

    return groups
