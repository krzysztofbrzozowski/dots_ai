"""Board values and neighborhood helpers for Dots.

Internal board representation
-----------------------------
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1

The board stores dots only. Captured territory is stored separately so
captured dots do not have to be deleted from the board and captured empty
intersections cannot become playable again.
"""

# --------------------------------------------------------------------------
# Board representation
# --------------------------------------------------------------------------
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1

PLAYERS = (PLAYER_1, PLAYER_2)

# --------------------------------------------------------------------------
# Neighborhoods
# --------------------------------------------------------------------------
# Values set as -> (row, col)
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS_8 = ORTHOGONAL + DIAGONAL

# NEIGHBORS_8 = (
#     (-1, -1), (-1, 0), (-1, 1),
#     ( 0, -1),          ( 0, 1),
#     ( 1, -1), ( 1, 0), ( 1, 1)
# )


def opponent_of(player):
    """Return the opponent of ``player``."""
    if player not in PLAYERS:
        raise ValueError("player must be PLAYER_1 or PLAYER_2")
    return -player


def get_neighbors(row, col, shape, include_diagonals=True):
    """
    Return neighbors of ``(row, col) coordinates``.
    Takes a predefined array of positions and adds it to row and col position.
    """
    rows, cols = shape
    deltas = NEIGHBORS_8 if include_diagonals else ORTHOGONAL

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
