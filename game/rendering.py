"""Text rendering for Dots boards"""

import numpy as np

try:  # Support package and direct imports
    from .board import EMPTY, PLAYER_1, PLAYER_2, PLAYERS
except ImportError:  # pragma: no cover - exercised by the direct test runner
    from board import EMPTY, PLAYER_1, PLAYER_2, PLAYERS


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
# Rendering
# --------------------------------------------------------------------------
def render_board(board, territory=None, colorize=False):
    """Render the board without changing its internal representation"""
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
