"""Mutable Dots game state and move application."""

import numpy as np

try:  # Support both package and direct imports.
    from .board import EMPTY, PLAYER_1, PLAYER_2, PLAYERS, get_neighbors
    from .capture import detect_capture_info
    from .groups import UnionFind, rebuild_groups
    from .rendering import render_board
except ImportError:  # pragma: no cover - exercised by the direct test runner
    from board import EMPTY, PLAYER_1, PLAYER_2, PLAYERS, get_neighbors
    from capture import detect_capture_info
    from groups import UnionFind, rebuild_groups
    from rendering import render_board


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

        # Territory stores areas that have already been captured by some player
        # and cannot be captured again
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
        """Return True when a dot may be placed at ``(row, col)``."""
        # Verify if row, col is inside the play area
        # If not, return False, move not allowed
        if not (0 <= row < self.board.shape[0] and 0 <= col < self.board.shape[1]):
            return False

        # Return True if the place where you want to put a dot is
        # -> still empty
        # -> and territory is not taken by any player
        return (
            self.board[row, col] == EMPTY
            and self.territory[row, col] == EMPTY
        )

    def legal_moves(self):
        """Return all currently legal move coordinates."""
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
    #     └── groups still represents active connectivity BEFORE the move
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
        """Place one dot and return opponent dots captured by this move."""
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

        # 2. Detect capture from the board state after the move
        #
        #    board  = state AFTER the move
        #    groups = active connectivity state BEFORE the move
        #
        #    The new dot is already visible on board, but it has NOT been
        #    added to UnionFind yet. That update happens in step 3 below.
        #
        #    _could_have_closed_loop() is only a cheap local pre-check.
        #    If it returns True, flood fill performs the real geometric check.
        #    Territory is passed for game-state handling such as avoiding
        #    duplicate scoring, but it is not part of enclosure geometry.
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
            # Skip previously captured territory for active group tracking
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
            # Mark newly captured cells as unavailable territory. Existing
            # territory keeps its owner when a larger enclosure contains it.
            for region in capture.captured_regions:
                for cell in region:
                    if self.territory[cell] == EMPTY:
                        self.territory[cell] = player

            # Add newly captured opponent dots to the player's score
            self.score[player] += len(capture.captured_dots)

            # Rebuild UnionFind because captured dots must no longer
            # participate in active connected groups
            self.groups = rebuild_groups(self.board, self.territory)

        return list(capture.captured_dots)

    def render(self, colorize=False):
        """Render this game board."""
        return render_board(
            self.board,
            territory=self.territory,
            colorize=colorize,
        )
