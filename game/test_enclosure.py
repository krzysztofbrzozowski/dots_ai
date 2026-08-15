"""Tests for the Dots enclosure and capture detection.

Run directly:
    python test_enclosure.py
"""

import numpy as np

from enclosure import (
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    DotsGame,
    detect_capture,
    find_enclosed_regions,
    render_board,
)


def _board(rows, cols, cells):
    """Create an empty board and place the provided dots on it."""
    board = np.zeros((rows, cols), dtype=int)
    for value, (row, col) in cells:
        board[row, col] = value
    return board


def _territory(rows, cols):
    """Create an empty territory map."""
    return np.zeros((rows, cols), dtype=int)


def test_no_enclosure():
    # A straight line of dots encloses nothing
    board = _board(
        5,
        5,
        [
            (PLAYER_1, (2, 1)),
            (PLAYER_1, (2, 2)),
            (PLAYER_1, (2, 3)),
        ],
    )

    assert detect_capture(board, (2, 3), PLAYER_1) == []


def test_empty_loop_is_not_capture():
    # A closed loop containing no opponent dot does not create captured territory
    ring = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    board = _board(3, 3, [(PLAYER_1, cell) for cell in ring])
    territory = _territory(3, 3)

    assert detect_capture(board, (1, 2), PLAYER_1, territory=territory) == []
    assert find_enclosed_regions(board, territory, (1, 2), PLAYER_1) == []


def test_square_enclosure_captures_one():
    ring = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    board = _board(
        3,
        3,
        [(PLAYER_1, cell) for cell in ring] + [(PLAYER_2, (1, 1))],
    )

    assert detect_capture(board, (1, 2), PLAYER_1) == [(1, 1)]


def test_enclosure_captures_multiple():
    # A 5x5 border encloses three opponent dots
    board = np.zeros((5, 5), dtype=int)

    for i in range(5):
        for cell in [(0, i), (4, i), (i, 0), (i, 4)]:
            board[cell] = PLAYER_1

    board[1, 2] = PLAYER_2
    board[2, 2] = PLAYER_2
    board[2, 3] = PLAYER_2

    captured = detect_capture(board, (4, 2), PLAYER_1)

    assert captured == [(1, 2), (2, 2), (2, 3)]


def test_diamond_enclosure():
    # Diagonal player connectivity closes the orthogonal center
    diamond = [(0, 1), (1, 0), (1, 2), (2, 1)]
    board = _board(
        3,
        3,
        [(PLAYER_1, cell) for cell in diamond] + [(PLAYER_2, (1, 1))],
    )

    assert detect_capture(board, (2, 1), PLAYER_1) == [(1, 1)]


def test_cycle_but_no_capture():
    # A graph cycle alone is not enough if no geometric enclosure is created
    board = _board(
        5,
        5,
        [
            (PLAYER_1, (2, 1)),
            (PLAYER_1, (2, 2)),
            (PLAYER_1, (2, 3)),
            (PLAYER_1, (3, 2)),
        ],
    )

    assert detect_capture(board, (3, 2), PLAYER_1) == []


def test_enclosure_near_edge():
    # The boundary dots may touch the board edge while the enclosed region does not
    ring = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    board = _board(
        5,
        5,
        [(PLAYER_1, cell) for cell in ring] + [(PLAYER_2, (1, 1))],
    )

    assert detect_capture(board, (2, 1), PLAYER_1) == [(1, 1)]


def test_render_board():
    board = np.array([
        [0, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ])

    expected = "· ● · ·\n○ · ● ·\n· ○ · ·\n· · · ●"

    assert render_board(board) == expected


def test_incremental_capture_keeps_captured_dot():
    # Build an enclosure move by move and capture PLAYER_2 at (2, 2)
    game = DotsGame(4, 4)

    assert game.place_dot(2, 2, PLAYER_2) == []

    for cell in [
        (1, 1), (1, 2), (1, 3),
        (2, 1),
        (3, 1), (3, 2), (3, 3),
    ]:
        assert game.place_dot(*cell, PLAYER_1) == []

    assert game.place_dot(2, 3, PLAYER_1) == [(2, 2)]

    # Captured dots remain on the board instead of becoming EMPTY
    assert game.board[2, 2] == PLAYER_2

    # The captured region is owned by PLAYER_1 and cannot be played again
    assert game.territory[2, 2] == PLAYER_1
    assert not game.is_legal_move(2, 2)

    # One opponent dot was captured
    assert game.score[PLAYER_1] == 1


def test_entire_captured_region_becomes_unplayable():
    # A larger enclosure contains one opponent dot and several empty intersections
    game = DotsGame(5, 5)

    assert game.place_dot(2, 2, PLAYER_2) == []

    border = []
    for i in range(5):
        border.extend([(0, i), (4, i), (i, 0), (i, 4)])

    # Remove duplicates while preserving order
    border = list(dict.fromkeys(border))

    # Leave a real opening in the top edge until the final move
    # so the interior can still reach the outside before closure
    closing_move = (0, 2)
    border.remove(closing_move)

    for cell in border:
        assert game.place_dot(*cell, PLAYER_1) == []

    assert game.place_dot(*closing_move, PLAYER_1) == [(2, 2)]

    # The whole 3x3 interior is captured territory
    for row in range(1, 4):
        for col in range(1, 4):
            assert game.territory[row, col] == PLAYER_1
            assert not game.is_legal_move(row, col)

    # The captured opponent dot is still present on the board
    assert game.board[2, 2] == PLAYER_2

    # Empty intersections inside captured territory remain EMPTY in board,
    # but they are still illegal because territory marks them as blocked
    assert game.board[1, 1] == EMPTY
    assert not game.is_legal_move(1, 1)


def test_render_captured_empty_territory():
    board = np.zeros((3, 3), dtype=int)
    territory = np.zeros((3, 3), dtype=int)
    territory[1, 1] = PLAYER_1

    expected = "· · ·\n· × ·\n· · ·"

    assert render_board(board, territory=territory) == expected


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]

    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print(f"\nAll {len(tests)} tests passed")
