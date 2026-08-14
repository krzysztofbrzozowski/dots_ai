"""Tests / examples for the Dots enclosure & capture detection.

Run directly:  python test_enclosure.py   (from the game/ directory)
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
    """cells: iterable of (value, (row, col)) placed on an empty board."""
    b = np.zeros((rows, cols), dtype=int)
    for value, (r, c) in cells:
        b[r, c] = value
    return b


def test_no_enclosure():
    # A straight line of dots encloses nothing.
    b = _board(5, 5, [(PLAYER_1, (2, 1)), (PLAYER_1, (2, 2)), (PLAYER_1, (2, 3))])
    assert detect_capture(b, (2, 3), PLAYER_1) == []


def test_square_enclosure_empty():
    # 3x3 ring of PLAYER_1 around an empty center: a real enclosure, but with
    # no opponent inside, so nothing is captured.
    ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    b = _board(3, 3, [(PLAYER_1, p) for p in ring])
    assert detect_capture(b, (1, 2), PLAYER_1) == []
    assert find_enclosed_regions(b, (1, 2), PLAYER_1) == [[(1, 1)]]


def test_square_enclosure_captures_one():
    ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    b = _board(3, 3, [(PLAYER_1, p) for p in ring] + [(PLAYER_2, (1, 1))])
    assert detect_capture(b, (1, 2), PLAYER_1) == [(1, 1)]


def test_enclosure_captures_multiple():
    # 5x5 border with three opponent dots inside.
    b = np.zeros((5, 5), dtype=int)
    for i in range(5):
        for p in [(0, i), (4, i), (i, 0), (i, 4)]:
            b[p] = PLAYER_1
    b[2, 2] = PLAYER_2
    b[2, 3] = PLAYER_2
    b[1, 2] = PLAYER_2
    captured = detect_capture(b, (4, 2), PLAYER_1)
    assert captured == [(1, 2), (2, 2), (2, 3)]


def test_diamond_enclosure():
    # Diagonal / diamond-shaped enclosure: the four dots are only pairwise
    # diagonal (8-connected), yet they seal the orthogonal center.
    diamond = [(0, 1), (1, 0), (1, 2), (2, 1)]
    b = _board(3, 3, [(PLAYER_1, p) for p in diamond] + [(PLAYER_2, (1, 1))])
    assert detect_capture(b, (2, 1), PLAYER_1) == [(1, 1)]


def test_cycle_but_no_capture():
    # A dot placed below the middle of a 3-dot row closes a *graph* cycle
    # (the row + the new dot form a triangle in the 8-connected group graph)
    # but geometrically encloses nothing, so no capture may be reported.
    b = _board(5, 5, [(PLAYER_1, (2, 1)), (PLAYER_1, (2, 2)),
                      (PLAYER_1, (2, 3)), (PLAYER_1, (3, 2))])
    assert detect_capture(b, (3, 2), PLAYER_1) == []


def test_enclosure_near_edge():
    # A 3x3 ring sitting in the top-left corner.  The ring's dots touch the
    # edge, but the *region* inside does not, so the opponent is captured.
    ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    b = _board(5, 5, [(PLAYER_1, p) for p in ring] + [(PLAYER_2, (1, 1))])
    assert detect_capture(b, (2, 1), PLAYER_1) == [(1, 1)]


def test_render_board():
    b = np.array([
        [0, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ])
    expected = "· ● · ·\n○ · ● ·\n· ○ · ·\n· · · ●"
    assert render_board(b) == expected


def test_incremental_game():
    # Build a square enclosure move by move through DotsGame; verify captures
    # only happen on the closing move and that the DSU is maintained correctly.
    game = DotsGame(4, 4)
    assert game.place_dot(2, 2, PLAYER_2) == []          # opponent dot inside
    for p in [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 2), (3, 3)]:
        assert game.place_dot(*p, PLAYER_1) == []
    # Final dot closes the ring and captures the opponent at (2, 2).
    assert game.place_dot(2, 3, PLAYER_1) == [(2, 2)]
    assert game.board[2, 2] == EMPTY


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
