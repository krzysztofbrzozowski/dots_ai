"""Tests for the Dots enclosure and capture detection

Run directly from the game folder
"""

import numpy as np

from enclosure import (
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    DotsGame,
    UnionFind,
    _could_have_closed_loop,
    detect_capture,
    detect_surrounded_move_info,
    find_candidate_regions,
    find_enclosed_regions,
    flood_fill_region,
    rebuild_groups,
    render_board,
)


def _board(rows, cols, cells):
    """Create an empty board and place the provided dots on it"""
    board = np.zeros((rows, cols), dtype=int)
    for value, (row, col) in cells:
        board[row, col] = value
    return board


def _territory(rows, cols):
    """Create an empty territory map"""
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


def test_connecting_separate_groups_does_not_close_cycle():
    last_move = (2, 2)
    board = _board(
        5,
        5,
        [
            (PLAYER_1, (2, 1)),
            (PLAYER_1, last_move),
            (PLAYER_1, (2, 3)),
        ],
    )
    territory = _territory(5, 5)
    groups = UnionFind()
    groups.add((2, 1))
    groups.add((2, 3))

    assert not _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
        groups,
    )
    assert detect_capture(
        board,
        last_move,
        PLAYER_1,
        territory=territory,
        groups=groups,
    ) == []


def test_active_neighbors_connected_before_move_form_cycle_candidate():
    last_move = (2, 2)
    active_path = [
        (2, 1),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    board = _board(
        5,
        5,
        [(PLAYER_1, cell) for cell in active_path + [last_move]],
    )
    territory = _territory(5, 5)
    pre_move_board = board.copy()
    pre_move_board[last_move] = EMPTY
    groups = rebuild_groups(pre_move_board, territory)

    assert _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
        groups,
    )


def test_inactive_bridge_does_not_connect_active_neighbors():
    last_move = (2, 2)
    left_neighbor = (2, 1)
    right_neighbor = (2, 3)
    inactive_bridge = (1, 2)
    board = _board(
        5,
        5,
        [
            (PLAYER_1, left_neighbor),
            (PLAYER_1, inactive_bridge),
            (PLAYER_1, right_neighbor),
            (PLAYER_1, last_move),
        ],
    )
    territory = _territory(5, 5)
    territory[inactive_bridge] = PLAYER_2
    groups = UnionFind()
    groups.add(left_neighbor)
    groups.add(right_neighbor)

    assert not _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
        groups,
    )


def test_inactive_adjacent_player_dot_is_not_cycle_neighbor():
    last_move = (2, 2)
    active_neighbor = (2, 1)
    inactive_neighbor = (2, 3)
    board = _board(
        5,
        5,
        [
            (PLAYER_1, active_neighbor),
            (PLAYER_1, inactive_neighbor),
            (PLAYER_1, last_move),
        ],
    )
    territory = _territory(5, 5)
    territory[inactive_neighbor] = PLAYER_2
    groups = UnionFind()
    groups.add(active_neighbor)

    assert not _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
        groups,
    )


def test_rebuilt_pre_move_groups_ignore_inactive_bridge():
    last_move = (2, 2)
    inactive_bridge = (1, 2)
    board = _board(
        5,
        5,
        [
            (PLAYER_1, (2, 1)),
            (PLAYER_1, inactive_bridge),
            (PLAYER_1, (2, 3)),
            (PLAYER_1, last_move),
        ],
    )
    territory = _territory(5, 5)
    territory[inactive_bridge] = PLAYER_2

    assert not _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
    )


def test_provided_groups_remain_authoritative_with_visible_inactive_dots():
    class TrackingGroups:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.find_calls = []

        def find(self, cell):
            self.find_calls.append(cell)
            return self.wrapped.find(cell)

    last_move = (2, 2)
    inactive_dot = (4, 4)
    active_path = [
        (2, 1),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    board = _board(
        5,
        5,
        [(PLAYER_1, cell) for cell in active_path + [last_move, inactive_dot]],
    )
    territory = _territory(5, 5)
    territory[inactive_dot] = PLAYER_2
    pre_move_board = board.copy()
    pre_move_board[last_move] = EMPTY
    groups = TrackingGroups(rebuild_groups(pre_move_board, territory))

    assert _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
        groups,
    )
    assert groups.find_calls


def test_empty_loop_is_not_capture():
    # A closed loop containing no opponent dot does not create captured territory
    ring = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    board = _board(3, 3, [(PLAYER_1, cell) for cell in ring])
    territory = _territory(3, 3)

    assert _could_have_closed_loop(
        board,
        territory,
        (1, 2),
        PLAYER_1,
    )
    assert detect_capture(board, (1, 2), PLAYER_1, territory=territory) == []
    assert find_enclosed_regions(board, territory, (1, 2), PLAYER_1) == []


def test_dot_entering_an_existing_empty_enclosure_is_captured():
    game = DotsGame(5, 5)
    center = (2, 2)
    diamond = [(1, 2), (2, 1), (2, 3), (3, 2)]

    for cell in diamond:
        assert game.place_dot(*cell, PLAYER_1) == []

    # The empty loop remains unclaimed and may still be entered.
    assert game.territory[center] == EMPTY
    assert game.is_legal_move(*center)

    # Entering it triggers a capture by the player who owns the old boundary.
    assert game.place_dot(*center, PLAYER_2) == []
    assert game.board[center] == PLAYER_2
    assert game.territory[center] == PLAYER_1
    assert game.score[PLAYER_1] == 1
    assert game.score[PLAYER_2] == 0
    assert game.last_captured_dots == [center]
    assert game.last_capture_player == PLAYER_1
    assert center not in game.groups


def test_surrounded_move_diagnostic_finds_the_entering_dot():
    center = (2, 2)
    board = _board(
        5,
        5,
        [
            (PLAYER_1, (1, 2)),
            (PLAYER_1, (2, 1)),
            (PLAYER_1, (2, 3)),
            (PLAYER_1, (3, 2)),
            (PLAYER_2, center),
        ],
    )

    capture = detect_surrounded_move_info(board, center, PLAYER_2)

    assert capture.captured_dots == (center,)
    assert capture.captured_regions == ((center,),)


def test_inactive_boundary_dots_do_not_capture_an_entering_opponent():
    game = DotsGame(5, 5)
    center = (2, 2)
    diamond = [(1, 2), (2, 1), (2, 3), (3, 2)]
    inactive_boundary = [(1, 2), (2, 3)]

    for cell in diamond:
        assert game.place_dot(*cell, PLAYER_1) == []

    # Captured dots stay visible, but they are no longer active enclosure walls.
    for cell in inactive_boundary:
        game.territory[cell] = PLAYER_2
    game.groups = rebuild_groups(game.board, game.territory)

    assert game.place_dot(*center, PLAYER_2) == []
    assert game.board[center] == PLAYER_2
    assert game.territory[center] == EMPTY
    assert game.score[PLAYER_1] == 0
    assert game.last_captured_dots == []
    assert game.last_capture_player is None
    assert all(game.board[cell] == PLAYER_1 for cell in inactive_boundary)


def test_dot_entering_an_open_shape_is_not_captured():
    game = DotsGame(5, 5)
    center = (2, 2)

    for cell in [(1, 2), (2, 1), (2, 3)]:
        assert game.place_dot(*cell, PLAYER_1) == []

    assert game.place_dot(*center, PLAYER_2) == []
    assert game.territory[center] == EMPTY
    assert game.score[PLAYER_1] == 0
    assert game.last_captured_dots == []
    assert game.last_capture_player is None


def test_square_enclosure_captures_one():
    ring = [
        (row, col)
        for row in range(4)
        for col in range(4)
        if row in (0, 3) or col in (0, 3)
    ]
    board = _board(
        4,
        4,
        [(PLAYER_1, cell) for cell in ring] + [(PLAYER_2, (2, 2))],
    )
    territory = _territory(4, 4)
    visited = np.zeros(board.shape, dtype=bool)

    assert _could_have_closed_loop(
        board,
        territory,
        (2, 3),
        PLAYER_1,
    )
    region, reaches_edge, opponent_cells = flood_fill_region(
        board,
        (2, 2),
        PLAYER_1,
        visited,
    )

    assert set(region) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert reaches_edge is False
    assert opponent_cells == [(2, 2)]
    assert detect_capture(
        board,
        (2, 3),
        PLAYER_1,
        territory=territory,
    ) == [(2, 2)]


def test_open_region_reaching_board_edge_is_not_captured():
    # The opponent's region can escape through the open left side
    wall_with_opening = [
        (0, 0), (0, 1), (0, 2), (0, 3),
                                    (1, 3),
                                    (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]
    board = _board(
        4,
        4,
        [(PLAYER_1, cell) for cell in wall_with_opening]
        + [(PLAYER_2, (2, 2))],
    )
    territory = _territory(4, 4)
    visited = np.zeros(board.shape, dtype=bool)

    assert _could_have_closed_loop(
        board,
        territory,
        (2, 3),
        PLAYER_1,
    )
    region, reaches_edge, opponent_cells = flood_fill_region(
        board,
        (2, 2),
        PLAYER_1,
        visited,
    )

    assert (2, 0) in region
    assert reaches_edge is True
    assert opponent_cells == [(2, 2)]
    assert detect_capture(
        board,
        (2, 3),
        PLAYER_1,
        territory=territory,
    ) == []


def test_existing_territory_does_not_complete_active_dot_wall():
    # The opponent is bounded on three sides by active dots, while the fourth
    # side only appears closed because flood fill encounters old territory
    board = _board(
        5,
        5,
        [
            (PLAYER_1, (1, 2)),
            (PLAYER_1, (2, 1)),
            (PLAYER_1, (3, 2)),
            (PLAYER_1, (3, 3)),
            (PLAYER_2, (2, 2)),
        ],
    )
    territory = _territory(5, 5)
    territory[2, 3] = PLAYER_1
    visited = np.zeros(board.shape, dtype=bool)

    region, reaches_edge, opponent_cells = flood_fill_region(
        board,
        (2, 2),
        PLAYER_1,
        visited,
    )

    assert (2, 3) in region
    assert (2, 4) in region
    assert reaches_edge is True
    assert opponent_cells == [(2, 2)]
    assert find_enclosed_regions(
        board,
        territory,
        (3, 2),
        PLAYER_1,
    ) == []
    assert detect_capture(
        board,
        (3, 2),
        PLAYER_1,
        territory=territory,
    ) == []


def test_visible_inactive_dot_opens_an_otherwise_closed_region():
    ring = [
        (row, col)
        for row in range(5)
        for col in range(5)
        if row in (0, 4) or col in (0, 4)
    ]
    last_move = (0, 3)
    inactive_wall = (0, 1)
    opponent_dot = (2, 2)
    board = _board(
        5,
        5,
        [(PLAYER_1, cell) for cell in ring]
        + [(PLAYER_2, opponent_dot)],
    )
    territory = _territory(5, 5)
    territory[inactive_wall] = PLAYER_2
    visited = np.zeros(board.shape, dtype=bool)

    # The active graph still has a cycle candidate, so flood-fill geometry is
    # responsible for rejecting this apparent enclosure.
    assert _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
    )
    region, reaches_edge, opponent_cells = flood_fill_region(
        board,
        opponent_dot,
        PLAYER_1,
        visited,
        territory=territory,
    )

    assert inactive_wall in region
    assert reaches_edge is True
    assert opponent_cells == [opponent_dot]
    assert find_enclosed_regions(
        board,
        territory,
        last_move,
        PLAYER_1,
    ) == []
    assert detect_capture(
        board,
        last_move,
        PLAYER_1,
        territory=territory,
    ) == []
    assert board[inactive_wall] == PLAYER_1


def test_inactive_same_player_neighbor_is_a_flood_fill_candidate():
    last_move = (2, 2)
    inactive_neighbor = (2, 3)
    board = _board(
        5,
        5,
        [
            (PLAYER_1, last_move),
            (PLAYER_1, inactive_neighbor),
        ],
    )
    territory = _territory(5, 5)
    territory[inactive_neighbor] = PLAYER_2

    assert inactive_neighbor in find_candidate_regions(
        board,
        territory,
        last_move,
        PLAYER_1,
    )


def test_previously_captured_opponent_dot_is_not_reported_again():
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
    territory = _territory(3, 3)
    territory[1, 1] = PLAYER_1

    assert detect_capture(
        board,
        (1, 2),
        PLAYER_1,
        territory=territory,
    ) == []


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


def test_last_move_must_create_the_enclosure():
    # The diamond already encloses the opponent
    # The new diagonal dot has two same-player neighbors
    # but the enclosed region is not orthogonally adjacent to last_move
    diamond = [(1, 2), (2, 1), (2, 3), (3, 2)]
    last_move = (1, 1)
    board = _board(
        5,
        5,
        [(PLAYER_1, cell) for cell in diamond + [last_move]]
        + [(PLAYER_2, (2, 2))],
    )
    territory = _territory(5, 5)

    assert _could_have_closed_loop(
        board,
        territory,
        last_move,
        PLAYER_1,
    )
    assert find_enclosed_regions(
        board,
        territory,
        last_move,
        PLAYER_1,
    ) == []
    assert detect_capture(
        board,
        last_move,
        PLAYER_1,
        territory=territory,
    ) == []


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


def test_new_capture_preserves_existing_territory_owner():
    game = DotsGame(5, 5)
    assert game.place_dot(2, 2, PLAYER_2) == []
    game.territory[1, 1] = PLAYER_2

    border = [
        (row, col)
        for row in range(5)
        for col in range(5)
        if row in (0, 4) or col in (0, 4)
    ]
    closing_move = (0, 2)
    border.remove(closing_move)

    for cell in border:
        assert game.place_dot(*cell, PLAYER_1) == []

    assert game.place_dot(*closing_move, PLAYER_1) == [(2, 2)]
    assert game.territory[1, 1] == PLAYER_2
    assert game.territory[2, 2] == PLAYER_1
    assert game.score[PLAYER_1] == 1


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
