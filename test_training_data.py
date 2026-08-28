"""Tests for action-aligned MCTS self-play training records."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from game.enclosure import PLAYER_1, PLAYER_2, DotsGame
from main_mcts import run_mcts_game
from mcts.enclosure import SearchStats, TwoPlayerMCTSNode
from training import SCHEMA_VERSION, SelfPlayTrajectory, load_self_play_game


def test_raw_q_and_n_are_aligned_by_child_action():
    game = DotsGame(2, 2)
    root = TwoPlayerMCTSNode(game)
    children = [root.expand() for _ in range(4)]
    results_by_action = {
        (1, 1): (3, 1),
        (1, 0): (1, 4),
        (0, 1): (1, 1),
        (0, 0): (1, 0),
    }
    for child in children:
        player_1_wins, player_2_wins = results_by_action[child.action]
        child._results[PLAYER_1] = player_1_wins
        child._results[PLAYER_2] = player_2_wins
        child._number_of_visits = player_1_wins + player_2_wins

    trajectory = SelfPlayTrajectory()
    trajectory.record_search(
        state=game,
        root=root,
        selected_action=(0, 0),
        search_stats=SearchStats(
            completed_rollouts=12,
            completed_batches=0,
            elapsed_seconds=0.25,
        ),
    )

    # Recording must own an isolated snapshot of the pre-move state.
    game.board[0, 0] = PLAYER_2

    with TemporaryDirectory() as directory:
        path = trajectory.save(directory, final_result=PLAYER_1)
        stored = load_self_play_game(path)

        assert path.parent.name == "2x2"
        assert list(path.parent.glob("*.tmp")) == []
        assert int(stored["schema_version"]) == SCHEMA_VERSION
        assert stored["boards"].shape == (1, 2, 2)
        assert np.count_nonzero(stored["boards"]) == 0
        assert stored["q_values"][0].tolist() == [[1.0, 0.0], [-3.0, 2.0]]
        assert stored["visit_counts"][0].tolist() == [[1, 2], [5, 4]]
        assert stored["legal_masks"][0].tolist() == [[1, 1], [1, 1]]
        assert stored["selected_actions"].tolist() == [[0, 0]]
        assert stored["completed_rollouts"].tolist() == [12]
        assert stored["q_perspective"].item() == "player_to_move"
        assert stored["final_result_perspective"].item() == "player_1"
        assert int(stored["final_result"]) == PLAYER_1


def test_absolute_player_state_is_preserved_for_later_loader_conversion():
    game = DotsGame(1, 2).move((0, 0))
    assert game.next_to_move == PLAYER_2
    root = TwoPlayerMCTSNode(game)
    child = root.expand()
    child._results[PLAYER_1] = 1
    child._number_of_visits = 1

    trajectory = SelfPlayTrajectory()
    trajectory.record_search(
        state=game,
        root=root,
        selected_action=child.action,
        search_stats=SearchStats(1, 0, 0.01),
    )

    with TemporaryDirectory() as directory:
        stored = load_self_play_game(
            trajectory.save(directory, final_result=PLAYER_1)
        )

    assert stored["boards"][0].tolist() == [[PLAYER_1, 0]]
    assert stored["next_players"].tolist() == [PLAYER_2]
    assert int(stored["final_result"]) == PLAYER_1
    # A future canonical loader can derive a loss for the player to move.
    z_current_player = int(stored["final_result"]) * stored["next_players"]
    assert z_current_player.tolist() == [PLAYER_2]


def test_complete_game_writes_one_npz_with_only_the_played_trajectory():
    with TemporaryDirectory() as directory:
        final_state = run_mcts_game(
            DotsGame(1, 2),
            simulations_number=4,
            simulation_seconds=None,
            move_delay=0,
            publisher=lambda *args, **kwargs: None,
            training_data_directory=directory,
        )

        files = list((Path(directory) / "1x2").glob("*.npz"))
        assert len(files) == 1
        stored = load_self_play_game(files[0])

    assert final_state.game_result == 0
    assert stored["boards"].shape == (2, 1, 2)
    assert stored["territories"].shape == (2, 1, 2)
    assert stored["next_players"].tolist() == [PLAYER_1, PLAYER_2]
    assert np.count_nonzero(stored["boards"][0]) == 0
    first_action = tuple(stored["selected_actions"][0])
    assert stored["boards"][1][first_action] == PLAYER_1
    assert stored["visit_counts"].sum(axis=(1, 2)).tolist() == [4, 4]
    assert stored["completed_rollouts"].tolist() == [4, 4]
    assert stored["legal_masks"].sum(axis=(1, 2)).tolist() == [2, 1]
    assert int(stored["final_result"]) == 0
    assert stored["search_budget_type"].item() == "simulations"
    assert int(stored["requested_simulations"]) == 4
    assert np.isnan(stored["requested_simulation_seconds"])


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} training-data tests passed")
