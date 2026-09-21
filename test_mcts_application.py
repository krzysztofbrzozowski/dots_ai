"""Integration tests for immutable MCTS states and the shared live GUI."""

from concurrent.futures import Future
from threading import Thread
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from GUI.presentation import move_message
from GUI.server import LiveGameStore, app
from game.enclosure import EMPTY, PLAYER_1, PLAYER_2, DotsGame
from main_mcts import run_mcts_game, run_parallel_mcts_game
from mcts.enclosure import MonteCarloTreeSearch, TwoPlayerMCTSNode


class ImmediateExecutor:
    """Executor test double that records submissions and resolves inline."""

    def __init__(self, fail_on_submission=None):
        self.fail_on_submission = fail_on_submission
        self.submissions = []

    def submit(self, function, *args):
        future = Future()
        self.submissions.append((function, args))

        if len(self.submissions) == self.fail_on_submission:
            future.set_exception(RuntimeError("simulated rollout failure"))
            return future

        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


def walk_nodes(root):
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(node.children)


def test_move_returns_independent_state_and_switches_player():
    game = DotsGame(3, 3)

    moved = game.move((1, 1))

    assert game.board[1, 1] == EMPTY
    assert game.next_to_move == PLAYER_1
    assert game.last_move is None
    assert moved.board[1, 1] == PLAYER_1
    assert moved.next_to_move == PLAYER_2
    assert moved.last_move == (1, 1)
    assert moved.groups is not game.groups
    assert not np.shares_memory(moved.board, game.board)
    assert not np.shares_memory(moved.territory, game.territory)

    moved.score[PLAYER_1] = 7
    moved.groups.add((0, 0))
    assert game.score[PLAYER_1] == 0
    assert (0, 0) not in game.groups


def test_move_uses_the_player_whose_turn_it_is():
    first = DotsGame(2, 2).move((0, 0))
    second = first.move((0, 1))

    assert first.board[0, 1] == EMPTY
    assert second.board[0, 0] == PLAYER_1
    assert second.board[0, 1] == PLAYER_2
    assert second.next_to_move == PLAYER_1


def test_move_rejects_malformed_or_non_integer_actions():
    game = DotsGame(3, 3)
    invalid_actions = [
        ((1,), ValueError),
        ((1, 2, 3), ValueError),
        ("12", ValueError),
        ((1.9, 2), TypeError),
        ((1, 2.1), TypeError),
        ((True, 2), TypeError),
    ]

    for action, expected_error in invalid_actions:
        try:
            game.move(action)
        except expected_error:
            pass
        else:
            raise AssertionError(
                f"move({action!r}) did not raise {expected_error.__name__}"
            )

    moved = game.move((np.int64(1), np.int32(2)))
    assert moved.board[1, 2] == PLAYER_1
    assert np.count_nonzero(game.board) == 0


def test_get_legal_actions_matches_the_engine_legal_moves():
    game = DotsGame(2, 2).move((0, 0))

    assert game.get_legal_actions() == game.legal_moves()
    assert (0, 0) not in game.get_legal_actions()
    assert set(game.get_legal_actions()) == {(0, 1), (1, 0), (1, 1)}


def test_expanded_node_records_the_action_that_created_its_state():
    game = DotsGame(3, 3)
    child = TwoPlayerMCTSNode(game).expand()

    changed_cells = np.argwhere(child.state.board != game.board)
    assert changed_cells.shape == (1, 2)
    assert tuple(int(value) for value in changed_cells[0]) == child.action
    assert child.parent is not None


def test_parallel_search_batches_exact_simulation_count():
    executor = ImmediateExecutor()
    root = TwoPlayerMCTSNode(DotsGame(3, 3))

    search = MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        rollout_batch_size=3,
        random_seed=123,
    )
    best = search.best_action(simulations_number=10)

    seeds = [args[1] for _, args in executor.submissions]
    stats = search.last_search_stats
    assert len(executor.submissions) == 10
    assert len(set(seeds)) == 10
    assert root.n == 10
    assert sum(child.n for child in root.children) == 10
    assert best.action in DotsGame(3, 3).get_legal_actions()
    assert all(node.virtual_visits == 0 for node in walk_nodes(root))
    assert stats.completed_rollouts == 10
    assert stats.completed_batches == 4
    assert stats.elapsed_seconds >= 0
    assert stats.rollouts_per_second >= 0


def test_sequential_search_records_completed_rollouts_without_batches():
    search = MonteCarloTreeSearch(TwoPlayerMCTSNode(DotsGame(2, 2)))

    search.best_action(simulations_number=3)

    stats = search.last_search_stats
    assert stats.completed_rollouts == 3
    assert stats.completed_batches == 0
    assert stats.elapsed_seconds >= 0


def test_time_based_parallel_search_records_completed_batch_statistics():
    executor = ImmediateExecutor()
    root = TwoPlayerMCTSNode(DotsGame(1, 2))
    search = MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        rollout_batch_size=2,
        random_seed=789,
    )

    clock_values = [100.0, 100.0, 100.0, 100.3, 100.3]
    with patch("mcts.search.time.monotonic", side_effect=clock_values):
        search.best_action(total_simulation_seconds=0.25)

    stats = search.last_search_stats
    assert stats.completed_rollouts == 2
    assert stats.completed_batches == 1
    assert abs(stats.elapsed_seconds - 0.3) < 1e-9
    assert abs(stats.rollouts_per_second - (2 / 0.3)) < 1e-9


def test_virtual_loss_spreads_a_batch_across_equivalent_children():
    executor = ImmediateExecutor()
    root = TwoPlayerMCTSNode(DotsGame(1, 2))

    MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        rollout_batch_size=2,
        random_seed=456,
    ).best_action(simulations_number=4)

    assert sorted(child.n for child in root.children) == [2, 2]


def test_parallel_search_releases_reservations_after_worker_failure():
    executor = ImmediateExecutor(fail_on_submission=2)
    root = TwoPlayerMCTSNode(DotsGame(2, 2))
    search = MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        rollout_batch_size=3,
    )

    try:
        search.best_action(simulations_number=3)
    except RuntimeError as error:
        assert str(error) == "simulated rollout failure"
    else:
        raise AssertionError("parallel rollout failure was not propagated")

    assert root.n == 0
    assert all(node.virtual_visits == 0 for node in walk_nodes(root))


def test_parallel_search_requires_an_executor_for_multiple_rollouts_per_batch():
    try:
        MonteCarloTreeSearch(
            TwoPlayerMCTSNode(DotsGame(2, 2)),
            rollout_batch_size=2,
        )
    except ValueError as error:
        assert "requires a rollout executor" in str(error)
    else:
        raise AssertionError("parallel search accepted no rollout executor")


def test_process_pool_lifecycle_inside_game_thread():
    results = []
    errors = []

    def run_game():
        try:
            results.append(
                run_parallel_mcts_game(
                    DotsGame(2, 2),
                    simulations_number=4,
                    simulation_seconds=None,
                    move_delay=0,
                    publisher=lambda *args, **kwargs: None,
                    workers=2,
                )
            )
        except BaseException as error:
            errors.append(error)

    game_thread = Thread(target=run_game, name="parallel-mcts-test")
    game_thread.start()
    game_thread.join(timeout=15)

    assert not game_thread.is_alive()
    assert not errors
    assert len(results) == 1
    assert results[0].game_result is not None


def test_live_store_serializes_search_timeline_and_isolates_reads():
    game = DotsGame(2, 2)
    root = TwoPlayerMCTSNode(game)
    search = MonteCarloTreeSearch(root, random_seed=123)
    selected = search.best_action(simulations_number=4)
    store = LiveGameStore()
    initial = store.start(
        game,
        simulation_seconds=None,
        simulations_number=4,
        rollout_batch_size=1,
    )
    published = store.publish(
        state=game,
        root=root,
        selected_action=selected.action,
        search_stats=search.last_search_stats,
        resulting_state=selected.state,
        move_number=1,
        message="Move complete",
    )

    initial["current_frame"]["board"][0][0] = PLAYER_2
    published["analysis"]["timeline"][0]["selected_action"][0] = 99
    stored = store.read()

    assert stored["revision"] == initial["revision"] + 1
    assert stored["analysis"]["frame_count"] == 1
    assert stored["analysis"]["timeline"][0]["selected_action"] == list(
        selected.action
    )
    assert store.frame(0)["board"] == [[0, 0], [0, 0]]
    row, col = selected.action
    assert stored["current_frame"]["board"][row][col] == PLAYER_1
    assert stored["status"] == "searching"


def test_message_credits_a_surrounded_move_to_the_opponent():
    game = DotsGame(5, 5)
    center = (2, 2)
    for cell in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        game.place_dot(*cell, PLAYER_1)
    game.place_dot(*center, PLAYER_2)

    message = move_message(game, center, PLAYER_2)

    assert game.last_capture_player == PLAYER_1
    assert center in game.last_captured_dots
    assert "Player 1 captured 1 trapped dot" in message


def test_live_server_uses_shared_gui_and_has_no_mutating_game_routes():
    client = TestClient(app)
    methods_by_path = {
        route.path: route.methods
        for route in app.routes
        if hasattr(route, "methods")
    }

    assert methods_by_path["/api/runtime"] == {"GET"}
    assert methods_by_path["/api/live"] == {"GET"}
    assert methods_by_path["/api/live/frames/{frame_index}"] == {"GET"}
    assert "/api/move" not in methods_by_path
    assert "/api/reset" not in methods_by_path
    assert "/api/mcts/step" not in methods_by_path
    assert client.get("/api/runtime").json() == {"mode": "live"}
    assert "Decision timeline" in client.get("/").text
    assert "refreshLiveGame" in client.get("/assets/analysis.js").text
    assert client.get("/game.js").status_code == 404
    assert client.get("/styles.css").status_code == 404


def test_main_loop_publishes_the_action_selected_by_mcts():
    publications = []

    def collect(**publication):
        publications.append(publication)

    final_state = run_mcts_game(
        DotsGame(1, 1),
        simulations_number=1,
        move_delay=0,
        publisher=collect,
        simulation_seconds=None,
    )

    assert final_state.game_result == 0
    assert len(publications) == 1
    publication = publications[0]
    assert publication["selected_action"] == (0, 0)
    assert publication["state"].board[0, 0] == EMPTY
    assert publication["resulting_state"].board[0, 0] == PLAYER_1
    assert publication["move_number"] == 1
    assert "Game over: Draw" in publication["message"]


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]

    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print(f"\nAll {len(tests)} MCTS application tests passed")
