"""Integration tests for immutable MCTS states and the read-only display."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread

import numpy as np

from GUI.presentation import move_message
from GUI.server import SnapshotStore, app, publish_state, snapshot_store
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

    best = MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        parallelism=3,
        random_seed=123,
    ).best_action(simulations_number=10)

    seeds = [args[1] for _, args in executor.submissions]
    assert len(executor.submissions) == 10
    assert len(set(seeds)) == 10
    assert root.n == 10
    assert sum(child.n for child in root.children) == 10
    assert best.action in DotsGame(3, 3).get_legal_actions()
    assert all(node.virtual_visits == 0 for node in walk_nodes(root))


def test_virtual_loss_spreads_a_batch_across_equivalent_children():
    executor = ImmediateExecutor()
    root = TwoPlayerMCTSNode(DotsGame(1, 2))

    MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        parallelism=2,
        random_seed=456,
    ).best_action(simulations_number=4)

    assert sorted(child.n for child in root.children) == [2, 2]


def test_parallel_search_releases_reservations_after_worker_failure():
    executor = ImmediateExecutor(fail_on_submission=2)
    root = TwoPlayerMCTSNode(DotsGame(2, 2))
    search = MonteCarloTreeSearch(
        root,
        rollout_executor=executor,
        parallelism=3,
    )

    try:
        search.best_action(simulations_number=3)
    except RuntimeError as error:
        assert str(error) == "simulated rollout failure"
    else:
        raise AssertionError("parallel rollout failure was not propagated")

    assert root.n == 0
    assert all(node.virtual_visits == 0 for node in walk_nodes(root))


def test_parallel_search_requires_an_executor_for_multiple_workers():
    try:
        MonteCarloTreeSearch(TwoPlayerMCTSNode(DotsGame(2, 2)), parallelism=2)
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


def test_published_snapshot_is_serialized_and_isolated():
    game = DotsGame(2, 2)
    published = publish_state(
        game,
        last_move=None,
        move_number=0,
        message="Initial state",
    )
    version = published["version"]

    game.board[0, 0] = PLAYER_1
    published["board"][0][1] = PLAYER_2
    stored = snapshot_store.read()

    assert stored["version"] == version
    assert stored["board"] == [[0, 0], [0, 0]]
    assert stored["legal_moves"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert stored["game_over"] is False
    assert stored["winner"] is None
    assert stored["capture_player"] is None


def test_snapshot_and_message_credit_a_surrounded_move_to_the_opponent():
    game = DotsGame(5, 5)
    center = (2, 2)
    for cell in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        game.place_dot(*cell, PLAYER_1)
    game.place_dot(*center, PLAYER_2)

    published = publish_state(
        game,
        last_move=center,
        move_number=5,
        message=move_message(game, center, PLAYER_2),
    )

    assert published["capture_player"] == PLAYER_1
    assert published["last_captured_dots"] == [[2, 2]]
    assert "Player 1 captured 1 trapped dot" in published["message"]


def test_snapshot_store_is_thread_safe_and_versions_every_publish():
    store = SnapshotStore()

    with ThreadPoolExecutor(max_workers=8) as executor:
        published = list(executor.map(lambda value: store.publish({"value": value}), range(40)))

    assert sorted(snapshot["version"] for snapshot in published) == list(range(1, 41))
    latest = store.read()
    latest["value"] = "changed outside the store"
    assert store.read()["value"] != "changed outside the store"


def test_display_server_has_no_mutating_game_routes():
    methods_by_path = {
        route.path: route.methods
        for route in app.routes
        if hasattr(route, "methods")
    }

    assert methods_by_path["/api/state"] == {"GET"}
    assert "/api/move" not in methods_by_path
    assert "/api/reset" not in methods_by_path
    assert "/api/mcts/step" not in methods_by_path


def test_main_loop_publishes_the_action_selected_by_mcts():
    publications = []

    def collect(game, last_move, move_number, message):
        publications.append((game.copy(), last_move, move_number, message))

    final_state = run_mcts_game(
        DotsGame(1, 1),
        simulations_number=1,
        move_delay=0,
        publisher=collect,
        simulation_seconds=None,
    )

    assert final_state.game_result == 0
    assert len(publications) == 1
    published_state, action, move_number, message = publications[0]
    assert action == (0, 0)
    assert published_state.board[action] == PLAYER_1
    assert move_number == 1
    assert "Game over: Draw" in message


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
