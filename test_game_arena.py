"""Tests for independently configured MCTS arena players."""

from concurrent.futures import Future
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from _game_arena import (
    ArenaConfig,
    PlayerMCTSConfig,
    SearchMode,
    run_arena,
)
from analysis import load_analysis_path
from game.enclosure import PLAYER_1, PLAYER_2
from training import load_self_play_game


class ImmediateExecutor:
    """Process-pool test double that resolves rollout tasks inline."""

    instances = []

    def __init__(self, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submissions = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        return False

    def submit(self, function, *args):
        self.submissions.append((function, args))
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


def test_config_selects_independent_player_settings_and_first_player():
    player_1 = PlayerMCTSConfig.serial(simulation_seconds=0.01)
    player_2 = PlayerMCTSConfig.parallel(
        simulation_seconds=0.02,
        workers=3,
    )
    config = ArenaConfig(
        rows=3,
        cols=4,
        next_to_move=PLAYER_2,
        player_1=player_1,
        player_2=player_2,
    )

    assert config.next_to_move == PLAYER_2
    assert config.player_config(PLAYER_1) is player_1
    assert config.player_config(PLAYER_2) is player_2


def test_arena_honors_starting_player_on_a_serial_game():
    ImmediateExecutor.instances.clear()
    seen_moves = []
    config = ArenaConfig(
        rows=1,
        cols=1,
        next_to_move=PLAYER_2,
        player_1=PlayerMCTSConfig.serial(0.001),
        player_2=PlayerMCTSConfig.serial(0.001),
    )

    result = run_arena(
        config,
        on_move=lambda _state, move: seen_moves.append(move),
        training_data_directory=None,
    )

    assert result.final_result == 0
    assert len(result.moves) == 1
    assert seen_moves[0].player == PLAYER_2
    assert seen_moves[0].search_mode is SearchMode.SERIAL
    assert seen_moves[0].search_stats.completed_batches == 0
    assert ImmediateExecutor.instances == []


def test_arena_switches_between_serial_and_parallel_player_searches():
    ImmediateExecutor.instances.clear()
    config = ArenaConfig(
        rows=1,
        cols=2,
        next_to_move=PLAYER_1,
        player_1=PlayerMCTSConfig.serial(0.001),
        player_2=PlayerMCTSConfig.parallel(0.002, workers=2),
    )

    result = run_arena(
        config,
        executor_factory=ImmediateExecutor,
        training_data_directory=None,
    )

    assert [move.player for move in result.moves] == [PLAYER_1, PLAYER_2]
    assert [move.search_mode for move in result.moves] == [
        SearchMode.SERIAL,
        SearchMode.PARALLEL,
    ]
    assert result.moves[0].requested_simulation_seconds == 0.001
    assert result.moves[1].requested_simulation_seconds == 0.002
    assert result.moves[0].search_stats.completed_batches == 0
    assert result.moves[1].search_stats.completed_batches > 0
    assert len(ImmediateExecutor.instances) == 1
    assert ImmediateExecutor.instances[0].max_workers == 2


def test_arena_also_supports_parallel_player_1_against_serial_player_2():
    ImmediateExecutor.instances.clear()
    config = ArenaConfig(
        rows=1,
        cols=2,
        next_to_move=PLAYER_1,
        player_1=PlayerMCTSConfig.parallel(0.001, workers=2),
        player_2=PlayerMCTSConfig.serial(0.002),
    )

    result = run_arena(
        config,
        executor_factory=ImmediateExecutor,
        training_data_directory=None,
    )

    assert [move.player for move in result.moves] == [PLAYER_1, PLAYER_2]
    assert [move.search_mode for move in result.moves] == [
        SearchMode.PARALLEL,
        SearchMode.SERIAL,
    ]
    assert result.moves[0].search_stats.completed_batches > 0
    assert result.moves[1].search_stats.completed_batches == 0


def test_parallel_players_share_one_pool_sized_for_the_larger_player():
    ImmediateExecutor.instances.clear()
    config = ArenaConfig(
        rows=1,
        cols=2,
        next_to_move=PLAYER_1,
        player_1=PlayerMCTSConfig.parallel(0.001, workers=2),
        player_2=PlayerMCTSConfig.parallel(0.001, workers=3),
    )

    result = run_arena(
        config,
        executor_factory=ImmediateExecutor,
        training_data_directory=None,
    )

    assert all(move.search_mode is SearchMode.PARALLEL for move in result.moves)
    assert len(ImmediateExecutor.instances) == 1
    assert ImmediateExecutor.instances[0].max_workers == 3


def test_arena_saves_a_categorized_trajectory_with_config_in_filename():
    ImmediateExecutor.instances.clear()
    config = ArenaConfig(
        rows=1,
        cols=2,
        next_to_move=PLAYER_2,
        player_1=PlayerMCTSConfig.serial(0.001),
        player_2=PlayerMCTSConfig.parallel(0.002, workers=2),
    )

    with TemporaryDirectory() as directory:
        arena_directory = Path(directory) / "training_data" / "_game_arena"
        result = run_arena(
            config,
            executor_factory=ImmediateExecutor,
            training_data_directory=arena_directory,
        )
        path = result.training_data_path
        stored = load_self_play_game(path)
        analyzed = load_analysis_path(path)

        assert path.parent == arena_directory / "1x2"
        assert re.fullmatch(
            r"\d{8}T\d{6}\.\d{6}Z_[0-9a-f]{32}_"
            r"next-p2__p1-serial-0p001s-w1__"
            r"p2-parallel-0p002s-w2\.npz",
            path.name,
        )
        assert "arena" not in path.name
        assert "1x2" not in path.name
        assert path.suffix == ".npz"
        assert list(path.parent.glob("*.tmp")) == []
        assert stored["boards"].shape == (2, 1, 2)
        assert stored["next_players"].tolist() == [PLAYER_2, PLAYER_1]
        assert stored["search_budget_type"].item() == "seconds"
        assert not any(name.startswith("arena_") for name in stored)
        assert analyzed.schema_version == 1


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} game arena tests passed")
