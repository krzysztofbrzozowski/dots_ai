"""Run one game using independent MCTS settings for each player."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from game.enclosure import DotsGame
from mcts.enclosure import (
    MonteCarloTreeSearch,
    SearchStats,
    TwoPlayerMCTSNode,
    rollout_state,
)
from training import SelfPlayTrajectory

from .config import ArenaConfig, SearchMode
from .recording import (
    DEFAULT_ARENA_TRAINING_DATA_DIRECTORY,
    arena_filename_suffix,
)


@dataclass(frozen=True, slots=True)
class ArenaMove:
    """Configuration and measured search statistics for one played move."""

    move_number: int
    player: int
    action: tuple[int, int]
    search_mode: SearchMode
    requested_simulation_seconds: float
    search_stats: SearchStats


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """Terminal state and the complete list of arena moves."""

    final_state: DotsGame
    moves: tuple[ArenaMove, ...]
    training_data_path: Path | None

    @property
    def final_result(self):
        return self.final_state.game_result


def _parallel_worker_limit(config):
    parallel_workers = [
        player.workers
        for player in (config.player_1, config.player_2)
        if player.mode is SearchMode.PARALLEL
    ]
    return max(parallel_workers, default=0)


def _executor_context(config, executor_factory):
    worker_limit = _parallel_worker_limit(config)
    if worker_limit == 0:
        return nullcontext(None)
    return executor_factory(
        max_workers=worker_limit,
        mp_context=get_context("spawn"),
    )


def _warm_parallel_workers(executor, workers):
    """Start worker processes before the first timed player search."""
    if executor is None:
        return
    terminal_state = DotsGame(1, 1).move((0, 0))
    warmups = [
        executor.submit(rollout_state, terminal_state, seed)
        for seed in range(workers)
    ]
    for warmup in warmups:
        warmup.result()


def run_arena(
    config,
    on_move=None,
    executor_factory=ProcessPoolExecutor,
    training_data_directory=DEFAULT_ARENA_TRAINING_DATA_DIRECTORY,
):
    """Play one configured game and return its terminal state and move log.

    ``on_move`` is an optional callback receiving ``(state_after_move, move)``.
    Completed games are recorded by default; pass ``training_data_directory=None``
    to disable persistence. The core arena does not depend on the GUI.
    """
    if not isinstance(config, ArenaConfig):
        raise TypeError("config must be an ArenaConfig")

    state = DotsGame(config.rows, config.cols)
    state.next_to_move = config.next_to_move
    moves = []
    player_1_batch_size = (
        config.player_1.workers
        if config.player_1.mode is SearchMode.PARALLEL
        else 1
    )
    # Schema v1 has one legacy search-budget scalar for the whole file. Keep
    # its shape unchanged and encode the complete two-player setup only in the
    # filename, as required by the arena dataset layout.
    trajectory = (
        SelfPlayTrajectory(
            simulation_seconds=config.player_1.simulation_seconds,
            rollout_batch_size=player_1_batch_size,
        )
        if training_data_directory is not None
        else None
    )

    with _executor_context(config, executor_factory) as rollout_executor:
        _warm_parallel_workers(
            rollout_executor,
            _parallel_worker_limit(config),
        )

        while state.game_result is None:
            moving_player = state.next_to_move
            player_config = config.player_config(moving_player)
            uses_parallel_rollouts = player_config.mode is SearchMode.PARALLEL

            root = TwoPlayerMCTSNode(state=state)
            search = MonteCarloTreeSearch(
                root,
                rollout_executor=(
                    rollout_executor if uses_parallel_rollouts else None
                ),
                rollout_batch_size=(
                    player_config.workers if uses_parallel_rollouts else 1
                ),
            )
            best_node = search.best_action(
                total_simulation_seconds=player_config.simulation_seconds,
            )
            if best_node.action is None:
                raise RuntimeError("MCTS returned a node without an action")

            if trajectory is not None:
                trajectory.record_search(
                    state=state,
                    root=root,
                    selected_action=best_node.action,
                    search_stats=search.last_search_stats,
                )

            state = best_node.state
            move = ArenaMove(
                move_number=len(moves) + 1,
                player=moving_player,
                action=best_node.action,
                search_mode=player_config.mode,
                requested_simulation_seconds=player_config.simulation_seconds,
                search_stats=search.last_search_stats,
            )
            moves.append(move)

            if on_move is not None:
                on_move(state, move)

    training_data_path = None
    if trajectory is not None:
        training_data_path = trajectory.save(
            training_data_directory,
            final_result=state.game_result,
            filename_suffix=arena_filename_suffix(config),
        )

    return ArenaResult(
        final_state=state,
        moves=tuple(moves),
        training_data_path=training_data_path,
    )
