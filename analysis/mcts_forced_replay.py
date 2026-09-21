"""Run disposable forced MCTS replays from saved analysis frames.

The saved ``AnalysisGame`` remains immutable.  An experiment owns a playable
copy of one pre-move frame and advances only that copy with the existing MCTS
implementation.
"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
from multiprocessing import get_context
from threading import RLock, Thread

from analysis.diagnostics import PRINT_T
from analysis.service import mcts_decision_frame
from game.enclosure import DotsGame
from mcts.enclosure import (
    MonteCarloTreeSearch,
    TwoPlayerMCTSNode,
    rollout_state,
)
from ml.predictor import analysis_frame_to_game_state


class ExperimentNotFoundError(LookupError):
    """The requested analysis has no experiment branch."""


class ExperimentBusyError(RuntimeError):
    """An experiment search is already running."""


class ExperimentFinishedError(RuntimeError):
    """The experiment branch already represents a terminal position."""


@dataclass(frozen=True, slots=True)
class ExperimentSearchConfig:
    """Search settings copied from the imported NPZ metadata."""

    simulation_seconds: float | None
    simulations_number: int | None
    rollout_batch_size: int
    uct_c_param: float

    @classmethod
    def from_game(cls, game):
        return cls(
            simulation_seconds=game.requested_simulation_seconds,
            simulations_number=game.requested_simulations,
            rollout_batch_size=int(game.rollout_batch_size),
            uct_c_param=float(
                signature(TwoPlayerMCTSNode.best_child)
                .parameters["c_param"]
                .default
            ),
        )

    @property
    def budget_type(self):
        return "seconds" if self.simulation_seconds is not None else "simulations"

    def as_dict(self):
        return {
            "budget_type": self.budget_type,
            "simulation_seconds": self.simulation_seconds,
            "simulations_number": self.simulations_number,
            "rollout_batch_size": self.rollout_batch_size,
            "uct_c_param": self.uct_c_param,
        }


@contextmanager
def _rollout_executor(config):
    """Reuse one process pool across every move in a continuation."""
    if config.rollout_batch_size == 1:
        yield None
        return

    with ProcessPoolExecutor(
        max_workers=config.rollout_batch_size,
        mp_context=get_context("spawn"),
    ) as executor:
        terminal_state = DotsGame(1, 1).move((0, 0))
        warmups = [
            executor.submit(rollout_state, terminal_state, seed)
            for seed in range(config.rollout_batch_size)
        ]
        for warmup in warmups:
            warmup.result()
        yield executor


def _run_search(state, config, rollout_executor, move_number):
    root = TwoPlayerMCTSNode(state=state)
    search = MonteCarloTreeSearch(
        root,
        rollout_executor=rollout_executor,
        rollout_batch_size=(
            config.rollout_batch_size if rollout_executor is not None else 1
        ),
    )
    if config.simulation_seconds is not None:
        selected = search.best_action(
            total_simulation_seconds=config.simulation_seconds,
        )
    else:
        selected = search.best_action(
            simulations_number=config.simulations_number,
        )
    if selected.action is None:
        raise RuntimeError("MCTS returned a node without an action")
    return (
        selected.state,
        mcts_decision_frame(
            state,
            root,
            selected.action,
            search.last_search_stats,
            move_number,
            experiment=True,
        ),
    )


class _Experiment:
    def __init__(self, analysis_id, game, frame_index):
        self.analysis_id = analysis_id
        self.source_frame_index = int(frame_index)
        self.source_move_number = int(frame_index) + 1
        self.config = ExperimentSearchConfig.from_game(game)
        self.state = analysis_frame_to_game_state(game, frame_index)
        self.next_move_number = self.source_move_number
        self.moves_completed = 0
        self.frames = []
        self.latest_frame = None
        self.status = "complete" if self.state.game_result is not None else "ready"
        self.mode = None
        self.error = None
        self.cancel_requested = False
        self.revision = 1
        self.lock = RLock()

    def snapshot(self):
        with self.lock:
            return {
                "analysis_id": self.analysis_id,
                "source_frame_index": self.source_frame_index,
                "source_move_number": self.source_move_number,
                "next_move_number": self.next_move_number,
                "moves_completed": self.moves_completed,
                "status": self.status,
                "mode": self.mode,
                "error": self.error,
                "revision": self.revision,
                "search": self.config.as_dict(),
                "current_state": {
                    "player_to_move": int(self.state.next_to_move),
                    "scores": {
                        "player_1": int(self.state.score[1]),
                        "player_2": int(self.state.score[-1]),
                    },
                    "legal_move_count": len(self.state.get_legal_actions()),
                    "game_over": self.state.game_result is not None,
                    "winner": (
                        int(self.state.game_result)
                        if self.state.game_result is not None
                        else None
                    ),
                },
                "frames": deepcopy(self.frames),
                "latest_frame": deepcopy(self.latest_frame),
            }


class ExperimentManager:
    """Own one disposable branch per imported analysis session."""

    def __init__(self, search_runner=_run_search):
        self._search_runner = search_runner
        self._experiments = {}
        self._lock = RLock()

    def start_branch(self, analysis_id, game, frame_index):
        if not 0 <= frame_index < game.frame_count:
            raise IndexError(frame_index)
        with self._lock:
            existing = self._experiments.get(analysis_id)
            if existing is not None and existing.status == "running":
                raise ExperimentBusyError("The current experiment is still running")
            experiment = _Experiment(analysis_id, game, frame_index)
            self._experiments[analysis_id] = experiment
        PRINT_T(
            f"Forced replay ready before move {frame_index + 1}",
            level="success",
            source="MCTS",
        )
        return experiment.snapshot()

    def get(self, analysis_id):
        with self._lock:
            experiment = self._experiments.get(analysis_id)
        if experiment is None:
            raise ExperimentNotFoundError(analysis_id)
        return experiment

    def snapshot(self, analysis_id):
        return self.get(analysis_id).snapshot()

    def run_step(self, analysis_id):
        return self._start_job(analysis_id, "step")

    def continue_game(self, analysis_id):
        return self._start_job(analysis_id, "continue")

    def _start_job(self, analysis_id, mode):
        experiment = self.get(analysis_id)
        with experiment.lock:
            if experiment.status == "running":
                raise ExperimentBusyError("The current experiment is still running")
            if experiment.state.game_result is not None:
                raise ExperimentFinishedError("The experiment game is already over")
            experiment.status = "running"
            experiment.mode = mode
            experiment.error = None
            experiment.revision += 1
            snapshot = experiment.snapshot()

        Thread(
            target=self._run_job,
            args=(experiment, mode),
            name=f"mcts-experiment-{analysis_id[:8]}",
            daemon=True,
        ).start()
        return snapshot

    def _run_job(self, experiment, mode):
        try:
            with _rollout_executor(experiment.config) as rollout_executor:
                while True:
                    with experiment.lock:
                        state = experiment.state
                        move_number = experiment.next_move_number
                    if state.game_result is not None:
                        break

                    next_state, frame = self._search_runner(
                        state,
                        experiment.config,
                        rollout_executor,
                        move_number,
                    )
                    selected = frame["selected_action"]
                    with experiment.lock:
                        frame["index"] = len(experiment.frames)
                        experiment.state = next_state
                        experiment.frames.append(frame)
                        experiment.latest_frame = frame
                        experiment.moves_completed += 1
                        experiment.next_move_number += 1
                        experiment.revision += 1
                        cancel_requested = experiment.cancel_requested

                    PRINT_T(
                        f"Forced replay move {move_number}: selected "
                        f"({selected[0]}, {selected[1]}) after "
                        f"{frame['completed_rollouts']} rollouts",
                        level="success",
                        source="MCTS",
                    )
                    if mode == "step" or cancel_requested:
                        break

            with experiment.lock:
                if experiment.cancel_requested:
                    experiment.status = "cancelled"
                else:
                    experiment.status = (
                        "complete"
                        if experiment.state.game_result is not None
                        else "ready"
                    )
                experiment.mode = None
                experiment.revision += 1
        except BaseException as error:
            with experiment.lock:
                experiment.status = "failed"
                experiment.mode = None
                experiment.error = str(error) or error.__class__.__name__
                experiment.revision += 1
            PRINT_T(
                f"Forced replay failed: {experiment.error}",
                level="error",
                source="MCTS",
            )

    def remove(self, analysis_id):
        with self._lock:
            experiment = self._experiments.get(analysis_id)
            if experiment is None:
                return
            if experiment.status == "running":
                # A running rollout batch cannot be interrupted safely, but a
                # continuation can stop before it starts another real move.
                with experiment.lock:
                    experiment.cancel_requested = True
                self._experiments.pop(analysis_id, None)
                return
            self._experiments.pop(analysis_id, None)
