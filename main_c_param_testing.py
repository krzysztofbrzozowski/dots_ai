"""Run a resumable, paired-seed MCTS exploration-constant experiment.

The default run plays 13 matched seed blocks.  Every block starts from the
same empty 10x10 position and reuses the same per-move random seeds for each
``c_param`` value from 0.0 through 3.0 in 0.2 increments.  Games run one at a
time so every two-second search receives the full machine.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import signal
import tempfile
from time import monotonic

import numpy as np

from _game_arena_data_collection.search import CollectionNode
from _game_arena_data_collection.audit import inspect_game
from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch, rollout_state
from training import SelfPlayTrajectory, load_self_play_game


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "training_data" / "10_10_c_param"
DEFAULT_WORKERS = os.cpu_count() or 1
FILE_PATTERN = re.compile(
    r"_c_param-(?P<c_param>\d+p\d+)_seed-(?P<seed>\d+)\.npz$"
)


@dataclass(frozen=True, slots=True)
class CParamExperimentConfig:
    rows: int = 10
    cols: int = 10
    simulation_seconds: float = 2.0
    games_per_c_param: int = 13
    c_param_start: float = 0.0
    c_param_end: float = 3.0
    c_param_step: float = 0.2
    base_seed: int = 42
    workers: int = DEFAULT_WORKERS

    def __post_init__(self):
        for name in ("rows", "cols", "games_per_c_param", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError("base_seed must be a non-negative integer")
        for name in (
            "simulation_seconds",
            "c_param_start",
            "c_param_end",
            "c_param_step",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.simulation_seconds <= 0:
            raise ValueError("simulation_seconds must be positive")
        if self.c_param_start < 0 or self.c_param_end < self.c_param_start:
            raise ValueError("c_param range must satisfy 0 <= start <= end")
        if self.c_param_step <= 0:
            raise ValueError("c_param_step must be positive")
        # Evaluate the grid now so an invalid non-divisible range fails before
        # an output directory or process pool is created.
        self.c_params()

    def c_params(self):
        start = Decimal(str(self.c_param_start))
        end = Decimal(str(self.c_param_end))
        step = Decimal(str(self.c_param_step))
        span = end - start
        if span % step:
            raise ValueError("c_param range must be exactly divisible by c_param_step")
        count = int(span / step) + 1
        return tuple(float(start + index * step) for index in range(count))

    def seeds(self):
        return tuple(
            self.base_seed + index
            for index in range(self.games_per_c_param)
        )

    def settings(self):
        values = asdict(self)
        values["c_params"] = list(self.c_params())
        return values


def _c_param_slug(c_param):
    return f"{c_param:.1f}".replace(".", "p")


def _filename_suffix(c_param, seed):
    return f"c_param-{_c_param_slug(c_param)}_seed-{seed:010d}"


def _parse_game_key(path):
    match = FILE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(
            f"unrecognized NPZ in c_param experiment: {Path(path).name}"
        )
    c_param = float(match["c_param"].replace("p", "."))
    return int(match["seed"]), c_param


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _write_json(path, data):
    path = Path(path)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


@contextmanager
def _experiment_lock(directory):
    with (directory / ".collector.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another c_param experiment is already writing to this directory"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _worker_init():
    # The coordinator handles Ctrl-C and lets an active rollout batch finish.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _executor_context(config):
    if config.workers == 1:
        return nullcontext(None)
    return ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
    )


def _warm_workers(executor, workers):
    if executor is None:
        return
    terminal_state = DotsGame(1, 1).move((0, 0))
    warmups = [
        executor.submit(rollout_state, terminal_state, seed)
        for seed in range(workers)
    ]
    for warmup in warmups:
        warmup.result()


def _saved_game_record(path, config):
    inspected = inspect_game(path)
    game = load_self_play_game(path)
    expected_shape = np.asarray((config.rows, config.cols), dtype=np.int16)
    if not np.array_equal(game["board_shape"], expected_shape):
        raise ValueError(f"{Path(path).name}: unexpected board shape")
    if not math.isclose(
        float(game["requested_simulation_seconds"]),
        config.simulation_seconds,
    ):
        raise ValueError(f"{Path(path).name}: unexpected search budget")
    if int(game["rollout_batch_size"]) != config.workers:
        raise ValueError(f"{Path(path).name}: unexpected rollout worker count")

    seed, c_param = _parse_game_key(path)
    return {
        "seed": seed,
        "c_param": c_param,
        "file": Path(path).name,
        "game_id": str(game["game_id"].item()),
        "positions": inspected["positions"],
        "final_result": inspected["final_result"],
        "final_scores": inspected["final_scores"],
        "rollouts": int(game["completed_rollouts"].sum()),
        "search_seconds": float(game["search_elapsed_seconds"].sum()),
        "bytes": Path(path).stat().st_size,
    }


def _load_existing_records(directory, config):
    allowed = {
        (seed, c_param)
        for seed in config.seeds()
        for c_param in config.c_params()
    }
    records = []
    seen = set()
    for path in sorted(directory.glob("*.npz")):
        record = _saved_game_record(path, config)
        key = (record["seed"], record["c_param"])
        if key not in allowed:
            raise ValueError(f"{path.name}: seed/c_param is outside this experiment")
        if key in seen:
            raise ValueError(
                f"duplicate game for seed={key[0]}, c_param={key[1]:.1f}"
            )
        seen.add(key)
        records.append(record)
    records.sort(key=lambda item: (item["seed"], item["c_param"]))
    return records


def _game_plan(config):
    # One complete paired block is produced before moving to the next seed.
    for seed in config.seeds():
        for c_param in config.c_params():
            yield seed, c_param


def play_c_param_game(config, c_param, seed, executor, output_directory, progress=print):
    """Play and save one full game with deterministic per-move seed streams."""
    state = DotsGame(config.rows, config.cols)
    move_seed_source = random.Random(seed)
    trajectory = SelfPlayTrajectory(
        simulation_seconds=config.simulation_seconds,
        rollout_batch_size=config.workers,
    )

    while state.game_result is None:
        move_number = len(trajectory) + 1
        move_seed = move_seed_source.getrandbits(128)
        root = CollectionNode(state, np.random.default_rng(move_seed))
        search = MonteCarloTreeSearch(
            root,
            rollout_executor=executor,
            rollout_batch_size=config.workers if executor is not None else 1,
            random_seed=move_seed,
            c_param=c_param,
        )
        selected = search.best_action(
            total_simulation_seconds=config.simulation_seconds,
        )
        if selected.action is None:
            raise RuntimeError("MCTS returned a node without an action")
        trajectory.record_search(
            state=state,
            root=root,
            selected_action=selected.action,
            search_stats=search.last_search_stats,
        )
        state = selected.state

        if progress is not None and (move_number == 1 or move_number % 10 == 0):
            stats = search.last_search_stats
            progress(
                f"    move {move_number:3d}/{config.rows * config.cols}: "
                f"action={selected.action}, rollouts={stats.completed_rollouts}, "
                f"throughput={stats.rollouts_per_second:.0f}/s"
            )

    path = trajectory.save(
        output_directory,
        final_result=state.game_result,
        filename_suffix=_filename_suffix(c_param, seed),
        board_size_subdirectory=False,
    )
    return path


def run_experiment(
    config=CParamExperimentConfig(),
    *,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    max_games=None,
    dry_run=False,
    progress=print,
):
    """Run or resume the paired c_param experiment."""
    if not isinstance(config, CParamExperimentConfig):
        raise TypeError("config must be a CParamExperimentConfig")
    if max_games is not None and (
        isinstance(max_games, bool)
        or not isinstance(max_games, int)
        or max_games <= 0
    ):
        raise ValueError("max_games must be a positive integer or None")

    plan = list(_game_plan(config))
    theoretical_seconds = (
        len(plan)
        * config.rows
        * config.cols
        * config.simulation_seconds
    )
    if progress is not None:
        progress(
            f"c_param experiment: {len(config.c_params())} values × "
            f"{config.games_per_c_param} paired seeds = {len(plan)} games"
        )
        progress(
            f"Budget: {config.simulation_seconds:g}s/move; "
            f"workers: {config.workers}; theoretical search time: "
            f"{_format_duration(theoretical_seconds)}"
        )
        progress(f"Output: {Path(output_directory).resolve()}")
    if dry_run:
        return {
            "planned_games": len(plan),
            "theoretical_seconds": theoretical_seconds,
            "directory": str(Path(output_directory).resolve()),
        }

    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _experiment_lock(directory):
        settings_path = directory / "experiment.json"
        settings = {"version": 1, "settings": config.settings()}
        existing_paths = list(directory.glob("*.npz"))
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise ValueError(
                    "experiment settings differ; use another output directory"
                )
        elif existing_paths:
            raise ValueError("existing NPZ files have no experiment.json")
        else:
            _write_json(settings_path, settings)

        records = _load_existing_records(directory, config)
        completed = {(item["seed"], item["c_param"]) for item in records}
        remaining = [item for item in plan if item not in completed]
        if max_games is not None:
            remaining = remaining[:max_games]
        _write_json(directory / "manifest.json", records)

        if progress is not None:
            progress(
                f"Already complete: {len(completed)}/{len(plan)}; "
                f"running now: {len(remaining)}"
            )

        started_at = datetime.now(timezone.utc).isoformat()
        started = monotonic()
        interrupted = False
        added = 0
        with _executor_context(config) as executor:
            _warm_workers(executor, config.workers)
            try:
                for run_index, (seed, c_param) in enumerate(remaining, start=1):
                    if progress is not None:
                        progress(
                            f"[{run_index}/{len(remaining)}] Starting "
                            f"seed={seed}, c_param={c_param:.1f}"
                        )
                    game_started = monotonic()
                    path = play_c_param_game(
                        config,
                        c_param,
                        seed,
                        executor,
                        directory,
                        progress,
                    )
                    record = _saved_game_record(path, config)
                    record["wall_seconds"] = monotonic() - game_started
                    records.append(record)
                    records.sort(key=lambda item: (item["seed"], item["c_param"]))
                    _write_json(directory / "manifest.json", records)
                    added += 1

                    if progress is not None:
                        mean_game_seconds = (monotonic() - started) / added
                        eta = mean_game_seconds * (len(remaining) - added)
                        progress(
                            f"    saved {path.name}; result={record['final_result']:+d}; "
                            f"ETA {_format_duration(eta)}"
                        )
            except KeyboardInterrupt:
                interrupted = True
                if progress is not None:
                    progress(
                        "Interrupted. Completed games are saved; rerun the same "
                        "command to resume. The partial game was discarded."
                    )

        session = {
            "started_at_utc": started_at,
            "elapsed_seconds": monotonic() - started,
            "games_before": len(completed),
            "games_added": added,
            "total_games": len(records),
            "planned_games": len(plan),
            "interrupted": interrupted,
            "directory": str(directory),
        }
        _write_json(directory / "last_session.json", session)
        return session


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run or resume paired-seed MCTS c_param games.",
    )
    parser.add_argument("--simulation-seconds", type=float, default=2.0)
    parser.add_argument("--games-per-c-param", type=int, default=13)
    parser.add_argument("--c-param-start", type=float, default=0.0)
    parser.add_argument("--c-param-end", type=float, default=3.0)
    parser.add_argument("--c-param-step", type=float, default=0.2)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Run only this many remaining games, then exit cleanly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and estimate without creating files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = CParamExperimentConfig(
        simulation_seconds=args.simulation_seconds,
        games_per_c_param=args.games_per_c_param,
        c_param_start=args.c_param_start,
        c_param_end=args.c_param_end,
        c_param_step=args.c_param_step,
        base_seed=args.base_seed,
        workers=args.workers,
    )
    run_experiment(
        config,
        output_directory=args.output_directory,
        max_games=args.max_games,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
