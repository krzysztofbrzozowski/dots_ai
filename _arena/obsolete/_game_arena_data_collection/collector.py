"""Collect reproducible games with exact rollout budgets."""

from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time
from time import monotonic

import numpy as np

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch
from training import SelfPlayTrajectory

from .config import COLLECTION_CONFIG, DEFAULT_OUTPUT_DIRECTORY
from .search import CollectionNode


SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,32}$")
COLLECTION_FILE_PATTERN = re.compile(
    r"_collection-(?P<source>[A-Za-z0-9_]+)-s(?P<seed>\d+)-g(?P<index>\d+)\.npz$"
)


def validate_source_id(source_id):
    if not isinstance(source_id, str) or SOURCE_ID_PATTERN.fullmatch(source_id) is None:
        raise ValueError(
            "source_id must contain 1-32 letters, digits, or underscores"
        )
    return source_id


def _source_entropy(source_id):
    """Return stable seed words; Python's built-in hash is process-randomized."""
    digest = hashlib.blake2b(source_id.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value & 0xFFFFFFFF, value >> 32


def game_plan(config, index, source_id="local"):
    """Return a deterministic equal-strength plan for one source-local index."""
    validate_source_id(source_id)
    source_low, source_high = _source_entropy(source_id)
    schedule_rng = np.random.default_rng(
        np.random.SeedSequence(
            [config.seed, source_low, source_high, index // 2, 0]
        )
    )
    starters = [1, -1]
    schedule_rng.shuffle(starters)
    starter = starters[index % 2]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [config.seed, source_low, source_high, index, 1]
        )
    )
    return {
        "index": index,
        "source_id": source_id,
        "seed": config.seed,
        "matchup": "equal",
        "starting_player": starter,
        "mode": "serial",
        "workers": 1,
        "search_seed": int(rng.integers(2**63)),
        "opening_moves": [],
    }


def rollout_budget(config, legal_action_count):
    """Scale exact simulations with branching while keeping useful bounds."""
    if legal_action_count <= 0:
        raise ValueError("legal_action_count must be positive")
    return min(
        config.maximum_rollouts,
        max(
            config.minimum_rollouts,
            config.rollouts_per_legal_action * legal_action_count,
        ),
    )


def select_child_from_visits(root, rng, move_index, config):
    """Sample early moves from MCTS visits; use most-visited moves later."""
    if not root.children:
        raise ValueError("cannot select a move from an unexpanded root")
    visits = np.asarray([child.n for child in root.children], dtype=np.float64)
    if np.any(visits <= 0):
        raise ValueError("every expanded root child must have a completed visit")

    if move_index < config.temperature_moves:
        log_weights = np.log(visits) / config.visit_temperature
        weights = np.exp(log_weights - log_weights.max())
        probabilities = weights / weights.sum()
        selected_index = int(rng.choice(len(root.children), p=probabilities))
    else:
        candidates = np.flatnonzero(visits == visits.max())
        selected_index = int(rng.choice(candidates))
    return root.children[selected_index]


def play_game(config, plan, output_directory, progress=None):
    """Play one game using exact, position-dependent simulation counts."""
    state = DotsGame(config.rows, config.cols)
    state.next_to_move = plan["starting_player"]
    rng = np.random.default_rng(plan["search_seed"])
    # Schema v1 stores one scalar requested budget. The cap is stored there;
    # completed_rollouts records the exact adaptive budget for every position.
    trajectory = SelfPlayTrajectory(simulations_number=config.maximum_rollouts)
    while state.game_result is None:
        move_index = len(trajectory)
        root = CollectionNode(state, rng)
        legal_action_count = len(root.untried_actions)
        budget = rollout_budget(config, legal_action_count)
        search = MonteCarloTreeSearch(root)
        # best_action runs the search. Collection move choice deliberately uses
        # visits below so behavior matches the saved policy target.
        search.best_action(simulations_number=budget)
        stats = search.last_search_stats
        selected = select_child_from_visits(root, rng, move_index, config)
        trajectory.record_search(state, root, selected.action, stats)
        state = selected.state
        if progress is not None and len(trajectory) % 20 == 0:
            progress(
                f"  game {plan['index']}: move {len(trajectory)}, "
                f"rollouts={budget}, "
                f"score +1/-1={state.score[1]}/{state.score[-1]}"
            )
    return trajectory.save(
        output_directory,
        final_result=state.game_result,
        filename_suffix=(
            f"collection-{plan['source_id']}-s{plan['seed']}"
            f"-g{plan['index']:08d}"
        ),
    )


def write_json(path, data):
    """Replace metadata atomically, as the existing trajectory writer does."""
    path = Path(path)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".", suffix=".tmp", delete=False) as stream:
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
def collection_lock(directory):
    # OS releases the lock on exit or crash. The harmless file can remain.
    # Windows msvcrt locks a byte range, so the lock file must contain one byte.
    with (directory / ".collector.lock").open("a+b") as stream:
        if os.name == "nt":
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another collector is writing to this directory") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def game_identity(path):
    match = COLLECTION_FILE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"unrecognized NPZ in collection directory: {Path(path).name}")
    return (
        match.group("source"),
        int(match.group("seed")),
        int(match.group("index")),
    )


def game_index(path):
    return game_identity(path)[2]


def collection_files(directory):
    indexed = [(game_identity(path), path) for path in Path(directory).glob("*.npz")]
    indexed.sort()
    if len({identity for identity, _ in indexed}) != len(indexed):
        raise ValueError("collection has duplicate source/seed/game identities")
    return [path for _, path in indexed]


def available_indices(existing, source_id, seed):
    """Fill gaps in one source stream without colliding with other machines."""
    index = 0
    while True:
        if (source_id, seed, index) not in existing:
            yield index
        index += 1


def _worker_init():
    # Only the coordinator handles Ctrl-C; workers finish their current games.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def _graceful_sigint(request_stop):
    # A flag avoids interrupting a submit or manifest update halfway through.
    # Python only permits installing signal handlers from the main thread.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, lambda *_: request_stop())
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _play_worker(config, plan, output_directory):
    started = monotonic()
    cpu_started = time.process_time()
    path = play_game(config, plan, output_directory)
    return path, {
        "elapsed_seconds": monotonic() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "worker_pid": os.getpid(),
    }


def _parallel_games(config, source_id, output_directory, indices, workers, started,
                    duration_seconds, max_games, on_start, on_complete, progress):
    """Keep at most `workers` games in flight; one coordinator owns metadata."""
    pending = {}
    submitted = 0
    interrupted = False
    stop_announced = False
    failure = None

    def request_stop():
        nonlocal interrupted
        interrupted = True

    with _graceful_sigint(request_stop), ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
    ) as executor:
        while True:
            try:
                if interrupted and not stop_announced:
                    stop_announced = True
                    if progress is not None:
                        progress("Stopping new games; waiting for active games to finish and save.")
                while not interrupted and failure is None and len(pending) < workers:
                    if max_games is not None and submitted >= max_games:
                        break
                    if duration_seconds is not None and monotonic() - started >= duration_seconds:
                        break
                    plan = game_plan(config, next(indices), source_id)
                    on_start(plan)
                    future = executor.submit(_play_worker, config, plan, output_directory)
                    pending[future] = plan
                    submitted += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    plan = pending.pop(future)
                    try:
                        path, statistics = future.result()
                        on_complete(plan, path, statistics)
                    except Exception as error:
                        # Drain other active games, preserving their completed files.
                        failure = failure or error
            except KeyboardInterrupt:
                request_stop()
    if failure is not None:
        raise failure
    return interrupted


def collect(config=COLLECTION_CONFIG, *, output_directory=DEFAULT_OUTPUT_DIRECTORY,
            duration_seconds=600, max_games=None, workers=1,
            source_id="local", progress=print):
    """Resume this collection. Limits apply to this invocation, not old games.

    Time is checked before starting games; active games finish before stopping.
    Parallel Ctrl-C drains active games; serial Ctrl-C discards its partial game.
    """
    if duration_seconds is not None and (
        isinstance(duration_seconds, bool) or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be finite and positive")
    if max_games is not None and (
        isinstance(max_games, bool) or not isinstance(max_games, int) or max_games <= 0
    ):
        raise ValueError("max_games must be a positive integer")
    if duration_seconds is None and max_games is None:
        raise ValueError("provide a time or game-count limit")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    validate_source_id(source_id)

    from .audit import inspect_game

    directory = Path(output_directory).resolve() / f"{config.rows}x{config.cols}"
    directory.mkdir(parents=True, exist_ok=True)
    with collection_lock(directory):
        settings_path = directory / "collection.json"
        settings = {"version": 2, "settings": config.generation_settings()}
        paths = collection_files(directory)
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise ValueError("collection settings differ; use another output directory")
        elif paths:
            raise ValueError("existing NPZ files have no collection.json")
        else:
            write_json(settings_path, settings)

        # Recover even if the last process stopped after saving NPZ but before
        # saving the manifest. Plans derive from the persisted config + index.
        records = []
        for path in paths:
            file_source_id, file_seed, index = game_identity(path)
            plan = game_plan(
                replace(config, seed=file_seed), index, file_source_id,
            )
            record = inspect_game(path, replace(config, seed=file_seed))
            verify_plan(record, plan)
            records.append({**plan, **record})
        write_json(directory / "manifest.json", records)

        started_at = datetime.now(timezone.utc).isoformat()
        started = monotonic()
        old_count = len(records)
        identities = {
            (record["source_id"], record["seed"], record["index"])
            for record in records
        }
        indices = available_indices(identities, source_id, config.seed)
        interrupted = False
        cpu_seconds = 0.0

        def on_start(plan):
            if progress is not None:
                progress(
                    f"Starting {plan['source_id']} game {plan['index']}: "
                    f"starter={plan['starting_player']:+d}, "
                    f"adaptive rollouts={config.rollouts_per_legal_action}x legal "
                    f"clipped to {config.minimum_rollouts}-{config.maximum_rollouts}"
                )

        def on_complete(plan, path, statistics):
            nonlocal cpu_seconds
            record = inspect_game(path, config)
            verify_plan(record, plan)
            records.append({**plan, **record, **statistics})
            records.sort(
                key=lambda item: (
                    item["source_id"], item["seed"], item["index"],
                )
            )
            cpu_seconds += statistics.get("cpu_seconds", 0.0)
            write_json(directory / "manifest.json", records)
            if progress is not None:
                progress(
                    f"Saved {plan['source_id']} game {plan['index']}: "
                    f"result={record['final_result']:+d}, "
                    f"positions={record['positions']}, "
                    f"collection elapsed={monotonic() - started:.1f}s"
                )

        if progress is not None:
            progress(f"Collecting with up to {workers} concurrent game(s), one CPU process per game.")
        try:
            if workers > 1:
                interrupted = _parallel_games(
                    config, source_id, output_directory, indices, workers, started,
                    duration_seconds, max_games, on_start, on_complete, progress,
                )
            while workers == 1 and (max_games is None or len(records) - old_count < max_games):
                if duration_seconds is not None and monotonic() - started >= duration_seconds:
                    break
                plan = game_plan(config, next(indices), source_id)
                on_start(plan)
                game_started = monotonic()
                cpu_started = time.process_time()
                path = play_game(config, plan, output_directory, progress)
                on_complete(plan, path, {
                    "elapsed_seconds": monotonic() - game_started,
                    "cpu_seconds": time.process_time() - cpu_started,
                    "worker_pid": os.getpid(),
                })
        except KeyboardInterrupt:
            interrupted = True
            if progress is not None:
                progress("Interrupted; completed games are saved. Resume with the same settings.")
        session = {
            "started_at_utc": started_at,
            "elapsed_seconds": monotonic() - started,
            "requested_seconds": duration_seconds,
            "games_before": old_count,
            "games_added": len(records) - old_count,
            "total_games": len(records),
            "interrupted": interrupted,
            "workers": workers,
            "source_id": source_id,
            "seed": config.seed,
            "game_cpu_seconds": cpu_seconds,
            "directory": str(directory),
        }
        write_json(directory / "last_session.json", session)
        return session


def verify_plan(record, plan):
    if record["starting_player"] != plan["starting_player"]:
        raise ValueError("recorded starter differs from game plan")
    if record["opening_moves"]:
        raise ValueError("rollout collection must not contain random openings")
