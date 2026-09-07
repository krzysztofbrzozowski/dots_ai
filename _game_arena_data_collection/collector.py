"""Collect complete games with balanced matchups and resumable metadata."""

from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
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

from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch, SearchStats
from training import SelfPlayTrajectory

from .config import COLLECTION_CONFIG, DEFAULT_OUTPUT_DIRECTORY
from .search import CollectionNode


def game_plan(config, index):
    """Every shuffled block of eight balances starter and 50/25/25 matchups."""
    slots = [
        ("equal", 1), ("equal", -1), ("equal", 1), ("equal", -1),
        ("p1_advantage", 1), ("p1_advantage", -1),
        ("p2_advantage", 1), ("p2_advantage", -1),
    ]
    schedule_rng = np.random.default_rng(np.random.SeedSequence([config.seed, index // 8, 0]))
    schedule_rng.shuffle(slots)
    matchup, starter = slots[index % 8]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, index, 1]))
    base = float(rng.uniform(config.base_seconds_min, config.base_seconds_max))
    ratio = float(rng.uniform(config.advantage_min, config.advantage_max))
    stronger = min(config.max_move_seconds, base * ratio)
    p1_seconds = stronger if matchup == "p1_advantage" else base
    p2_seconds = stronger if matchup == "p2_advantage" else base
    opening_length = int(rng.choice(config.opening_lengths))
    state = DotsGame(config.rows, config.cols)
    state.next_to_move = starter
    opening = []
    for _ in range(opening_length):
        legal_moves = state.legal_moves()
        if not legal_moves:
            break
        move = legal_moves[int(rng.integers(len(legal_moves)))]
        opening.append(list(move))
        state = state.move(move)
    return {
        "index": index,
        "matchup": matchup,
        "starting_player": starter,
        "p1_seconds": p1_seconds,
        "p2_seconds": p2_seconds,
        "mode": "serial",
        "workers": 1,
        "search_seed": int(rng.integers(2**63)),
        "opening_moves": opening,
    }


def play_game(config, plan, output_directory, progress=None):
    """Run this game's fixed time budgets; record random opening moves honestly."""
    state = DotsGame(config.rows, config.cols)
    state.next_to_move = plan["starting_player"]
    rng = np.random.default_rng(plan["search_seed"])
    trajectory = SelfPlayTrajectory(simulation_seconds=plan["p1_seconds"])
    while state.game_result is None:
        move_index = len(trajectory)
        root = CollectionNode(state, rng)
        if move_index < len(plan["opening_moves"]):
            action = tuple(plan["opening_moves"][move_index])
            if action not in root.untried_actions:
                raise ValueError(f"illegal opening move: {action}")
            root.untried_actions.remove(action)
            root.untried_actions.append(action)
            selected = root.expand()
            # Random openings did not run MCTS; do not fabricate q/n targets.
            stats = SearchStats(0, 0, 0.0)
        else:
            budget = plan["p1_seconds"] if state.next_to_move == 1 else plan["p2_seconds"]
            search = MonteCarloTreeSearch(root)
            selected = search.best_action(total_simulation_seconds=budget)
            stats = search.last_search_stats
        trajectory.record_search(state, root, selected.action, stats)
        state = selected.state
        if progress is not None and len(trajectory) % 20 == 0:
            progress(
                f"  game {plan['index']}: move {len(trajectory)}, "
                f"score +1/-1={state.score[1]}/{state.score[-1]}"
            )
    return trajectory.save(
        output_directory,
        final_result=state.game_result,
        filename_suffix=f"collection-g{plan['index']:06d}",
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
    with (directory / ".collector.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another collector is writing to this directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def game_index(path):
    match = re.search(r"_collection-g(\d+)\.npz$", Path(path).name)
    if match is None:
        raise ValueError(f"unrecognized NPZ in collection directory: {Path(path).name}")
    return int(match[1])


def collection_files(directory):
    indexed = [(game_index(path), path) for path in Path(directory).glob("*.npz")]
    indexed.sort()
    if len({index for index, _ in indexed}) != len(indexed):
        raise ValueError("collection has duplicate game indices")
    return [path for _, path in indexed]


def available_indices(existing):
    """Fill gaps left by interrupted workers before allocating new indices."""
    index = 0
    while True:
        if index not in existing:
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


def _parallel_games(config, output_directory, indices, workers, started,
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
                    plan = game_plan(config, next(indices))
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
            duration_seconds=600, max_games=None, workers=1, progress=print):
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

    from .audit import inspect_game

    directory = Path(output_directory).resolve() / f"{config.rows}x{config.cols}"
    directory.mkdir(parents=True, exist_ok=True)
    with collection_lock(directory):
        settings_path = directory / "collection.json"
        settings = {"version": 1, "settings": config.settings()}
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
            plan = game_plan(config, game_index(path))
            record = inspect_game(path)
            verify_plan(record, plan)
            records.append({**plan, **record})
        write_json(directory / "manifest.json", records)

        started_at = datetime.now(timezone.utc).isoformat()
        started = monotonic()
        old_count = len(records)
        indices = available_indices({record["index"] for record in records})
        interrupted = False
        cpu_seconds = 0.0

        def on_start(plan):
            if progress is not None:
                progress(
                    f"Starting game {plan['index']}: {plan['matchup']}, "
                    f"starter={plan['starting_player']:+d}, "
                    f"seconds +1/-1={plan['p1_seconds']:.3f}/{plan['p2_seconds']:.3f}, "
                    f"opening={len(plan['opening_moves'])} moves"
                )

        def on_complete(plan, path, statistics):
            nonlocal cpu_seconds
            record = inspect_game(path)
            verify_plan(record, plan)
            records.append({**plan, **record, **statistics})
            records.sort(key=lambda item: item["index"])
            cpu_seconds += statistics.get("cpu_seconds", 0.0)
            write_json(directory / "manifest.json", records)
            if progress is not None:
                progress(
                    f"Saved game {plan['index']}: result={record['final_result']:+d}, "
                    f"positions={record['positions']}, "
                    f"collection elapsed={monotonic() - started:.1f}s"
                )

        if progress is not None:
            progress(f"Collecting with up to {workers} concurrent game(s), one CPU process per game.")
        try:
            if workers > 1:
                interrupted = _parallel_games(
                    config, output_directory, indices, workers, started,
                    duration_seconds, max_games, on_start, on_complete, progress,
                )
            while workers == 1 and (max_games is None or len(records) - old_count < max_games):
                if duration_seconds is not None and monotonic() - started >= duration_seconds:
                    break
                plan = game_plan(config, next(indices))
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
            "game_cpu_seconds": cpu_seconds,
            "directory": str(directory),
        }
        write_json(directory / "last_session.json", session)
        return session


def verify_plan(record, plan):
    if record["starting_player"] != plan["starting_player"]:
        raise ValueError("recorded starter differs from game plan")
    if record["opening_moves"] != plan["opening_moves"]:
        raise ValueError("recorded random opening differs from game plan")
