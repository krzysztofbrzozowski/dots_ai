"""Run with python -m unittest _game_arena_data_collection.test_collection."""

from collections import Counter
from concurrent.futures import wait as wait_for_futures
from dataclasses import replace
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch, TwoPlayerMCTSNode
from training import load_self_play_game

from .audit import audit_collection, inspect_game
from .collector import (
    available_indices,
    collect,
    collection_files,
    collection_lock,
    game_identity,
    game_index,
    game_plan,
    rollout_budget,
    select_child_from_visits,
)
from .config import COLLECTION_CONFIG, CollectionConfig
from .search import CollectionNode


SMALL_CONFIG = replace(
    COLLECTION_CONFIG,
    rows=3,
    cols=3,
    rollouts_per_legal_action=2,
    minimum_rollouts=10,
    maximum_rollouts=18,
    temperature_moves=2,
)


def _fail_first_game_worker(config, plan, output_directory):
    """Picklable failure injection for the real spawn-process pool."""
    if plan["index"] == 0:
        raise RuntimeError("injected worker failure")
    from .collector import _play_worker
    return _play_worker(config, plan, output_directory)


class CollectionTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        cases = (
            {"rows": 0}, {"seed": -1}, {"seed": True},
            {"rollouts_per_legal_action": 0}, {"minimum_rollouts": 0},
            {"minimum_rollouts": 401}, {"maximum_rollouts": 99},
            {"temperature_moves": -1}, {"visit_temperature": 0},
            {"visit_temperature": float("nan")},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(COLLECTION_CONFIG, **values)

    def test_schedule_balances_starters_and_sources_are_distinct(self):
        for block in range(3):
            indexes = range(block * 2, block * 2 + 2)
            plans = [game_plan(COLLECTION_CONFIG, index, "laptop") for index in indexes]
            self.assertEqual(Counter(plan["starting_player"] for plan in plans), {1: 1, -1: 1})
            for plan in plans:
                self.assertEqual(
                    plan,
                    game_plan(COLLECTION_CONFIG, plan["index"], "laptop"),
                )
                self.assertEqual(plan["matchup"], "equal")
                self.assertEqual(plan["opening_moves"], [])
        self.assertNotEqual(
            game_plan(COLLECTION_CONFIG, 0, "laptop")["search_seed"],
            game_plan(COLLECTION_CONFIG, 0, "pc2")["search_seed"],
        )

    def test_adaptive_budget_is_exact_and_bounded(self):
        self.assertEqual(rollout_budget(COLLECTION_CONFIG, 100), 400)
        self.assertEqual(rollout_budget(COLLECTION_CONFIG, 70), 280)
        self.assertEqual(rollout_budget(COLLECTION_CONFIG, 50), 200)
        self.assertEqual(rollout_budget(COLLECTION_CONFIG, 20), 128)
        self.assertEqual(rollout_budget(COLLECTION_CONFIG, 5), 128)

    def test_source_streams_use_independent_local_indices(self):
        existing = {("laptop", 42, 0), ("laptop", 42, 2), ("pc2", 42, 0)}
        laptop = available_indices(existing, "laptop", 42)
        pc2 = available_indices(existing, "pc2", 42)
        self.assertEqual([next(laptop) for _ in range(3)], [1, 3, 4])
        self.assertEqual([next(pc2) for _ in range(3)], [1, 2, 3])

    def test_randomized_nodes_propagate_rng_and_leave_original_mcts_unchanged(self):
        state = DotsGame(4, 4)
        original = TwoPlayerMCTSNode(state)
        self.assertEqual(original.untried_actions, state.legal_moves())
        a = CollectionNode(state, np.random.default_rng(17))
        b = CollectionNode(state, np.random.default_rng(17))
        c = CollectionNode(state, np.random.default_rng(18))
        self.assertEqual(a.untried_actions, b.untried_actions)
        self.assertNotEqual(a.untried_actions, c.untried_actions)
        self.assertCountEqual(a.untried_actions, state.legal_moves())
        child = a.expand()
        self.assertIsInstance(child, CollectionNode)
        self.assertIs(child.rng, a.rng)
        self.assertFalse(np.any(state.board))

    def test_seeded_fixed_iteration_search_is_repeatable(self):
        outputs = []
        for _ in range(2):
            root = CollectionNode(DotsGame(4, 4), np.random.default_rng(123))
            selected = MonteCarloTreeSearch(root).best_action(simulations_number=30)
            outputs.append((selected.action, [(child.action, child.n, child.q) for child in root.children]))
        self.assertEqual(outputs[0], outputs[1])

    def test_eight_real_games_replay_and_match_existing_loaders(self):
        with TemporaryDirectory() as directory:
            session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None, max_games=8, progress=None)
            report = audit_collection(Path(directory) / "3x3")
            self.assertEqual(session["games_added"], 8)
            self.assertEqual(report["games"], 8)
            self.assertEqual(report["starting_players"], {-1: 4, 1: 4})
            self.assertEqual(sum(report["labels_loss_draw_win"]), report["positions"])
            self.assertEqual(report["replay_validation"], "passed")
            self.assertFalse(report["augmentation"])
            self.assertGreater(report["rollouts"], 0)
            self.assertEqual(report["sources"], {"local": 8})
            self.assertEqual(report["opening_lengths"], {0: 8})
            self.assertEqual(
                report["visited_legal_fraction"]["median"], 1.0,
            )
            for path in collection_files(Path(directory) / "3x3"):
                record = inspect_game(path, SMALL_CONFIG)
                game = load_self_play_game(path)
                self.assertEqual(record["opening_moves"], [])
                expected = [
                    rollout_budget(SMALL_CONFIG, int(mask.sum()))
                    for mask in game["legal_masks"]
                ]
                self.assertEqual(game["completed_rollouts"].tolist(), expected)
                self.assertTrue(np.all(game["visit_counts"] > -1))
                self.assertTrue(np.all(game["completed_rollouts"] > 0))

    def test_resume_recovers_manifest_from_npz_without_overwriting_games(self):
        with TemporaryDirectory() as directory:
            collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None, max_games=2, progress=None)
            board_directory = Path(directory) / "3x3"
            old_files = {path.name: path.read_bytes() for path in collection_files(board_directory)}
            (board_directory / "manifest.json").write_text("interrupted metadata write")
            session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None, max_games=1, progress=None)
            self.assertEqual(session["games_before"], 2)
            self.assertEqual(session["total_games"], 3)
            manifest = json.loads((board_directory / "manifest.json").read_text())
            self.assertEqual([record["index"] for record in manifest], [0, 1, 2])
            self.assertEqual({record["source_id"] for record in manifest}, {"local"})
            for name, data in old_files.items():
                self.assertEqual((board_directory / name).read_bytes(), data)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                collect(
                    replace(SMALL_CONFIG, rollouts_per_legal_action=3),
                    output_directory=directory,
                    max_games=1,
                    progress=None,
                )

    def test_time_limit_stops_before_another_game_and_interrupt_preserves_files(self):
        with TemporaryDirectory() as directory:
            with patch("_game_arena_data_collection.collector.monotonic", side_effect=[10, 11, 12]):
                session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=0.5, progress=None)
            self.assertEqual(session["games_added"], 0)
            collect(SMALL_CONFIG, output_directory=directory, max_games=1, progress=None)
            with patch("_game_arena_data_collection.collector.play_game", side_effect=KeyboardInterrupt):
                session = collect(SMALL_CONFIG, output_directory=directory, max_games=1, progress=None)
            self.assertTrue(session["interrupted"])
            self.assertEqual(session["total_games"], 1)
            self.assertEqual(session["games_added"], 0)

    def test_replay_detects_corrupt_scores_and_result(self):
        with TemporaryDirectory() as directory:
            collect(SMALL_CONFIG, output_directory=directory, max_games=1, progress=None)
            path = collection_files(Path(directory) / "3x3")[0]
            original = load_self_play_game(path)
            corrupted = {key: value.copy() for key, value in original.items()}
            corrupted["scores"][0, 0] += 1
            np.savez_compressed(path, **corrupted)
            with self.assertRaises(ValueError):
                inspect_game(path)
            corrupted = {key: value.copy() for key, value in original.items()}
            corrupted["final_result"] = np.asarray(1 if int(original["final_result"]) != 1 else -1, dtype=np.int8)
            np.savez_compressed(path, **corrupted)
            with self.assertRaisesRegex(ValueError, "recorded result"):
                inspect_game(path)

    def test_concurrent_writer_is_rejected(self):
        with TemporaryDirectory() as directory:
            with collection_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "another collector"):
                    with collection_lock(Path(directory)):
                        self.fail("second writer acquired the lock")

    def test_parallel_games_use_multiple_processes_and_replay(self):
        with TemporaryDirectory() as directory:
            session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None,
                              max_games=8, workers=2, progress=None)
            board_directory = Path(directory) / "3x3"
            records = json.loads((board_directory / "manifest.json").read_text())
            self.assertEqual(session["games_added"], 8)
            self.assertEqual(session["workers"], 2)
            self.assertGreater(session["game_cpu_seconds"], 0)
            self.assertEqual(len({record["worker_pid"] for record in records}), 2)
            self.assertEqual([record["index"] for record in records], list(range(8)))
            report = audit_collection(board_directory)
            self.assertEqual(report["games"], 8)
            self.assertEqual(report["starting_players"], {-1: 4, 1: 4})

    def test_resume_fills_missing_indices_and_can_change_worker_count(self):
        with TemporaryDirectory() as directory:
            collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None,
                    max_games=3, workers=2, progress=None)
            board_directory = Path(directory) / "3x3"
            paths = collection_files(board_directory)
            paths[0].unlink()  # Simulate a worker failing while later games completed.
            preserved = {path.name: path.read_bytes() for path in paths[1:]}
            self.assertEqual(audit_collection(board_directory)["games"], 2)
            (board_directory / "manifest.json").unlink()
            session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None,
                              max_games=2, workers=1, progress=None)
            self.assertEqual(session["games_before"], 2)
            self.assertEqual([game_index(path) for path in collection_files(board_directory)], [0, 1, 2, 3])
            for name, content in preserved.items():
                self.assertEqual((board_directory / name).read_bytes(), content)
            self.assertEqual(audit_collection(board_directory)["games"], 4)

    def test_different_sources_and_seeds_can_share_one_directory(self):
        with TemporaryDirectory() as directory:
            collect(
                SMALL_CONFIG,
                output_directory=directory,
                duration_seconds=None,
                max_games=1,
                source_id="laptop",
                progress=None,
            )
            collect(
                replace(SMALL_CONFIG, seed=43),
                output_directory=directory,
                duration_seconds=None,
                max_games=1,
                source_id="pc2",
                progress=None,
            )
            board_directory = Path(directory) / "3x3"
            paths = collection_files(board_directory)
            self.assertEqual(len(paths), 2)
            self.assertEqual(
                {game_identity(path) for path in paths},
                {("laptop", 42, 0), ("pc2", 43, 0)},
            )
            report = audit_collection(board_directory)
            self.assertEqual(report["sources"], {"laptop": 1, "pc2": 1})

    def test_parallel_interrupt_drains_active_games_without_starting_more(self):
        first_wait = True

        def interrupted_wait(*args, **kwargs):
            nonlocal first_wait
            if first_wait:
                first_wait = False
                signal.raise_signal(signal.SIGINT)
            return wait_for_futures(*args, **kwargs)

        with TemporaryDirectory() as directory:
            with patch("_game_arena_data_collection.collector.wait", side_effect=interrupted_wait):
                session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None,
                                  max_games=10, workers=2, progress=None)
            self.assertTrue(session["interrupted"])
            self.assertEqual(session["games_added"], 2)
            self.assertEqual(audit_collection(Path(directory) / "3x3")["games"], 2)

    def test_parallel_time_limit_finishes_only_already_started_games(self):
        clock = [100.0]

        def elapsed_wait(*args, **kwargs):
            result = wait_for_futures(*args, **kwargs)
            clock[0] = 200.0
            return result

        with TemporaryDirectory() as directory:
            with patch("_game_arena_data_collection.collector.monotonic", side_effect=lambda: clock[0]), \
                 patch("_game_arena_data_collection.collector.wait", side_effect=elapsed_wait):
                session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=1,
                                  workers=2, progress=None)
            self.assertEqual(session["games_added"], 2)
            self.assertFalse(session["interrupted"])
            self.assertEqual(audit_collection(Path(directory) / "3x3")["games"], 2)

    def test_invalid_worker_count_is_rejected(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                collect(SMALL_CONFIG, workers=workers, progress=None)

    def test_worker_failure_preserves_other_completed_games_and_allows_resume(self):
        with TemporaryDirectory() as directory:
            with patch("_game_arena_data_collection.collector._play_worker", _fail_first_game_worker):
                with self.assertRaisesRegex(RuntimeError, "injected worker failure"):
                    collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None,
                            max_games=4, workers=2, progress=None)
            board_directory = Path(directory) / "3x3"
            paths = collection_files(board_directory)
            self.assertTrue(paths)
            self.assertNotIn(0, [game_index(path) for path in paths])
            preserved = {path.name: path.read_bytes() for path in paths}
            self.assertEqual(audit_collection(board_directory)["games"], len(paths))
            session = collect(SMALL_CONFIG, output_directory=directory, duration_seconds=None,
                              max_games=1, workers=2, progress=None)
            self.assertEqual(session["games_before"], len(paths))
            self.assertEqual(session["games_added"], 1)
            self.assertEqual(game_index(collection_files(board_directory)[0]), 0)
            for name, content in preserved.items():
                self.assertEqual((board_directory / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
