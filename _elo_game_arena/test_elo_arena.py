"""Focused tests for Elo persistence and human identity handling."""

import json
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analysis import load_analysis_path
from training import load_self_play_game

from . import arena as arena_module
from .arena import EloArenaSession, resolve_competitor
from .ratings import RatingStore


@dataclass(frozen=True)
class StubCompetitor:
    rating_id: str
    display_name: str
    kind: str = "test"


class RatingStoreTests(unittest.TestCase):
    def test_win_updates_both_players_and_persists_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            store = RatingStore(path)
            first = StubCompetitor("first", "First")
            second = StubCompetitor("second", "Second")

            update = store.record_match("25x25", first, second, 1, match_id="match")

            self.assertEqual(update["player_1"]["rating"], 1216.0)
            self.assertEqual(update["player_2"]["rating"], 1184.0)
            self.assertEqual(store.snapshot("25x25")["best"]["id"], "first")
            with path.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["pools"]["25x25"]["matches"][0]["id"], "match")

    def test_draw_keeps_equal_ratings(self):
        with TemporaryDirectory() as directory:
            store = RatingStore(Path(directory) / "ratings.json")
            first = StubCompetitor("first", "First")
            second = StubCompetitor("second", "Second")
            store.record_match("25x25", first, second, 0)
            ratings = store.snapshot("25x25")["players"]
            self.assertEqual([player["rating"] for player in ratings], [1200.0, 1200.0])
            self.assertTrue(all(player["draws"] == 1 for player in ratings))

    def test_human_identity_is_case_and_whitespace_insensitive(self):
        first = resolve_competitor("human", "  Ada   Lovelace ")
        second = resolve_competitor("human", "ada lovelace")
        self.assertEqual(first.rating_id, second.rating_id)
        self.assertEqual(first.display_name, "Ada Lovelace")

    def test_two_humans_can_exchange_a_valid_move(self):
        with TemporaryDirectory() as directory:
            store = RatingStore(Path(directory) / "ratings.json")
            games_directory = Path(directory) / "games"
            session = EloArenaSession(store, games_directory)
            session.start_match("human", "human", "Alice", "Bob")

            deadline = time.monotonic() + 1.0
            while session.snapshot()["status"] != "waiting_human":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)

            session.submit_human_move(0, 0)
            while session.snapshot()["move_count"] != 1:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)

            snapshot = session.snapshot()
            self.assertEqual(snapshot["board"]["board"][0][0], 1)
            self.assertEqual(snapshot["next_to_move"], -1)
            self.assertTrue(snapshot["human_can_move"])
            stopped = session.end_match()
            self.assertEqual(stopped["status"], "stopped")
            self.assertFalse(stopped["human_can_move"])
            self.assertEqual(stopped["game_result"], 0)
            self.assertEqual(stopped["ratings"]["match_count"], 1)
            self.assertTrue(
                all(player["games"] == 1 for player in stopped["ratings"]["players"])
            )
            self.assertEqual(
                stopped["ratings"]["recent_matches"][0]["termination"], "stopped"
            )
            saved_path = Path(stopped["training_data_path"])
            self.assertTrue(saved_path.is_file())
            self.assertEqual(saved_path.parent, games_directory / "25x25")
            self.assertIn("_elo-stopped__", saved_path.name)
            saved = load_self_play_game(saved_path)
            self.assertEqual(saved["boards"].shape, (1, 25, 25))
            self.assertEqual(saved["selected_actions"].tolist(), [[0, 0]])
            self.assertEqual(saved["completed_rollouts"].tolist(), [0])
            self.assertEqual(saved["has_policy_priors"].tolist(), [0])
            self.assertEqual(int(saved["final_result"]), 0)
            self.assertEqual(load_analysis_path(saved_path).frame_count, 1)

    def test_naturally_completed_game_is_saved(self):
        with TemporaryDirectory() as directory:
            with (
                patch.object(arena_module, "ROWS", 1),
                patch.object(arena_module, "COLS", 1),
                patch.object(arena_module, "POOL_ID", "1x1"),
            ):
                store = RatingStore(Path(directory) / "ratings.json")
                games_directory = Path(directory) / "games"
                session = EloArenaSession(store, games_directory)
                session.start_match("human", "human", "Alice", "Bob")

                deadline = time.monotonic() + 1.0
                while session.snapshot()["status"] != "waiting_human":
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.005)
                session.submit_human_move(0, 0)
                while session.snapshot()["status"] != "complete":
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.005)

                completed = session.snapshot()
                saved_path = Path(completed["training_data_path"])
                self.assertTrue(saved_path.is_file())
                self.assertEqual(saved_path.parent, games_directory / "1x1")
                self.assertIn("_elo-completed__", saved_path.name)
                saved = load_self_play_game(saved_path)
                self.assertEqual(saved["boards"].shape, (1, 1, 1))
                self.assertEqual(saved["selected_actions"].tolist(), [[0, 0]])
                self.assertEqual(int(saved["final_result"]), 0)
                self.assertEqual(load_analysis_path(saved_path).frame_count, 1)
                self.assertEqual(completed["ratings"]["match_count"], 1)


if __name__ == "__main__":
    unittest.main()
