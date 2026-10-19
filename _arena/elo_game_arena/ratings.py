"""Persistent Elo ratings with atomic JSON writes."""

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .config import BASE_RATING, K_FACTOR


class RatingStore:
    def __init__(self, path, base_rating=BASE_RATING, k_factor=K_FACTOR):
        self.path = Path(path)
        self.base_rating = float(base_rating)
        self.k_factor = float(k_factor)
        self._lock = RLock()
        self._data = self._load()

    def _empty_data(self):
        return {
            "schema_version": 1,
            "base_rating": self.base_rating,
            "k_factor": self.k_factor,
            "pools": {},
        }

    def _load(self):
        if not self.path.exists():
            return self._empty_data()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not load ratings from {self.path}: {error}") from error
        if data.get("schema_version") != 1 or not isinstance(data.get("pools"), dict):
            raise ValueError("ratings.json has an unsupported schema")
        return data

    def _pool(self, pool_id):
        return self._data["pools"].setdefault(
            pool_id,
            {"players": {}, "matches": []},
        )

    def _ensure_player(self, pool, competitor):
        return pool["players"].setdefault(
            competitor.rating_id,
            {
                "id": competitor.rating_id,
                "display_name": competitor.display_name,
                "kind": competitor.kind,
                "rating": self.base_rating,
                "games": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
            },
        )

    def record_match(
        self,
        pool_id,
        player_1,
        player_2,
        result,
        match_id=None,
        termination="completed",
    ):
        if result not in (-1, 0, 1):
            raise ValueError("result must be -1, 0, or 1")
        if termination not in {"completed", "stopped"}:
            raise ValueError("termination must be 'completed' or 'stopped'")
        if player_1.rating_id == player_2.rating_id:
            raise ValueError("a rated match requires two distinct player identities")

        with self._lock:
            pool = self._pool(pool_id)
            first = self._ensure_player(pool, player_1)
            second = self._ensure_player(pool, player_2)
            first_before = float(first["rating"])
            second_before = float(second["rating"])
            expected_first = 1.0 / (1.0 + 10.0 ** ((second_before - first_before) / 400.0))
            actual_first = 1.0 if result == 1 else 0.0 if result == -1 else 0.5
            delta = self.k_factor * (actual_first - expected_first)
            first["rating"] = round(first_before + delta, 2)
            second["rating"] = round(second_before - delta, 2)

            for player in (first, second):
                player["games"] += 1
            if result == 1:
                first["wins"] += 1
                second["losses"] += 1
            elif result == -1:
                second["wins"] += 1
                first["losses"] += 1
            else:
                first["draws"] += 1
                second["draws"] += 1

            pool["matches"].append(
                {
                    "id": match_id or uuid4().hex,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "player_1": player_1.rating_id,
                    "player_2": player_2.rating_id,
                    "result": result,
                    "termination": termination,
                    "ratings_before": {
                        player_1.rating_id: first_before,
                        player_2.rating_id: second_before,
                    },
                    "ratings_after": {
                        player_1.rating_id: first["rating"],
                        player_2.rating_id: second["rating"],
                    },
                }
            )
            self._save_unlocked()
            return {
                "player_1": deepcopy(first),
                "player_2": deepcopy(second),
                "delta": round(delta, 2),
            }

    def snapshot(self, pool_id):
        with self._lock:
            pool = deepcopy(self._data.get("pools", {}).get(pool_id, {}))
        players = list(pool.get("players", {}).values())
        players.sort(key=lambda player: (-player["rating"], player["display_name"].lower()))
        ranked = [player for player in players if player["games"] > 0]
        return {
            "pool_id": pool_id,
            "base_rating": self.base_rating,
            "k_factor": self.k_factor,
            "players": players,
            "best": deepcopy(ranked[0]) if ranked else None,
            "worst": deepcopy(ranked[-1]) if ranked else None,
            "match_count": len(pool.get("matches", [])),
            "recent_matches": list(reversed(pool.get("matches", [])[-10:])),
        }

    def _save_unlocked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
