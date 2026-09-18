"""Editable defaults for deterministic rollout-based data collection."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "training_data"
    / "_game_arena_data_collection_rollouts"
)
# Execution setting, separate from generation settings so an existing
# collection can be resumed with a different number of concurrent games.
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 1) - 2))


@dataclass(frozen=True)
class CollectionConfig:
    rows: int = 10
    cols: int = 10
    seed: int = 42
    rollouts_per_legal_action: int = 4
    minimum_rollouts: int = 128
    maximum_rollouts: int = 400
    temperature_moves: int = 20
    visit_temperature: float = 1.0

    def __post_init__(self):
        for name in (
            "rows",
            "cols",
            "seed",
            "rollouts_per_legal_action",
            "minimum_rollouts",
            "maximum_rollouts",
            "temperature_moves",
        ):
            value = getattr(self, name)
            minimum = 0 if name in ("seed", "temperature_moves") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.minimum_rollouts > self.maximum_rollouts:
            raise ValueError("minimum_rollouts must be <= maximum_rollouts")
        if self.maximum_rollouts < self.rows * self.cols:
            raise ValueError(
                "maximum_rollouts must cover every cell on an empty board"
            )
        if (
            isinstance(self.visit_temperature, bool)
            or not math.isfinite(self.visit_temperature)
            or self.visit_temperature <= 0
        ):
            raise ValueError("visit_temperature must be finite and positive")

    def settings(self):
        return asdict(self)

    def generation_settings(self):
        """Settings that must match when several sources share a directory."""
        values = self.settings()
        del values["seed"]
        return values


COLLECTION_CONFIG = CollectionConfig()
