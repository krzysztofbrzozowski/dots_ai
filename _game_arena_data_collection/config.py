"""Editable defaults for collecting games; no augmentation is applied."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "training_data" / "_game_arena_data_collection"
)
# Execution setting, separate from generation settings so an existing
# collection can be resumed with a different number of concurrent games.
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 1) - 2))


@dataclass(frozen=True)
class CollectionConfig:
    rows: int = 10
    cols: int = 10
    seed: int = 42
    base_seconds_min: float = 0.10
    base_seconds_max: float = 0.20
    advantage_min: float = 2.0
    advantage_max: float = 4.0
    max_move_seconds: float = 0.80
    opening_lengths: tuple[int, ...] = (0, 2, 4, 6)

    def __post_init__(self):
        for name in ("rows", "cols", "seed"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "base_seconds_min", "base_seconds_max", "advantage_min",
            "advantage_max", "max_move_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.base_seconds_min > self.base_seconds_max:
            raise ValueError("base_seconds_min must be <= base_seconds_max")
        if not 1 < self.advantage_min <= self.advantage_max:
            raise ValueError("advantage must satisfy 1 < min <= max")
        if self.max_move_seconds <= self.base_seconds_max:
            raise ValueError("max_move_seconds must exceed base_seconds_max")
        if not self.opening_lengths or any(
            isinstance(length, bool) or not isinstance(length, int)
            or not 0 <= length < self.rows * self.cols
            for length in self.opening_lengths
        ):
            raise ValueError("opening_lengths must contain integers in [0, board area)")

    def settings(self):
        values = asdict(self)
        values["opening_lengths"] = list(self.opening_lengths)
        return values


COLLECTION_CONFIG = CollectionConfig()
