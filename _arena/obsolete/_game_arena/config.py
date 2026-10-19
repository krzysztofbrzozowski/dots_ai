"""Editable configuration for an MCTS-vs-MCTS arena game."""

import math
import os
from dataclasses import dataclass
from enum import Enum

from game.enclosure import PLAYER_1, PLAYER_2


class SearchMode(str, Enum):
    """How one player's rollout simulations are executed."""

    SERIAL = "serial"
    PARALLEL = "parallel"


def _default_parallel_workers():
    cpu_count = os.cpu_count() or 2
    return min(8, max(2, cpu_count - 1))


@dataclass(frozen=True, slots=True)
class PlayerMCTSConfig:
    """Search settings used whenever this player is on the move."""

    mode: SearchMode
    simulation_seconds: float
    workers: int = 1

    def __post_init__(self):
        try:
            normalized_mode = SearchMode(self.mode)
        except (TypeError, ValueError) as error:
            choices = ", ".join(mode.value for mode in SearchMode)
            raise ValueError(f"mode must be one of: {choices}") from error
        object.__setattr__(self, "mode", normalized_mode)

        if isinstance(self.simulation_seconds, bool) or not isinstance(
            self.simulation_seconds, (int, float)
        ):
            raise TypeError("simulation_seconds must be a number")
        if not math.isfinite(self.simulation_seconds) or self.simulation_seconds <= 0:
            raise ValueError("simulation_seconds must be positive and finite")

        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise TypeError("workers must be an integer")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.mode is SearchMode.SERIAL and self.workers != 1:
            raise ValueError("serial mode must use exactly one worker")

    @classmethod
    def serial(cls, simulation_seconds):
        return cls(
            mode=SearchMode.SERIAL,
            simulation_seconds=simulation_seconds,
            workers=1,
        )

    @classmethod
    def parallel(cls, simulation_seconds, workers=None):
        return cls(
            mode=SearchMode.PARALLEL,
            simulation_seconds=simulation_seconds,
            workers=workers if workers is not None else _default_parallel_workers(),
        )


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Board and independent Player 1 / Player 2 search settings."""

    rows: int
    cols: int
    next_to_move: int
    player_1: PlayerMCTSConfig
    player_2: PlayerMCTSConfig

    def __post_init__(self):
        for name, value in (("rows", self.rows), ("cols", self.cols)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.next_to_move not in (PLAYER_1, PLAYER_2):
            raise ValueError("next_to_move must be PLAYER_1 or PLAYER_2")
        if not isinstance(self.player_1, PlayerMCTSConfig):
            raise TypeError("player_1 must be a PlayerMCTSConfig")
        if not isinstance(self.player_2, PlayerMCTSConfig):
            raise TypeError("player_2 must be a PlayerMCTSConfig")

    def player_config(self, player):
        """Return the settings assigned to an absolute player value."""
        if player == PLAYER_1:
            return self.player_1
        if player == PLAYER_2:
            return self.player_2
        raise ValueError("player must be PLAYER_1 or PLAYER_2")


# Edit this object to choose the arena matchup.
ARENA_CONFIG = ArenaConfig(
    rows=10,
    cols=10,
    next_to_move=PLAYER_1,
    player_1=PlayerMCTSConfig.serial(
        simulation_seconds=1,
    ),
    player_2=PlayerMCTSConfig.parallel(
        simulation_seconds=30,
        workers=14,
    ),
)

# Serial: 0.05 vs Parallel: 3, W: 14 -> 2/2 Parallel win
# Serial: 0.10 vs Parallel: 3, W: 14 -> 2/4 Parallel win, 2/4 Draw
#   -> Changed start position
# Serial: 0.10 vs Parallel: 4, W: 14 -> 2/4 Parallel win, 2/4 Draw
#   -> Changed start position
# Serial: 0.15 vs Parallel: 3, W: 14 -> 3/3 Parallel win