"""Configurable MCTS-vs-MCTS game arena."""

from .arena import ArenaMove, ArenaResult, run_arena
from .config import (
    ARENA_CONFIG,
    ArenaConfig,
    PlayerMCTSConfig,
    SearchMode,
)
from .recording import DEFAULT_ARENA_TRAINING_DATA_DIRECTORY

__all__ = [
    "ARENA_CONFIG",
    "DEFAULT_ARENA_TRAINING_DATA_DIRECTORY",
    "ArenaConfig",
    "ArenaMove",
    "ArenaResult",
    "PlayerMCTSConfig",
    "SearchMode",
    "run_arena",
]
