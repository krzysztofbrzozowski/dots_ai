"""Self-play training-data collection and persistence."""

from .data import SCHEMA_VERSION, SelfPlayTrajectory, load_self_play_game

__all__ = [
    "SCHEMA_VERSION",
    "SelfPlayTrajectory",
    "load_self_play_game",
]
