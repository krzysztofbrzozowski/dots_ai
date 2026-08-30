"""Arena-specific trajectory location, filename, and search metadata."""

from decimal import Decimal
from pathlib import Path

from game.enclosure import PLAYER_1


DEFAULT_ARENA_TRAINING_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "training_data" / "_game_arena"
)


def _seconds_slug(value):
    decimal = Decimal(str(value))
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def arena_filename_suffix(config):
    """Encode every ARENA_CONFIG choice in a filesystem-safe suffix."""
    next_player = 1 if config.next_to_move == PLAYER_1 else 2

    def player_slug(number, player_config):
        return (
            f"p{number}-{player_config.mode.value}-"
            f"{_seconds_slug(player_config.simulation_seconds)}s-"
            f"w{player_config.workers}"
        )

    return "__".join(
        (
            f"_next-p{next_player}",
            player_slug(1, config.player_1),
            player_slug(2, config.player_2),
        )
    )
