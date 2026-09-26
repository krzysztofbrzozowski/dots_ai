"""Configuration and stable player presets for the 25x25 Elo arena."""

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIRECTORY.parent
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"
RATINGS_PATH = PACKAGE_DIRECTORY / "ratings.json"
TRAINING_DATA_DIRECTORY = PROJECT_ROOT / "training_data" / "_elo_game_arena"

ROWS = 25
COLS = 25
HOST = "0.0.0.0"
PORT = 8003
BASE_RATING = 1200.0
K_FACTOR = 32.0

# Classic rollout MCTS settings.
CLASSIC_MCTS_SIMULATION_SECONDS = 90.0
CLASSIC_MCTS_WORKERS = 14

# Neural policy/value MCTS settings.
NEURAL_MCTS_SIMULATION_SECONDS = 15.0
NEURAL_MCTS_C_PUCT = 1.5
NEURAL_MCTS_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "25x25_088622_new_data_dual_head_v1.keras"
)


def _number_slug(value):
    return format(value, "g").replace(".", "p")


@dataclass(frozen=True, slots=True)
class PlayerPreset:
    """One immutable AI identity used by the rating table."""

    id: str
    label: str
    kind: str
    simulation_seconds: float
    workers: int = 1
    c_puct: float = 1.5
    model_path: Path | None = None


AI_PRESETS = (
    PlayerPreset(
        id=(
            f"classic-mcts-{_number_slug(CLASSIC_MCTS_SIMULATION_SECONDS)}s-"
            f"w{CLASSIC_MCTS_WORKERS}"
        ),
        label=(
            f"Classic MCTS · {CLASSIC_MCTS_SIMULATION_SECONDS:g} s · "
            f"{CLASSIC_MCTS_WORKERS} workers"
        ),
        kind="classic_mcts",
        simulation_seconds=CLASSIC_MCTS_SIMULATION_SECONDS,
        workers=CLASSIC_MCTS_WORKERS,
    ),
    PlayerPreset(
        id=(
            f"neural-mcts-{_number_slug(NEURAL_MCTS_SIMULATION_SECONDS)}s-"
            f"c{_number_slug(NEURAL_MCTS_C_PUCT)}-25x25-v1"
        ),
        label=f"Neural MCTS · {NEURAL_MCTS_SIMULATION_SECONDS:g} s · 25x25 v1",
        kind="neural_mcts",
        simulation_seconds=NEURAL_MCTS_SIMULATION_SECONDS,
        c_puct=NEURAL_MCTS_C_PUCT,
        model_path=NEURAL_MCTS_MODEL_PATH,
    ),
)

PRESETS_BY_ID = {preset.id: preset for preset in AI_PRESETS}
HUMAN_PRESET_ID = "human"
