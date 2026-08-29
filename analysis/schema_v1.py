"""Validation and canonicalization for self-play NPZ schema version 1."""

from pathlib import Path

import numpy as np

from analysis.errors import AnalysisFileError
from analysis.models import AnalysisGame
from game.board import EMPTY, PLAYER_1, PLAYER_2


SCHEMA_VERSION = 1

REQUIRED_FIELDS = {
    "schema_version",
    "game_id",
    "created_at_utc",
    "board_shape",
    "boards",
    "territories",
    "next_players",
    "scores",
    "q_values",
    "q_perspective",
    "visit_counts",
    "legal_masks",
    "selected_actions",
    "completed_rollouts",
    "search_elapsed_seconds",
    "final_result",
    "final_result_perspective",
    "search_budget_type",
    "requested_simulation_seconds",
    "requested_simulations",
    "rollout_batch_size",
}


def _scalar(arrays, name):
    value = arrays[name]
    if value.shape != ():
        raise AnalysisFileError(f"'{name}' must be one scalar value")
    return value.item()


def _require_shape(name, array, expected_shape):
    if array.shape != expected_shape:
        raise AnalysisFileError(
            f"'{name}' has shape {array.shape}, expected {expected_shape}"
        )


def _require_integer_array(name, array):
    if not np.issubdtype(array.dtype, np.integer):
        raise AnalysisFileError(f"'{name}' must contain integer values")


def _require_finite_array(name, array):
    if not np.issubdtype(array.dtype, np.number):
        raise AnalysisFileError(f"'{name}' must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise AnalysisFileError(f"'{name}' contains NaN or infinite values")


def _require_values(name, array, allowed_values):
    invalid_values = set(np.unique(array).tolist()) - set(allowed_values)
    if invalid_values:
        shown_values = ", ".join(str(value) for value in sorted(invalid_values))
        raise AnalysisFileError(
            f"'{name}' contains unsupported values: {shown_values}"
        )


def _make_arrays_read_only(arrays):
    """Prevent an API request from accidentally mutating a loaded session."""
    for array in arrays.values():
        array.setflags(write=False)


def adapt_schema_v1(arrays, file_name):
    """Validate schema v1 arrays and return the canonical analysis model."""
    missing_fields = sorted(REQUIRED_FIELDS - arrays.keys())
    if missing_fields:
        raise AnalysisFileError(
            "The NPZ file is missing required fields: " + ", ".join(missing_fields)
        )

    schema_version = int(_scalar(arrays, "schema_version"))
    if schema_version != SCHEMA_VERSION:
        raise AnalysisFileError(
            f"Schema adapter v1 cannot read schema version {schema_version}"
        )

    board_shape_array = arrays["board_shape"]
    _require_shape("board_shape", board_shape_array, (2,))
    _require_integer_array("board_shape", board_shape_array)
    rows, cols = (int(value) for value in board_shape_array)
    if rows <= 0 or cols <= 0:
        raise AnalysisFileError("'board_shape' dimensions must be positive")

    boards = arrays["boards"]
    if boards.ndim != 3:
        raise AnalysisFileError("'boards' must have shape [frames, rows, cols]")
    frame_count = int(boards.shape[0])
    if frame_count <= 0:
        raise AnalysisFileError("The saved trajectory does not contain any frames")

    board_stack_shape = (frame_count, rows, cols)
    shapes = {
        "boards": board_stack_shape,
        "territories": board_stack_shape,
        "q_values": board_stack_shape,
        "visit_counts": board_stack_shape,
        "legal_masks": board_stack_shape,
        "next_players": (frame_count,),
        "scores": (frame_count, 2),
        "selected_actions": (frame_count, 2),
        "completed_rollouts": (frame_count,),
        "search_elapsed_seconds": (frame_count,),
    }
    for name, expected_shape in shapes.items():
        _require_shape(name, arrays[name], expected_shape)

    integer_fields = (
        "boards",
        "territories",
        "next_players",
        "scores",
        "visit_counts",
        "legal_masks",
        "selected_actions",
        "completed_rollouts",
    )
    for name in integer_fields:
        _require_integer_array(name, arrays[name])

    _require_finite_array("q_values", arrays["q_values"])
    _require_finite_array(
        "search_elapsed_seconds",
        arrays["search_elapsed_seconds"],
    )

    player_values = (PLAYER_2, EMPTY, PLAYER_1)
    _require_values("boards", boards, player_values)
    _require_values("territories", arrays["territories"], player_values)
    _require_values("next_players", arrays["next_players"], (PLAYER_1, PLAYER_2))
    _require_values("legal_masks", arrays["legal_masks"], (0, 1))

    if np.any(arrays["scores"] < 0):
        raise AnalysisFileError("'scores' cannot contain negative values")
    if np.any(arrays["visit_counts"] < 0):
        raise AnalysisFileError("'visit_counts' cannot contain negative values")
    if np.any(arrays["completed_rollouts"] < 0):
        raise AnalysisFileError("'completed_rollouts' cannot be negative")
    if np.any(arrays["search_elapsed_seconds"] < 0):
        raise AnalysisFileError("'search_elapsed_seconds' cannot be negative")

    selected_actions = arrays["selected_actions"]
    selected_rows = selected_actions[:, 0]
    selected_cols = selected_actions[:, 1]
    if (
        np.any(selected_rows < 0)
        or np.any(selected_rows >= rows)
        or np.any(selected_cols < 0)
        or np.any(selected_cols >= cols)
    ):
        raise AnalysisFileError("'selected_actions' contains a coordinate off the board")

    frame_indexes = np.arange(frame_count)
    selected_are_legal = arrays["legal_masks"][
        frame_indexes,
        selected_rows,
        selected_cols,
    ]
    if not np.all(selected_are_legal == 1):
        raise AnalysisFileError("Every selected action must be legal in its frame")

    selected_board_values = boards[
        frame_indexes,
        selected_rows,
        selected_cols,
    ]
    selected_territory_values = arrays["territories"][
        frame_indexes,
        selected_rows,
        selected_cols,
    ]
    if np.any(selected_board_values != EMPTY) or np.any(
        selected_territory_values != EMPTY
    ):
        raise AnalysisFileError(
            "Every selected action must point to an empty, active board position"
        )

    final_result = int(_scalar(arrays, "final_result"))
    if final_result not in (PLAYER_1, 0, PLAYER_2):
        raise AnalysisFileError("'final_result' must be 1, 0, or -1")

    q_perspective = str(_scalar(arrays, "q_perspective"))
    if q_perspective != "player_to_move":
        raise AnalysisFileError(
            "Schema v1 expects q_values from the 'player_to_move' perspective"
        )

    final_result_perspective = str(
        _scalar(arrays, "final_result_perspective")
    )
    if final_result_perspective != "player_1":
        raise AnalysisFileError(
            "Schema v1 expects final_result from the 'player_1' perspective"
        )

    search_budget_type = str(_scalar(arrays, "search_budget_type"))
    if search_budget_type not in ("seconds", "simulations"):
        raise AnalysisFileError(
            "'search_budget_type' must be 'seconds' or 'simulations'"
        )

    requested_seconds_value = float(
        _scalar(arrays, "requested_simulation_seconds")
    )
    requested_simulations_value = int(_scalar(arrays, "requested_simulations"))
    if search_budget_type == "seconds":
        if not np.isfinite(requested_seconds_value) or requested_seconds_value <= 0:
            raise AnalysisFileError(
                "A seconds-based search must have a positive requested duration"
            )
        requested_simulation_seconds = requested_seconds_value
        requested_simulations = None
    else:
        if requested_simulations_value <= 0:
            raise AnalysisFileError(
                "A simulation-based search must have a positive simulation count"
            )
        requested_simulation_seconds = None
        requested_simulations = requested_simulations_value

    rollout_batch_size = int(_scalar(arrays, "rollout_batch_size"))
    if rollout_batch_size <= 0:
        raise AnalysisFileError("'rollout_batch_size' must be positive")

    _make_arrays_read_only(arrays)

    return AnalysisGame(
        file_name=Path(file_name).name,
        schema_version=schema_version,
        game_id=str(_scalar(arrays, "game_id")),
        created_at_utc=str(_scalar(arrays, "created_at_utc")),
        board_shape=(rows, cols),
        final_result=final_result,
        final_result_perspective=final_result_perspective,
        q_perspective=q_perspective,
        search_budget_type=search_budget_type,
        requested_simulation_seconds=requested_simulation_seconds,
        requested_simulations=requested_simulations,
        rollout_batch_size=rollout_batch_size,
        boards=arrays["boards"],
        territories=arrays["territories"],
        next_players=arrays["next_players"],
        scores=arrays["scores"],
        q_values=arrays["q_values"],
        visit_counts=arrays["visit_counts"],
        legal_masks=arrays["legal_masks"],
        selected_actions=arrays["selected_actions"],
        completed_rollouts=arrays["completed_rollouts"],
        search_elapsed_seconds=arrays["search_elapsed_seconds"],
    )

