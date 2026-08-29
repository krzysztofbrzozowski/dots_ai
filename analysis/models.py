"""Canonical data model used by every saved-game schema adapter.

The browser never receives raw NPZ field names directly from the loader. Each
supported storage schema is converted to this model first, which gives the GUI
one stable representation even when the on-disk schema changes in the future.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AnalysisGame:
    """One validated self-play trajectory held in read-only NumPy arrays."""

    file_name: str
    schema_version: int
    game_id: str
    created_at_utc: str
    board_shape: tuple[int, int]
    final_result: int
    final_result_perspective: str
    q_perspective: str
    search_budget_type: str
    requested_simulation_seconds: float | None
    requested_simulations: int | None
    rollout_batch_size: int
    boards: np.ndarray
    territories: np.ndarray
    next_players: np.ndarray
    scores: np.ndarray
    q_values: np.ndarray
    visit_counts: np.ndarray
    legal_masks: np.ndarray
    selected_actions: np.ndarray
    completed_rollouts: np.ndarray
    search_elapsed_seconds: np.ndarray

    @property
    def rows(self):
        return self.board_shape[0]

    @property
    def cols(self):
        return self.board_shape[1]

    @property
    def frame_count(self):
        """Number of saved pre-move decision states in the trajectory."""
        return int(self.boards.shape[0])

