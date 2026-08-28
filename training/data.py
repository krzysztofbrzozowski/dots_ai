"""Collect one played MCTS trajectory and persist it as a compressed NPZ.

Files deliberately retain the absolute Player 1 / Player 2 representation.
Canonicalization to the player-to-move perspective belongs in the future
training loader, so the same self-play data can support either perspective.
"""

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # Support package and direct imports
    from game.board import PLAYER_1, PLAYER_2
except ImportError:  # pragma: no cover - direct module execution
    from ..game.board import PLAYER_1, PLAYER_2


SCHEMA_VERSION = 1


class SelfPlayTrajectory:
    """Buffer the actual states and root statistics from one played game."""

    def __init__(
        self,
        simulation_seconds=None,
        simulations_number=None,
        rollout_batch_size=1,
    ):
        self._simulation_seconds = simulation_seconds
        self._simulations_number = simulations_number
        self._rollout_batch_size = rollout_batch_size
        self._board_shape = None
        self._boards = []
        self._territories = []
        self._next_players = []
        self._scores = []
        self._q_values = []
        self._visit_counts = []
        self._legal_masks = []
        self._selected_actions = []
        self._completed_rollouts = []
        self._search_elapsed_seconds = []

    def __len__(self):
        return len(self._boards)

    def record_search(self, state, root, selected_action, search_stats):
        """Record the pre-move state and completed statistics at the root.

        Root children are mapped by ``child.action``. Their list order is not
        used, so each stored value is aligned with its physical board cell.
        ``q_values`` use the root's player-to-move perspective, matching the
        current MCTS node definition; the board itself remains absolute.
        """
        shape = tuple(int(size) for size in state.board.shape)
        if self._board_shape is None:
            self._board_shape = shape
        elif shape != self._board_shape:
            raise ValueError("board shape cannot change during one trajectory")
        if root.state is not state:
            raise ValueError("MCTS root must represent the recorded pre-move state")

        action = tuple(int(value) for value in selected_action)
        if action not in {child.action for child in root.children}:
            raise ValueError("selected action must identify an expanded root child")

        legal_mask = np.zeros(shape, dtype=np.uint8)
        for legal_action in state.get_legal_actions():
            legal_mask[legal_action] = 1

        # Save the q and n as 2D map
        # Each position will show q, n
        q_values = np.zeros(shape, dtype=np.float32)
        visit_counts = np.zeros(shape, dtype=np.int64)
        for child in root.children:
            q_values[child.action] = child.q
            visit_counts[child.action] = int(child.n)

        self._boards.append(np.array(state.board, dtype=np.int8, copy=True))
        self._territories.append(
            np.array(state.territory, dtype=np.int8, copy=True)
        )
        self._next_players.append(int(state.next_to_move))
        self._scores.append(
            (int(state.score[PLAYER_1]), int(state.score[PLAYER_2]))
        )
        self._q_values.append(q_values)
        self._visit_counts.append(visit_counts)
        self._legal_masks.append(legal_mask)
        self._selected_actions.append(action)
        self._completed_rollouts.append(int(search_stats.completed_rollouts))
        self._search_elapsed_seconds.append(float(search_stats.elapsed_seconds))

    def save(self, output_directory, final_result):
        """Atomically write this completed trajectory and return its path."""
        if not self._boards:
            raise ValueError("cannot save an empty self-play trajectory")
        if final_result not in (PLAYER_1, 0, PLAYER_2):
            raise ValueError("final_result must be PLAYER_1, draw, or PLAYER_2")

        rows, cols = self._board_shape
        size_directory = Path(output_directory) / f"{rows}x{cols}"
        size_directory.mkdir(parents=True, exist_ok=True)

        game_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc)
        timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        destination = size_directory / f"game_{timestamp}_{game_id}.npz"

        arrays = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
            "game_id": np.asarray(game_id),
            "created_at_utc": np.asarray(created_at.isoformat()),
            "board_shape": np.asarray(self._board_shape, dtype=np.int16),
            "boards": np.stack(self._boards),
            "territories": np.stack(self._territories),
            "next_players": np.asarray(self._next_players, dtype=np.int8),
            "scores": np.asarray(self._scores, dtype=np.int32),
            "q_values": np.stack(self._q_values),
            "q_perspective": np.asarray("player_to_move"),
            "visit_counts": np.stack(self._visit_counts),
            "legal_masks": np.stack(self._legal_masks),
            "selected_actions": np.asarray(
                self._selected_actions,
                dtype=np.int16,
            ),
            "completed_rollouts": np.asarray(
                self._completed_rollouts,
                dtype=np.int64,
            ),
            "search_elapsed_seconds": np.asarray(
                self._search_elapsed_seconds,
                dtype=np.float64,
            ),
            "final_result": np.asarray(final_result, dtype=np.int8),
            "final_result_perspective": np.asarray("player_1"),
            "search_budget_type": np.asarray(
                "seconds"
                if self._simulation_seconds is not None
                else "simulations"
            ),
            "requested_simulation_seconds": np.asarray(
                self._simulation_seconds
                if self._simulation_seconds is not None
                else np.nan,
                dtype=np.float64,
            ),
            "requested_simulations": np.asarray(
                self._simulations_number
                if self._simulation_seconds is None
                and self._simulations_number is not None
                else -1,
                dtype=np.int64,
            ),
            "rollout_batch_size": np.asarray(
                self._rollout_batch_size,
                dtype=np.int32,
            ),
        }

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{game_id}_",
            suffix=".tmp",
            dir=size_directory,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                np.savez_compressed(temporary_file, **arrays)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

        return destination


def load_self_play_game(path):
    """Load one NPZ without pickle and return independent NumPy arrays."""
    with np.load(path, allow_pickle=False) as stored:
        schema_version = int(stored["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported self-play schema {schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        return {name: stored[name].copy() for name in stored.files}
