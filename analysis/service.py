"""In-memory sessions and JSON-ready views for saved-game analysis."""

from collections import OrderedDict
from threading import RLock
from uuid import uuid4

import numpy as np


class AnalysisNotFoundError(LookupError):
    """The requested analysis session is no longer present in memory."""


class FrameNotFoundError(IndexError):
    """The requested frame does not exist in the selected game."""


class AnalysisStore:
    """Keep a small number of immutable analysis sessions for local browser tabs."""

    def __init__(self, maximum_sessions=4):
        if maximum_sessions <= 0:
            raise ValueError("maximum_sessions must be positive")
        self._maximum_sessions = maximum_sessions
        self._sessions = OrderedDict()
        self._lock = RLock()

    def add(self, game):
        analysis_id = uuid4().hex
        with self._lock:
            self._sessions[analysis_id] = game
            while len(self._sessions) > self._maximum_sessions:
                self._sessions.popitem(last=False)
        return analysis_id

    def get(self, analysis_id):
        with self._lock:
            game = self._sessions.get(analysis_id)
            if game is None:
                raise AnalysisNotFoundError(analysis_id)
            self._sessions.move_to_end(analysis_id)
            return game

    def remove(self, analysis_id):
        with self._lock:
            if self._sessions.pop(analysis_id, None) is None:
                raise AnalysisNotFoundError(analysis_id)

    def clear(self):
        """Remove all sessions; primarily useful for isolated tests."""
        with self._lock:
            self._sessions.clear()


def _coordinate(values):
    return [int(values[0]), int(values[1])]


def selected_action_statistics(
    q_values,
    visit_counts,
    legal_mask,
    action,
    policy_priors=None,
):
    """Return the action metrics shared by saved, replay, and live frames."""
    row, col = (int(value) for value in action)
    visits = int(visit_counts[row, col])
    raw_q = float(q_values[row, col])
    total_visits = int(visit_counts.sum())
    mean_value = raw_q / visits if visits else None
    policy = visits / total_visits if total_visits else 0.0
    prior = None
    prior_rank = None
    policy_delta = None
    policy_amplification = None
    if policy_priors is not None:
        prior = float(policy_priors[row, col])
        legal_priors = policy_priors[legal_mask == 1]
        prior_rank = 1 + int(np.count_nonzero(legal_priors > prior))
        policy_delta = policy - prior
        if prior > 0:
            policy_amplification = policy / prior

    visited_legal = (
        (legal_mask == 1)
        & (visit_counts > 0)
    )
    compared_visits = visit_counts[visited_legal]
    compared_mean_values = (
        q_values[visited_legal] / compared_visits
    )

    value_rank = None
    visit_rank = None
    if visits:
        value_rank = 1 + int(np.count_nonzero(compared_mean_values > mean_value))
        visit_rank = 1 + int(np.count_nonzero(compared_visits > visits))

    return {
        "coordinate": [row, col],
        "raw_q": raw_q,
        "visits": visits,
        "mean_value": mean_value,
        "policy": policy,
        "visit_share": policy,
        "prior": prior,
        "prior_rank": prior_rank,
        "policy_delta": policy_delta,
        "policy_amplification": policy_amplification,
        "value_rank": value_rank,
        "visit_rank": visit_rank,
    }


def _root_policy_priors(root, shape):
    """Return the exact root priors used by neural MCTS, when available."""

    if not root.children or not all(
        hasattr(child, "prior") for child in root.children
    ):
        return None

    policy_priors = np.zeros(shape, dtype=np.float32)
    for child in root.children:
        prior = float(child.prior)
        if not np.isfinite(prior) or prior < 0:
            raise ValueError("MCTS child priors must be finite and non-negative")
        policy_priors[child.action] = prior
    return policy_priors


def _saved_policy_priors(game, frame_index):
    """Read optional search-time priors from a backward-compatible game."""

    if game.policy_priors is None or game.has_policy_priors is None:
        return None
    if not bool(game.has_policy_priors[frame_index]):
        return None
    return game.policy_priors[frame_index]


def mcts_decision_frame(
    state,
    root,
    selected_action,
    search_stats,
    move_number,
    *,
    experiment=False,
):
    """Serialize one completed search for every GUI timeline source."""
    shape = state.board.shape
    q_values = np.zeros(shape, dtype=np.float32)
    visit_counts = np.zeros(shape, dtype=np.int64)
    policy_priors = _root_policy_priors(root, shape)
    legal_mask = np.zeros(shape, dtype=np.uint8)
    for action in state.get_legal_actions():
        legal_mask[action] = 1
    for child in root.children:
        q_values[child.action] = child.q
        visit_counts[child.action] = int(child.n)

    action = tuple(int(value) for value in selected_action)
    elapsed_seconds = float(search_stats.elapsed_seconds)
    completed_rollouts = int(search_stats.completed_rollouts)
    throughput = completed_rollouts / elapsed_seconds if elapsed_seconds else 0.0
    rows, cols = shape

    return {
        "experiment": bool(experiment),
        "index": None,
        "move_number": int(move_number),
        "rows": int(rows),
        "cols": int(cols),
        "board": state.board.tolist(),
        "territory": state.territory.tolist(),
        "player_to_move": int(state.next_to_move),
        "scores": {
            "player_1": int(state.score[1]),
            "player_2": int(state.score[-1]),
        },
        "q_values": q_values.tolist(),
        "q_perspective": "player_to_move",
        "visit_counts": visit_counts.tolist(),
        "policy_priors": (
            policy_priors.tolist() if policy_priors is not None else None
        ),
        "legal_mask": legal_mask.tolist(),
        "legal_move_count": int(np.count_nonzero(legal_mask)),
        "selected_action": list(action),
        "selected_action_statistics": selected_action_statistics(
            q_values,
            visit_counts,
            legal_mask,
            action,
            policy_priors,
        ),
        "completed_rollouts": completed_rollouts,
        "elapsed_seconds": elapsed_seconds,
        "rollouts_per_second": throughput,
        "is_first_frame": int(move_number) == 1,
        "is_last_frame": False,
    }


def analysis_summary(analysis_id, game):
    """Return metadata and lightweight descriptors for the complete timeline."""
    timeline = []
    for frame_index in range(game.frame_count):
        policy_priors = _saved_policy_priors(game, frame_index)
        selected = selected_action_statistics(
            game.q_values[frame_index],
            game.visit_counts[frame_index],
            game.legal_masks[frame_index],
            game.selected_actions[frame_index],
            policy_priors,
        )
        timeline.append(
            {
                "index": frame_index,
                "move_number": frame_index + 1,
                "player_to_move": int(game.next_players[frame_index]),
                "scores": {
                    "player_1": int(game.scores[frame_index, 0]),
                    "player_2": int(game.scores[frame_index, 1]),
                },
                "selected_action": selected["coordinate"],
                "selected_mean_value": selected["mean_value"],
                "selected_visits": selected["visits"],
                "selected_prior": selected["prior"],
                "selected_visit_share": selected["visit_share"],
                "completed_rollouts": int(
                    game.completed_rollouts[frame_index]
                ),
                "elapsed_seconds": float(
                    game.search_elapsed_seconds[frame_index]
                ),
            }
        )

    total_rollouts = int(game.completed_rollouts.sum())
    total_elapsed_seconds = float(game.search_elapsed_seconds.sum())
    average_throughput = (
        total_rollouts / total_elapsed_seconds if total_elapsed_seconds else 0.0
    )

    return {
        "analysis_id": analysis_id,
        "file_name": game.file_name,
        "schema_version": game.schema_version,
        "game_id": game.game_id,
        "created_at_utc": game.created_at_utc,
        "board": {
            "rows": game.rows,
            "cols": game.cols,
        },
        "frame_count": game.frame_count,
        "final_result": game.final_result,
        "final_result_perspective": game.final_result_perspective,
        "q_perspective": game.q_perspective,
        "search": {
            "budget_type": game.search_budget_type,
            "requested_simulation_seconds": game.requested_simulation_seconds,
            "requested_simulations": game.requested_simulations,
            "rollout_batch_size": game.rollout_batch_size,
            "total_rollouts": total_rollouts,
            "total_elapsed_seconds": total_elapsed_seconds,
            "average_rollouts_per_second": average_throughput,
        },
        "timeline": timeline,
    }


def analysis_frame(game, frame_index):
    """Return one board-sized decision frame and its derived statistics."""
    if not 0 <= frame_index < game.frame_count:
        raise FrameNotFoundError(frame_index)

    elapsed_seconds = float(game.search_elapsed_seconds[frame_index])
    completed_rollouts = int(game.completed_rollouts[frame_index])
    throughput = completed_rollouts / elapsed_seconds if elapsed_seconds else 0.0
    policy_priors = _saved_policy_priors(game, frame_index)

    return {
        "index": frame_index,
        "move_number": frame_index + 1,
        "rows": game.rows,
        "cols": game.cols,
        "board": game.boards[frame_index].tolist(),
        "territory": game.territories[frame_index].tolist(),
        "player_to_move": int(game.next_players[frame_index]),
        "scores": {
            "player_1": int(game.scores[frame_index, 0]),
            "player_2": int(game.scores[frame_index, 1]),
        },
        "q_values": game.q_values[frame_index].tolist(),
        "q_perspective": game.q_perspective,
        "visit_counts": game.visit_counts[frame_index].tolist(),
        "policy_priors": (
            policy_priors.tolist() if policy_priors is not None else None
        ),
        "legal_mask": game.legal_masks[frame_index].tolist(),
        "legal_move_count": int(np.count_nonzero(game.legal_masks[frame_index])),
        "selected_action": _coordinate(game.selected_actions[frame_index]),
        "selected_action_statistics": selected_action_statistics(
            game.q_values[frame_index],
            game.visit_counts[frame_index],
            game.legal_masks[frame_index],
            game.selected_actions[frame_index],
            policy_priors,
        ),
        "completed_rollouts": completed_rollouts,
        "elapsed_seconds": elapsed_seconds,
        "rollouts_per_second": throughput,
        "is_first_frame": frame_index == 0,
        "is_last_frame": frame_index == game.frame_count - 1,
    }
