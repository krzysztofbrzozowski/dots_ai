"""Dual-head inference shared by the command-line example and analysis GUI."""

from pathlib import Path
from threading import Lock
import sys

import numpy as np


# --- START PATHS AND SETTINGS
# Keeping the project root on sys.path lets this file run both as a module and
# as a regular Python file during quick local experiments.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "25x25_088622_new_data_dual_head_v1.keras"
)
# --- END PATHS AND SETTINGS


from game.board import PLAYER_1, PLAYER_2
from game.enclosure import DotsGame
from game.groups import rebuild_groups


class DualHeadUnavailableError(RuntimeError):
    """The configured dual-head model could not be loaded."""


# Keep the old public name working for code that only uses the value endpoint.
ValueHeadUnavailableError = DualHeadUnavailableError


# --- START MODEL INPUT PREPARATION
def game_state_to_model_input(state):
    """Convert one ``DotsGame`` position to the model's two input tensors.

    The five board channels are always relative to the player whose turn it is:
    my dots, opponent dots, my territory, opponent territory, and legal moves.
    Scores use the same perspective and board-area normalization as training.
    """
    player = state.next_to_move
    legal_moves = (state.board == 0) & (state.territory == 0)

    board_features = np.stack(
        (
            state.board == player,
            state.board == -player,
            state.territory == player,
            state.territory == -player,
            legal_moves,
        ),
        axis=-1,
    ).astype(np.float32)

    board_area = np.float32(state.board.size)
    score_features = np.asarray(
        (state.score[player], state.score[-player]),
        dtype=np.float32,
    ) / board_area

    # Keras expects the first dimension to identify the sample in a batch.
    return (
        board_features[np.newaxis, ...],
        score_features[np.newaxis, ...],
    )


def analysis_frame_to_game_state(saved_game, frame_index):
    """Rebuild a playable ``DotsGame`` from one stored analysis frame."""
    if not 0 <= frame_index < saved_game.frame_count:
        raise IndexError(f"Frame {frame_index} does not exist")

    state = DotsGame(saved_game.rows, saved_game.cols)
    state.board = np.array(saved_game.boards[frame_index], dtype=int, copy=True)
    state.territory = np.array(
        saved_game.territories[frame_index],
        dtype=int,
        copy=True,
    )
    state.groups = rebuild_groups(state.board, state.territory)
    state.score = {
        PLAYER_1: int(saved_game.scores[frame_index, 0]),
        PLAYER_2: int(saved_game.scores[frame_index, 1]),
    }
    state.next_to_move = int(saved_game.next_players[frame_index])
    return state
# --- END MODEL INPUT PREPARATION


# --- START DUAL-HEAD PREDICTION
def _validate_model_inputs(model, model_input):
    """Fail with a useful message when a checkpoint has a different schema."""
    input_shapes = model.input_shape
    if not (
        isinstance(input_shapes, (tuple, list))
        and len(input_shapes) == 2
        and all(isinstance(shape, (tuple, list)) for shape in input_shapes)
    ):
        raise ValueError(
            "Model must have board (rows, cols, 5) and score (2,) inputs; "
            "retrain legacy single-input checkpoints"
        )

    expected_shapes = tuple(tuple(shape[1:]) for shape in input_shapes)
    actual_shapes = tuple(value.shape[1:] for value in model_input)
    if actual_shapes != expected_shapes:
        raise ValueError(f"Model expects {expected_shapes}, got {actual_shapes}")


def _named_model_output(model, predictions, output_name):
    """Return one named output from either dict- or list-style Keras results."""
    if isinstance(predictions, dict):
        if output_name not in predictions:
            raise ValueError(f"Model does not provide a {output_name!r} output")
        return predictions[output_name]

    if isinstance(predictions, (tuple, list)):
        output_names = list(getattr(model, "output_names", ()))
        if output_name not in output_names:
            raise ValueError(f"Model does not provide a {output_name!r} output")
        return predictions[output_names.index(output_name)]

    # A single tensor is accepted only for the legacy value-only test model.
    if output_name == "value":
        return predictions
    raise ValueError(f"Model does not provide a {output_name!r} output")


def _predict_outputs(model, state):
    """Validate and run one model inference for a game state."""
    model_input = game_state_to_model_input(state)
    _validate_model_inputs(model, model_input)
    return model.predict(model_input, verbose=0)


def _value_result(model, predictions):
    """Convert the named value output into loss/draw/win fields."""
    probabilities = np.asarray(
        _named_model_output(model, predictions, "value")
    )[0]

    # Training labels are 0=loss, 1=draw, and 2=win for the player to move.
    if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Model must return three finite loss/draw/win probabilities")

    loss, draw, win = (float(value) for value in probabilities)
    return {
        "loss": loss,
        "draw": draw,
        "win": win,
        "value": win - loss,
        "source": "model",
    }


def predict_game_state(model, state):
    """Predict loss/draw/win probabilities for the player whose turn it is."""
    return _value_result(model, _predict_outputs(model, state))


def predict_policy(model, state):
    """Return a legal-move probability map for the current position.

    The network produces one raw logit per board cell. Illegal cells are
    excluded before a numerically stable softmax, so the returned legal
    probabilities sum to one and every illegal cell is exactly zero.
    """
    # Get the policy head predictions
    predictions = _predict_outputs(model, state)
    # returns
    # predictions['policy'] -> array[[2.8634408, 0.24973416, -1.2134645, -0.7428496, ...]]
    # ...
    # predictions['value'] -> array[[[0.36447012, 0.045201804, 0.5903281]]
    # ...
    # GET THE VALUES ONLY FOR policy FROM predictions
    logits = np.asarray(
        _named_model_output(model, predictions, "policy")
    )[0]
    action_count = state.board.size
    if logits.shape != (action_count,) or not np.all(np.isfinite(logits)):
        raise ValueError(
            f"Model must return {action_count} finite policy logits"
        )
    # Create logical mask from lega/possible moves -> before: (25, 25) -> reshape(-1) -> after: (625,)
    legal_mask = ((state.board == 0) & (state.territory == 0)).reshape(-1)
    if not np.any(legal_mask):
        raise ValueError("Cannot predict policy for a position without legal moves")

    # NumPy boolean masking/indexing
    legal_logits = logits[legal_mask]
    # Converts legal-move logits into positive weights
    # Subtracting the maximum prevents numerical overflow 
    # when calculating the exponential function
    legal_exponentials = np.exp(legal_logits - np.max(legal_logits))

    # Creates a zero-filled vector with one entry for every possible board action
    probabilities = np.zeros(action_count, dtype=np.float32)

    # Normalizes the legal moves weights (legal_expotentials) into probabilities
    # and assigns them only to legal-action positions
    # Probabilities sum = 1
    probabilities[legal_mask] = legal_exponentials / legal_exponentials.sum()
    policy_map = probabilities.reshape(state.board.shape)

    top_indices = np.flatnonzero(legal_mask)
    top_indices = top_indices[np.argsort(probabilities[top_indices])[::-1]][:5]
    top_moves = []
    for flat_index in top_indices:
        row, col = np.unravel_index(flat_index, state.board.shape)
        top_moves.append(
            {
                "coordinate": [int(row), int(col)],
                "probability": float(probabilities[flat_index]),
            }
        )

    return {
        "policy": policy_map.tolist(),
        "top_moves": top_moves,
        "source": "model",
    }


def predict_move(model, state, move):
    """Evaluate a legal candidate move from the moving player's perspective.

    The model sees the position after the move, when the opponent is next to
    play. Swapping win and loss converts that result back to the perspective of
    the player who selected the candidate move.
    """
    moving_player = state.next_to_move
    next_state = state.move(move)

    # A terminal move has an exact game result, so no neural estimate is needed.
    if next_state.game_result is not None:
        result_for_player = next_state.game_result * moving_player
        return {
            "loss": float(result_for_player == -1),
            "draw": float(result_for_player == 0),
            "win": float(result_for_player == 1),
            "value": float(result_for_player),
            "source": "terminal_result",
        }

    opponent_prediction = predict_game_state(model, next_state)
    return {
        "loss": opponent_prediction["win"],
        "draw": opponent_prediction["draw"],
        "win": opponent_prediction["loss"],
        "value": -opponent_prediction["value"],
        "source": "model",
    }


class DualHeadPredictor:
    """Load one Keras checkpoint lazily and reuse it across GUI requests."""

    def __init__(self, model_path=MODEL_PATH, model_loader=None):
        self.model_path = Path(model_path)
        self._model_loader = model_loader
        self._model = None
        self._load_lock = Lock()
        self._prediction_lock = Lock()

    @property
    def model_name(self):
        return self.model_path.name

    def _load_model(self):
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise DualHeadUnavailableError(
                    f"Dual-head model was not found: {self.model_path}"
                )

            try:
                if self._model_loader is None:
                    import keras

                    model_loader = keras.models.load_model
                else:
                    model_loader = self._model_loader
                self._model = model_loader(self.model_path, compile=False)
            except Exception as error:
                raise DualHeadUnavailableError(
                    f"Could not load dual-head model {self.model_name}: {error}"
                ) from error

        return self._model

    def predict_analysis_move(self, saved_game, frame_index, move):
        """Predict one candidate move selected in the analysis GUI."""
        state = analysis_frame_to_game_state(saved_game, frame_index)
        return self.predict_state_move(state, move)

    def predict_state_move(self, state, move):
        """Predict one move from an already constructed game state."""
        with self._prediction_lock:
            prediction = predict_move(self._load_model(), state, move)
        return {
            **prediction,
            "coordinate": [int(move[0]), int(move[1])],
            "player": int(state.next_to_move),
            "model": self.model_name,
        }

    def predict_analysis_policy(self, saved_game, frame_index):
        """Predict the complete policy map for one saved analysis frame."""
        state = analysis_frame_to_game_state(saved_game, frame_index)
        return self.predict_state_policy(state)

    def predict_state_policy(self, state):
        """Predict policy from an already constructed game state."""
        with self._prediction_lock:
            prediction = predict_policy(self._load_model(), state)
        return {
            **prediction,
            "player": int(state.next_to_move),
            "model": self.model_name,
        }
# --- END DUAL-HEAD PREDICTION


# Preserve the previous class name for callers that only request head value.
ValueHeadPredictor = DualHeadPredictor


# --- START MANUAL EXAMPLE
def main():
    """Run one small prediction example from the command line."""
    predictor = DualHeadPredictor()

    # Build the position before the candidate move with zero-based coordinates.
    state = DotsGame(10, 10)
    manual_moves = [(4, 4), (2, 1), (9, 9), (2, 3), (9, 8), (3, 2)]
    for move in manual_moves:
        state = state.move(move)

    candidate_move = (0, 0)
    prediction = predictor.predict_state_move(state, candidate_move)

    print("Position before the candidate move:")
    print(state.render())
    print(f"Player to move: {state.next_to_move:+d}")
    print(f"Scores: {state.score}")
    print(f"Candidate move (row, column): {candidate_move}")
    print(f"Model: {predictor.model_name}")
    print(f"Loss:  {prediction['loss']:.2%}")
    print(f"Draw:  {prediction['draw']:.2%}")
    print(f"Win:   {prediction['win']:.2%}")
    print(f"Value: {prediction['value']:+.4f}")


if __name__ == "__main__":
    main()
# --- END MANUAL EXAMPLE
