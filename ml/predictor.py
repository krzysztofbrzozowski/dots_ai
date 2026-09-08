"""Manual value-model POC. Run with ``python -m ml.predictor``.

Edit manual_moves and move_to_evaluate in main() to evaluate a candidate move.
The resulting probabilities describe the player who makes that move.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "10x10_64322_5conv_d4_normalized.keras"
)

from game.enclosure import DotsGame


def game_state_to_model_input(state):
    """Encode a DotsGame as board and scalar score inputs matching data_loader.

    Board channels are my dots, opponent dots, my territory, opponent territory,
    and legal moves. "My" means state.next_to_move. The two scalar scores are
    normalized by board area, exactly as in the training loader.
    """
    player = state.next_to_move
    legal_mask = (state.board == 0) & (state.territory == 0)
    score_scale = np.float32(state.board.size)
    board_sample = np.stack(
        (
            state.board == player,
            state.board == -player,
            state.territory == player,
            state.territory == -player,
            legal_mask,
        ),
        axis=-1,
    ).astype(np.float32)
    score_features = np.asarray(
        (state.score[player], state.score[-player]),
        dtype=np.float32,
    ) / score_scale
    # Keras expects a batch dimension, even for a single position.
    return board_sample[np.newaxis, ...], score_features[np.newaxis, ...]


def predict_game_state(model, state):
    """Return loss/draw/win probabilities and the player-to-move value.

    Load the model once and reuse it when evaluating more positions.
    This is a raw network prediction, including for terminal positions.
    Future MCTS should use the exact engine result for terminal positions.
    """
    model_input = game_state_to_model_input(state)
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
        raise ValueError(
            f"Model expects {expected_shapes}, got {actual_shapes}"
        )

    # Training labels are 0=loss, 1=draw, 2=win, relative to the player to move.
    probabilities = model.predict(model_input, verbose=0)[0]
    p_loss, p_draw, p_win = map(float, probabilities)
    return {
        "loss": p_loss,
        "draw": p_draw,
        "win": p_win,
        "value": p_win - p_loss,
    }


def predict_move(model, state, move):
    """Evaluate a legal move from the moving player's perspective.

    move() validates the action and creates a new state, leaving state intact.
    The network sees the position AFTER the move, when it is the opponent's
    turn. Swap win/loss probabilities and negate value to report our chances.
    For a game-ending move, use the exact engine result instead of the network.
    """
    next_state = state.move(move)
    result = next_state.game_result
    if result is not None:
        result_for_player = result * state.next_to_move
        return {
            "loss": float(result_for_player == -1),
            "draw": float(result_for_player == 0),
            "win": float(result_for_player == 1),
            "value": float(result_for_player),
        }

    opponent_prediction = predict_game_state(model, next_state)
    return {
        "loss": opponent_prediction["win"],
        "draw": opponent_prediction["draw"],
        "win": opponent_prediction["loss"],
        "value": -opponent_prediction["value"],
    }


def main():
    import keras

    # This loads the checkpoint saved by main_ml.py; no training is started.
    test_model = keras.models.load_model(MODEL_PATH, compile=False)

    # Build the position BEFORE the candidate move using zero-based coordinates.
    # Using move() keeps territory, scores and next_to_move consistent.
    some_test_state = DotsGame(10, 10)
    some_test_state.next_to_move = 1
    manual_moves = [(4, 4), (2, 1), (9, 9), (2, 3), (9, 8), (3, 2)]
    for move in manual_moves:
        some_test_state = some_test_state.move(move)

    # This is the next move we want to evaluate, made by state.next_to_move.
    move_to_evaluate = (0, 0)
    prediction = predict_move(test_model, some_test_state, move_to_evaluate)
    print("Position before the candidate move:")
    print(some_test_state.render())
    print(f"Player to move: {some_test_state.next_to_move:+d}")
    print(f"Scores: {some_test_state.score}")
    print(f"Evaluated move (row, column, zero-based): {move_to_evaluate}")
    print(
        f"Outcome probabilities for player {some_test_state.next_to_move:+d} "
        "after making this move:"
    )
    print(f"  Loss: {prediction['loss']:.2%}")
    print(f"  Draw: {prediction['draw']:.2%}")
    print(f"  Win:  {prediction['win']:.2%}")
    print(f"  Value = P(win) - P(loss): {prediction['value']:+.4f}")


if __name__ == "__main__":
    main()
