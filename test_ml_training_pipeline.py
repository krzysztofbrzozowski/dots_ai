"""Tests for exact square-board augmentation and batched array input."""

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from game.enclosure import DotsGame
from ml.data_loader import game_to_samples
from ml.predictor import (
    analysis_frame_to_game_state,
    game_state_to_model_input,
    predict_move,
)
from ml.training_pipeline import (
    batched_array_dataset,
    d4_symmetries,
    random_d4_augmentation,
)


def test_training_and_prediction_use_matching_board_and_scalar_score_inputs():
    game = {
        "boards": np.zeros((2, 2, 2), dtype=np.int8),
        "territories": np.zeros((2, 2, 2), dtype=np.int8),
        "next_players": np.asarray([1, -1], dtype=np.int8),
        "scores": np.asarray([[2, 1], [3, 4]], dtype=np.int32),
        "legal_masks": np.ones((2, 2, 2), dtype=np.uint8),
        "final_result": np.asarray(1, dtype=np.int8),
    }
    (board_samples, score_features), labels = game_to_samples(game)

    np.testing.assert_array_equal(labels, [2, 0])
    assert board_samples.shape == (2, 2, 2, 5)
    np.testing.assert_allclose(score_features, [[0.5, 0.25], [1.0, 0.75]])

    state = DotsGame(2, 2)
    state.score[1] = 2
    state.score[-1] = 1
    prediction_board, prediction_scores = game_state_to_model_input(state)
    assert prediction_board.shape == (1, 2, 2, 5)
    np.testing.assert_allclose(prediction_scores, [[0.5, 0.25]])


def test_candidate_prediction_is_returned_for_the_player_making_the_move():
    class FakeValueModel:
        input_shape = [(None, 2, 2, 5), (None, 2)]

        def predict(self, model_input, verbose=0):
            assert model_input[0].shape == (1, 2, 2, 5)
            assert model_input[1].shape == (1, 2)
            assert verbose == 0
            # The model sees the opponent after the candidate move.
            return np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32)

    state = DotsGame(2, 2)
    prediction = predict_move(FakeValueModel(), state, (0, 0))

    # Swapping loss and win converts the opponent-facing model output back to
    # the perspective of the player who made the candidate move.
    assert abs(prediction["loss"] - 0.5) < 1e-6
    assert abs(prediction["draw"] - 0.3) < 1e-6
    assert abs(prediction["win"] - 0.2) < 1e-6
    assert abs(prediction["value"] - (-0.3)) < 1e-6
    assert prediction["source"] == "model"


def test_saved_analysis_frame_becomes_an_independent_playable_state():
    saved_game = SimpleNamespace(
        rows=3,
        cols=3,
        frame_count=1,
        boards=np.asarray(
            [[[1, 1, 0], [0, -1, 0], [0, 0, -1]]],
            dtype=np.int8,
        ),
        territories=np.asarray(
            [[[0, 0, 0], [0, 0, 0], [0, 0, -1]]],
            dtype=np.int8,
        ),
        scores=np.asarray([[4, 2]], dtype=np.int32),
        next_players=np.asarray([-1], dtype=np.int8),
    )

    state = analysis_frame_to_game_state(saved_game, 0)

    np.testing.assert_array_equal(state.board, saved_game.boards[0])
    np.testing.assert_array_equal(state.territory, saved_game.territories[0])
    assert state.score == {1: 4, -1: 2}
    assert state.next_to_move == -1
    assert state.groups.connected((0, 0), (0, 1))
    assert (2, 2) not in state.groups

    state.board[0, 0] = 0
    assert saved_game.boards[0, 0, 0] == 1


def test_d4_symmetries_are_exact_rotations_and_reflections():
    board = np.arange(9, dtype=np.float32).reshape(3, 3)
    samples = tf.constant(board[None, :, :, None])

    actual = d4_symmetries(samples).numpy()[0, :, :, :, 0]
    rotations = [np.rot90(board, turns) for turns in range(4)]
    expected = np.stack(
        rotations + [np.fliplr(rotated) for rotated in rotations]
    )

    np.testing.assert_array_equal(actual, expected)
    assert len({candidate.tobytes() for candidate in actual}) == 8


def test_random_augmentation_preserves_shape_targets_and_discrete_values():
    board = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    board_samples = tf.constant(np.repeat(board[None, ...], 32, axis=0))
    score_features = tf.constant(np.arange(64, dtype=np.float32).reshape(32, 2))
    targets = tf.constant(np.arange(32), dtype=tf.int64)

    (augmented, returned_scores), returned_targets = random_d4_augmentation(
        (board_samples, score_features), targets
    )
    valid_candidates = {
        candidate.tobytes()
        for candidate in d4_symmetries(board_samples[:1]).numpy()[0]
    }

    assert augmented.shape == board_samples.shape
    np.testing.assert_array_equal(returned_scores.numpy(), score_features.numpy())
    np.testing.assert_array_equal(returned_targets.numpy(), targets.numpy())
    assert all(sample.tobytes() in valid_candidates for sample in augmented.numpy())


def test_batched_array_dataset_is_finite_and_preserves_every_sample():
    board_samples = np.arange(35, dtype=np.float32).reshape(5, 7)
    score_features = np.arange(10, dtype=np.float32).reshape(5, 2)
    labels = np.arange(5, dtype=np.int64)
    dataset = batched_array_dataset(
        (board_samples, score_features), labels, batch_size=2
    )

    batches = list(dataset.as_numpy_iterator())

    assert len(batches) == 3
    np.testing.assert_array_equal(
        np.concatenate([batch_inputs[0] for batch_inputs, _ in batches]),
        board_samples,
    )
    np.testing.assert_array_equal(
        np.concatenate([batch_inputs[1] for batch_inputs, _ in batches]),
        score_features,
    )
    np.testing.assert_array_equal(
        np.concatenate([batch_labels for _, batch_labels in batches]),
        labels,
    )


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} ML training-pipeline tests passed")
