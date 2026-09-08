"""Tests for exact square-board augmentation and batched array input."""

import numpy as np
import tensorflow as tf

from game.enclosure import DotsGame
from ml.data_loader import game_to_samples
from ml.predictor import game_state_to_model_input
from ml.training_pipeline import (
    batched_array_dataset,
    d4_symmetries,
    random_d4_augmentation,
)


def test_training_and_prediction_normalize_scores_by_board_area():
    game = {
        "boards": np.zeros((2, 2, 2), dtype=np.int8),
        "territories": np.zeros((2, 2, 2), dtype=np.int8),
        "next_players": np.asarray([1, -1], dtype=np.int8),
        "scores": np.asarray([[2, 1], [3, 4]], dtype=np.int32),
        "legal_masks": np.ones((2, 2, 2), dtype=np.uint8),
        "final_result": np.asarray(1, dtype=np.int8),
    }
    samples, labels = game_to_samples(game)

    np.testing.assert_array_equal(labels, [2, 0])
    np.testing.assert_allclose(samples[0, :, :, 5], 0.5)
    np.testing.assert_allclose(samples[0, :, :, 6], 0.25)
    np.testing.assert_allclose(samples[1, :, :, 5], 1.0)
    np.testing.assert_allclose(samples[1, :, :, 6], 0.75)

    state = DotsGame(2, 2)
    state.score[1] = 2
    state.score[-1] = 1
    prediction_sample = game_state_to_model_input(state)[0]
    np.testing.assert_allclose(prediction_sample[:, :, 5], 0.5)
    np.testing.assert_allclose(prediction_sample[:, :, 6], 0.25)


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
    samples = tf.constant(np.repeat(board[None, ...], 32, axis=0))
    targets = tf.constant(np.arange(32), dtype=tf.int64)

    augmented, returned_targets = random_d4_augmentation(samples, targets)
    valid_candidates = {
        candidate.tobytes()
        for candidate in d4_symmetries(samples[:1]).numpy()[0]
    }

    assert augmented.shape == samples.shape
    np.testing.assert_array_equal(returned_targets.numpy(), targets.numpy())
    assert all(sample.tobytes() in valid_candidates for sample in augmented.numpy())


def test_batched_array_dataset_is_finite_and_preserves_every_sample():
    samples = np.arange(35, dtype=np.float32).reshape(5, 7)
    labels = np.arange(5, dtype=np.int64)
    dataset = batched_array_dataset(samples, labels, batch_size=2)

    batches = list(dataset.as_numpy_iterator())

    assert len(batches) == 3
    np.testing.assert_array_equal(
        np.concatenate([batch_samples for batch_samples, _ in batches]),
        samples,
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
