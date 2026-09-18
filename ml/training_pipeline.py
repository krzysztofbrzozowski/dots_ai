"""TensorFlow input-pipeline helpers for value and policy/value training."""

import math

import numpy as np
import tensorflow as tf


def batched_array_dataset(
    samples,
    labels,
    batch_size,
    *,
    sample_weights=None,
    shuffle=False,
    seed=None,
):
    """Expose nested NumPy inputs and targets as finite shuffled batches.

    The generator copies only the current batch into a TensorFlow tensor. This
    avoids making another full in-memory copy of the multi-million-position
    training set, as ``Dataset.from_tensor_slices`` would do. Dictionaries are
    supported for multi-output targets and per-output sample weights.
    """
    sample_arrays = tf.nest.flatten(samples)
    label_arrays = tf.nest.flatten(labels)
    weight_arrays = (
        [] if sample_weights is None else tf.nest.flatten(sample_weights)
    )
    if not sample_arrays:
        raise ValueError("samples cannot be empty")
    if not label_arrays:
        raise ValueError("labels cannot be empty")
    number_of_samples = len(sample_arrays[0])
    all_arrays = (*sample_arrays, *label_arrays, *weight_arrays)
    if any(len(array) != number_of_samples for array in all_arrays):
        raise ValueError(
            "all inputs, targets, and weights must contain the same number of items"
        )
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if number_of_samples == 0:
        raise ValueError("samples and labels cannot be empty")

    random_generator = np.random.default_rng(seed)

    def select(structure, indices):
        return tf.nest.map_structure(lambda array: array[indices], structure)

    def batches():
        if shuffle:
            indices = np.arange(number_of_samples)
            random_generator.shuffle(indices)
            for start in range(0, number_of_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = (
                    select(samples, batch_indices),
                    select(labels, batch_indices),
                )
                if sample_weights is not None:
                    batch = (*batch, select(sample_weights, batch_indices))
                yield batch
        else:
            for start in range(0, number_of_samples, batch_size):
                stop = start + batch_size
                batch_indices = slice(start, stop)
                batch = (
                    select(samples, batch_indices),
                    select(labels, batch_indices),
                )
                if sample_weights is not None:
                    batch = (*batch, select(sample_weights, batch_indices))
                yield batch

    def tensor_spec(array):
        return tf.TensorSpec(
            shape=(None, *array.shape[1:]),
            dtype=tf.as_dtype(array.dtype),
        )

    output_signature = (
        tf.nest.map_structure(tensor_spec, samples),
        tf.nest.map_structure(tensor_spec, labels),
    )
    if sample_weights is not None:
        output_signature = (
            *output_signature,
            tf.nest.map_structure(tensor_spec, sample_weights),
        )
    dataset = tf.data.Dataset.from_generator(
        batches,
        output_signature=output_signature,
    )
    batch_count = math.ceil(number_of_samples / batch_size)
    return dataset.apply(tf.data.experimental.assert_cardinality(batch_count))


def d4_symmetries(samples):
    """Return all eight square-board symmetries for a batch of samples.

    The returned shape is ``(batch, 8, rows, columns, channels)``. Every
    spatial input plane is transformed together, while feature channels keep
    their meaning.
    """
    rotations = tuple(tf.image.rot90(samples, k=turns) for turns in range(4))
    reflections = tuple(
        tf.image.flip_left_right(rotated) for rotated in rotations
    )
    return tf.stack((*rotations, *reflections), axis=1)


def random_d4_augmentation(samples, targets):
    """Choose one exact D4 symmetry independently for every board in a batch."""
    board_samples, score_features = samples
    candidates = d4_symmetries(board_samples)
    symmetry_indices = tf.random.uniform(
        shape=(tf.shape(board_samples)[0],),
        minval=0,
        maxval=8,
        dtype=tf.int32,
    )
    augmented_samples = tf.gather(
        candidates,
        symmetry_indices,
        axis=1,
        batch_dims=1,
    )
    augmented_samples.set_shape(board_samples.shape)
    return (augmented_samples, score_features), targets


def random_dual_head_d4_augmentation(samples, targets, sample_weights):
    """Apply the same random D4 symmetry to a board and its policy map."""
    board_samples, score_features = samples
    board_shape = tf.shape(board_samples)
    policy_maps = tf.reshape(
        targets["policy"],
        tf.stack((board_shape[0], board_shape[1], board_shape[2], 1)),
    )
    board_candidates = d4_symmetries(board_samples)
    policy_candidates = d4_symmetries(policy_maps)
    symmetry_indices = tf.random.uniform(
        shape=(board_shape[0],),
        minval=0,
        maxval=8,
        dtype=tf.int32,
    )
    augmented_boards = tf.gather(
        board_candidates,
        symmetry_indices,
        axis=1,
        batch_dims=1,
    )
    augmented_policy = tf.gather(
        policy_candidates,
        symmetry_indices,
        axis=1,
        batch_dims=1,
    )
    augmented_boards.set_shape(board_samples.shape)
    augmented_policy = tf.reshape(augmented_policy, tf.shape(targets["policy"]))
    augmented_policy.set_shape(targets["policy"].shape)

    return (
        (augmented_boards, score_features),
        {"policy": augmented_policy, "value": targets["value"]},
        sample_weights,
    )
