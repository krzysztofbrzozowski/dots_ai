"""TensorFlow input-pipeline helpers used by value-model training."""

import math

import numpy as np
import tensorflow as tf


def batched_array_dataset(samples, labels, batch_size, *, shuffle=False, seed=None):
    """Expose NumPy arrays as a finite, optionally shuffled batched dataset.

    The generator copies only the current batch into a TensorFlow tensor. This
    avoids making another full in-memory copy of the multi-million-position
    training set, as ``Dataset.from_tensor_slices`` would do.
    """
    sample_arrays = samples if isinstance(samples, (tuple, list)) else (samples,)
    if not sample_arrays:
        raise ValueError("samples cannot be empty")
    number_of_samples = len(sample_arrays[0])
    if any(len(array) != number_of_samples for array in sample_arrays):
        raise ValueError("all sample inputs must contain the same number of items")
    if number_of_samples != len(labels):
        raise ValueError("samples and labels must contain the same number of items")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if number_of_samples == 0:
        raise ValueError("samples and labels cannot be empty")

    random_generator = np.random.default_rng(seed)

    def select_samples(indices):
        selected = tuple(array[indices] for array in sample_arrays)
        return selected if isinstance(samples, (tuple, list)) else selected[0]

    def batches():
        if shuffle:
            indices = np.arange(number_of_samples)
            random_generator.shuffle(indices)
            for start in range(0, number_of_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                yield select_samples(batch_indices), labels[batch_indices]
        else:
            for start in range(0, number_of_samples, batch_size):
                stop = start + batch_size
                yield select_samples(slice(start, stop)), labels[start:stop]

    sample_signature = tuple(
        tf.TensorSpec(
            shape=(None, *array.shape[1:]),
            dtype=tf.as_dtype(array.dtype),
        )
        for array in sample_arrays
    )
    if not isinstance(samples, (tuple, list)):
        sample_signature = sample_signature[0]
    output_signature = (
        sample_signature,
        tf.TensorSpec(
            shape=(None, *labels.shape[1:]),
            dtype=tf.as_dtype(labels.dtype),
        ),
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
