"""TensorFlow input-pipeline helpers for value and policy/value training."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import itertools
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.data_loader import load_dual_head_game_samples


def batched_array_dataset(
    samples,
    targets,
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
    target_arrays = tf.nest.flatten(targets)
    weight_arrays = (
        [] if sample_weights is None else tf.nest.flatten(sample_weights)
    )
    if not sample_arrays:
        raise ValueError("samples cannot be empty")
    if not target_arrays:
        raise ValueError("targets cannot be empty")
    number_of_samples = len(sample_arrays[0])
    all_arrays = (*sample_arrays, *target_arrays, *weight_arrays)
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
        raise ValueError("samples and targets cannot be empty")

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
                    select(targets, batch_indices),
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
                    select(targets, batch_indices),
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
        tf.nest.map_structure(tensor_spec, targets),
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


def _parallel_game_samples(paths, worker_count):
    """Load a bounded number of game files concurrently."""

    if worker_count == 1:
        for path in paths:
            yield load_dual_head_game_samples(path)
        return

    path_iterator = iter(paths)
    maximum_pending = worker_count * 2
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = {}

        def submit_next():
            try:
                path = next(path_iterator)
            except StopIteration:
                return False
            future = executor.submit(load_dual_head_game_samples, path)
            pending[future] = path
            return True

        for _ in range(maximum_pending):
            if not submit_next():
                break

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                result = future.result()
                submit_next()
                yield result


def streaming_dual_head_npz_dataset(
    game_paths,
    board_shape,
    batch_size,
    *,
    position_count=None,
    shuffle=False,
    seed=None,
    shuffle_buffer_size=16_384,
    file_workers=4,
):
    """Stream complete NPZ games into shuffled, position-level batches.

    Only a bounded number of games and shuffled positions are resident in
    memory. Every NPZ is opened once per dataset iteration, and several files
    can be decoded concurrently while TensorFlow consumes previous positions.
    """

    paths = tuple(Path(path) for path in game_paths)
    if not paths:
        raise ValueError("game_paths cannot be empty")
    board_shape = tuple(int(value) for value in board_shape)
    if len(board_shape) != 2 or any(value <= 0 for value in board_shape):
        raise ValueError("board_shape must contain two positive dimensions")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        not isinstance(file_workers, int)
        or isinstance(file_workers, bool)
        or file_workers <= 0
    ):
        raise ValueError("file_workers must be a positive integer")
    if shuffle and shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be positive when shuffling")
    if position_count is not None and (
        not isinstance(position_count, int)
        or isinstance(position_count, bool)
        or position_count <= 0
    ):
        raise ValueError("position_count must be a positive integer")

    rows, columns = board_shape
    action_count = rows * columns
    generation = itertools.count()

    # Training shuffle buffer: 1,588,963 positions (~32 GiB tensor payload)
    # Counting positions for exact epoch progress...
    # Position split: 11,494,130 training, 639,641 validation, 642,538 test

    # Get the game samples one by one
    # -> We got 10 file workers runing in parallel
    # -> 20 are queued
    # next(game_blocks_generator) -> returns one game npz and pushing it to RAM memory
    def game_blocks():
        # Get all the training paths - 0.05 validation and 0.05 test ones
        # 30x30 -> 81 457 (*.npz files)
        ordered_paths = list(paths)
        if shuffle:
            generation_index = next(generation)
            effective_seed = None if seed is None else seed + generation_index
            np.random.default_rng(effective_seed).shuffle(ordered_paths)
        # Yield the game samples when the function is called
        # -> Get the game paths as 300 * below structure in one yield
        # (
        #     n -> positions amount (liczba pozycji)
        #     (board_samples, score_features),
        #           -> board_samples.shape  = (n, 30, 30, 5)
        #                   my_dots,
        #                   opponent_dots,
        #                   my_territory,
        #                   opponent_territory,
        #                   game["legal_masks"],
        #           -> score_features.shape = (n, 2)
        #                   my_scores
        #                   opponent_scores
        #     {
        #         "policy": policy_targets,
        #            -> policy_targets.shape == (n, 900)
        #               -> this is exactly one move created during round
        #               -> policy_targets[20].sum() == 1.0 or policy_targets[21].sum() == 1.0
        #               ->  
        #                   board_samples[20]  -> board state before the enclosing dot is placed
        #                   policy_targets[20] -> 1 at the position where the enclosing dot will be placed
        #                   board_samples[21]  -> board state containing that dot and the resulting capture
        #         "value": value_targets,
        #           -> value_targets.shape == (n,)
        #     },
        #     {
        #         "policy": policy_weights,
        #         "value": value_weights,
        #     },
        # )
        # ---
        # file_workers      -> max threads to load the *.npz games
        # file_workers * 2  -> queue for loading the *.npz games
        yield from _parallel_game_samples(ordered_paths, file_workers)

    # Contracted structure of the data returned by generator game_blocks
    # If contract will be broken, TF will raise the error
    # This is used further in tf.data.Dataset.from_generator to change data from NumPy values to TensorFlow
    game_signature = (
        (
            tf.TensorSpec(
                shape=(None, rows, columns, 5),
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        ),
        {
            "policy": tf.TensorSpec(
                shape=(None, action_count),
                dtype=tf.float32,
            ),
            "value": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        },
        {
            "policy": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "value": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        },
    )
    # Get the 1 npz data, cast NumPy data to TensorFlow structures
    # Unbatch it to 300 separate examples/training samples using unbatch
    dataset = tf.data.Dataset.from_generator(
        game_blocks,
        output_signature=game_signature,
    ).unbatch()

    # Shuffles the data in assigned buffer with approprate size
    # It shuffles the data from dataset already unbatched so e.g. 300 training samples are shuffeled
    if shuffle:
        dataset = dataset.shuffle(
            shuffle_buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )
    # Fetch the data from shuffle buffer by batches
    # One batch requires 6.8 *.npz games to be filled in
    # BATCH_SIZE = 2048
    # 2048 / 300 ≈ 6.8 npz files
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Calculate batches amount for training based on position count
    # 11494130 / 2048 = 5613
    # During the training we can see the actual step and time required for training
    #   12/5613 ━━━━━━━━━━━━━━━━━━━━ 2:19:58 1s/step
    if position_count is not None:
        batch_count = math.ceil(position_count / batch_size)
        dataset = dataset.apply(
            tf.data.experimental.assert_cardinality(batch_count)
        )
    # tf.data.AUTOTUNE automatically finds out the best value of buffer 
    #   which needs to be filled in to provide enough data for GPU to consume
    return dataset.prefetch(tf.data.AUTOTUNE)


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
