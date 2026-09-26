"""Load saved games as value-only or policy/value training samples."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from training import load_self_play_game


STREAMING_REQUIRED_FIELDS = (
    "boards",
    "territories",
    "next_players",
    "scores",
    "legal_masks",
    "selected_actions",
    "final_result",
)
STREAMING_OPTIONAL_FIELDS = (
    "data_source",
    "has_mcts_policy",
)


def game_to_samples(game):
    """Convert one loaded game into model inputs and sparse value targets."""
    # Copy loaded game (np arrays) object to local one
    # Boards -> each move board state
    # PLAYER_1 = 1
    # PLAYER_2 = -1
    # EMPTY = 0
    # [0:25]
    #   -> [[0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0],
    #       [0, 0, 1, 0, 0],
    #       [0, 0, 0, 0, 0],
    #       [0, 0, 0, 0, 0]],
    boards = game["boards"]
    territories = game["territories"]
    next_players = game["next_players"]
    scores = game["scores"]

    # player_planes = 
    # [0:25]
    #   -> [[+1]],
    #   -> [[-1]],
    player_planes = next_players[:, None, None]
    # NumPy boroadcasting
    # [0:25]
    #   -> [[0, 0, 0, 0, 0],   ==  [[1]]     ->  [[False, False, False, False, False],
    #       [0, 0, 0, 0, 0],   ==  [[1]]     ->   [False, False, False, False, False],
    #       [0, 0, 1, 0, 0],   ==  [[1]]     ->   [False, False, True,  False, False],
    #       [0, 0, 0, 0, 0],   ==  [[1]]     ->   [False, False, False, False, False],
    #       [0, 0, 0, 0, 0]],  ==  [[1]]     ->   [False, False, False, False, False]],
    my_dots = boards == player_planes
    opponent_dots = boards == -player_planes
    my_territory = territories == player_planes
    opponent_territory = territories == -player_planes

    player_1_to_move = next_players == 1
    # my_scores = np.where(condittion, if True, if False)
    # -> scores[:, 0]
    #   -> all rows
    #   -> column 0
    score_scale = np.float32(boards.shape[1] * boards.shape[2])
    my_scores = (
        np.where(player_1_to_move, scores[:, 0], scores[:, 1]).astype(np.float32)
        / score_scale
    )
    opponent_scores = (
        np.where(player_1_to_move, scores[:, 1], scores[:, 0]).astype(np.float32)
        / score_scale
    )
    # Scores are global position features, so keep them as two scalar inputs
    # instead of repeating each value over every board cell.
    score_features = np.stack((my_scores, opponent_scores), axis=-1)

    # Brain imagination
    # 2 próbki,
    # plansza 2×2,
    # 3 cechy na każde pole
    #
    # ->
    #
    #    [
    #        # próbka 0
    #        [
    #            # row 0
    #            [
    #                [1, 0, 5],   # col 0 -> 3 cechy (like depth)
    #                [0, 1, 2],   # col 1 -> 3 cechy
    #            ],
    #    
    #            # row 1
    #            [
    #                [0, 0, 3],
    #                [1, 0, 4],
    #            ],
    #        ],
    #    
    #        # próbka 1
    #        [
    #            [
    #                [0, 1, 7],
    #                [1, 0, 6],
    #            ],
    #            [
    #                [0, 0, 1],
    #                [0, 1, 8],
    #            ],
    #        ],
    #    ]
    # -> Stack it, create depth
    board_samples = np.stack(
        (
            my_dots,
            opponent_dots,
            my_territory,
            opponent_territory,
            game["legal_masks"],
        ),
        axis=-1,
    ).astype(np.float32)

    # Populate result e.g. [-1, 1, -1...]
    result_for_current_player = int(game["final_result"]) * next_players
    # Change result to sparse_categorical_crossentropy -> 0 lose, 1 draw, 2 win
    value_targets = (result_for_current_player + 1).astype(np.int64)
    return (board_samples, score_features), value_targets


def load_game_samples(path):
    """Load one NPZ game and convert all of its moves into samples"""
    # Get dictionary with 1 game details -> game details are np arrays
    game = load_self_play_game(path)
    return game_to_samples(game)


def load_streaming_game(path):
    """Load only arrays needed for policy/value training from one NPZ."""

    with np.load(path, allow_pickle=False) as stored:
        missing = [
            name for name in STREAMING_REQUIRED_FIELDS if name not in stored.files
        ]
        if missing:
            raise ValueError(
                f"{Path(path).name} is missing required training fields: "
                f"{', '.join(missing)}"
            )
        game = {
            name: stored[name].copy() for name in STREAMING_REQUIRED_FIELDS
        }
        for name in STREAMING_OPTIONAL_FIELDS:
            if name in stored.files:
                game[name] = stored[name].copy()
        data_source = str(np.asarray(game.get("data_source", "")).item())
        if data_source != "new_data" and "visit_counts" in stored.files:
            game["visit_counts"] = stored["visit_counts"].copy()
    return game


def _position_count(path):
    """Read the number of saved positions without loading board tensors."""

    with np.load(path, allow_pickle=False) as stored:
        if "next_players" not in stored.files:
            raise ValueError(f"{Path(path).name} is missing next_players")
        return len(stored["next_players"])


def count_game_positions(game_paths, worker_count=4):
    """Count positions across NPZ games with bounded parallel file access"""

    paths = tuple(Path(path) for path in game_paths)
    if not paths:
        raise ValueError("game_paths cannot be empty")
    if (
        not isinstance(worker_count, int)
        or isinstance(worker_count, bool)
        or worker_count <= 0
    ):
        raise ValueError("worker_count must be a positive integer")

    path_iterator = iter(paths)
    total = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = set()

        def submit_next():
            try:
                path = next(path_iterator)
            except StopIteration:
                return False
            pending.add(executor.submit(_position_count, path))
            return True

        for _ in range(worker_count * 2):
            if not submit_next():
                break

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                total += future.result()
                submit_next()

    return total


def split_game_paths(directory, validation_fraction=0.2, seed=42, test_fraction=0.0):
    """Shuffle and split complete games without leaking positions between sets"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be between 0 (inclusive) and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    # Get all game paths from directory
    game_paths = sorted(Path(directory).glob("*.npz"))
    minimum_games = 3 if test_fraction > 0 else 2
    if len(game_paths) < minimum_games:
        raise ValueError(f"at least {minimum_games} game files are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(game_paths)

    test_count = 0
    if test_fraction > 0:
        test_count = max(1, round(len(game_paths) * test_fraction))
        test_count = min(test_count, len(game_paths) - 2)
    validation_count = max(1, round(len(game_paths) * validation_fraction))
    validation_count = min(validation_count, len(game_paths) - test_count - 1)
    # Keep every move from a given game in the same split to avoid data leakage.
    test_paths = game_paths[:test_count]
    validation_paths = game_paths[test_count:test_count + validation_count]
    training_paths = game_paths[test_count + validation_count:]
    return training_paths, validation_paths, test_paths


def load_many_game_samples(paths):
    """Concatenate value-model samples from a collection of complete games"""
    loaded = [load_game_samples(path) for path in paths]
    board_samples = np.concatenate([item[0][0] for item in loaded], axis=0)
    score_features = np.concatenate([item[0][1] for item in loaded], axis=0)
    value_targets = np.concatenate([item[1] for item in loaded], axis=0)
    # e.g.
    # board_samples.shape
    # (295, 10, 10, 5)
    # 295 -> total number of positions from all loaded games (100, 100, 95)
    # 10  -> number of board rows
    # 10  -> number of board columns
    # 5   -> number of features/channels for each board cell:
    #        0: current player's dots
    #        1: opponent's dots
    #        2: current player's territory
    #        3: opponent's territory
    #        4: legal moves

    # score_features.shape
    # (295, 2)
    # 295 -> one score pair for each position
    # 2   -> [current player's score, opponent's score]

    # value_targets.shape
    # (295,)
    # 295 -> one value label for each position
    # Each label describes the final game result from the perspective
    # of the player whose turn it is in that position:
    # 0 -> loss
    # 1 -> draw
    # 2 -> win
    return (board_samples, score_features), value_targets


def policy_targets_for_game(game, board_shape):
    """Build one game's human one-hot or MCTS visit policy targets."""

    board_shape = tuple(int(value) for value in board_shape)
    action_count = int(np.prod(board_shape))
    legal_mask = np.asarray(game["legal_masks"], dtype=np.float32)
    if legal_mask.ndim != 3 or legal_mask.shape[1:] != board_shape:
        raise ValueError("policy board shape does not match value samples")

    frame_count = len(legal_mask)
    policy_targets = np.zeros(
        (frame_count, action_count),
        dtype=np.float32,
    )
    policy_weights = np.zeros(frame_count, dtype=np.float32)
    data_source = str(np.asarray(game.get("data_source", "")).item())

    if data_source == "new_data":
        selected_actions = np.asarray(game["selected_actions"])
        if selected_actions.shape != (frame_count, 2):
            raise ValueError("human selected actions do not match policy samples")

        rows = selected_actions[:, 0].astype(np.int64, copy=False)
        columns = selected_actions[:, 1].astype(np.int64, copy=False)
        if (
            np.any(rows < 0)
            or np.any(rows >= board_shape[0])
            or np.any(columns < 0)
            or np.any(columns >= board_shape[1])
        ):
            raise ValueError("human selected action is outside the board")

        frame_indexes = np.arange(frame_count)
        if not np.all(legal_mask[frame_indexes, rows, columns] == 1):
            raise ValueError("human selected action must be legal")

        flat_actions = rows * board_shape[1] + columns
        policy_targets[frame_indexes, flat_actions] = 1.0
        policy_weights.fill(1.0)
        return policy_targets, policy_weights

    visit_counts = game.get("visit_counts")
    if visit_counts is None:
        return policy_targets, policy_weights

    visit_counts = np.asarray(visit_counts, dtype=np.float32)
    if visit_counts.shape != legal_mask.shape:
        raise ValueError("MCTS visit counts do not match policy samples")

    legal_visits = visit_counts * legal_mask
    visit_totals = legal_visits.sum(axis=(1, 2), keepdims=True)
    valid_policy = visit_totals[:, 0, 0] > 0
    normalized_visits = np.divide(
        legal_visits,
        visit_totals,
        out=np.zeros_like(legal_visits),
        where=visit_totals > 0,
    )
    policy_targets[:] = normalized_visits.reshape(-1, action_count)
    policy_weights[:] = valid_policy.astype(np.float32)
    return policy_targets, policy_weights


def load_dual_head_game_samples(path):
    """Load one NPZ as one game-sized dual-head training sample block."""

    game = load_streaming_game(path)
    inputs, value_targets = game_to_samples(game)
    board_shape = inputs[0].shape[1:3]
    policy_targets, policy_weights = policy_targets_for_game(game, board_shape)
    targets = {
        "policy": policy_targets,
        "value": value_targets,
    }
    sample_weights = {
        "policy": policy_weights,
        "value": np.ones(len(value_targets), dtype=np.float32),
    }
    return inputs, targets, sample_weights


def load_policy_targets(game_paths, expected_sample_count, board_shape):
    """Return human one-hot or normalized MCTS policy targets and weights.

    New data positions use the selected action with weight one. Other games
    use normalized legal MCTS visits. A zero weight marks a position without
    either target, such as a random opening without search statistics.
    """

    action_count = int(np.prod(board_shape))
    # Create array
    # [
    #   0           -> [0...100]
    #   ...
    #   5756458     -> [0...100]
    # ]
    policy_targets = np.zeros(
        (expected_sample_count, action_count), dtype=np.float32,
    )
    policy_weights = np.zeros(expected_sample_count, dtype=np.float32)
    offset = 0

    for path in game_paths:
        game = load_streaming_game(path)
        game_targets, game_weights = policy_targets_for_game(game, board_shape)

        stop = offset + len(game_targets)
        if stop > expected_sample_count:
            raise ValueError("policy samples do not match value samples")
        policy_targets[offset:stop] = game_targets
        policy_weights[offset:stop] = game_weights
        offset = stop

    if offset != expected_sample_count:
        raise ValueError(
            f"loaded {offset} policy samples; expected {expected_sample_count}"
        )
    return policy_targets, policy_weights


def policy_and_value_heads_targets(game_paths, value_data):
    """Attach policy targets and independent head weights to value samples."""
    # inputs = (
    #     board_samples, 
    #     score_features,
    # )
    inputs, value_targets = value_data
    # inputs[0].shape
    # (295, 10, 10, 5)
    # inputs[0].shape[1:3] -> 10, 10
    board_shape = inputs[0].shape[1:3]
    policy_targets, policy_weights = load_policy_targets(
        game_paths,
        len(value_targets),
        board_shape,
    )
    targets = {
        "policy": policy_targets,
        "value": value_targets,
    }
    sample_weights = {
        "policy": policy_weights,
        "value": np.ones(len(value_targets), dtype=np.float32),
    }
    return inputs, targets, sample_weights


def load_training_data(directory, validation_fraction=0.2, seed=42, test_fraction=0.0):
    """Load value-only train/validation and, optionally, test sets."""
    training_paths, validation_paths, test_paths = split_game_paths(
        directory,
        validation_fraction=validation_fraction,
        seed=seed,
        test_fraction=test_fraction,
    )
    datasets = (
        load_many_game_samples(training_paths),
        load_many_game_samples(validation_paths),
    )
    if test_fraction > 0:
        return (*datasets, load_many_game_samples(test_paths))
    return datasets


def load_dual_head_training_data(
    directory,
    validation_fraction=0.2,
    seed=42,
    test_fraction=0.0,
):
    """Load inputs, policy/value targets, and per-head sample weights."""
    training_paths, validation_paths, test_paths = split_game_paths(
        directory,
        validation_fraction=validation_fraction,
        seed=seed,
        test_fraction=test_fraction,
    )
    datasets = (
        policy_and_value_heads_targets(
            training_paths,
            load_many_game_samples(training_paths),
        ),
        policy_and_value_heads_targets(
            validation_paths,
            load_many_game_samples(validation_paths),
        ),
    )
    if test_fraction > 0:
        return (
            *datasets,
            policy_and_value_heads_targets(
                test_paths,
                load_many_game_samples(test_paths),
            ),
        )
    return datasets
