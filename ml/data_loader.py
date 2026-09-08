"""Load saved games as value-model samples."""

from pathlib import Path

import numpy as np

from training import load_self_play_game


def game_to_samples(game):
    """Convert one loaded game into model inputs and sparse class labels."""
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
    labels = (result_for_current_player + 1).astype(np.int64)
    return (board_samples, score_features), labels


def load_game_samples(path):
    """Load one NPZ game and convert all of its moves into samples"""
    # Get dictionary with 1 game details -> game details are np arrays
    game = load_self_play_game(path)
    return game_to_samples(game)


def load_training_data(directory, validation_fraction=0.2, seed=42, test_fraction=0.0):
    """Split complete games into train/validation and, optionally, test sets."""
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

    def load_many(paths):
        loaded = [load_game_samples(path) for path in paths]
        board_samples = np.concatenate([item[0][0] for item in loaded], axis=0)
        score_features = np.concatenate([item[0][1] for item in loaded], axis=0)
        labels = np.concatenate([item[1] for item in loaded], axis=0)
        return (board_samples, score_features), labels

    datasets = load_many(training_paths), load_many(validation_paths)
    if test_fraction > 0:
        return (*datasets, load_many(test_paths))
    return datasets
