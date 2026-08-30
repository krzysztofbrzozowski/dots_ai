"""Load saved games as value-model samples."""

from pathlib import Path

import numpy as np

from training import load_self_play_game


def game_to_samples(game):
    """Convert one loaded game into CNN inputs and sparse class labels."""
    boards = game["boards"]
    territories = game["territories"]
    next_players = game["next_players"]
    scores = game["scores"]

    player_planes = next_players[:, None, None]
    my_dots = boards == player_planes
    opponent_dots = boards == -player_planes
    my_territory = territories == player_planes
    opponent_territory = territories == -player_planes

    player_1_to_move = next_players == 1
    my_scores = np.where(player_1_to_move, scores[:, 0], scores[:, 1])
    opponent_scores = np.where(player_1_to_move, scores[:, 1], scores[:, 0])
    my_score_planes = np.broadcast_to(my_scores[:, None, None], boards.shape)
    opponent_score_planes = np.broadcast_to(
        opponent_scores[:, None, None],
        boards.shape,
    )

    samples = np.stack(
        (
            my_dots,
            opponent_dots,
            my_territory,
            opponent_territory,
            game["legal_masks"],
            my_score_planes,
            opponent_score_planes,
        ),
        axis=-1,
    ).astype(np.float32)

    result_for_current_player = int(game["final_result"]) * next_players
    labels = (result_for_current_player + 1).astype(np.int64)
    return samples, labels


def load_game_samples(path):
    """Load one NPZ game and convert all of its moves into samples."""
    return game_to_samples(load_self_play_game(path))


def load_training_data(directory, validation_fraction=0.2, seed=42):
    """Load one board-size directory and split complete games into two sets."""
    game_paths = sorted(Path(directory).glob("*.npz"))
    if len(game_paths) < 2:
        raise ValueError("at least two game files are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(game_paths)

    validation_count = max(1, round(len(game_paths) * validation_fraction))
    validation_count = min(validation_count, len(game_paths) - 1)
    validation_paths = game_paths[:validation_count]
    training_paths = game_paths[validation_count:]

    def load_many(paths):
        loaded = [load_game_samples(path) for path in paths]
        samples = np.concatenate([item[0] for item in loaded], axis=0)
        labels = np.concatenate([item[1] for item in loaded], axis=0)
        return samples, labels

    return load_many(training_paths), load_many(validation_paths)
