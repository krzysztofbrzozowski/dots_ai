"""Replay saved games and report dataset diversity without training a model."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import load_analysis_path
from game.enclosure import DotsGame
from ml.data_loader import game_to_samples
from training import load_self_play_game


def inspect_game(path):
    """Verify each pre-move frame, every legal action, and the terminal result."""
    path = Path(path)
    load_analysis_path(path)  # Verify compatibility with the existing analysis GUI.
    game = load_self_play_game(path)
    rows, cols = map(int, game["board_shape"])
    state = DotsGame(rows, cols)
    state.next_to_move = int(game["next_players"][0])
    starter = state.next_to_move
    for index, action in enumerate(game["selected_actions"]):
        expected = {
            "boards": state.board,
            "territories": state.territory,
            "scores": [state.score[1], state.score[-1]],
            "legal_masks": (state.board == 0) & (state.territory == 0),
            "next_players": state.next_to_move,
        }
        for key, value in expected.items():
            if not np.array_equal(game[key][index], value):
                raise ValueError(f"{path.name}, frame {index}: inconsistent {key}")
        visits = game["visit_counts"][index]
        q_values = game["q_values"][index]
        if int(visits.sum()) != int(game["completed_rollouts"][index]):
            raise ValueError(f"{path.name}, frame {index}: visits != rollouts")
        if np.any(np.abs(q_values) > visits) or np.any(visits[~expected["legal_masks"]]):
            raise ValueError(f"{path.name}, frame {index}: invalid search statistics")
        state = state.move(tuple(map(int, action)))
    if state.game_result is None or state.game_result != int(game["final_result"]):
        raise ValueError(f"{path.name}: recorded result differs from replayed result")
    zero_search = game["completed_rollouts"] == 0
    opening_length = int(zero_search.sum())
    if not np.all(zero_search[:opening_length]) or np.any(zero_search[opening_length:]):
        raise ValueError(f"{path.name}: zero-search moves outside the opening")
    if np.any(game["search_elapsed_seconds"][:opening_length] != 0):
        raise ValueError(f"{path.name}: random opening has fabricated search time")
    return {
        "file": path.name,
        "game_id": str(game["game_id"].item()),
        "starting_player": starter,
        "opening_moves": game["selected_actions"][:opening_length].tolist(),
        "positions": len(game["boards"]),
        "final_result": int(state.game_result),
        "final_scores": {"1": state.score[1], "-1": state.score[-1]},
        "rollouts": int(game["completed_rollouts"].sum()),
        "rollouts_by_player": {
            str(player): int(game["completed_rollouts"][game["next_players"] == player].sum())
            for player in (1, -1)
        },
        "search_seconds": float(game["search_elapsed_seconds"].sum()),
        "bytes": path.stat().st_size,
    }


def audit_collection(directory):
    """Reopen every NPZ, replay it, and write JSON + readable Markdown reports."""
    from .collector import collection_files, game_index, game_plan, verify_plan, write_json
    from .config import CollectionConfig

    directory = Path(directory)
    config = CollectionConfig(**json.loads((directory / "collection.json").read_text())["settings"])
    results = Counter({-1: 0, 0: 0, 1: 0})
    starters = Counter({-1: 0, 1: 0})
    matchups = {name: Counter({-1: 0, 0: 0, 1: 0}) for name in ("equal", "p1_advantage", "p2_advantage")}
    opening_lengths = Counter()
    quadrants = Counter({"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0})
    trajectories = set()
    positions = set()
    later_positions = set()
    label_counts = np.zeros(3, dtype=np.int64)
    search_counts = []
    early_search_counts = []
    records = []
    later_count = 0
    for path in collection_files(directory):
        record = inspect_game(path)
        plan = game_plan(config, game_index(path))
        verify_plan(record, plan)
        records.append(record)
        results[record["final_result"]] += 1
        starters[record["starting_player"]] += 1
        matchups[plan["matchup"]][record["final_result"]] += 1
        opening_lengths[len(record["opening_moves"])] += 1
        game = load_self_play_game(path)
        (board_samples, score_features), labels = game_to_samples(game)
        if (
            board_samples.dtype != np.float32
            or board_samples.shape[1:] != (config.rows, config.cols, 5)
            or score_features.dtype != np.float32
            or score_features.shape[1:] != (2,)
        ):
            raise ValueError(f"{path.name}: incompatible value-model input")
        label_counts += np.bincount(labels, minlength=3)
        signature = bytes([record["starting_player"] + 1]) + game["selected_actions"].tobytes()
        trajectories.add(hashlib.sha256(signature).hexdigest())
        for board_sample, scores in zip(board_samples, score_features):
            digest = hashlib.sha256(
                board_sample.tobytes() + scores.tobytes()
            ).hexdigest()
            positions.add(digest)
            if np.count_nonzero(
                board_sample[:, :, 0] + board_sample[:, :, 1]
            ) >= 10:
                later_count += 1
                later_positions.add(digest)
        for row, col in game["selected_actions"][:10]:
            name = ("top" if row < config.rows / 2 else "bottom")
            name += "_left" if col < config.cols / 2 else "_right"
            quadrants[name] += 1
        counts = game["completed_rollouts"]
        search_counts.extend(counts[counts > 0].tolist())
        early_search_counts.extend(counts[:20][counts[:20] > 0].tolist())
    summary = {
        "games": len(records),
        "positions": sum(record["positions"] for record in records),
        "results": dict(results),
        "starting_players": dict(starters),
        "results_by_matchup": {name: dict(counts) for name, counts in matchups.items()},
        "opening_lengths": dict(opening_lengths),
        "early_move_quadrants": dict(quadrants),
        "unique_trajectories": len(trajectories),
        "unique_canonical_positions": len(positions),
        "duplicate_positions_after_10_dots": later_count - len(later_positions),
        "labels_loss_draw_win": label_counts.tolist(),
        "rollouts": sum(record["rollouts"] for record in records),
        "median_rollouts_per_search": float(np.median(search_counts)) if search_counts else 0,
        "median_rollouts_first_20_moves": float(np.median(early_search_counts)) if early_search_counts else 0,
        "total_bytes": sum(record["bytes"] for record in records),
        "replay_validation": "passed",
        "augmentation": False,
    }
    write_json(directory / "audit.json", summary)
    lines = [
        "# Collection audit", "",
        f"Games: **{summary['games']}**. Positions: **{summary['positions']}**.",
        "All recorded positions, territories, scores, legal masks, actions and final results passed engine replay.",
        "NPZ files also passed the existing analysis loader and value-model encoder.", "",
        "| Matchup | Player +1 wins | Draws | Player -1 wins |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, counts in matchups.items():
        lines.append(f"| {name} | {counts[1]} | {counts[0]} | {counts[-1]} |")
    lines.extend([
        "", f"Starting players: +1 **{starters[1]}**, -1 **{starters[-1]}**.",
        f"Opening lengths: {dict(opening_lengths)}.",
        f"First ten moves by quadrant: {dict(quadrants)}.",
        f"Unique complete trajectories: **{len(trajectories)}**.",
        f"Repeated canonical positions with at least 10 dots: **{summary['duplicate_positions_after_10_dots']}**.",
        f"Median rollouts per MCTS move: **{summary['median_rollouts_per_search']}**; in first 20 moves: **{summary['median_rollouts_first_20_moves']}**.",
        "", "No augmentation was applied. Short search budgets produce a diversity pilot, not a guarantee of strong-play labels.",
        "An unfinished block of eight games may not have exact 50/25/25 matchup proportions.",
        "The legacy per-file requested_simulation_seconds field describes Player +1; both budgets and opening moves are recorded in manifest.json.",
    ])
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return summary
