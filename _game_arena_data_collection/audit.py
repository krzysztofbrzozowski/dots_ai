"""Replay saved games and report dataset diversity without training a model."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import load_analysis_path
from game.enclosure import DotsGame
from ml.data_loader import game_to_samples
from training import load_self_play_game


def inspect_game(path, config=None):
    """Verify each pre-move frame, every legal action, and the terminal result."""
    path = Path(path)
    load_analysis_path(path)  # Verify compatibility with the shared MCTS GUI.
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
        if config is not None:
            legal_action_count = int(expected["legal_masks"].sum())
            expected_rollouts = min(
                config.maximum_rollouts,
                max(
                    config.minimum_rollouts,
                    config.rollouts_per_legal_action * legal_action_count,
                ),
            )
            if int(game["completed_rollouts"][index]) != expected_rollouts:
                raise ValueError(
                    f"{path.name}, frame {index}: expected "
                    f"{expected_rollouts} rollouts"
                )
        state = state.move(tuple(map(int, action)))
    if state.game_result is None or state.game_result != int(game["final_result"]):
        raise ValueError(f"{path.name}: recorded result differs from replayed result")
    zero_search = game["completed_rollouts"] == 0
    opening_length = int(zero_search.sum())
    if not np.all(zero_search[:opening_length]) or np.any(zero_search[opening_length:]):
        raise ValueError(f"{path.name}: zero-search moves outside the opening")
    if np.any(game["search_elapsed_seconds"][:opening_length] != 0):
        raise ValueError(f"{path.name}: random opening has fabricated search time")
    if config is not None and opening_length:
        raise ValueError(f"{path.name}: rollout collection cannot contain openings")
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
    from .collector import (
        collection_files,
        game_identity,
        game_plan,
        verify_plan,
        write_json,
    )
    from .config import CollectionConfig

    directory = Path(directory)
    metadata = json.loads((directory / "collection.json").read_text())
    if metadata.get("version") != 2:
        raise ValueError("this collector audits rollout collection version 2")
    config = CollectionConfig(seed=0, **metadata["settings"])
    results = Counter({-1: 0, 0: 0, 1: 0})
    starters = Counter({-1: 0, 1: 0})
    sources = Counter()
    source_seeds = set()
    opening_lengths = Counter()
    quadrants = Counter({"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0})
    trajectories = set()
    positions = set()
    later_positions = set()
    label_counts = np.zeros(3, dtype=np.int64)
    search_counts = []
    early_search_counts = []
    legal_counts = []
    rollout_ratios = []
    visited_fractions = []
    revisited_fractions = []
    normalized_entropies = []
    top_visit_shares = []
    phase_metrics = {
        name: {"rollouts": [], "ratios": [], "visited": [], "revisited": []}
        for name in ("1-20", "21-40", "41-60", "61-80", "81-100")
    }
    records = []
    later_count = 0
    for path in collection_files(directory):
        source_id, seed, index = game_identity(path)
        file_config = replace(config, seed=seed)
        record = inspect_game(path, file_config)
        plan = game_plan(file_config, index, source_id)
        verify_plan(record, plan)
        records.append(record)
        results[record["final_result"]] += 1
        starters[record["starting_player"]] += 1
        sources[source_id] += 1
        source_seeds.add((source_id, seed))
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
        for move_index, (visits, legal_mask, rollouts) in enumerate(zip(
            game["visit_counts"], game["legal_masks"], counts,
        )):
            legal = legal_mask.astype(bool)
            legal_count = int(legal.sum())
            legal_visits = visits[legal]
            visited_count = int(np.count_nonzero(legal_visits))
            revisited_count = int(np.count_nonzero(legal_visits >= 2))
            ratio = float(rollouts / legal_count)
            visited_fraction = float(visited_count / legal_count)
            revisited_fraction = float(revisited_count / legal_count)
            probabilities = legal_visits[legal_visits > 0].astype(np.float64)
            probabilities /= probabilities.sum()
            entropy = float(-(probabilities * np.log(probabilities)).sum())
            normalized_entropy = (
                entropy / np.log(legal_count) if legal_count > 1 else 0.0
            )
            top_share = float(legal_visits.max() / rollouts)
            legal_counts.append(legal_count)
            rollout_ratios.append(ratio)
            visited_fractions.append(visited_fraction)
            revisited_fractions.append(revisited_fraction)
            normalized_entropies.append(normalized_entropy)
            top_visit_shares.append(top_share)
            phase_start = min((move_index // 20) * 20 + 1, 81)
            phase_name = f"{phase_start}-{min(phase_start + 19, 100)}"
            phase = phase_metrics[phase_name]
            phase["rollouts"].append(int(rollouts))
            phase["ratios"].append(ratio)
            phase["visited"].append(visited_fraction)
            phase["revisited"].append(revisited_fraction)

    def percentiles(values):
        if not values:
            return {"p10": 0.0, "median": 0.0, "p90": 0.0}
        p10, median, p90 = np.percentile(values, (10, 50, 90))
        return {
            "p10": float(p10),
            "median": float(median),
            "p90": float(p90),
        }

    phase_summary = {
        name: {
            "positions": len(values["rollouts"]),
            "rollouts": percentiles(values["rollouts"]),
            "rollouts_per_legal_action": percentiles(values["ratios"]),
            "visited_legal_fraction": percentiles(values["visited"]),
            "revisited_legal_fraction": percentiles(values["revisited"]),
        }
        for name, values in phase_metrics.items()
    }
    summary = {
        "games": len(records),
        "positions": sum(record["positions"] for record in records),
        "results": dict(results),
        "starting_players": dict(starters),
        "sources": dict(sources),
        "source_seeds": [
            {"source_id": source_id, "seed": seed}
            for source_id, seed in sorted(source_seeds)
        ],
        "opening_lengths": dict(opening_lengths),
        "early_move_quadrants": dict(quadrants),
        "unique_trajectories": len(trajectories),
        "unique_canonical_positions": len(positions),
        "duplicate_positions_after_10_dots": later_count - len(later_positions),
        "labels_loss_draw_win": label_counts.tolist(),
        "rollouts": sum(record["rollouts"] for record in records),
        "median_rollouts_per_search": float(np.median(search_counts)) if search_counts else 0,
        "median_rollouts_first_20_moves": float(np.median(early_search_counts)) if early_search_counts else 0,
        "legal_actions": percentiles(legal_counts),
        "rollouts_per_legal_action": percentiles(rollout_ratios),
        "visited_legal_fraction": percentiles(visited_fractions),
        "revisited_legal_fraction": percentiles(revisited_fractions),
        "normalized_policy_entropy": percentiles(normalized_entropies),
        "top_visit_share": percentiles(top_visit_shares),
        "search_quality_by_move": phase_summary,
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
        f"Sources: {dict(sources)}. Source/seed pairs: {summary['source_seeds']}.",
        "All positions used their exact configured adaptive rollout budget.", "",
        "| Moves | Positions | Median rollouts | Median rollouts/legal | Median visited | Median revisited |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in phase_summary.items():
        lines.append(
            f"| {name} | {values['positions']} | "
            f"{values['rollouts']['median']:.0f} | "
            f"{values['rollouts_per_legal_action']['median']:.2f} | "
            f"{values['visited_legal_fraction']['median']:.1%} | "
            f"{values['revisited_legal_fraction']['median']:.1%} |"
        )
    lines.extend([
        "", f"Starting players: +1 **{starters[1]}**, -1 **{starters[-1]}**.",
        f"Opening lengths: {dict(opening_lengths)}.",
        f"First ten moves by quadrant: {dict(quadrants)}.",
        f"Unique complete trajectories: **{len(trajectories)}**.",
        f"Repeated canonical positions with at least 10 dots: **{summary['duplicate_positions_after_10_dots']}**.",
        f"Median rollouts per MCTS move: **{summary['median_rollouts_per_search']}**; in first 20 moves: **{summary['median_rollouts_first_20_moves']}**.",
        "", "No augmentation or random opening moves were applied.",
        "Early actions are sampled from MCTS visits; later actions use the most-visited move.",
        "The schema-v1 requested_simulations scalar stores the configured maximum; completed_rollouts stores the exact budget of every position.",
    ])
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return summary
