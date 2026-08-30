"""Run the editable arena configuration with ``python -m _game_arena``."""

from GUI.presentation import player_label, result_label

from .arena import run_arena
from .config import ARENA_CONFIG


def _print_move(_state, move):
    stats = move.search_stats
    print(
        f"Move {move.move_number}: {player_label(move.player)} "
        f"played {move.action} using {move.search_mode.value}; "
        f"budget={move.requested_simulation_seconds:.3f}s, "
        f"rollouts={stats.completed_rollouts}, "
        f"elapsed={stats.elapsed_seconds:.3f}s"
    )


def main():
    for _ in range(30):
        print(
            f"Arena {ARENA_CONFIG.rows}x{ARENA_CONFIG.cols}; "
            f"{player_label(ARENA_CONFIG.next_to_move)} moves first."
        )
        result = run_arena(ARENA_CONFIG, on_move=_print_move)
        print(f"Game over: {result_label(result.final_result)}")
        print(f"Saved training data: {result.training_data_path}")


if __name__ == "__main__":
    main()
