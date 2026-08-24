"""Presentation-only text helpers for the MCTS game observer."""

from game.enclosure import PLAYER_1, PLAYER_2


def player_label(player):
    """Return the display name for a player value."""
    return "Player 1" if player == PLAYER_1 else "Player 2"


def result_label(result):
    """Return a human-readable final result."""
    if result == PLAYER_1:
        return "Player 1 wins"
    if result == PLAYER_2:
        return "Player 2 wins"
    return "Draw"


def move_message(game, action, moving_player):
    """Describe an MCTS-selected move for the GUI status area."""
    row, col = action
    captured_count = len(game.last_captured_dots)
    capture_text = (
        f" and captured {captured_count} dot"
        f"{'s' if captured_count != 1 else ''}"
        if captured_count
        else ""
    )

    if game.game_result is not None:
        return (
            f"{player_label(moving_player)} selected ({row}, {col})"
            f"{capture_text}. Game over: {result_label(game.game_result)}."
        )

    return (
        f"{player_label(moving_player)} selected ({row}, {col})"
        f"{capture_text}. {player_label(game.next_to_move)} is searching."
    )
