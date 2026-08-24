"""Run an MCTS-vs-MCTS Dots game and publish it to the read-only GUI."""

import time
from threading import Thread

import uvicorn

from GUI.presentation import move_message
from GUI.server import app, publish_state
from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch, TwoPlayerMCTSNode


DEFAULT_ROWS = 10
DEFAULT_COLS = 10
DEFAULT_SIMULATIONS = 12
DEFAULT_MOVE_DELAY = 0.4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
SIMULATION_SECONDS = 1


def run_mcts_game(
    board_state,
    simulations_number=DEFAULT_SIMULATIONS,
    move_delay=DEFAULT_MOVE_DELAY,
    publisher=publish_state,
):
    """Own and run the complete game loop, publishing after every MCTS move."""
    if simulations_number <= 0:
        raise ValueError("simulations_number must be positive")
    if move_delay < 0:
        raise ValueError("move_delay cannot be negative")

    move_number = 0

    while board_state.game_result is None:
        moving_player = board_state.next_to_move
        root = TwoPlayerMCTSNode(state=board_state)
        mcts = MonteCarloTreeSearch(root)
        best_node = mcts.best_action(total_simulation_seconds=SIMULATION_SECONDS)

        if best_node.action is None:
            raise RuntimeError("MCTS returned a node without an action")

        board_state = best_node.state
        move_number += 1
        publisher(
            board_state,
            last_move=best_node.action,
            move_number=move_number,
            message=move_message(board_state, best_node.action, moving_player),
        )

        if move_delay and board_state.game_result is None:
            time.sleep(move_delay)

    return board_state


def main():
    board_state = DotsGame(DEFAULT_ROWS, DEFAULT_COLS)
    publish_state(
        board_state,
        last_move=None,
        move_number=0,
        message="MCTS game ready. Player 1 is searching.",
    )

    game_thread = Thread(
        target=run_mcts_game,
        kwargs={
            "board_state": board_state,
            "simulations_number": DEFAULT_SIMULATIONS,
            "move_delay": DEFAULT_MOVE_DELAY,
        },
        name="mcts-game-loop",
        daemon=True,
    )
    game_thread.start()

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
