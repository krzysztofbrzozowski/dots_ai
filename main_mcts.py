"""Run an MCTS-vs-MCTS Dots game and publish it to the read-only GUI."""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from threading import Thread

import uvicorn

from GUI.presentation import move_message
from GUI.server import app, publish_state
from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch, TwoPlayerMCTSNode, rollout_state


DEFAULT_ROWS = 10
DEFAULT_COLS = 10
DEFAULT_SIMULATIONS = 12
DEFAULT_MOVE_DELAY = 0.4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
SIMULATION_SECONDS = 30
DEFAULT_MCTS_WORKERS = os.cpu_count()


def run_mcts_game(
    board_state,
    simulations_number=DEFAULT_SIMULATIONS,
    move_delay=DEFAULT_MOVE_DELAY,
    publisher=publish_state,
    simulation_seconds=SIMULATION_SECONDS,
    rollout_executor=None,
    parallelism=1,
):
    """Own and run the complete game loop, publishing after every MCTS move."""
    if simulations_number <= 0:
        raise ValueError("simulations_number must be positive")
    if move_delay < 0:
        raise ValueError("move_delay cannot be negative")
    if simulation_seconds is not None and simulation_seconds <= 0:
        raise ValueError("simulation_seconds must be positive or None")

    move_number = 0

    while board_state.game_result is None:
        moving_player = board_state.next_to_move
        root = TwoPlayerMCTSNode(state=board_state)
        # -> 1st option it to run rollouts as sequention -> rollout_executor = None and parallelism = 1
        # -> 2nd option is to run rollouts in parallel via rollout_executor -> ProcessPoolExecutor
        # Based on aviablle CPU count, parallel processes of rollout are started
        # When each process finish, result is backpropagated to the top
        mcts = MonteCarloTreeSearch(
            root,
            rollout_executor=rollout_executor,
            parallelism=parallelism,
        )
        if simulation_seconds is None:
            best_node = mcts.best_action(simulations_number=simulations_number)
        else:
            best_node = mcts.best_action(total_simulation_seconds=simulation_seconds)

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


def run_parallel_mcts_game(
    board_state,
    simulations_number=DEFAULT_SIMULATIONS,
    move_delay=DEFAULT_MOVE_DELAY,
    publisher=publish_state,
    simulation_seconds=SIMULATION_SECONDS,
    workers=DEFAULT_MCTS_WORKERS,
):
    """Run a game using one persistent process pool for all MCTS moves"""
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers <= 0:
        raise ValueError("workers must be positive")

    if workers == 1:
        return run_mcts_game(
            board_state,
            simulations_number=simulations_number,
            move_delay=move_delay,
            publisher=publisher,
            simulation_seconds=simulation_seconds,
        )

    # ``spawn`` is safe when this function runs in the GUI's game thread and
    # behaves consistently across macOS, Linux, and Windows.
    # Creating separate processes to run rollouts in parallel on independent CPU
    # Game thread
    # ├── runs the game loop
    # ├── owns the MCTS tree
    # ├── selects leaf nodes
    # │
    # └── ProcessPoolExecutor
    #     ├── worker process 1 → rollout_state(...)
    #     ├── worker process 2 → rollout_state(...)
    #     └── worker process 3 → rollout_state(...)
    with ProcessPoolExecutor(
        max_workers=workers,
        # Worker processes:
        # Process creation methods:
        # ├── spawn → starts a new, clean Python interpreter
        # |  -> have their own memory
        # |  -> run their own Python interpreter
        # |  -> do not have direct access to the MCTS tree
        # |  -> communicate with the parent process through serialized data
        # └── fork  → creates a copy of the existing process
        mp_context=get_context("spawn"),
    ) as rollout_executor:
        # ProcessPoolExecutor starts lazily -> when some task will be assigned
        # We want to have exact simulation_seconds to be executed
        # Interpreter startup is not charged to the first move
        terminal_state = DotsGame(1, 1).move((0, 0))
        warmups = [
            rollout_executor.submit(rollout_state, terminal_state, seed)
            for seed in range(workers)
        ]
        # Wait for the execution ends and Future objects will finish
        for warmup in warmups:
            warmup.result()

        return run_mcts_game(
            board_state,
            simulations_number=simulations_number,
            move_delay=move_delay,
            publisher=publisher,
            simulation_seconds=simulation_seconds,
            # Passing the ProcessPoolExecutor object to run
            rollout_executor=rollout_executor,
            parallelism=workers,
        )


def main():
    # Initial board state
    board_state = DotsGame(DEFAULT_ROWS, DEFAULT_COLS)
    publish_state(
        board_state,
        last_move=None,
        move_number=0,
        message="MCTS game ready. Player 1 is searching.",
    )

    # Pararell thread for the game
    game_thread = Thread(
        target=run_parallel_mcts_game,
        kwargs={
            "board_state": board_state,
            "simulations_number": DEFAULT_SIMULATIONS,
            "move_delay": DEFAULT_MOVE_DELAY,
            "workers": DEFAULT_MCTS_WORKERS,
        },
        name="mcts-game-loop",
        daemon=True,
    )
    game_thread.start()

    # Main thread for the app (displaying the game state)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
