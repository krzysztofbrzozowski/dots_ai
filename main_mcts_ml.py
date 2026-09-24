"""Run a sequential policy/value MCTS game with the shared live GUI."""

import time
from pathlib import Path
from threading import Thread

import uvicorn

from GUI.presentation import move_message
from GUI.server import app, initialize_live_game, publish_search
from analysis.diagnostics import PRINT_T
from game.enclosure import DotsGame
from mcts.neural_search import NeuralMCTSNode, NeuralMonteCarloTreeSearch
from ml.predictor import MODEL_PATH, DualHeadPredictor
from training import SelfPlayTrajectory


ROWS = 30
COLS = 30
DEFAULT_SIMULATIONS = 128
DEFAULT_MOVE_DELAY = 0.4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002
SIMULATION_SECONDS = 30
C_PUCT = 1.5
TRAINING_DATA_DIRECTORY = (
    Path(__file__).resolve().parent / "training_data" / "neural_mcts"
)


def run_neural_mcts_game(
    board_state,
    evaluator,
    simulations_number=DEFAULT_SIMULATIONS,
    move_delay=DEFAULT_MOVE_DELAY,
    publisher=publish_search,
    simulation_seconds=SIMULATION_SECONDS,
    c_puct=C_PUCT,
    training_data_directory=None,
):
    """Play one game with sequential neural MCTS and optionally save it."""
    if simulations_number <= 0:
        raise ValueError("simulations_number must be positive")
    if move_delay < 0:
        raise ValueError("move_delay cannot be negative")
    if simulation_seconds is not None and simulation_seconds <= 0:
        raise ValueError("simulation_seconds must be positive or None")

    # Trajectory -> npz recorder object
    trajectory = (
        SelfPlayTrajectory(
            simulation_seconds=simulation_seconds,
            simulations_number=simulations_number,
            rollout_batch_size=1,
        )
        if training_data_directory is not None
        else None
    )
    move_number = 0

    while board_state.game_result is None:
        moving_player = board_state.next_to_move
        root = NeuralMCTSNode(state=board_state)
        search = NeuralMonteCarloTreeSearch(
            node=root,
            # evaluator -> returns prediction of policy and value
            evaluator=evaluator,
            c_puct=c_puct,
        )
        # best_action -> MAIN RUNNER
        # Run simulation NUMBER based
        if simulation_seconds is None:
            best_node = search.best_action(
                simulations_number=simulations_number,
            )
        # Run simulation TIME based
        else:
            best_node = search.best_action(
                total_simulation_seconds=simulation_seconds,
            )

        stats = search.last_search_stats
        print(
            f"Neural MCTS move {move_number + 1}: "
            f"simulations={stats.completed_rollouts}, "
            f"elapsed={stats.elapsed_seconds:.2f}s, "
            f"throughput={stats.rollouts_per_second:.1f} simulations/s"
        )

        if best_node.action is None:
            raise RuntimeError("neural MCTS returned a node without an action")

        if trajectory is not None:
            trajectory.record_search(
                state=board_state,
                root=root,
                selected_action=best_node.action,
                search_stats=stats,
            )

        move_number += 1
        publisher(
            state=board_state,
            root=root,
            selected_action=best_node.action,
            search_stats=stats,
            resulting_state=best_node.state,
            move_number=move_number,
            message=move_message(best_node.state, best_node.action, moving_player),
        )
        board_state = best_node.state

        if move_delay and board_state.game_result is None:
            time.sleep(move_delay)

    if trajectory is not None:
        training_data_path = trajectory.save(
            training_data_directory,
            final_result=board_state.game_result,
            filename_suffix="neural-mcts",
        )
        print(f"Saved neural self-play training data: {training_data_path}")

    return board_state


def main():
    board_state = DotsGame(ROWS, COLS)
    # Get the model
    predictor = DualHeadPredictor(MODEL_PATH)
    initialize_live_game(
        board_state,
        simulation_seconds=SIMULATION_SECONDS,
        simulations_number=DEFAULT_SIMULATIONS,
        rollout_batch_size=1,
    )

    def game_loop():
        PRINT_T(
            f"Loading and warming {predictor.model_name}",
            source="MODEL",
        )
        started_at = time.perf_counter()
        predictor.predict_state_policy_value(board_state)
        PRINT_T(
            f"{predictor.model_name} ready in "
            f"{time.perf_counter() - started_at:.2f}s",
            level="success",
            source="MODEL",
        )
        # Main game loop
        run_neural_mcts_game(
            board_state,
            evaluator=predictor.predict_state_policy_value,
            simulations_number=DEFAULT_SIMULATIONS,
            move_delay=DEFAULT_MOVE_DELAY,
            simulation_seconds=SIMULATION_SECONDS,
            c_puct=C_PUCT,
            training_data_directory=TRAINING_DATA_DIRECTORY,
        )

    Thread(
        target=game_loop,
        name="neural-mcts-game-loop",
        daemon=True,
    ).start()
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, access_log=False)


if __name__ == "__main__":
    main()
