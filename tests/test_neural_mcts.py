"""Tests for the isolated sequential policy/value MCTS path."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from game.enclosure import DotsGame, PLAYER_1
from main_mcts_ml import run_neural_mcts_game
from mcts.neural_search import NeuralMCTSNode, NeuralMonteCarloTreeSearch
from ml.predictor import predict_policy_value
from training import load_self_play_game


class RecordingEvaluator:
    def __init__(self, preferred_action=(1, 1), child_value=-0.6):
        self.preferred_action = preferred_action
        self.child_value = child_value
        self.calls = []

    def __call__(self, state):
        self.calls.append(state.copy())
        policy = np.zeros(state.board.shape, dtype=np.float32)
        legal_actions = state.get_legal_actions()
        for action in legal_actions:
            policy[action] = 1.0
        if self.preferred_action in legal_actions:
            policy[self.preferred_action] = 20.0
        value = 0.0 if np.count_nonzero(state.board) == 0 else self.child_value
        return {"policy": policy, "value": value}


def test_dual_head_prediction_reads_policy_and_value_with_one_inference():
    class FakeDualHeadModel:
        input_shape = [(None, 2, 2, 5), (None, 2)]

        def __init__(self):
            self.calls = 0

        def predict(self, model_input, verbose=0):
            self.calls += 1
            assert model_input[0].shape == (1, 2, 2, 5)
            assert model_input[1].shape == (1, 2)
            assert verbose == 0
            return {
                "policy": np.asarray(
                    [[1.0, 2.0, 100.0, 3.0]],
                    dtype=np.float32,
                ),
                "value": np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32),
            }

    model = FakeDualHeadModel()
    state = DotsGame(2, 2)
    state.board[1, 0] = PLAYER_1

    prediction = predict_policy_value(model, state)
    policy = np.asarray(prediction["policy"])

    assert model.calls == 1
    assert policy[1, 0] == 0
    assert abs(policy.sum() - 1.0) < 1e-6
    assert abs(prediction["value"] - 0.3) < 1e-6


def test_neural_search_uses_policy_prior_and_selects_by_visit_count():
    evaluator = RecordingEvaluator(preferred_action=(1, 1))
    root = NeuralMCTSNode(DotsGame(2, 2))
    search = NeuralMonteCarloTreeSearch(root, evaluator, c_puct=1.5)

    selected = search.best_action(simulations_number=2)

    assert selected.action == (1, 1)
    assert selected.n == 1
    assert root.n == 2
    assert search.last_search_stats.completed_rollouts == 2
    assert len(evaluator.calls) == 2


def test_neural_value_changes_sign_for_the_parent_player():
    evaluator = RecordingEvaluator(
        preferred_action=(1, 1),
        child_value=-0.6,
    )
    root = NeuralMCTSNode(DotsGame(2, 2))
    search = NeuralMonteCarloTreeSearch(root, evaluator)

    search.best_action(simulations_number=2)
    preferred_child = next(
        child for child in root.children if child.action == (1, 1)
    )

    assert preferred_child.n == 1
    assert abs(preferred_child.q - 0.6) < 1e-9


def test_expansion_keeps_unvisited_child_game_states_lazy():
    root = NeuralMCTSNode(DotsGame(3, 3))
    search = NeuralMonteCarloTreeSearch(root, RecordingEvaluator())

    search.best_action(simulations_number=1)

    assert len(root.children) == 9
    assert all(child._state is None for child in root.children)


def test_terminal_leaf_uses_exact_result_without_model_inference():
    evaluator = RecordingEvaluator(preferred_action=(0, 0))
    root = NeuralMCTSNode(DotsGame(1, 1))
    search = NeuralMonteCarloTreeSearch(root, evaluator)

    selected = search.best_action(simulations_number=2)

    assert selected.state.game_result == 0
    assert selected.n == 1
    assert selected.q == 0
    assert len(evaluator.calls) == 1


def test_neural_game_runner_publishes_and_saves_compatible_trajectory():
    publications = []

    with TemporaryDirectory() as directory:
        final_state = run_neural_mcts_game(
            DotsGame(1, 1),
            evaluator=RecordingEvaluator(preferred_action=(0, 0)),
            simulations_number=2,
            simulation_seconds=None,
            move_delay=0,
            publisher=lambda **payload: publications.append(payload),
            training_data_directory=directory,
        )
        paths = list((Path(directory) / "1x1").glob("*_neural-mcts.npz"))
        stored = load_self_play_game(paths[0])

    assert final_state.game_result == 0
    assert len(publications) == 1
    assert publications[0]["selected_action"] == (0, 0)
    assert publications[0]["resulting_state"].board[0, 0] == PLAYER_1
    assert stored["selected_actions"].tolist() == [[0, 0]]
    assert stored["visit_counts"][0, 0, 0] == 1
    assert stored["has_policy_priors"].tolist() == [1]
    assert abs(float(stored["policy_priors"][0].sum()) - 1.0) < 1e-6
    assert stored["policy_priors"][0, 0, 0] == 1.0


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} neural MCTS tests passed")
