"""Move adapters that let different player implementations share one arena."""

from dataclasses import dataclass

from mcts.enclosure import MonteCarloTreeSearch, TwoPlayerMCTSNode
from mcts.neural_search import NeuralMCTSNode, NeuralMonteCarloTreeSearch
from ml.predictor import DualHeadPredictor

from .config import PlayerPreset


@dataclass(frozen=True, slots=True)
class MoveDecision:
    action: tuple[int, int]
    completed_rollouts: int
    elapsed_seconds: float
    root: object


class ClassicMCTSPlayer:
    def __init__(self, preset: PlayerPreset):
        self.preset = preset

    def choose_move(self, state, rollout_executor=None):
        uses_parallel = self.preset.workers > 1
        if uses_parallel and rollout_executor is None:
            raise RuntimeError("parallel classic MCTS requires a rollout executor")

        root = TwoPlayerMCTSNode(state=state)
        search = MonteCarloTreeSearch(
            root,
            rollout_executor=rollout_executor if uses_parallel else None,
            rollout_batch_size=self.preset.workers if uses_parallel else 1,
        )
        selected = search.best_action(
            total_simulation_seconds=self.preset.simulation_seconds,
        )
        if selected.action is None:
            raise RuntimeError("classic MCTS returned no action")
        stats = search.last_search_stats
        return MoveDecision(
            action=tuple(selected.action),
            completed_rollouts=stats.completed_rollouts,
            elapsed_seconds=stats.elapsed_seconds,
            root=root,
        )


class NeuralMCTSPlayer:
    def __init__(self, preset: PlayerPreset):
        self.preset = preset
        self.predictor = DualHeadPredictor(preset.model_path)

    def choose_move(self, state, rollout_executor=None):
        root = NeuralMCTSNode(state=state)
        search = NeuralMonteCarloTreeSearch(
            node=root,
            evaluator=self.predictor.predict_state_policy_value,
            c_puct=self.preset.c_puct,
        )
        selected = search.best_action(
            total_simulation_seconds=self.preset.simulation_seconds,
        )
        if selected.action is None:
            raise RuntimeError("neural MCTS returned no action")
        stats = search.last_search_stats
        return MoveDecision(
            action=tuple(selected.action),
            completed_rollouts=stats.completed_rollouts,
            elapsed_seconds=stats.elapsed_seconds,
            root=root,
        )


def build_runtime_player(preset: PlayerPreset):
    if preset.kind == "classic_mcts":
        return ClassicMCTSPlayer(preset)
    if preset.kind == "neural_mcts":
        return NeuralMCTSPlayer(preset)
    raise ValueError(f"unsupported AI player kind: {preset.kind}")
