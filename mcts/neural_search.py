"""Small sequential policy/value MCTS using the project's dual-head model.

The structure deliberately mirrors the original ``node + search`` MCTS
implementation.  A simulation selects one path with PUCT, evaluates one leaf,
expands it with policy priors, and backpropagates the value before the next
simulation starts.
"""

import math
import time

import numpy as np

from .search import SearchStats


class NeuralMCTSNode:
    """One lazily materialized node in a two-player neural MCTS tree."""

    def __init__(self, state=None, parent=None, action=None, prior=1.0):
        if state is None and parent is None:
            raise ValueError("a root node requires a game state")
        self._state = state
        self.parent = parent
        self.action = action
        self.prior = float(prior)
        self.children = []
        self._number_of_visits = 0
        self._value_sum = 0.0
        self._expanded = False

    @property
    def state(self):
        """Create a child state only when the search actually visits it."""
        if self._state is None:
            self._state = self.parent.state.move(self.action)
        return self._state

    @property
    def n(self):
        return self._number_of_visits

    @property
    def q(self):
        """Accumulated value from the parent player's perspective.

        The model evaluates a position for ``state.next_to_move``.  The parent
        player is the opponent, so a child's public Q has the opposite sign.
        This keeps GUI/training statistics compatible with classic MCTS.
        """
        if self.parent is None:
            return self._value_sum
        return -self._value_sum

    @property
    def is_expanded(self):
        return self._expanded

    def is_terminal_node(self):
        return self.state.is_game_over()

    def expand(self, policy):
        """Create lightweight children with normalized legal policy priors."""
        if self._expanded:
            return

        policy = np.asarray(policy, dtype=np.float64)
        if policy.shape != self.state.board.shape:
            raise ValueError(
                f"policy shape {policy.shape} does not match board "
                f"shape {self.state.board.shape}"
            )
        if not np.all(np.isfinite(policy)) or np.any(policy < 0):
            raise ValueError("policy must contain finite non-negative values")

        legal_actions = self.state.get_legal_actions()
        if not legal_actions:
            self._expanded = True
            return
        legal_total = sum(float(policy[action]) for action in legal_actions)
        if legal_total <= 0:
            raise ValueError("policy must assign positive mass to legal moves")

        self.children = [
            NeuralMCTSNode(
                parent=self,
                action=action,
                prior=float(policy[action]) / legal_total,
            )
            for action in legal_actions
        ]
        self._expanded = True

    def best_child(self, c_puct):
        """Choose the child with the largest PUCT score."""
        if not self.children:
            raise RuntimeError("cannot select a child before expanding the node")
        parent_scale = math.sqrt(max(self.n, 1))

        def puct_score(child):
            mean_value = child.q / child.n if child.n else 0.0
            exploration = (
                c_puct * child.prior * parent_scale / (1 + child.n)
            )
            return mean_value + exploration

        return max(self.children, key=puct_score)

    def backpropagate(self, value):
        """Propagate a leaf value, changing perspective at every level."""
        current_node = self
        current_value = float(value)
        while current_node is not None:
            current_node._number_of_visits += 1
            current_node._value_sum += current_value
            current_value = -current_value
            current_node = current_node.parent


class NeuralMonteCarloTreeSearch:
    """Sequential PUCT search with one dual-head evaluation per new leaf."""

    def __init__(self, node, evaluator, c_puct=1.5):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        if not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError("c_puct must be positive and finite")
        self.root = node
        self.evaluator = evaluator
        self.c_puct = float(c_puct)
        self.last_search_stats = None

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """Run sequential simulations and return the most visited root child."""
        if simulations_number is None:
            if total_simulation_seconds is None:
                raise ValueError("a simulation count or time budget is required")
            if (
                not math.isfinite(total_simulation_seconds)
                or total_simulation_seconds <= 0
            ):
                raise ValueError("total_simulation_seconds must be positive")
        else:
            if isinstance(simulations_number, bool) or not isinstance(
                simulations_number,
                int,
            ):
                raise TypeError("simulations_number must be an integer")
            if simulations_number <= 0:
                raise ValueError("simulations_number must be positive")

        started_at = time.monotonic()
        completed_simulations = 0
        if simulations_number is not None:
            for _ in range(simulations_number):
                self._run_simulation()
                completed_simulations += 1
        else:
            deadline = started_at + total_simulation_seconds
            while time.monotonic() < deadline:
                self._run_simulation()
                completed_simulations += 1

        elapsed_seconds = time.monotonic() - started_at
        self.last_search_stats = SearchStats(
            completed_rollouts=completed_simulations,
            completed_batches=0,
            elapsed_seconds=elapsed_seconds,
        )
        if not self.root.children:
            raise RuntimeError("neural MCTS did not expand the root")

        # Visit count is the robust policy-improvement target used by neural MCTS.
        # Prior probability only resolves the one-simulation all-zero tie.
        return max(self.root.children, key=lambda child: (child.n, child.prior))

    def _run_simulation(self):
        node = self.root
        while node.is_expanded and node.children:
            node = node.best_child(self.c_puct)

        if node.is_terminal_node():
            value = float(node.state.game_result * node.state.next_to_move)
        else:
            prediction = self.evaluator(node.state)
            try:
                policy = prediction["policy"]
                value = float(prediction["value"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "evaluator must return finite 'policy' and 'value' fields"
                ) from error
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("evaluator value must be finite and within [-1, 1]")
            node.expand(policy)

        node.backpropagate(value)
