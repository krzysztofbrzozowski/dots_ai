"""Monte Carlo tree search with optional process-backed leaf rollouts"""

import random
import time

try:  # Support package and direct imports
    from .nodes import rollout_state
except ImportError:  # pragma: no cover - direct module execution
    from nodes import rollout_state


class MonteCarloTreeSearch:
    """Own an MCTS tree and optionally dispatch rollout batches to an executor.

    Selection, expansion, reservations, and backpropagation always happen in
    the calling process. Only immutable game states cross the executor boundary,
    so worker processes never mutate or copy the search tree itself.
    """

    def __init__(
        self,
        node,
        rollout_executor=None,
        parallelism=1,
        random_seed=None,
    ):
        if isinstance(parallelism, bool) or not isinstance(parallelism, int):
            raise TypeError("parallelism must be an integer")
        if parallelism <= 0:
            raise ValueError("parallelism must be positive")
        if rollout_executor is None and parallelism != 1:
            raise ValueError("parallelism requires a rollout executor")

        self.root = node
        self.rollout_executor = rollout_executor
        self.parallelism = parallelism
        self._seed_source = random.Random(random_seed)

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """Run the configured search budget and return the best root child"""
        # Run time based search
        if simulations_number is None:
            if total_simulation_seconds is None:
                raise ValueError("a simulation count or time budget is required")
            if total_simulation_seconds <= 0:
                raise ValueError("total_simulation_seconds must be positive")
            self._search_for_seconds(total_simulation_seconds)
            
        # Run iterations number based search
        else:
            if isinstance(simulations_number, bool) or not isinstance(
                simulations_number, int
            ):
                raise TypeError("simulations_number must be an integer")
            if simulations_number <= 0:
                raise ValueError("simulations_number must be positive")
            self._search_simulations(simulations_number)

        # Final selection is exploitation-only. All reservations have been
        # released before this point, so only completed results are considered.
        return self.root.best_child(c_param=0.0)

    def _search_for_seconds(self, total_simulation_seconds):
        deadline = time.monotonic() + total_simulation_seconds

        if self.rollout_executor is None:
            first_simulation = True
            while first_simulation or time.monotonic() < deadline:
                self._run_sequential_simulation()
                first_simulation = False
            return

        first_batch = True
        while first_batch or time.monotonic() < deadline:
            self._run_parallel_batch(self.parallelism)
            first_batch = False

    def _search_simulations(self, simulations_number):
        if self.rollout_executor is None:
            for _ in range(simulations_number):
                self._run_sequential_simulation()
            return

        remaining = simulations_number
        while remaining:
            batch_size = min(self.parallelism, remaining)
            self._run_parallel_batch(batch_size)
            remaining -= batch_size

    def _run_sequential_simulation(self):
        leaf = self._tree_policy()
        reward = leaf.rollout()
        leaf.backpropagate(reward)

    def _run_parallel_batch(self, batch_size):
        leaves = []
        futures = []

        try:
            for _ in range(batch_size):
                leaf = self._tree_policy()
                leaf.reserve_path()
                leaves.append(leaf)

            for leaf in leaves:
                seed = self._seed_source.getrandbits(128)
                futures.append(
                    self.rollout_executor.submit(rollout_state, leaf.state, seed)
                )

            rewards = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            for leaf in leaves:
                leaf.release_path()

        for leaf, reward in zip(leaves, rewards):
            leaf.backpropagate(reward)

    def _tree_policy(self):
        current_node = self.root

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            current_node = current_node.best_child()

        return current_node
