"""Monte Carlo tree search with optional process-backed leaf rollouts"""

import random
import time
from dataclasses import dataclass

try:  # Support package and direct imports
    from .nodes import rollout_state
except ImportError:  # pragma: no cover - direct module execution
    from nodes import rollout_state


@dataclass(frozen=True)
class SearchStats:
    """Performance statistics from one completed MCTS search"""

    completed_rollouts: int
    completed_batches: int
    elapsed_seconds: float

    @property
    def rollouts_per_second(self):
        """Average completed rollout throughput"""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.completed_rollouts / self.elapsed_seconds


class MonteCarloTreeSearch:
    """Own an MCTS tree and optionally dispatch rollout batches to an executor

    Selection, expansion, reservations, and backpropagation always happen in
    the calling process. Only immutable game states cross the executor boundary,
    so worker processes never mutate or copy the search tree itself.
    """

    def __init__(
        self,
        node,
        rollout_executor=None,
        rollout_batch_size=1,
        random_seed=None,
    ):
        if isinstance(rollout_batch_size, bool) or not isinstance(
            rollout_batch_size, int
        ):
            raise TypeError("rollout_batch_size must be an integer")
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if rollout_executor is None and rollout_batch_size != 1:
            raise ValueError("rollout_batch_size requires a rollout executor")

        self.root = node
        self.rollout_executor = rollout_executor
        self.rollout_batch_size = rollout_batch_size
        self._seed_source = random.Random(random_seed)
        self.last_search_stats = None

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """Run the configured search budget and return the best root child"""
        # Time based search
        if simulations_number is None:
            if total_simulation_seconds is None:
                raise ValueError("a simulation count or time budget is required")
            if total_simulation_seconds <= 0:
                raise ValueError("total_simulation_seconds must be positive")

        # Iterations number based search
        else:
            if isinstance(simulations_number, bool) or not isinstance(
                simulations_number, int
            ):
                raise TypeError("simulations_number must be an integer")
            if simulations_number <= 0:
                raise ValueError("simulations_number must be positive")

        started_at = time.monotonic()
        if simulations_number is None:
            completed_rollouts, completed_batches = self._time_based_search(
                total_simulation_seconds
            )
        else:
            completed_rollouts, completed_batches = self._iteration_based_search(
                simulations_number
            )
        elapsed_seconds = time.monotonic() - started_at

        self.last_search_stats = SearchStats(
            completed_rollouts=completed_rollouts,
            completed_batches=completed_batches,
            elapsed_seconds=elapsed_seconds,
        )

        # Final selection is exploitation-only
        # All reservations have been released before this point
        # Only completed results are considered
        # TODO: For more diverse self-play training data, consider sampling
        # early-game moves from the MCTS visit distribution instead of always
        # selecting the exploitation-only argmax.
        return self.root.best_child(c_param=0.0)

    def _time_based_search(self, total_simulation_seconds):
        """Time based search"""
        deadline = time.monotonic() + total_simulation_seconds
        completed_rollouts = 0

        # Sequential simulation
        if self.rollout_executor is None:
            while time.monotonic() < deadline:
                self._run_sequential_simulation()
                completed_rollouts += 1
            return completed_rollouts, 0

        # Parallel simulation
        completed_batches = 0
        while time.monotonic() < deadline:
            # rollout_batch_size controls how many rollout tasks are submitted together
            # The executor worker count controls how many tasks can run at the same time
            self._run_parallel_batch(self.rollout_batch_size)
            completed_rollouts += self.rollout_batch_size
            completed_batches += 1
        return completed_rollouts, completed_batches

    def _iteration_based_search(self, simulations_number):
        """Iteration based search"""

        # Sequential simulation
        if self.rollout_executor is None:
            for _ in range(simulations_number):
                self._run_sequential_simulation()
            return simulations_number, 0

        # Parallel simulation
        remaining = simulations_number
        completed_rollouts = 0
        completed_batches = 0
        while remaining:
            # Use a smaller final batch when fewer rollouts remain than configured
            batch_size = min(self.rollout_batch_size, remaining)
            self._run_parallel_batch(batch_size)
            remaining -= batch_size
            completed_rollouts += batch_size
            completed_batches += 1
        return completed_rollouts, completed_batches

    def _run_sequential_simulation(self):
        # Expand the current tree -> return child node
        #     with next_state <- independent game-state (board) object
        leaf = self._tree_policy()
        #   For that child node play game until termination happen 
        reward = leaf.rollout()
        # Each child contains statistics only from simulations 
        # that passed through that child
        leaf.backpropagate(reward)

    def _run_parallel_batch(self, batch_size):
        leaves = []
        futures = []

        try:
            # Select one leaf for each rollout in the current batch
            for _ in range(batch_size):
                # Select the child from untried_actions
                leaf = self._tree_policy()
                # Reserve the selected path so subsequent tree policy calls in this
                # batch are less likely to select a path with an assigned rollout
                leaf.reserve_path()
                # Add to the list of children
                leaves.append(leaf)

            for leaf in leaves:
                # Generate a 128-bit seed for an independent rollout random number generator
                seed = self._seed_source.getrandbits(128)
                # Submit one rollout task per leaf to the executor and store its future result
                #   -> submit one rollout task per leaf to the process pool
                #   -> executor assigns queued tasks to its available workers
                #
                # rollout_state -> make a random.move based on the seed
                # TODO -> maybe here we might do some improvement, but i think it is already self fine -> get_legal_actions reutn only possible moves
                futures.append(
                    self.rollout_executor.submit(rollout_state, leaf.state, seed)
                )

            # rewards list with terminal_state results
            rewards = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        # At the end release leaves for future usage
        finally:
            for leaf in leaves:
                leaf.release_path()

        # Backpropagate each result
        for leaf, reward in zip(leaves, rewards):
            leaf.backpropagate(reward)

    def _tree_policy(self):
        current_node = self.root

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            current_node = current_node.best_child()

        return current_node
