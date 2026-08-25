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
        # Time based search
        if simulations_number is None:
            if total_simulation_seconds is None:
                raise ValueError("a simulation count or time budget is required")
            if total_simulation_seconds <= 0:
                raise ValueError("total_simulation_seconds must be positive")
            # Wrapper for time based search
            self._time_based_search(total_simulation_seconds)
            
        # Iterations number based search
        else:
            if isinstance(simulations_number, bool) or not isinstance(
                simulations_number, int
            ):
                raise TypeError("simulations_number must be an integer")
            if simulations_number <= 0:
                raise ValueError("simulations_number must be positive")
            self._iteration_based_search(simulations_number)

        # Final selection is exploitation-only
        # All reservations have been released before this point
        # Only completed results are considered
        return self.root.best_child(c_param=0.0)

    def _time_based_search(self, total_simulation_seconds):
        """Time based search"""
        deadline = time.monotonic() + total_simulation_seconds

        # Sequential simulation
        if self.rollout_executor is None:
            while time.monotonic() < deadline:
                self._run_sequential_simulation()
            return

        # Parallel simulation
        while time.monotonic() < deadline:
            # parallelism = DEFAULT_MCTS_WORKERS -> number of CPU cores in general
            self._run_parallel_batch(self.parallelism)

    def _iteration_based_search(self, simulations_number):
        """Iteration based search"""

        # Sequential simulation
        if self.rollout_executor is None:
            for _ in range(simulations_number):
                self._run_sequential_simulation()
            return

        # Parallel simulation
        remaining = simulations_number
        while remaining:
            batch_size = min(self.parallelism, remaining)
            self._run_parallel_batch(batch_size)
            remaining -= batch_size

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
            # batch_size -> parallelism = DEFAULT_MCTS_WORKERS -> CPU cores assigned
            for _ in range(batch_size):
                # Select the child from untried_actions
                leaf = self._tree_policy()
                # Reserve the current child to the root to make other workers
                # less likely select the same path
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
