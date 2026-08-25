"""Public API for the Monte Carlo tree search implementation.

Implementation lives in focused modules; imports from ``mcts.enclosure``
provide callers with the same stable facade pattern used by ``game.enclosure``.
"""

try:  # Package imports using the mcts enclosure module
    from .nodes import MCTSNode, TwoPlayerMCTSNode, rollout_state
    from .search import MonteCarloTreeSearch, SearchStats
except ImportError:  # Direct imports when running from inside ``mcts``
    from nodes import MCTSNode, TwoPlayerMCTSNode, rollout_state
    from search import MonteCarloTreeSearch, SearchStats


__all__ = [
    "MCTSNode",
    "MonteCarloTreeSearch",
    "SearchStats",
    "TwoPlayerMCTSNode",
    "rollout_state",
]
