"""Collection-only randomized MCTS nodes; the original MCTS is unchanged."""

from mcts.enclosure import TwoPlayerMCTSNode, rollout_state


class CollectionNode(TwoPlayerMCTSNode):
    def __init__(self, state, rng, parent=None, action=None):
        super().__init__(state, parent=parent, action=action)
        self.rng = rng

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = list(self.state.get_legal_actions())
            self.rng.shuffle(self._untried_actions)
        return self._untried_actions

    def expand(self):
        action = self.untried_actions.pop()
        child = CollectionNode(
            self.state.move(action), self.rng, parent=self, action=action,
        )
        self.children.append(child)
        return child

    def rollout(self):
        return rollout_state(self.state, seed=int(self.rng.integers(2**63)))
