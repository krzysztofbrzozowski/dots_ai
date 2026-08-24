import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


class MCTSNode(ABC):

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        pass
    # current_node.q AttributeError("'NoneType' object has no attribute 'state'")
    @property
    @abstractmethod
    def q(self):
        pass
    
    # Proerty gives us during the initial state the value of it
    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # c_param controls how strongly MCTS prefers exploration
        # 1.4 is a common default value because it is close to sqrt(2) ≈ 1.414
        # ---
        # c.q / c.n
        #   -> EXPLOITATION - korzystanie z ruchów, które już wyglądają na dobre
        #   -> tells how good this child was in previous simulations
        #   -> q (wins - loses) / n (number of visits of this child)
        #   -> gives a hint how good or bad general this child is
        #
        # c_param * sqrt(2 * log(self.n) / c.n)
        #   -> EXPLORATION - sprawdzanie ruchów, które były jeszcze mało testowane 
        #   -> gives extra score to children that were visited less often
        #   -> np.log(self.n) = number of visits of the parent node
                # >>> np.log(1)
                # np.float64(0.0)
                # >>> np.log(2)
                # np.float64(0.6931471805599453)
                # >>> np.log(100)
                # np.float64(4.605170185988092)
                # >>> np.log(1000)
                # np.float64(6.907755278982137)
                # >>> np.log(10000)
                # np.float64(9.210340371976184)
                # >>> np.log(100000)
                # np.float64(11.512925464970229)
        #   -> c.n = number of visits of this child
        #       odwrotnie proporcjonalny składnik
        #       - dzielenie przez dużą liczbę -> mniejszy wynik końcowy
        #       - dzielenie przez małą liczbę -> wiekszy wynik końcowy
        #   -> bigger c_param -> more exploration
        #   -> c_param = 0 -> use exploitation (first part of equation) only
        #
        # Final intuition:
        #   good child + not explored enough child can both get selected
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]


class TwoPlayerMCTSNode(MCTSNode):

    def __init__(self, state, parent=None, action=None):
        super().__init__(state, parent, action)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = list(self.state.get_legal_actions())
            # np.random.shuffle(self._untried_actions)
        return self._untried_actions

    @property
    def q(self):
        # For wins:
        #   -> parent will give next move to -1 or 1
        #   -> took all results for [-1 or +1]
        # For loses:
        #   -> parent will give next move to -1 or 1
        #   -> took all results for -1 (opposite) * [-1 or +1]
        # Return:
        #   general balance wins - loses
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        # From possible moves -> get last one / pop -> assign to action
        # e.g. action = x:2 y:2 v:1 -> v (next player to move)
        action = self.untried_actions.pop()
        # Inside create copy od current state
        # and return new obiect with applied move, old one remains unchanged
        next_state = self.state.move(action)
        # Create new object of TwoPlayerMCTSNode and assign to it:
        #   -> next_state - independent game-state (board) object
        #   -> parent - current_node will be parent (in first iteration root)
        child_node = TwoPlayerMCTSNode(
            state=next_state,
            parent=self,
            action=action,
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        # Assign the child_node.state -> next_state from above methid
        current_rollout_state = self.state
        # Play the game until a terminal state is reached
        # Here is the same logic, move is only done in new object
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
