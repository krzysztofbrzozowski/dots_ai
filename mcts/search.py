import time
class MonteCarloTreeSearch(object):

    def __init__(self, node):
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else :
            for _ in range(0, simulations_number):
                # Initial state:
                #   Expand the current tree -> return child node 
                #     with next_state <- independent game-state (board) object
                #   Assign it to v for now
                # Further state:
                #   -> in 3x3 board only 9 child aviable
                #   -> later iterations will return self.root.bestchild with c_param 1.4
                # And end state of choosing best_child
                v = self._tree_policy()
                # Initial state:
                #   For that child node play game until termination happen 
                #   (some game result)
                #   reward <- game_result
                reward = v.rollout()
                # Initial state:
                #   For that child node add statistics
                #   -> increment _number_of_visits +1
                #   -> for each result (-1 -> O win, 0 -> draw, 1 -> X win)
                #       increment number of wins
                #   -> for root also do the same - root contains statistics from all simulations below it
                # Backpropagation example:
                # root: visits=30, results={1: 18, -1: 10, 0: 2}
                # ├─ child_0: visits=10, results={1: 8, -1: 2}
                # ├─ child_1: visits=5,  results={1: 2, -1: 3}
                # ├─ child_2: visits=15, results={1: 8, -1: 5, 0: 2}
                # Each child contains statistics only from simulations that passed through that child
                v.backpropagate(reward)
        # to select best child go for exploitation only
        # After MCTS:
        #   "Which move looks best based on collected statistics?" -> c_param = 0
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        # This is current root <- TwoPlayerMCTSNode(state=board_state)
        # Initial state:
        #   children = []
        #   n = 0.0 -> because of @property
        current_node = self.root
        # While not game_over() -> self.game_result is not None (False)
        while not current_node.is_terminal_node():
            # Initial state:
            #   len(self.untried_actions) == 0 -> at the beginning 9 -> 9 == 0 ? -> False
            # untried_actions -> get_legal_actions (in this case)
            #
            #   See the Notes -> Note for the game coordinates to see game coordinates
            #
            # not False? -> go to current_node.expand()
            # With 3x3 game only 9 children from root aviable
            # Further state:
            #   -> After all 9 root children are expanded:
            #   -> len(self.untried_actions) == 0 -> 0 == 0 ? -> True
            #      not True -> False -> go to else
            #   -> current_node = current_node.best_child()
            #   Example: best_child() returns child_3
            #
            #   Next while iteration:
            #      -> current_node is now child_3
            #      -> If child_3 still has untried actions -> expand one of its children
            #      -> Example: child_3 -> child_3_0
            #      -> child_3_0 is returned from _tree_policy() and rollout starts from it
            if not current_node.is_fully_expanded():
                # -> Add child_node to the children of current_node
                # -> Child node has proposed virutal move with next_state (board_state)
                return current_node.expand()
            # Further state:
            #   All of untried_actions -> get_legal_actions poped
            #   No more actions to try, return best_child anyway
            else:
                # For each child of current node calculate best_child
                # During MCTS:
                #   "Which moves should I explore more?" -> c_param = 1.4   
                current_node = current_node.best_child()
        return current_node
