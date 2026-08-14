import numpy as np
from common import TwoPlayersAbstractGameState

class DotsGameState(TwoPlayersAbstractGameState):

    def __init__(self, map_state, next_to_move=1, win=None):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Only 2D square boards allowed")
        self.board = map_state

    @property
    def game_result(self):
        # Here some clever algorith has to be figured out
        pass