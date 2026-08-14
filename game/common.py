from abc import ABC, abstractmethod

class TwoPlayersAbstractGameState(ABC):

    @abstractmethod
    def game_result(self):
        """
        Return the result of the game

        Returns
        -------
        int or None
            1 if player #1 wins
            -1 if player #2 wins
            0 if there is a draw
            None if the game is not over yet
        """
        pass