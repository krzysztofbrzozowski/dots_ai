from pathlib import Path

from game.enclosure import (
    PLAYER_1,
    PLAYER_2,
    DotsGame,
    could_have_closed_loop,
    detect_capture_info,
    find_candidate_regions,
    opponent_of,
)

from mcts.enclosure import (
    MCTSNode,
    MonteCarloTreeSearch,
    TwoPlayerMCTSNode,
)


GUI_DIR = Path(__file__).resolve().parent
DEFAULT_ROWS = 10
DEFAULT_COLS = 10

board_state = DotsGame(self.rows, self.cols)

# TODO Implement this method or think about some other condition \
# when the game is still running
# maybe Legal moves as in GUI?
# while game_state.anyone_has_move:
# calculate best move


while board_state.game_result is None:
    root = TwoPlayerMCTSNode(state=board_state)

    mcts = MonteCarloTreeSearch(root)

    best_node = mcts.best_action(simulations_number=12)

    board_state = best_node.state

