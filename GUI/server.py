"""FastAPI adapter for the browser-based Dots inspector.

The API owns turn metadata, while ``DotsGame`` remains responsible for move
validation, captures, territory, connectivity, and scoring.
"""

from pathlib import Path
from threading import RLock

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from game.enclosure import (
    PLAYER_1,
    PLAYER_2,
    DotsGame,
    could_have_closed_loop,
    detect_capture_info,
    find_candidate_regions,
    opponent_of,
)


GUI_DIR = Path(__file__).resolve().parent
DEFAULT_ROWS = 10
DEFAULT_COLS = 10


class MoveRequest(BaseModel):
    """Coordinates selected by the browser."""

    row: int
    col: int


def _coordinates(cells):
    """Convert NumPy-friendly coordinate tuples into JSON-friendly lists."""

    return [[int(row), int(col)] for row, col in cells]


def _regions(regions):
    return [_coordinates(region) for region in regions]


class GameSession:
    """Keep one local game and the small amount of turn/UI metadata around it."""

    def __init__(self, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
        self.rows = rows
        self.cols = cols
        self._lock = RLock()
        self.reset()

    @staticmethod
    def _empty_debug():
        return {
            "could_have_closed_loop": None,
            "candidate_region_count": 0,
            "candidate_regions": [],
            "enclosed_region_count": 0,
            "detected_enclosed_regions": [],
            "opponent_cells_found": [],
            "capture_happened": False,
            "captured_dots": [],
            "captured_regions": [],
        }

    def reset(self):
        with self._lock:
            self.game = DotsGame(self.rows, self.cols)
            self.current_player = PLAYER_1
            self.move_number = 0
            self.last_move = None
            self.last_captured_dots = []
            self.capture_happened = False
            self.debug = self._empty_debug()
            self.message = "New game ready. Player 1 to move."
            return self._state_unlocked()

    def state(self):
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self):
        legal_moves = self.game.legal_moves()
        return {
            "rows": int(self.game.board.shape[0]),
            "cols": int(self.game.board.shape[1]),
            "board": self.game.board.tolist(),
            "territory": self.game.territory.tolist(),
            "score": {
                "player_1": int(self.game.score[PLAYER_1]),
                "player_2": int(self.game.score[PLAYER_2]),
            },
            "current_player": int(self.current_player),
            "last_move": (
                _coordinates([self.last_move])[0] if self.last_move else None
            ),
            "legal_moves": _coordinates(legal_moves),
            "legal_move_count": len(legal_moves),
            "move_number": self.move_number,
            "last_captured_dots": _coordinates(self.last_captured_dots),
            "capture_happened": self.capture_happened,
            "debug": self.debug,
            "message": self.message,
        }

    def apply_move(self, row, col):
        with self._lock:
            if not self.game.is_legal_move(row, col):
                self.message = f"Illegal move at ({row}, {col})."
                return False, self._state_unlocked()

            moving_player = self.current_player
            player_label = "Player 1" if moving_player == PLAYER_1 else "Player 2"

            # Preserve the exact pre-move inputs used by the engine's cycle
            # check so the same helpers can also produce read-only diagnostics.
            pre_move_groups = self.game.groups
            pre_move_territory = self.game.territory.copy()
            last_move = (row, col)
            diagnostic_board = self.game.board.copy()
            diagnostic_board[last_move] = moving_player

            could_have_closed_loop_result = could_have_closed_loop(
                diagnostic_board,
                pre_move_territory,
                last_move,
                moving_player,
                groups=pre_move_groups,
            )
            candidate_regions = find_candidate_regions(
                diagnostic_board,
                pre_move_territory,
                last_move,
                moving_player,
            )
            capture_info = detect_capture_info(
                diagnostic_board,
                last_move,
                moving_player,
                territory=pre_move_territory,
                groups=pre_move_groups,
            )

            captured_dots = self.game.place_dot(row, col, moving_player)

            self.move_number += 1
            self.last_move = last_move
            self.last_captured_dots = list(captured_dots)
            self.capture_happened = bool(captured_dots)
            self.current_player = opponent_of(moving_player)
            next_player_label = (
                "Player 1" if self.current_player == PLAYER_1 else "Player 2"
            )

            captured_regions = list(capture_info.captured_regions)
            self.debug = {
                "could_have_closed_loop": could_have_closed_loop_result,
                # The existing helper returns flood-fill seeds around last_move.
                "candidate_region_count": len(candidate_regions),
                "candidate_regions": _coordinates(candidate_regions),
                "enclosed_region_count": len(captured_regions),
                "detected_enclosed_regions": _regions(captured_regions),
                "opponent_cells_found": _coordinates(capture_info.captured_dots),
                "capture_happened": capture_info.happened,
                "captured_dots": _coordinates(capture_info.captured_dots),
                "captured_regions": _regions(captured_regions),
            }

            if captured_dots:
                noun = "dot" if len(captured_dots) == 1 else "dots"
                self.message = (
                    f"{player_label} captured {len(captured_dots)} {noun} with "
                    f"({row}, {col}). {next_player_label} to move."
                )
            else:
                self.message = (
                    f"{player_label} placed a dot at ({row}, {col}). "
                    f"{next_player_label} to move."
                )

            return True, self._state_unlocked()


app = FastAPI(
    title="Dots GUI API",
    description="A presentation adapter around the existing DotsGame engine.",
    version="1.0.0",
)
session = GameSession()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(GUI_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def styles():
    return FileResponse(GUI_DIR / "styles.css", media_type="text/css")


@app.get("/game.js", include_in_schema=False)
def script():
    return FileResponse(GUI_DIR / "game.js", media_type="text/javascript")


@app.get("/api/state")
def get_state():
    return session.state()


@app.post("/api/move")
def post_move(move: MoveRequest):
    succeeded, state = session.apply_move(move.row, move.col)
    if not succeeded:
        return JSONResponse(
            status_code=409,
            content={"error": "Illegal move", "state": state},
        )
    return state


@app.post("/api/reset")
def reset_game():
    return session.reset()
