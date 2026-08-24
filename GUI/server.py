"""Read-only HTTP presentation layer for an MCTS-driven Dots game.

``main_mcts.py`` owns the game and publishes serialized snapshots here. This
module only stores the latest snapshot and serves it to browser clients.
"""

from copy import deepcopy
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


GUI_DIR = Path(__file__).resolve().parent


def _coordinates(cells):
    """Convert NumPy-friendly coordinates into JSON-compatible lists."""
    return [[int(row), int(col)] for row, col in cells]


class SnapshotStore:
    """Keep an isolated, versioned copy of the latest display state."""

    def __init__(self):
        self._lock = RLock()
        self._snapshot = None
        self._version = 0

    def publish(self, snapshot):
        """Store and return an independent copy of ``snapshot``."""
        with self._lock:
            self._version += 1
            published = deepcopy(snapshot)
            published["version"] = self._version
            self._snapshot = published
            return deepcopy(published)

    def read(self):
        """Return an independent copy of the latest snapshot, if available."""
        with self._lock:
            return deepcopy(self._snapshot)


snapshot_store = SnapshotStore()


def publish_state(game, last_move, move_number, message):
    """Serialize an MCTS-owned game and publish it for read-only display."""
    legal_moves = game.get_legal_actions()
    winner = game.game_result
    captured_dots = list(game.last_captured_dots)

    snapshot = {
        "rows": int(game.board.shape[0]),
        "cols": int(game.board.shape[1]),
        "board": game.board.tolist(),
        "territory": game.territory.tolist(),
        "score": {
            "player_1": int(game.score[1]),
            "player_2": int(game.score[-1]),
        },
        "current_player": int(game.next_to_move),
        "last_move": _coordinates([last_move])[0] if last_move else None,
        "legal_moves": _coordinates(legal_moves),
        "legal_move_count": len(legal_moves),
        "move_number": int(move_number),
        "last_captured_dots": _coordinates(captured_dots),
        "capture_happened": bool(captured_dots),
        "game_over": winner is not None,
        "winner": int(winner) if winner is not None else None,
        "message": str(message),
    }
    return snapshot_store.publish(snapshot)


app = FastAPI(
    title="Dots MCTS Display API",
    description="Read-only snapshots published by the MCTS application.",
    version="2.0.0",
)


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
    snapshot = snapshot_store.read()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="No MCTS state published yet")
    return snapshot
