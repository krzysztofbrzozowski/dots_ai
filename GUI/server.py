"""Live MCTS backend for the shared Dots analysis interface."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analysis.diagnostics import DIAGNOSTICS, PRINT_T
from analysis.service import mcts_decision_frame


GUI_DIRECTORY = Path(__file__).resolve().parent
APP_DIRECTORY = GUI_DIRECTORY / "analysis"
SHARED_DIRECTORY = GUI_DIRECTORY / "shared"


def _position_frame(game, move_number):
    """Build a renderer-compatible frame while the next search is pending."""
    rows, cols = (int(value) for value in game.board.shape)
    legal_actions = game.get_legal_actions()
    legal_mask = [[0 for _ in range(cols)] for _ in range(rows)]
    for row, col in legal_actions:
        legal_mask[int(row)][int(col)] = 1
    winner = game.game_result

    return {
        "live_position": True,
        "index": None,
        "move_number": int(move_number),
        "rows": rows,
        "cols": cols,
        "board": game.board.tolist(),
        "territory": game.territory.tolist(),
        "player_to_move": int(game.next_to_move),
        "scores": {
            "player_1": int(game.score[1]),
            "player_2": int(game.score[-1]),
        },
        "q_values": [[0.0 for _ in range(cols)] for _ in range(rows)],
        "q_perspective": "player_to_move",
        "visit_counts": [[0 for _ in range(cols)] for _ in range(rows)],
        "policy_priors": None,
        "legal_mask": legal_mask,
        "legal_move_count": len(legal_actions),
        "selected_action": None,
        "selected_action_statistics": None,
        "completed_rollouts": 0,
        "elapsed_seconds": 0.0,
        "rollouts_per_second": 0.0,
        "is_first_frame": int(move_number) == 1,
        "is_last_frame": winner is not None,
        "game_over": winner is not None,
        "winner": int(winner) if winner is not None else None,
    }


class LiveGameStore:
    """Keep an isolated timeline and current position for browser clients."""

    def __init__(self):
        self._lock = RLock()
        self._revision = 0
        self._created_at_utc = None
        self._frames = []
        self._position = None
        self._message = "Waiting for an MCTS game to start."
        self._simulation_seconds = None
        self._simulations_number = None
        self._rollout_batch_size = 1

    def start(
        self,
        game,
        *,
        simulation_seconds,
        simulations_number,
        rollout_batch_size,
    ):
        with self._lock:
            self._revision += 1
            self._created_at_utc = datetime.now(timezone.utc).isoformat()
            self._frames = []
            self._position = _position_frame(game, 1)
            self._message = "MCTS game ready. Player 1 is searching."
            self._simulation_seconds = simulation_seconds
            self._simulations_number = simulations_number
            self._rollout_batch_size = int(rollout_batch_size)
            snapshot = self._snapshot_unlocked()
        PRINT_T(self._message, level="success", source="MCTS")
        return snapshot

    def publish(
        self,
        *,
        state,
        root,
        selected_action,
        search_stats,
        resulting_state,
        move_number,
        message,
    ):
        frame = mcts_decision_frame(
            state,
            root,
            selected_action,
            search_stats,
            move_number,
        )
        with self._lock:
            if self._position is None:
                raise RuntimeError("initialize the live game before publishing moves")
            expected_move_number = len(self._frames) + 1
            if int(move_number) != expected_move_number:
                raise ValueError(
                    f"expected live move {expected_move_number}, got {move_number}"
                )
            frame["index"] = len(self._frames)
            self._frames.append(deepcopy(frame))
            self._position = _position_frame(resulting_state, int(move_number) + 1)
            self._message = str(message)
            self._revision += 1
            snapshot = self._snapshot_unlocked()
        PRINT_T(
            message,
            level="success" if resulting_state.game_result is not None else "info",
            source="MCTS",
        )
        return snapshot

    def read(self):
        with self._lock:
            if self._position is None:
                return None
            return self._snapshot_unlocked()

    def frame(self, frame_index):
        with self._lock:
            if not 0 <= frame_index < len(self._frames):
                raise IndexError(frame_index)
            return deepcopy(self._frames[frame_index])

    def _snapshot_unlocked(self):
        frames = self._frames
        position = self._position
        total_rollouts = sum(frame["completed_rollouts"] for frame in frames)
        total_elapsed = sum(frame["elapsed_seconds"] for frame in frames)
        timeline = [
            {
                "index": int(frame["index"]),
                "move_number": int(frame["move_number"]),
                "player_to_move": int(frame["player_to_move"]),
                "scores": deepcopy(frame["scores"]),
                "selected_action": deepcopy(frame["selected_action"]),
                "selected_mean_value": frame["selected_action_statistics"][
                    "mean_value"
                ],
                "selected_visits": int(
                    frame["selected_action_statistics"]["visits"]
                ),
                "selected_prior": frame["selected_action_statistics"]["prior"],
                "selected_visit_share": frame["selected_action_statistics"][
                    "visit_share"
                ],
                "completed_rollouts": int(frame["completed_rollouts"]),
                "elapsed_seconds": float(frame["elapsed_seconds"]),
            }
            for frame in frames
        ]
        final_result = position["winner"] if position["game_over"] else None
        status = "complete" if position["game_over"] else "searching"
        summary = {
            "analysis_id": "live",
            "source_type": "live",
            "file_name": "Live MCTS game",
            "schema_version": 1,
            "game_id": "live",
            "created_at_utc": self._created_at_utc,
            "board": {"rows": position["rows"], "cols": position["cols"]},
            "frame_count": len(frames),
            "final_result": final_result,
            "final_result_perspective": "player_1",
            "q_perspective": "player_to_move",
            "search": {
                "budget_type": (
                    "seconds"
                    if self._simulation_seconds is not None
                    else "simulations"
                ),
                "requested_simulation_seconds": self._simulation_seconds,
                "requested_simulations": (
                    self._simulations_number
                    if self._simulation_seconds is None
                    else None
                ),
                "rollout_batch_size": self._rollout_batch_size,
                "total_rollouts": total_rollouts,
                "total_elapsed_seconds": total_elapsed,
                "average_rollouts_per_second": (
                    total_rollouts / total_elapsed if total_elapsed else 0.0
                ),
            },
            "timeline": timeline,
        }
        return deepcopy(
            {
                "revision": self._revision,
                "status": status,
                "message": self._message,
                "analysis": summary,
                "current_frame": position,
            }
        )


live_game_store = LiveGameStore()


def initialize_live_game(
    game,
    *,
    simulation_seconds,
    simulations_number,
    rollout_batch_size,
):
    return live_game_store.start(
        game,
        simulation_seconds=simulation_seconds,
        simulations_number=simulations_number,
        rollout_batch_size=rollout_batch_size,
    )


def publish_search(**search_result):
    return live_game_store.publish(**search_result)


app = FastAPI(
    title="Dots live MCTS API",
    description="Publish a running MCTS game to the shared analysis interface.",
    version="3.0.0",
)


@app.middleware("http")
async def prevent_local_gui_asset_caching(request: Request, call_next):
    response = await call_next(request)
    if (
        request.url.path == "/"
        or request.url.path.startswith("/assets/")
        or request.url.path.startswith("/shared/")
    ):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/runtime", include_in_schema=False)
def runtime():
    return {"mode": "live"}


@app.get("/api/diagnostics", include_in_schema=False)
def get_diagnostics(after: int = Query(default=0, ge=0)):
    events, cursor = DIAGNOSTICS.read_after(after)
    return {"events": events, "cursor": cursor}


@app.get("/api/live")
def get_live_game():
    snapshot = live_game_store.read()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="No MCTS game has started yet")
    return snapshot


@app.get("/api/live/frames/{frame_index}")
def get_live_frame(frame_index: int):
    try:
        return live_game_store.frame(frame_index)
    except IndexError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Live frame {frame_index} does not exist.",
        ) from error


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(APP_DIRECTORY / "index.html")


app.mount(
    "/assets",
    StaticFiles(directory=APP_DIRECTORY),
    name="app-assets",
)
app.mount(
    "/shared",
    StaticFiles(directory=SHARED_DIRECTORY),
    name="shared-gui-assets",
)
