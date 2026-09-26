"""FastAPI server for the standalone Elo arena page."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .arena import EloArenaSession, POOL_ID
from .config import AI_PRESETS, COLS, HUMAN_PRESET_ID, RATINGS_PATH, ROWS, STATIC_DIRECTORY
from .ratings import RatingStore


SHARED_DIRECTORY = Path(__file__).resolve().parents[1] / "GUI" / "shared"
WORKSPACE_ASSETS_DIRECTORY = Path(__file__).resolve().parents[1] / "GUI" / "analysis"


class StartMatchRequest(BaseModel):
    player_1: str
    player_2: str
    human_name_1: str | None = None
    human_name_2: str | None = None


class HumanMoveRequest(BaseModel):
    row: int
    col: int


rating_store = RatingStore(RATINGS_PATH)
arena_session = EloArenaSession(rating_store)

app = FastAPI(
    title="Dots 25x25 Elo Arena",
    description="Classic MCTS, neural MCTS, and human rated matches.",
    version="1.0.0",
)


@app.middleware("http")
async def prevent_asset_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(
        ("/assets/", "/shared/", "/workspace-assets/")
    ):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/config")
def get_config():
    players = [
        {
            "id": preset.id,
            "label": preset.label,
            "kind": preset.kind,
        }
        for preset in AI_PRESETS
    ]
    players.append(
        {"id": HUMAN_PRESET_ID, "label": "Human", "kind": "human"}
    )
    return {
        "board": {"rows": ROWS, "cols": COLS},
        "pool_id": POOL_ID,
        "players": players,
    }


@app.get("/api/state")
def get_state():
    return arena_session.snapshot()


@app.post("/api/matches")
def start_match(request: StartMatchRequest):
    try:
        return arena_session.start_match(
            request.player_1,
            request.player_2,
            request.human_name_1,
            request.human_name_2,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/matches/end")
def end_match():
    try:
        return arena_session.end_match()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/moves")
def submit_move(request: HumanMoveRequest):
    try:
        return arena_session.submit_human_move(request.row, request.col)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIRECTORY / "index.html")


app.mount("/assets", StaticFiles(directory=STATIC_DIRECTORY), name="arena-assets")
app.mount("/shared", StaticFiles(directory=SHARED_DIRECTORY), name="shared-gui-assets")
app.mount(
    "/workspace-assets",
    StaticFiles(directory=WORKSPACE_ASSETS_DIRECTORY),
    name="workspace-assets",
)
