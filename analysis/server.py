"""Standalone HTTP application for the saved-game analyzer."""

from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analysis.errors import AnalysisFileError
from analysis.loader import MAX_UPLOAD_BYTES, load_analysis_bytes
from analysis.service import (
    AnalysisNotFoundError,
    AnalysisStore,
    FrameNotFoundError,
    analysis_frame,
    analysis_summary,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
ANALYSIS_GUI_DIRECTORY = PROJECT_DIRECTORY / "GUI" / "analysis"
SHARED_GUI_DIRECTORY = PROJECT_DIRECTORY / "GUI" / "shared"


async def _read_request_body_with_limit(request):
    """Read an upload incrementally so a large request is rejected early."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The selected NPZ file is larger than 32 MB",
            )

    file_bytes = bytearray()
    async for chunk in request.stream():
        file_bytes.extend(chunk)
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The selected NPZ file is larger than 32 MB",
            )
    return bytes(file_bytes)


def create_app(store=None):
    """Build an application with an injectable store for isolated tests."""
    analysis_store = store or AnalysisStore()
    application = FastAPI(
        title="Dots MCTS Analysis API",
        description="Read-only analysis of saved MCTS self-play games.",
        version="1.0.0",
    )

    @application.exception_handler(AnalysisFileError)
    async def analysis_file_error_handler(_request, error):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @application.get("/api/health", include_in_schema=False)
    def health():
        """Report that the independent analysis application is running."""
        return {"status": "ready"}

    @application.post("/api/analyses", status_code=201)
    async def import_analysis(request: Request):
        """Validate one uploaded NPZ file and start an in-memory session."""
        file_bytes = await _read_request_body_with_limit(request)
        encoded_file_name = request.headers.get("x-file-name", "game.npz")
        file_name = unquote(encoded_file_name)
        game = load_analysis_bytes(file_bytes, file_name)
        analysis_id = analysis_store.add(game)
        return analysis_summary(analysis_id, game)

    @application.get("/api/analyses/{analysis_id}")
    def get_analysis(analysis_id: str):
        try:
            game = analysis_store.get(analysis_id)
        except AnalysisNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail="The analysis session was not found. Import the file again.",
            ) from error
        return analysis_summary(analysis_id, game)

    @application.get("/api/analyses/{analysis_id}/frames/{frame_index}")
    def get_analysis_frame(analysis_id: str, frame_index: int):
        try:
            game = analysis_store.get(analysis_id)
            return analysis_frame(game, frame_index)
        except AnalysisNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail="The analysis session was not found. Import the file again.",
            ) from error
        except FrameNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_index} does not exist in this game.",
            ) from error

    @application.delete(
        "/api/analyses/{analysis_id}",
        status_code=204,
        response_class=Response,
    )
    def delete_analysis(analysis_id: str):
        try:
            analysis_store.remove(analysis_id)
        except AnalysisNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail="The analysis session was not found.",
            ) from error
        return Response(status_code=204)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(ANALYSIS_GUI_DIRECTORY / "index.html")

    application.mount(
        "/assets",
        StaticFiles(directory=ANALYSIS_GUI_DIRECTORY),
        name="analysis-assets",
    )
    application.mount(
        "/shared",
        StaticFiles(directory=SHARED_GUI_DIRECTORY),
        name="shared-gui-assets",
    )
    return application


app = create_app()
