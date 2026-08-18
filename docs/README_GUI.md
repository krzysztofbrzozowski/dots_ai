# Dots GUI

The Dots GUI is a local web interface for playing a game and inspecting the
state produced by the Python engine. It is intentionally simple: the board is
the main surface, while scores, turn information, capture details, and raw
arrays are available beside it.

The browser does not implement game rules.

```text
GUI
    presentation and user interaction

Python game engine
    source of truth for rules and state
```

## Architecture

```text
Browser
    ↓
HTML / CSS / JavaScript
    ↓
FastAPI (`GUI/server.py`)
    ↓
`game/enclosure.py` (public game API)
    ↓
board.py / capture.py / engine.py / groups.py / rendering.py
```

The frontend sends coordinates to FastAPI. `GUI/server.py` accesses the game
only through the public API exported by `game/enclosure.py`. Through that API,
the server asks the existing `DotsGame` object whether the move is legal and
then calls its `place_dot()` method. Capture detection, territory changes,
UnionFind updates, and score changes all remain in the engine under `game/`.

The GUI server does not import `game.board`, `game.capture`, `game.engine`,
`game.groups`, or `game.rendering`. Those modules are implementation details
behind `game/enclosure.py`.

FastAPI also owns the small amount of session information that is not part of
the current engine class: whose turn it is, the move number, the last move,
and the latest status message. Turn switching therefore happens in Python,
not in JavaScript.

The web server keeps one game in memory. Restarting the server clears it, and
all browser tabs connected to that server use the same local game.

## Files

```text
GUI/
├── index.html   page structure and accessible controls
├── styles.css   visual tokens, layout, and responsive styles
├── game.js      API calls, Canvas rendering, and browser interaction
└── server.py    FastAPI adapter around the public game API
```

- `GUI/index.html` contains the header, Canvas, game-state panel, collapsible
  debug sections, and status area.
- `GUI/styles.css` defines the light neutral design, player colors, territory
  colors, panels, and desktop/mobile layouts. Core colors and spacing tokens
  are CSS variables near the top of the file.
- `GUI/game.js` fetches engine state, translates pointer positions into board
  coordinates, draws the Canvas, and updates the information panels. It does
  not validate captures, change scores, or decide which moves are legal.
- `GUI/server.py` serves the three frontend files and exposes the HTTP API. It
  imports all game symbols exclusively from `game.enclosure`.

The related game entry point is:

```text
game/enclosure.py    public API used by external callers such as the GUI
```

`game/enclosure.py` intentionally exports `DotsGame`, player constants,
`opponent_of`, capture diagnostics, candidate-region lookup, and the public
`could_have_closed_loop` diagnostic. It decides which internal module provides
each symbol, so the GUI does not depend on the engine's internal file layout.

The detailed game algorithm remains documented separately in
`docs/README_GAME.md`.

## Move flow

```text
click board
    ↓
pixel position -> (row, col)
    ↓
POST /api/move
    ↓
game/enclosure.py public API
    ↓
DotsGame move logic
    ↓
capture detection / state update
    ↓
updated state returned as JSON
    ↓
redraw board and information panel
```

### Code example

The following simplified example shows how the layers connect. The browser
converts the pointer position into a coordinate and sends only that coordinate
to Python:

```javascript
canvas.addEventListener("click", (event) => {
  const cell = eventToCell(event); // pixel position -> [row, col]
  if (!cell) return;

  submitMove(cell[0], cell[1]);
});

async function submitMove(row, col) {
  const response = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });

  const payload = await response.json();
  const state = response.ok ? payload : payload.state;
  setGameState(state); // update the panel and redraw the Canvas
}
```

FastAPI receives the coordinate. The GUI server obtains `DotsGame` from the
public `game.enclosure` API and delegates validation and state changes to it:

```python
from game.enclosure import DotsGame, PLAYER_1, opponent_of


class GameSession:
    def __init__(self, rows, cols):
        self.game = DotsGame(rows, cols)
        self.current_player = PLAYER_1

    def apply_move(self, row, col):
        if not self.game.is_legal_move(row, col):
            return False, self._state_unlocked()

        moving_player = self.current_player
        captured = self.game.place_dot(row, col, moving_player)
        self.current_player = opponent_of(moving_player)

        # _state_unlocked() serializes board, territory, scores, and metadata.
        return True, self._state_unlocked()


@app.post("/api/move")
def post_move(move: MoveRequest):
    succeeded, state = session.apply_move(move.row, move.col)
    if not succeeded:
        return JSONResponse(status_code=409, content={
            "error": "Illegal move",
            "state": state,
        })
    return state
```

Finally, JavaScript uses the returned arrays and metadata to update the visible
interface:

```javascript
function setGameState(state) {
  view.game = state;
  updateInformationPanel();
  resizeAndDrawBoard();
}
```

The snippets omit diagnostic and status-message details for readability. The
important boundary remains the same: JavaScript sends user input, while the
Python game engine validates the move and performs all board, capture,
territory, connectivity, and scoring updates.

Coordinates are zero-based. `(0, 0)` is the top-left intersection. The first
number is the row and the second number is the column.

## API

### `GET /api/state`

Returns the current game and diagnostic state.

Example response, shortened for readability:

```json
{
  "rows": 10,
  "cols": 10,
  "board": [[0, 0, 0], [0, 1, 0]],
  "territory": [[0, 0, 0], [0, 0, 0]],
  "score": {
    "player_1": 0,
    "player_2": 0
  },
  "current_player": -1,
  "last_move": [1, 1],
  "legal_moves": [[0, 0], [0, 1]],
  "legal_move_count": 99,
  "move_number": 1,
  "last_captured_dots": [],
  "capture_happened": false,
  "debug": {
    "could_have_closed_loop": false,
    "candidate_region_count": 4,
    "candidate_regions": [[0, 1], [2, 1], [1, 0], [1, 2]],
    "enclosed_region_count": 0,
    "detected_enclosed_regions": [],
    "opponent_cells_found": []
  },
  "message": "Player 1 placed a dot at (1, 1). Player 2 to move."
}
```

The real `board`, `territory`, and `legal_moves` arrays cover the complete
board; they are shortened only in this example.

### `POST /api/move`

Requests a move.

```json
{
  "row": 2,
  "col": 4
}
```

The backend obtains `DotsGame` from `game.enclosure`, calls
`DotsGame.is_legal_move()`, and, for a legal coordinate,
`DotsGame.place_dot()`. It then switches the active player and returns the same
state shape as `GET /api/state`.

An occupied, blocked, or out-of-bounds coordinate returns HTTP `409`:

```json
{
  "error": "Illegal move",
  "state": {
    "message": "Illegal move at (2, 4)."
  }
}
```

The included `state` is the unchanged authoritative state (shortened above),
so the browser can stay synchronized after a rejected request.

### `POST /api/reset`

Creates a fresh board, sets Player 1 as the current player, clears scores and
move history, and returns the initial state. No request body is required.

The **New game** and **Reset** buttons both use this endpoint because the
engine does not have a separate concept for those actions.

## Board clicks and legal moves

The Canvas is resized to its displayed CSS size and scaled for the screen's
pixel density. JavaScript calculates a single grid step from the backend's
`rows` and `cols`; the frontend never hardcodes the board dimensions.

For a pointer event, JavaScript:

1. measures the pointer relative to the Canvas;
2. subtracts the grid origin;
3. divides by the grid step;
4. rounds to the nearest row and column;
5. checks whether that coordinate appears in the backend-provided
   `legal_moves` list; and
6. sends the coordinate to `POST /api/move`.

The legal-move list is used only to control hover and click presentation. The
backend validates every submitted move again. Occupied and captured
intersections do not receive the clickable hover state.

The Canvas can also be used with a keyboard. Focus it, move the selection with
the arrow keys, and press Enter or Space to submit the selected legal move.

## Canvas rendering

`game.js` redraws the complete visible board whenever fresh state arrives or
the Canvas changes size.

- Grid lines and row/column labels are drawn first.
- `board[row][col] == 1` is drawn with the muted Player 1 color.
- `board[row][col] == -1` is drawn with the muted Player 2 color.
- Nonzero `territory` values are drawn as soft colored tiles behind their grid
  intersections. Player 1 and Player 2 territory use different translucent
  colors.
- A dot whose `territory` value is nonzero is inactive/captured and is drawn
  with reduced opacity. It remains visible because the engine keeps captured
  dots in `board`.
- `last_move` receives an outer ring.
- The nearest legal intersection receives a player-colored hover ring.

Only display colors live in the frontend. The numeric values `0`, `1`, and
`-1` returned by Python remain the actual game representation.

## Information and scores

The side panel reads directly from each API response. It shows:

- current player;
- Player 1 and Player 2 scores;
- move number;
- last move;
- number of legal moves;
- dots captured by the last move; and
- whether the last move caused a capture.

The status bar uses the backend's plain-language `message`, for example
`Player 1 placed a dot at (3, 4)` or `Player 2 captured 2 dots`.

## Debug details

Open **Debug details**, or use the **Show debug** button, to inspect optional
data for the last move:

- the result of the public `could_have_closed_loop()` diagnostic;
- local candidate region seeds from `find_candidate_regions()`;
- captured/enclosed regions returned by the existing capture detector; and
- opponent dots found in those regions.

The API produces this information by calling helpers exported by
`game/enclosure.py` with the engine's pre-move connectivity and territory
snapshot. `could_have_closed_loop()` is the public API name for the engine's
historical `_could_have_closed_loop()` implementation. This is a read-only
diagnostic pass. It does not duplicate capture rules or change the game, and
the GUI never imports the internal capture module directly.

Open **Raw board state** to see the complete `board` and `territory` matrices
with row and column headers. These collapsible sections are closed by default
so they do not compete with normal play.

## Install and run

Python 3.10 or newer is recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn GUI.server:app --reload
```

If the existing `.venv` is already active, only the last two commands are
needed.

Open:

```text
http://127.0.0.1:8000
```

FastAPI serves the GUI directly, so there is no separate frontend server or
JavaScript build step. Interactive API documentation is also available at
`http://127.0.0.1:8000/docs` while the server is running.

The GUI adds two web dependencies to `requirements.txt`:

- `fastapi` for the HTTP API and file responses;
- `uvicorn` for the local ASGI server.

The frontend uses only browser-native HTML, CSS, JavaScript, Fetch, and Canvas.
