# MCTS game display

The browser interface is a read-only observer for the automated game owned by
`main_mcts.py`. It renders published states but does not make or apply moves.

## Architecture

```text
main_mcts.py
    owns DotsGame
    runs the MCTS game loop
    keeps selection and backpropagation in one authoritative tree
    dispatches immutable leaf states to rollout worker processes
    selects and applies each move
    publishes a serialized snapshot
            ↓
GUI/server.py
    stores the latest snapshot under a lock
    exposes GET /api/state
    serves the static browser files
            ↓
GUI/game.js
    polls for a newer snapshot version
    draws the board and information panel
```

The server and authoritative game loop run in the same Python process.
`main_mcts.py` starts the MCTS loop in a worker thread and runs Uvicorn in the
main thread. This lets the two components share the in-memory snapshot store
without making the server an owner of live game state.

CPU-bound rollouts run in a persistent process pool. Search-tree selection,
expansion, temporary virtual-loss reservations, and backpropagation stay in the
game process. Workers receive only independent `DotsGame` states and return a
game result, so the mutable MCTS tree is never shared between processes. The
pool is warmed before the first timed search and shut down when the match ends.
The complete search algorithm is documented in
[README_MCTS.md](README_MCTS.md).

`GUI/server.py` never retains a `DotsGame` reference. `publish_state()` converts
the supplied state into JSON-compatible values immediately, and the snapshot
store makes defensive copies on both publication and reading.

Human-readable player, result, and move messages live in
`GUI/presentation.py`, keeping presentation formatting out of the MCTS wrapper
and out of the HTTP transport.

## Responsibilities

### `main_mcts.py`

- creates the initial `DotsGame`;
- owns `board_state` throughout the match;
- constructs and executes each MCTS search;
- owns the persistent rollout process pool;
- replaces `board_state` with the selected child state;
- publishes the initial state and every selected move;
- controls the delay between visible moves;
- starts and keeps the HTTP server alive after the match finishes.

### `GUI/server.py`

- serializes published display state;
- assigns a monotonically increasing snapshot version;
- protects publication and reading with `RLock`;
- serves `index.html`, `styles.css`, and `game.js`;
- exposes the latest state through `GET /api/state`.

It does not expose move, reset, or MCTS-step endpoints.

### `GUI/game.js`

- polls `GET /api/state` every 300 milliseconds;
- prevents overlapping requests;
- redraws only when the snapshot version changes;
- displays the board, territory, scores, last move, capture information, and
  final result.

The canvas is not interactive. The frontend contains presentation mappings
such as player names and colors, but no game rules.

## Running the application

From the repository root:

```bash
python main_mcts.py
```

Then open:

```text
http://127.0.0.1:8000
```

Application settings are defined at the top of `main_mcts.py`:

```python
DEFAULT_ROWS = 10
DEFAULT_COLS = 10
DEFAULT_SIMULATIONS = 12
DEFAULT_MOVE_DELAY = 0.4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
SIMULATION_SECONDS = 30
DEFAULT_MCTS_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
```

`SIMULATION_SECONDS` is the wall-clock search budget for each move. The worker
count leaves one logical CPU available and is capped at eight. Passing
`simulation_seconds=None` to `run_mcts_game()` switches to the fixed
`simulations_number` budget instead.

The MCTS worker stops after the game reaches a result. Uvicorn continues
serving the final snapshot until the process is stopped with `Ctrl+C`.

## State endpoint

### `GET /api/state`

The endpoint returns the most recently published display snapshot. Before the
first publication it returns HTTP 503.

Example final response:

```json
{
  "version": 5,
  "rows": 2,
  "cols": 2,
  "board": [[-1, -1], [1, 1]],
  "territory": [[0, 0], [0, 0]],
  "score": {
    "player_1": 0,
    "player_2": 0
  },
  "current_player": 1,
  "last_move": [0, 0],
  "legal_moves": [],
  "legal_move_count": 0,
  "move_number": 4,
  "last_captured_dots": [],
  "capture_happened": false,
  "capture_player": null,
  "game_over": true,
  "winner": 0,
  "message": "Player 2 selected (0, 0). Game over: Draw."
}
```

Winner values use the same numeric representation as the game engine:

| Value | Meaning |
| ---: | --- |
| `1` | Player 1 wins |
| `-1` | Player 2 wins |
| `0` | Draw |
| `null` | Game still in progress |
