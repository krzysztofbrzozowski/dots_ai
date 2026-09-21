# Shared MCTS workspace

The project has one browser interface for both a running self-play game and a
saved NPZ trajectory. The HTML, CSS, JavaScript, board renderer, and artwork
live under `GUI/analysis` and `GUI/shared`; there is no separate live-game
frontend.

## Entry points

### Live self-play

Run:

```bash
python main_mcts.py
```

Open `http://127.0.0.1:8000`.

`main_mcts.py` owns the game and MCTS tree. After every completed search it
publishes the pre-move board, selected action, Q values, visit counts, legal
mask, timing, and resulting current position to `GUI/server.py`. The browser
shows the current position while the first search is running, grows the
decision timeline after every move, and switches to the terminal board when
the game finishes.

Live mode supports the saved-search overlays: Q / N, raw Q, visits, policy,
and no overlay. File import, neural-head requests, and forced replay are hidden
because those operations belong to the saved-analysis backend.

### Saved trajectory analysis

Run:

```bash
.venv/bin/python main_analysis.py
```

Open `http://127.0.0.1:8001` and choose an NPZ file. This mode also exposes
the neural-head overlays and disposable forced replay.

## Shared frontend

Both Python servers return the same assets:

```text
GUI/analysis/index.html
GUI/analysis/analysis.css
GUI/analysis/analysis.js
GUI/analysis/network.svg
GUI/shared/board_renderer.js
```

At startup, the frontend reads `GET /api/runtime`:

- `{"mode": "live"}` selects the continuously refreshed MCTS source;
- `{"mode": "analysis"}` selects NPZ import and analysis tools.

Rendering, timeline navigation, screenshots, diagnostics, responsive layout,
and theme styling remain shared. Only the data-source adapter and unavailable
controls differ by mode.

## Live data flow

```text
main_mcts.py
    runs a search and selects an action
        ↓
analysis.service.mcts_decision_frame
    creates the canonical GUI frame
        ↓
GUI.server.LiveGameStore
    appends the immutable frame and current position
        ↓
GET /api/live
GET /api/live/frames/{index}
        ↓
GUI/analysis/analysis.js
    updates the shared timeline and renderer
```

The in-memory store uses a lock and defensive copies. Browser clients cannot
mutate the authoritative `DotsGame`, MCTS tree, or stored timeline.

## Live endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/runtime` | Select live mode |
| `GET /api/live` | Read revision, summary, current position, and timeline descriptors |
| `GET /api/live/frames/{index}` | Read one completed MCTS decision frame |
| `GET /api/diagnostics?after={event_id}` | Read newer diagnostic events |

No endpoint applies moves or changes the running game.

## Application settings

Live settings remain at the top of `main_mcts.py`: board size, search budget,
worker count, move delay, host, port, and training-data directory. A completed
game is still saved as an NPZ trajectory for later inspection and training.
