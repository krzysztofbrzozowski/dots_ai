# Saved-game analysis GUI

`main_analysis.py` runs a local, read-only application for inspecting the NPZ
files written after MCTS self-play games. It is independent from
`main_mcts.py`: neither application imports the other, and both may run at the
same time.

## Running the analyzer

From the repository root:

```bash
.venv/bin/python main_analysis.py
```

Then open:

```text
http://127.0.0.1:8001
```

Use **Open NPZ game** or drag an `.npz` file onto the page. The browser sends
the selected bytes only to this local server. The analyzer does not modify or
persist the file.

After a frame is loaded, the camera button in the top bar downloads the current
board as a 1600 × 1600 PNG. The export includes the active overlay, the saved
action, and the currently selected cell. The board is rendered directly into
the square with a comfortable inset similar to the analysis view, without
stretching the grid.

## What the interface shows

The analysis workspace pairs an off-white page and header with dark plum board
and timeline panels, pink Player 1 dots, and cyan Player 2 dots. A warm gold ring
marks the selected action. Search values use a
pink-to-cyan scale for negative-to-positive results and a cyan intensity scale
for visits and policy. The legend updates with the selected overlay.

The timeline sits to the left of the larger board on desktop. They stack with
the board first on narrow screens.
Keyboard focus is visible, and timeline transitions respect reduced-motion
preferences. Theme colors are centralized in `GUI/analysis/analysis.css`.

The center board renders the state before the selected move. Five board modes
are available:

- **Head value** — click a legal move to evaluate the position after that move
  with `10x10_083769_3conv_d4_normalized.keras`;
- **Q / N** — the mean rollout result from the current player's perspective;
- **Raw Q** — the saved child win/loss balance;
- **Visits** — the raw child visit count;
- **Policy** — the visit share among root children.

Head value results appear below the board as loss, draw, and win probabilities,
plus `P(win) - P(loss)`. All four values use the perspective of the player making
the candidate move. The model is loaded on the first prediction and then reused.
Its current checkpoint accepts 10 × 10 positions.

The bright outer ring identifies the action chosen by MCTS. A small empty ring
identifies a legal action with no completed visit. This distinction matters
because schema v1 stores zero for both an unvisited action and a genuinely
neutral raw Q value.

Click any board intersection to inspect its dot, territory, legality, raw Q,
mean value, visits, and visit-policy share. In **Head value** mode, clicking an
empty legal intersection also runs the prediction.

The left timeline is a vertically snapping decision wheel. It supports:

- mouse, trackpad, and touch scrolling;
- clicking a decision card;
- the Up, Down, Home, and End keys;
- automatic playback through the saved trajectory.

The compact diagnostics message panel at the bottom records file validation,
session metadata, frame navigation, model predictions, cache hits, timing, and
errors. **Copy** exports the visible log and **Clear** resets the browser view.

Backend code can publish its own messages to this window with `PRINT_T`:

```python
from analysis import PRINT_T

PRINT_T("Preparing a custom calculation")
PRINT_T("Calculation complete", level="success", source="CUSTOM")
```

Supported levels are `info`, `success`, `warning`, and `error`. The optional
source label identifies the subsystem. Messages are also written to the Python
process output, and the in-memory server buffer retains the 500 most recent
events.

## Data flow

```text
Browser file picker
        ↓
POST /api/analyses
        ↓
analysis/loader.py
    checks archive limits and loads with allow_pickle=False
        ↓
analysis/schema_v1.py
    validates fields, shapes, values, and selected actions
        ↓
AnalysisGame
    canonical, read-only NumPy representation
        ↓
analysis/service.py
    summary and frame-specific derived statistics
        ↓
GUI/analysis
    timeline, board renderer, and cell inspector
```

For a Head value request, the server also reconstructs the selected pre-move
frame as a playable game state, applies the clicked move, and sends the resulting
position through `ml/predictor.py`. The response uses the moving player's
perspective even though the model evaluates the opponent's next turn.

Schema-specific NPZ names stop at the adapter. The service and browser consume
the canonical `AnalysisGame`, so a future schema can add another adapter
without teaching the GUI a second storage layout.

## Schema-v1 timeline semantics

Each schema-v1 index is a **pre-move decision state**. Frame zero is the state
before move one, and `selected_actions[0]` is the action selected from that
state. The file contains `T` decision states for `T` searches.

Schema v1 does not save the board after the final action. The analyzer
therefore shows the final result in the game header but does not fabricate a
terminal board with the current game engine.

## HTTP endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/analyses` | Validate an uploaded NPZ and create a local session |
| `GET /api/analyses/{id}` | Read game metadata and timeline descriptors |
| `GET /api/analyses/{id}/frames/{index}` | Read one canonical decision frame |
| `GET /api/analyses/{id}/frames/{index}/head-value?row={row}&col={col}` | Predict a legal candidate move with the configured value head |
| `GET /api/diagnostics?after={event_id}` | Read newer `PRINT_T` diagnostic events |
| `DELETE /api/analyses/{id}` | Release the in-memory session |
| `GET /api/health` | Check whether the analyzer is ready |

The request body for `POST /api/analyses` is the NPZ byte stream. The original
file name is URL-encoded in the `X-File-Name` header. Uploads are limited to
32 MB compressed and 256 MB uncompressed. At most four recent games are kept
in memory.
