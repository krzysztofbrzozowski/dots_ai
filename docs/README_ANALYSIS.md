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

## What the interface shows

The center board renders the state before the selected move. Four search
overlays are available:

- **Q / N** — the mean rollout result from the current player's perspective;
- **Raw Q** — the saved child win/loss balance;
- **Visits** — the raw child visit count;
- **Policy** — the visit share among root children.

The bright outer ring identifies the action chosen by MCTS. A small empty ring
identifies a legal action with no completed visit. This distinction matters
because schema v1 stores zero for both an unvisited action and a genuinely
neutral raw Q value.

Click any board intersection to inspect its dot, territory, legality, raw Q,
mean value, visits, and visit-policy share.

The right timeline is a vertically snapping decision wheel. It supports:

- mouse, trackpad, and touch scrolling;
- clicking a decision card;
- the Up, Down, Home, and End keys;
- automatic playback through the saved trajectory.

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
| `DELETE /api/analyses/{id}` | Release the in-memory session |
| `GET /api/health` | Check whether the analyzer is ready |

The request body for `POST /api/analyses` is the NPZ byte stream. The original
file name is URL-encoded in the `X-File-Name` header. Uploads are limited to
32 MB compressed and 256 MB uncompressed. At most four recent games are kept
in memory.

