# Elo arena 25x25

The current arena runs rated 25x25 matches between classic MCTS, neural MCTS,
and named human players. It lives in `_arena/elo_game_arena/`; the older
tools are archived under `obsolete/` in the [arena overview](../README.md).

## Starting the arena

Run from the repository root with the project virtual environment active:

```bash
python -m _arena.elo_game_arena
```

The host and port are configured in `config.py` (defaults: `0.0.0.0:8003`).
To override them for one invocation, launch the application directly:

```bash
python -m uvicorn _arena.elo_game_arena.server:app \
  --host 127.0.0.1 --port 8003 --no-access-log
```

Open [the local arena](http://127.0.0.1:8003). Choose two distinct competitors
and start the match. Enter a name for each human player, then click a legal
intersection when that human is to move. One match can be active at a time.

Use **End match** to adjudicate an unfinished game from the current score.
The higher score wins, an equal score is a draw, and Elo is updated immediately.

## Search settings and player identity

Classic and neural search parameters are configured independently in
[`config.py`](config.py). The current defaults are:

```python
CLASSIC_MCTS_SIMULATION_SECONDS = 90.0
CLASSIC_MCTS_WORKERS = 14

NEURAL_MCTS_SIMULATION_SECONDS = 30.0
NEURAL_MCTS_C_PUCT = 1.5
```

The neural preset uses `25x25_088622_new_data_dual_head_v1.keras` from
`ml/models/` at the repository root. A compatible dual-head checkpoint must
exist at `NEURAL_MCTS_MODEL_PATH`.

Preset IDs are derived from search settings: the classic ID includes time
and worker count, and the neural ID includes time, PUCT, and a fixed model
version suffix. Changing those numeric settings creates a separate Elo
identity automatically. When replacing the neural checkpoint, also change
the model version suffix in its preset ID and label so the new model does
not inherit the previous model's rating.

Human identities ignore letter case and normalize whitespace. Reusing the
same normalized name reuses that player's rating.

## Ratings and recordings

Ratings start at 1200 and use K=32. Completed and adjudicated games update
both competitors; draws score 0.5. The persistent table and match history
remain in [`ratings.json`](ratings.json) beside the arena code.

Every completed or adjudicated match with at least one played move is saved
as a schema-v1 NPZ trajectory under the repository root:

```text
training_data/_elo_game_arena/25x25/
```

The `_elo_game_arena` dataset name is retained independently of the new
Python package name. Filenames include `elo-completed` or `elo-stopped`,
the match ID, and both competitor identities. Ending a match before its first
move still updates ratings but creates no NPZ file.

Human moves contain zero search visits and rollouts. Stopped games use the
adjudicated score as their final result. Account for these differences when
selecting records for training; see the
[training data documentation](../../docs/README_TRAINING_DATA.md#arena-records).
Passing `training_data_directory=None` to `EloArenaSession` disables recording.

## Focused tests

From the repository root:

```bash
python -m unittest _arena.elo_game_arena.test_elo_arena
```

These tests cover rating persistence, human identities, trajectory recording,
and serving the arena page, shared GUI assets, and read-only API endpoints.
