# Arena tools

The arena tools are grouped here. The Elo arena is the current interface;
the older match runner and rollout collector are retained under `obsolete/`.

```text
_arena/
├── README.md
├── elo_game_arena/                    # Current 25x25 Elo arena
│   ├── config.py                      # Search presets and file locations
│   ├── server.py                      # FastAPI application
│   ├── ratings.json                   # Persistent ratings and match history
│   └── static/                        # Arena page assets
└── obsolete/
    ├── _game_arena/                   # Legacy timed MCTS match runner
    └── _game_arena_data_collection/   # Legacy exact-rollout collector
```

| Tool | Purpose | Documentation |
| --- | --- | --- |
| Elo arena | Rated classic MCTS, neural MCTS, and named human matches | [Elo arena](elo_game_arena/README.md) |
| Legacy game arena | Independent time budgets and execution modes for two MCTS players | [Legacy match runner](obsolete/_game_arena/README.md) |
| Legacy data collector | Reproducible self-play with exact rollout budgets | [Legacy rollout collector](obsolete/_game_arena_data_collection/README.md) |

Run commands from the repository root, with the project virtual environment
active. Python module names now follow this directory layout:
`_arena.elo_game_arena`, `_arena.obsolete._game_arena`, and
`_arena.obsolete._game_arena_data_collection`. The former top-level packages
are no longer present. Existing datasets under the repository's
`training_data/` directory are not moved by this source reorganization.

## Paths and commands

The Elo arena starts with `python -m _arena.elo_game_arena`. Its model and
shared GUI assets are loaded from the repository's `ml/models/` and `GUI/`
directories. Ratings remain in `elo_game_arena/ratings.json`.

All three tools write to the repository's existing `training_data/` tree,
independently of the working directory:

| Tool | Default output relative to the repository root |
| --- | --- |
| Elo arena | `training_data/_elo_game_arena/25x25/` |
| Legacy match runner | `training_data/_game_arena/<rows>x<cols>/` |
| Legacy rollout collector | `training_data/_game_arena_data_collection_rollouts/10x10/` |

The dataset directory names are retained for compatibility with existing
records. The collector accepts `--output-directory` to select another dataset
location. The legacy runner accepts `training_data_directory` in `run_arena()`.

## Verification

From the repository root:

```bash
python -m unittest tests.test_elo_arena \
  tests.test_collection
python -m tests.test_game_arena
```

The Elo tests also check that the page, shared board renderer, stylesheets,
and read-only API endpoints are served correctly.

See the [project documentation](../docs/README.md) for the engine, search,
training data schema, and saved-game analysis.
