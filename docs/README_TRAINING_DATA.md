# MCTS self-play training data

`main_mcts.py` records the actual trajectory played by MCTS and writes one
compressed NPZ file after the game finishes. Explored tree positions that were
not played are not included.

## Output layout

Generated files are grouped by board dimensions and ignored by Git:

```text
training_data/
├── 8x8/
│   └── game_<timestamp>_<id>.npz
└── 10x10/
    └── game_<timestamp>_<id>.npz
```

The writer first creates a temporary file in the destination directory and
atomically replaces it with the final `.npz`. An interrupted or failed game
therefore does not publish a partial training file.

Passing `training_data_directory=None` to `run_mcts_game()` or
`run_parallel_mcts_game()` disables recording. The application entry point
enables it and writes below `training_data/`.

## Schema version 1

For a game with `T` played moves on a `rows × cols` board, the file contains:

| Name | Shape | Meaning |
| --- | --- | --- |
| `boards` | `[T, rows, cols]` | Absolute dot ownership: Player 1 is `1`, Player 2 is `-1` |
| `territories` | `[T, rows, cols]` | Absolute captured-territory ownership |
| `next_players` | `[T]` | Absolute player to move in each pre-move state |
| `scores` | `[T, 2]` | Player 1 and Player 2 scores |
| `q_values` | `[T, rows, cols]` | Raw child `q`, aligned by `child.action` and evaluated for the player to move |
| `visit_counts` | `[T, rows, cols]` | Raw child visit count `n`, aligned by `child.action` |
| `policy_priors` | `[T, rows, cols]` | Neural root priors `P` used by PUCT, or zeros when unavailable |
| `has_policy_priors` | `[T]` | `1` when the corresponding prior map is available, otherwise `0` |
| `legal_masks` | `[T, rows, cols]` | `1` for legal actions and `0` otherwise |
| `selected_actions` | `[T, 2]` | The real `(row, col)` moves played by MCTS |
| `completed_rollouts` | `[T]` | Successful rollouts for each search |
| `search_elapsed_seconds` | `[T]` | Actual search duration for each move |
| `final_result` | scalar | Player 1 result: `1` win, `0` draw, `-1` loss |

The file also includes the schema version, game identifier, creation time,
board shape, perspective labels, configured search-budget values, and rollout
batch size.

The two policy-prior arrays are optional schema-v1 extensions, so existing NPZ
files remain readable. Neural MCTS writes a normalized map that sums to one over
legal actions; classic rollout MCTS writes an unavailable flag and a zero map.

The stored board remains absolute. A training loader chooses the model
perspective later. For a player-to-move value target:

```python
z_current_player = final_result * next_players
```

Because current `q_values` already use the player-to-move perspective, an
absolute Player 1 value can be derived when required:

```python
q_player_1 = q_values * next_players[:, None, None]
```

A visit-based policy target can be derived without discarding raw search data:

```python
policy = visit_counts / visit_counts.sum(axis=(1, 2), keepdims=True)
```

Model batches should group games by board dimensions. This allows one fully
convolutional model to support several `n × n` variants without padding every
sample to the largest configured board.

## Arena records

Arena tools now live under [`_arena/`](../_arena/README.md). The current
interface is the [Elo arena](../_arena/elo_game_arena/README.md); the timed
match runner and exact-rollout collector are archived under `_arena/obsolete/`.
Existing datasets are not relocated with the Python packages.

### Elo arena

Completed and adjudicated Elo matches with at least one played move are saved
using the same schema-v1 fields as self-play, beneath the repository root:

```text
training_data/_elo_game_arena/25x25/
```

The dataset directory keeps its `_elo_game_arena` name even though the package
is now `_arena.elo_game_arena`.

Filenames identify the competitors and whether the match finished naturally
(`elo-completed`) or was adjudicated with **End match** (`elo-stopped`).
Human turns have zero visit counts and completed rollouts, so the visit-based
policy normalization above needs a zero-total guard or must exclude those
turns. Adjudicated results are based on the score when the match was ended;
keep them distinguishable from naturally completed games when preparing
training targets.

### Legacy timed arena

The archived runner is launched with `python -m _arena.obsolete._game_arena`.
Its default output, relative to the repository root, is:

```text
training_data/_game_arena/<rows>x<cols>/
```

Existing records remain in this directory. The
[legacy runner documentation](../_arena/obsolete/_game_arena/README.md) explains
how to select a different output location.

The size directory identifies the board. The filename encodes the first player
and both players' MCTS mode, seconds, and worker count after its timestamp and
unique ID. The contents use the same unchanged schema-v1 fields listed above;
arena-specific parameters are not added to the NPZ schema.

### Legacy exact-rollout collector

The archived collector is launched with
`python -m _arena.obsolete._game_arena_data_collection`.
Its [documented commands](../_arena/obsolete/_game_arena_data_collection/README.md)
pass `--output-directory training_data/_game_arena_data_collection_rollouts`
to select the repository's existing collection explicitly. This is also the
default location when the option is omitted. Games are grouped by board size,
with `10x10/` as the current default.

## Loading

`load_self_play_game()` disables pickle and validates the schema version:

```python
from training import load_self_play_game

game = load_self_play_game("training_data/10x10/game_....npz")
print(game["boards"].shape)
print(game["selected_actions"])
```
