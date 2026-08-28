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
| `legal_masks` | `[T, rows, cols]` | `1` for legal actions and `0` otherwise |
| `selected_actions` | `[T, 2]` | The real `(row, col)` moves played by MCTS |
| `completed_rollouts` | `[T]` | Successful rollouts for each search |
| `search_elapsed_seconds` | `[T]` | Actual search duration for each move |
| `final_result` | scalar | Player 1 result: `1` win, `0` draw, `-1` loss |

The file also includes the schema version, game identifier, creation time,
board shape, perspective labels, configured search-budget values, and rollout
batch size.

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

## Loading

`load_self_play_game()` disables pickle and validates the schema version:

```python
from training import load_self_play_game

game = load_self_play_game("training_data/10x10/game_....npz")
print(game["boards"].shape)
print(game["selected_actions"])
```
