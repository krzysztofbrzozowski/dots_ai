# Game data collection

This package collects complete MCTS-versus-MCTS games without augmentation or
neural-network evaluation. It reuses the game engine, MCTS search/backup, NPZ
writer and analysis loader. Collection-specific randomization lives in this
folder; the existing `_game_arena` and `mcts` implementations are unchanged.

From the project root, with the virtual environment active:

```bash
python -m _game_arena_data_collection --minutes 10
python -m _game_arena_data_collection --minutes 10 --workers 8
python -m _game_arena_data_collection --minutes 10 --workers 1
python -m _game_arena_data_collection --minutes 10 --games 8
python -m _game_arena_data_collection --audit-only
python -m unittest _game_arena_data_collection.test_collection
```

The duration and game-count limits apply to each invocation. The collector
finishes active games when the time limit is reached, so a run may exceed
the requested duration by the longest active game's remaining time. With
multiple workers, Ctrl-C stops new games and finishes/saves the active games.
With one worker, Ctrl-C discards its unfinished game. Completed files remain
usable and the next run resumes.

## CPU parallelism

`--workers N` runs up to N independent games in separate processes. The CLI
defaults to `min(8, max(1, CPU count - 2))` (8 on this 14-core M4 Pro).
The Python `collect()` API retains `workers=1` by default. Each game uses
one serial MCTS search; no nested worker pools are created. This stage runs
game simulations on the CPU and does not use a neural network or GPU.

The coordinator alone updates the manifest; workers atomically save separate
NPZ files. Games can finish out of order. If a process crashes, resuming fills
missing game indices without overwriting completed games. The worker count
is an execution setting, so it can change when resuming an existing collection.
`last_session.json` records the requested worker count and total CPU time
spent playing games; the manifest records runtime statistics for new games.

Use `--workers 1` for sequential collection or increase the count to use more
CPU cores. More workers do not automatically make individual moves stronger:
the per-move wall-clock budgets stay the same. Oversubscribing the CPU can
reduce rollouts per move, so compare the audit's rollout counts as well as
games per minute. A small fixed batch may leave workers idle while its last,
more expensive games finish; a longer timed run keeps starting new games.

## Configuration and matchups

Edit `COLLECTION_CONFIG` in `config.py`. Default board size is 10×10. Both
players run serial MCTS within their game's process, so comparisons use the
same execution mode. Each game samples a base time between 0.10 and 0.20 seconds per move.
An advantaged player receives 2–4 times this budget, capped at 0.80 seconds.
These are short pilot budgets; stronger labels may need longer searches.

Every shuffled block of eight games contains:

| Matchup | Games | Starting players |
| --- | ---: | --- |
| Equal budgets | 4 | twice +1, twice -1 |
| Higher budget for +1 | 2 | once +1, once -1 |
| Higher budget for -1 | 2 | once +1, once -1 |

Budget and starter are fixed for a game. A partly completed block does not
necessarily have exact proportions. A higher budget does not guarantee a win,
and the collector never changes labels or discards games based on their result.

Each game chooses 0, 2, 4 or 6 random legal opening moves. MCTS then plays to
completion. `CollectionNode` shuffles the untried actions at every tree node
and seeds random rollouts. This removes the original fixed expansion order's
systematic preference for a particular end of the coordinate list.

Configuration selection and opening moves are reproducible from the collection
seed and game index. Timed MCTS games are not guaranteed to be bit-for-bit
reproducible, because machine load affects the completed rollout count.

## Output and resuming

Files are saved under `training_data/_game_arena_data_collection/10x10/`:

- One schema-v1 `.npz` per completed game, compatible with `load_self_play_game`,
  the existing analysis GUI and `ml.data_loader`.
- `collection.json`: generation settings, checked before resuming.
- `manifest.json`: per-game budget, starter, seed, actual opening, result,
  final scores and measured search statistics.
- `last_session.json`: duration, game count, concurrency and game CPU time
  from the latest invocation.
- `audit.json` and `report.md`: replay validation and collection statistics.

Use `--output-directory PATH` for another collection, and `--seed INTEGER`
to override the configuration seed. Reusing a directory with different
generation settings is rejected. An OS file lock prevents simultaneous writers.
The manifest is recoverable from completed NPZ files and the persisted settings
if a process stops between saving a game and updating its metadata.

The legacy NPZ `requested_simulation_seconds` scalar describes Player +1's
budget. Read the manifest for **both** players' budgets. Random opening moves
are included in the trajectory with zero visits, zero q values and zero search
time. They are valid value-training examples, but must be excluded or handled
separately when deriving a future policy target from visit counts.

## Verification

Every completed game is replayed from the empty board. The audit verifies each
stored board, territory, score, next player, legal mask, chosen action and final
result; it also checks visit/rollout counts and compatibility with the existing
analysis loader. No incomplete game receives a training label.

The final report includes outcomes by matchup, starting-player counts, opening
lengths, early move locations, duplicate trajectories/positions and rollout
counts. These checks establish format integrity and describe diversity; they
do not prove that the resulting model will play well. Split training and
validation by complete games. Any future augmentation belongs in the training
input pipeline after that split, not in this collector.
