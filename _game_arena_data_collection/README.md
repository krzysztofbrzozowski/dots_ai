# Rollout-based game data collection

This package collects complete MCTS-versus-MCTS games for policy/value
training. It uses exact simulation counts instead of per-move time limits, so
the search quality does not depend on CPU speed or temporary machine load.
No augmentation is applied during collection.

From the project root, with the virtual environment active:

```bash
python -m _game_arena_data_collection --minutes 10 --workers 8 \
  --source-id laptop --seed 42

python -m _game_arena_data_collection --games 100 --workers 8 \
  --source-id laptop --seed 42

python -m _game_arena_data_collection --audit-only
python -m unittest _game_arena_data_collection.test_collection
```

The default output is separate from the old seconds-based collection:

```text
training_data/_game_arena_data_collection_rollouts/10x10/
```

## Exact rollout budget

For a position with `L` legal actions, the default budget is:

```text
min(400, max(128, 4 * L))
```

Examples:

| Legal actions | Completed rollouts |
| ---: | ---: |
| 100 | 400 |
| 70 | 280 |
| 50 | 200 |
| 20 | 128 |
| 5 | 128 |

Every legal root action is therefore expanded at least once on a 10x10 board.
Unlike a wall-clock budget, an exact simulation budget is reproducible and
comparable between a laptop and another computer. The settings are editable in
`config.py`.

The schema-v1 `requested_simulations` scalar stores the maximum configured
budget for compatibility with the existing loader and analysis GUI. The
`completed_rollouts` array stores the exact number actually performed for every
position. The audit verifies every value against the formula above.

## Policy-consistent move selection

There are no completely random opening moves. Every saved position runs MCTS
and therefore has a real policy target.

- During the first 20 moves, the played action is sampled from the normalized
  root visit counts with temperature 1.0.
- From move 21 onward, the most-visited root action is played. Ties are broken
  with the game's seeded random generator.

This provides diverse openings while keeping played actions consistent with
the visit-count distribution used to train the policy head. Both players use
the same rollout formula; asymmetric strength experiments should be collected
separately from the main training dataset.

## Parallel computers: source ID and seed

`source-id` is a short name for the machine or collection stream. It accepts
letters, digits, and underscores. Both `source-id` and `seed` affect the random
game stream and are embedded in every filename.

For example:

```bash
# Laptop
python -m _game_arena_data_collection --minutes 480 --workers 8 \
  --source-id laptop --seed 42

# Another computer
python -m _game_arena_data_collection --minutes 480 --workers 8 \
  --source-id pc2 --seed 43
```

Example filenames:

```text
..._collection-laptop-s42-g00000000.npz
..._collection-pc2-s43-g00000000.npz
```

The local game index may be the same because the complete identity is
`(source-id, seed, game index)`. Files from several machines can therefore be
copied into the same `10x10` directory without name or manifest collisions, as
long as their rollout-generation settings are identical. Using distinct
source IDs is sufficient to create distinct deterministic streams; using a
different seed as well makes the intent explicit.

Do not reuse the same `source-id` and seed independently on two computers: both
would intentionally generate the same deterministic stream.

## CPU parallelism and stopping

`--workers N` runs up to N independent games in separate CPU processes. Each
game uses one serial MCTS search. The default is
`min(8, max(1, CPU count - 2))`.

The minute and game limits apply to one invocation. The collector checks the
time before starting another game and lets active parallel games finish and
save. Ctrl-C stops new submissions and drains active parallel games. Completed
files remain usable and a later invocation with the same source ID and seed
fills missing local indexes before allocating new ones.

The coordinator exclusively updates metadata, while workers atomically save
separate NPZ files. An OS lock prevents two collector processes from writing
to the same directory at the same time. Parallel collection on separate
computers uses separate local directories; the completed files can be merged
afterward.

## Files and verification

The output directory contains:

- one schema-v1 `.npz` per completed game;
- `collection.json` with rollout-generation settings;
- `manifest.json` with source, seed, index, result, runtime and aggregate
  search information for every game;
- `last_session.json` describing the last invocation;
- `audit.json` and `report.md` with replay and policy-target quality metrics.

Every game is replayed from the empty board. The audit checks boards,
territories, scores, player turns, legal masks, chosen actions, final results,
root visit totals and exact adaptive rollout budgets. It also reports quality
percentiles for the whole collection and for moves 1-20, 21-40, 41-60, 61-80
and 81-100, including:

- rollouts per position and per legal action;
- fraction of legal actions visited and revisited;
- normalized visit-distribution entropy;
- top-action visit share;
- results, starter balance, source counts and position diversity.

Split training, validation and test data by complete games. Any D4
augmentation belongs in the training input pipeline after that split, not in
this collector.
