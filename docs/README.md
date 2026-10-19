# Dots AI documentation

- [Tests](../tests/README.md) — centralized test modules and commands for
  running the engine, search, training, analysis, and arena suites.
- [Monte Carlo Tree Search](README_MCTS.md) — selection, expansion, random
  rollouts, backpropagation, UCT, and process-based parallelism.
- [Sequential neural MCTS](README_NEURAL_MCTS.md) — policy priors, PUCT,
  value-head leaf evaluation, and the isolated ML entry point.
- [Game engine](README_GAME.md) — board state, legal moves, captures,
  connectivity, scoring, and rendering.
- [Shared MCTS workspace](README_GUI.md) — one frontend for live self-play and
  saved-game analysis.
- [Self-play training data](README_TRAINING_DATA.md) — absolute game-state
  records, action-aligned root statistics, NPZ schema, and perspective handling.
- [Saved-game analysis GUI](README_ANALYSIS.md) — NPZ validation, the canonical
  analysis model, board overlays, and timeline navigation.
- [Arena tools](../_arena/README.md) — current layout, archived tools, default paths, and
  verification commands.
- [Elo arena](../_arena/elo_game_arena/README.md) — rated 25x25 matches between
  classic MCTS, neural MCTS, and named human players, with trajectory recording.
- [Legacy game arena](../_arena/obsolete/_game_arena/README.md) — archived
  MCTS match runner with independent time budgets and execution modes.
- [Legacy rollout collector](../_arena/obsolete/_game_arena_data_collection/README.md)
  — archived self-play collection with exact rollout budgets and dataset audits.
