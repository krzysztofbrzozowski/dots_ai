# Tests

The tracked project test modules live in this directory. Run commands from the repository
root with the project virtual environment active and dependencies installed.
Use module commands so imports resolve from the repository root.

## Running the tests

The function-based suites retain their standalone runners:

```bash
python -m tests.test_analysis
python -m tests.test_enclosure
python -m tests.test_game_arena
python -m tests.test_mcts_application
python -m tests.test_ml_training_pipeline
python -m tests.test_neural_mcts
python -m tests.test_training_data
```

The Elo arena and rollout collector use `unittest`:

```bash
python -m unittest tests.test_elo_arena tests.test_collection
```

Run all commands above to cover every suite. `unittest` discovery alone does
not execute the standalone function-based tests. The ML training pipeline
tests require TensorFlow; collector tests also exercise spawned worker
processes. Tests create their game records in temporary directories.

Local utility tests remain in `_tools/`, alongside their tools, and are
excluded from Git by the existing ignore rules.
