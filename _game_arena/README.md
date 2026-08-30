# Game arena

The arena runs one MCTS-vs-MCTS game with independent settings for both
players. Edit `ARENA_CONFIG` in `config.py`, then run:

```bash
python -m _game_arena
```

Example configuration:

```python
ARENA_CONFIG = ArenaConfig(
    rows=4,
    cols=4,
    next_to_move=PLAYER_2,
    player_1=PlayerMCTSConfig.serial(
        simulation_seconds=1,
    ),
    player_2=PlayerMCTSConfig.parallel(
        simulation_seconds=10,
        workers=14,
    ),
)
```

Each player independently supports `serial(...)` or `parallel(...)`, so the
same runner handles serial-vs-serial, serial-vs-parallel,
parallel-vs-serial, and parallel-vs-parallel games.

Every completed game is saved automatically below:

```text
training_data/_game_arena/<rows>x<cols>/
```

The filename contains the full arena configuration. For example:

```text
<timestamp>_<id>_next-p2__p1-serial-0p1s-w1__p2-parallel-4s-w14.npz
```

Here `0p1s` means `0.1` simulation seconds and `w14` means 14 parallel
workers. Passing `training_data_directory=None` to `run_arena()` disables
recording for tests or one-off games.
