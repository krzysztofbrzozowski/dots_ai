# Elo Arena 25x25

The arena runs rated 25x25 matches between classic MCTS, neural MCTS, and
named human players.

```bash
python -m _elo_game_arena
```

Open <http://127.0.0.1:8003>. Choose both competitors and start the match.
When a human is to move, click a legal intersection on the board.
Use **End match** to adjudicate an unfinished game from the current score.
The higher score wins, an equal score is a draw, and Elo is updated immediately.
Every match with at least one played move is also saved as a compatible NPZ
trajectory, including matches adjudicated with **End match**. Files are stored
under `training_data/_elo_game_arena/25x25`.

Classic and neural search parameters are configured independently at the top
of `config.py`:

```python
CLASSIC_MCTS_SIMULATION_SECONDS = 90.0
CLASSIC_MCTS_WORKERS = 14

NEURAL_MCTS_SIMULATION_SECONDS = 10.0
NEURAL_MCTS_C_PUCT = 1.5
NEURAL_MCTS_MODEL_PATH = ...
```

Preset ids are derived from these settings so a materially different AI
configuration receives a separate Elo identity.

Ratings use a base value of 1200, K=32, and are updated after completed or
adjudicated games. Draws score 0.5. The persistent table and match history are stored in
`ratings.json` in this directory. AI identity includes its fixed search preset,
so changing an AI budget requires a new preset id in `config.py`.
