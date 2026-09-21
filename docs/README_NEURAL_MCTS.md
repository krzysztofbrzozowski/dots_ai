# Sequential neural MCTS

`main_mcts_ml.py` is an isolated policy/value MCTS entry point. It uses
`ml/models/25x25_088622_new_data_dual_head_v1.keras` and the same live GUI as
the classic application, without changing `main_mcts.py` or its rollout-based
search.

## Running

From the repository root:

```bash
.venv/bin/python main_mcts_ml.py
```

Then open `http://127.0.0.1:8000`.

The first model load is performed in the game thread, so the GUI can start and
show the loading diagnostic. The loaded model is reused for the entire game.

## Search flow

The implementation follows the original project's simple node/search split:

```text
NeuralMCTSNode
    stores visits, accumulated value, action, and policy prior
        ↓
NeuralMonteCarloTreeSearch
    select with PUCT
    evaluate one leaf with policy + value
    expand legal children
    backpropagate with alternating perspective
        ↓
choose the root child with the most visits
```

Search is deliberately sequential. Each simulation finishes its selection,
single model inference, expansion, and backpropagation before the next one
begins. There is no process pool and the Keras model is never copied between
workers.

Children are lightweight until visited. Expanding the empty 25 × 25 root
creates 625 action/prior nodes, but does not create 625 independent game-state
copies. A child applies its move only when PUCT first selects it.

## Model semantics

One inference returns both heads:

- policy logits are masked to legal moves and normalized into PUCT priors;
- value is `P(win) - P(loss)` for the player to move;
- backpropagation reverses the sign at each parent;
- terminal positions use their exact result without model inference.

The public child `q` remains in the parent player's perspective. This keeps
the live overlays and schema-v1 training trajectory compatible with classic
MCTS.

## Settings

The main settings live at the top of `main_mcts_ml.py`:

- `ROWS` and `COLS` must remain 25 for the configured checkpoint;
- `SIMULATION_SECONDS` controls the per-move wall-clock budget;
- `DEFAULT_SIMULATIONS` is used when the time budget is set to `None`;
- `C_PUCT` controls the policy-prior exploration bonus;
- completed games are stored below `training_data/neural_mcts`.

The stored filename ends in `_neural-mcts.npz`. Array names and Q/visit
semantics remain compatible with the existing analyzer and training loader.
