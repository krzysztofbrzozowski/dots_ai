# Monte Carlo Tree Search

This document explains how Monte Carlo Tree Search (MCTS) chooses moves in this
project, how the search is parallelized, and how it connects to the Dots game
loop.

## Overview

MCTS estimates the value of legal moves by repeatedly playing randomized games
from the current position. It does not enumerate the complete game tree.
Instead, it grows the most useful parts of the tree and records the results of
completed random games.

One MCTS simulation has four phases:

```text
current board
     │
     ▼
1. Selection ── follow promising tree nodes
     │
     ▼
2. Expansion ── add one previously untried move
     │
     ▼
3. Rollout ──── play random legal moves until the game ends
     │
     ▼
4. Backpropagation ── update the selected node and all its ancestors
```

The search repeats these phases until its simulation or time budget is
exhausted. It then returns the root child with the best average result.

## Code map

| File | Responsibility |
| --- | --- |
| [`mcts/nodes.py`](../mcts/nodes.py) | Node data, expansion, UCT scoring, rollouts, virtual loss, and backpropagation |
| [`mcts/search.py`](../mcts/search.py) | Search budget, tree policy, sequential simulations, and parallel rollout batches |
| [`mcts/enclosure.py`](../mcts/enclosure.py) | Stable public imports for the MCTS package |
| [`game/engine.py`](../game/engine.py) | Immutable search-state transitions through `DotsGame.move()` |
| [`main_mcts.py`](../main_mcts.py) | Complete game loop and persistent rollout process pool |

## Search state and nodes

Every `TwoPlayerMCTSNode` represents one game position and contains:

| Field | Meaning |
| --- | --- |
| `state` | The `DotsGame` position represented by this node |
| `parent` | The node from which this position was reached |
| `action` | The move applied to the parent state to create this state |
| `children` | Expanded moves below this node |
| `_untried_actions` | Legal moves that have not yet been expanded |
| `_number_of_visits` | Number of completed simulations through this node |
| `_results` | Counts of results: Player 1 win, Player 2 win, or draw |
| `_virtual_visits` | Temporary reservations for parallel rollouts currently in flight |

`DotsGame.move(action)` returns an independent state. The parent board is not
modified when a child is expanded or when a rollout advances. This isolation
allows rollout states to be serialized and sent safely to worker processes.

See [README_GAME.md](README_GAME.md) for the game-state and capture rules.

## The four MCTS phases

### 1. Selection

Selection starts at the root node for the current real board position.

At each node:

1. If the game is over, that terminal node is returned.
2. If the node has an untried legal action, one action is expanded immediately.
3. Otherwise, `best_child()` chooses a child using the UCT score and selection
   continues from that child.

This means every legal action is expanded before the search relies entirely on
UCT to revisit existing children. Untried actions are currently taken with
`list.pop()`, so their initial expansion order follows the reverse order of
`DotsGame.get_legal_actions()`.

### 2. Expansion

Expansion removes one action from `untried_actions`, applies it to an
independent game state, and attaches the resulting child to the tree:

```python
action = node.untried_actions.pop()
next_state = node.state.move(action)
child = TwoPlayerMCTSNode(
    state=next_state,
    parent=node,
    action=action,
)
```

At most one node is added for each selected simulation. A selection that
reaches an existing terminal node does not need to expand another child.

### 3. Rollout

A rollout starts from the selected leaf state and plays until no legal moves
remain. Every rollout move is selected uniformly at random from the current
legal actions.

The terminal result is represented as:

| Result | Meaning |
| ---: | --- |
| `1` | Player 1 wins |
| `-1` | Player 2 wins |
| `0` | Draw |

Rollouts use the real `DotsGame.move()` and capture logic. They are random
simulations, not a simplified version of the game.

### 4. Backpropagation

The terminal result is recorded on the rollout leaf and every ancestor up to
the root:

```text
root                         visits += 1, result count += 1
└── selected child           visits += 1, result count += 1
    └── selected descendant  visits += 1, result count += 1
        └── rollout leaf     visits += 1, result count += 1
```

The root visit count therefore equals the number of completed simulations.
Each child visit count includes only simulations whose selected path passed
through that child.

## Node value and player perspective

For a child node, `q` is calculated from the perspective of the player who
chose that child from its parent:

```text
q = wins for parent.state.next_to_move
    - wins for the opposing player
```

A win contributes `+1`, a loss contributes `-1`, and a draw contributes `0` to
the value balance. Draws still increase the visit count, so they affect the
average value.

This parent-player perspective is important because turns alternate at every
tree level. Each parent evaluates its children for the player making the move
at that parent position.

## UCT: balancing exploitation and exploration

After all actions at a node have been expanded, `best_child()` uses an Upper
Confidence Bound for Trees (UCT) score:

```text
                       q(child) - virtual_visits(child)
UCT(child) =           --------------------------------
                       visits(child) + virtual_visits(child)

                       ┌ 2 × ln(effective parent visits) ┐
             + C × sqrt│ -------------------------------- │
                       └       effective child visits     ┘
```

Where:

- the first term is exploitation: prefer moves with better recorded results;
- the second term is exploration: prefer moves with fewer visits;
- `C` is the exploration constant, currently `1.4`;
- effective visits are completed visits plus temporary virtual visits.

During an ordinary sequential search, virtual visits are zero. The formula then
reduces to the standard win/loss average plus the exploration bonus.

Once the budget is exhausted, the final move uses `C = 0`. Exploration is
disabled and the child with the best completed average result is returned.

## Parallel rollouts

Python threads do not significantly speed up these CPU-bound rollouts because
most game logic executes under Python's Global Interpreter Lock. The project
therefore uses worker processes.

The mutable search tree always remains in one authoritative process. Only leaf
rollouts execute in parallel:

```text
Authoritative game/search process
    │
    ├── select leaf A ── reserve path A ── send state A + seed ─┐
    ├── select leaf B ── reserve path B ── send state B + seed ─┤
    ├── select leaf C ── reserve path C ── send state C + seed ─┤
    │                                                           ▼
    │                                                   rollout processes
    │                                                           │
    └── receive results ◀── release reservations ◀──────────────┘
             │
             └── backpropagate each completed result
```

For every batch, `MonteCarloTreeSearch`:

1. Selects up to `parallelism` leaves.
2. Reserves every selected path with a temporary virtual loss.
3. Submits only each leaf's `DotsGame` state and a unique random seed.
4. Waits for the rollout results.
5. Releases all reservations, even if a worker raises an exception.
6. Backpropagates the batch after every worker completes successfully. If one
   worker fails, the batch is aborted without partially updating the tree.

Virtual loss temporarily lowers the score of a branch that already has a
rollout in flight. This encourages the rest of the batch to explore other
branches instead of duplicating the same work.

Worker processes never receive MCTS nodes, parent references, statistics, or
the full search tree. Consequently, tree updates require no cross-process locks
and cannot race with worker code.

## Process-pool lifecycle

`run_parallel_mcts_game()` creates one `ProcessPoolExecutor` for the complete
match. The pool:

- uses the `spawn` process-start method for consistent behavior across macOS,
  Linux, and Windows;
- is warmed before the first timed search so process startup is not charged to
  the first move;
- is reused for every MCTS move;
- is shut down after the game reaches a terminal result.

The application defaults to:

```python
DEFAULT_MCTS_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
```

This normally leaves one logical CPU available for the GUI and operating
system, while limiting the pool to eight rollout workers.

## Search budgets

`MonteCarloTreeSearch.best_action()` supports two budget types.

### Fixed simulation count

```python
best_node = search.best_action(simulations_number=1_000)
```

The search completes exactly the requested number of simulations. The final
parallel batch is reduced when necessary so it does not exceed the count.

### Wall-clock time

```python
best_node = search.best_action(total_simulation_seconds=30)
```

The search starts batches until the deadline is reached. An in-flight final
batch is allowed to finish, so elapsed time can exceed the requested duration
by approximately one rollout batch.

The application currently uses a 30-second time budget for every real move:

```python
SIMULATION_SECONDS = 30
```

In `run_mcts_game()`, a non-`None` `simulation_seconds` value takes precedence.
Set `simulation_seconds=None` to use `simulations_number` instead:

```python
run_mcts_game(
    board_state,
    simulations_number=1_000,
    simulation_seconds=None,
)
```

## Complete application flow

For each visible move, [`main_mcts.py`](../main_mcts.py) performs the following
steps:

1. Creates a new root node for the current board.
2. Creates `MonteCarloTreeSearch` using the persistent rollout executor.
3. Runs simulations for the configured budget.
4. Takes the state stored in the selected root child as the new real board.
5. Publishes a serialized snapshot for the read-only GUI.
6. Waits for `DEFAULT_MOVE_DELAY` when the game is not over.
7. Starts a new tree for the next real move.

The search tree is currently rebuilt after every real move. Statistics from the
previous tree are not retained or re-rooted, even though the selected child
already contains a subtree.

See [README_GUI.md](README_GUI.md) for the server and browser-display flow.

## Sequential usage

The MCTS classes can still be used without a process pool:

```python
from game.enclosure import DotsGame
from mcts.enclosure import MonteCarloTreeSearch, TwoPlayerMCTSNode

game = DotsGame(10, 10)
root = TwoPlayerMCTSNode(game)
search = MonteCarloTreeSearch(root)

best_node = search.best_action(simulations_number=500)
print(best_node.action)
```

When no executor is supplied, selection, rollout, and backpropagation all run
sequentially in the calling process.

## Current limitations and tuning points

- Rollout moves are uniformly random; there is no tactical rollout policy,
  heuristic evaluation, or neural network.
- The tree is discarded after every real move instead of being re-rooted and
  reused.
- Time-based budgets produce different simulation counts on different
  machines and at different stages of the game.
- The initial expansion order is deterministic because untried actions are not
  shuffled.
- More workers increase throughput but also increase temporary virtual loss
  and coordination overhead. Eight is a measured practical cap, not an
  algorithmic requirement.
- MCTS results are stochastic. Parallel rollouts can use a fixed
  `random_seed` on `MonteCarloTreeSearch`; sequential callers can seed NumPy's
  global random generator when reproducibility is needed.

The most important tuning parameters are the search budget, worker count, UCT
exploration constant, and rollout policy.
