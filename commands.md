# Prompt for DeepSeek

Implement an efficient enclosure/capture detection algorithm for the game **Dots (Kropki)** in Python using NumPy.

Use this internal board representation:

```python
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1
```

Do not use `X`, `O`, `RED`, or `BLUE` in the game logic.

The board should be represented as a 2D NumPy array, for example:

```python
board = np.array([
    [0,  1,  0],
    [-1, 0,  1],
    [0, -1,  0],
])
```

Keep the game logic independent from presentation. The game engine should operate only on:

```python
EMPTY
PLAYER_1
PLAYER_2
```

Colors and terminal symbols should be handled separately by the rendering layer.

For example:

```python
SYMBOLS = {
    EMPTY: "·",
    PLAYER_1: "●",
    PLAYER_2: "○",
}
```

and optionally:

```python
COLORS = {
    PLAYER_1: "red",
    PLAYER_2: "blue",
}
```

The enclosure/capture detection algorithm should work **incrementally after each move** instead of scanning the entire board from scratch.

Use the following general idea:

```text
player places a new dot
        ↓
check its 8 neighboring cells
        ↓
find neighboring dots belonging to the same player
        ↓
connect/update their group
        ↓
check whether the new dot could have closed a loop
        ↓
if a loop/enclosure may have been created:
        ↓
perform a local flood-fill
        ↓
determine whether a region is actually enclosed
        ↓
check whether opponent dots are inside
        ↓
return captured opponent dots
```

Requirements:

1. Use **8-neighbor connectivity** for dots belonging to the same player:

   * up
   * down
   * left
   * right
   * four diagonals

2. Keep track of connected groups of dots

   * You may use Union-Find / DSU or another efficient structure
   * A newly placed dot should either:

     * create a new group
     * join an existing group
     * merge multiple groups
     * connect different parts of the same group

3. Detect when the newly placed dot may have closed an enclosure

   * Do not assume that every graph cycle automatically means a valid capture
   * A cycle should only trigger geometric verification

4. Use a **local flood-fill / BFS / DFS** to verify whether an area is actually enclosed

   * dots belonging to the current player act as the boundary
   * determine whether the candidate region can reach the outside
   * only a region that cannot escape should count as enclosed

5. Detect opponent dots inside the enclosed region

6. Determine the opponent using the numeric representation:

```python
opponent = -player
```

For example:

```python
player = PLAYER_1
opponent = PLAYER_2
```

7. Return captured coordinates, for example:

```python
[(2, 3), (2, 4), (3, 3)]
```

If nothing was captured:

```python
[]
```

8. Design the API around the last move, for example:

```python
captured = detect_capture(
    board,
    last_move=(row, col),
    player=PLAYER_1,
)
```

9. Avoid scanning the entire board after every move whenever possible. The algorithm should focus primarily on the neighborhood and connected group affected by `last_move`

10. Keep the implementation readable and split it into small functions or classes, for example:

```python
get_neighbors(...)
find_group(...)
merge_groups(...)
find_candidate_regions(...)
flood_fill_region(...)
detect_capture(...)
render_board(...)
```

11. Implement a simple terminal renderer:

```python
def render_board(board):
    ...
```

The renderer should translate the internal representation into display symbols.

Example:

```text
· ● · ·
○ · ● ·
· ○ · ·
· · · ●
```

The renderer must not affect the internal board representation.

12. Add clear English comments explaining the non-obvious parts of the algorithm

13. Include several tests/examples:

* no enclosure
* simple square enclosure
* diagonal/diamond-shaped enclosure
* enclosure containing one opponent dot
* enclosure containing multiple opponent dots
* a connected cycle that does NOT constitute a valid capture
* enclosure close to the edge of the board
* terminal rendering example

14. Do not implement MCTS or neural networks. Implement only:

* board representation
* connected groups
* enclosure detection
* capture detection
* terminal rendering

15. Keep game-state logic separate from presentation logic. Code responsible for detecting captures should never depend on colors or display symbols

Before writing the final code, briefly explain the algorithm and especially how you determine that an area is truly enclosed rather than merely detecting a cycle in the graph.
