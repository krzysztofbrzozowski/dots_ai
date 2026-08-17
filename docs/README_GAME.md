# Game logic overview

This document explains the implementation used by `DotsGame` in
`game/engine.py` and by the helpers in `game/board.py`, `game/groups.py`, and
`game/capture.py`

The implementation keeps four related ideas separate

```text
board
    stores the dots that were placed

territory
    stores which cells have already been captured

UnionFind
    stores connectivity between active dots

flood fill
    explores board regions and determines their enclosure geometry
```

This separation is important because a captured dot can remain visible on the
board while no longer participating in the active connectivity graph

At a high level, placing a move follows this flow

```text
place_dot()
    ↓
validate player and cell
    ↓
write last_move to board
    ↓
detect_capture_info()
    ↓
_could_have_closed_loop()
    ↓
False ──→ return an empty CaptureInfo
    ↓ True
find_enclosed_regions()
    ↓
find_candidate_regions()
    ↓
flood_fill_region()
    ↓
keep enclosed regions containing new opponent dots
    ↓
return CaptureInfo
    ↓
add last_move to UnionFind and join active same-player neighbors
    ↓
if capture happened
    ├── update territory
    ├── update score
    └── rebuild active UnionFind groups
```

The `board` has already been updated when capture detection runs, but the
provided `groups` object still describes connectivity from before the move
This timing allows the code to ask whether `last_move` created a new graph
cycle

## Source files

The game logic is divided into focused modules

| File | Responsibility |
| --- | --- |
| `game/board.py` | Cell values, player values, and neighbor lookup |
| `game/groups.py` | UnionFind and rebuilding active dot groups |
| `game/capture.py` | Cycle pre-check, flood fill, and capture results |
| `game/engine.py` | Mutable game state and move application |
| `game/rendering.py` | Text rendering without changing game state |
| `game/enclosure.py` | Backward-compatible imports for the public API |

## State created by `DotsGame`

`DotsGame(rows, cols)` requires positive dimensions and creates this mutable
state

| Attribute | Initial value | Purpose |
| --- | --- | --- |
| `self.board` | A `rows × cols` integer array filled with `0` | Stores placed dots |
| `self.territory` | A `rows × cols` integer array filled with `0` | Stores captured ownership |
| `self.groups` | An empty `UnionFind` | Stores active dot connectivity |
| `self.score` | `{PLAYER_1: 0, PLAYER_2: 0}` | Counts opponent dots captured by each player |

The main temporary name used during a move is `last_move`
It is the `(row, col)` coordinate just written to the board and is the move that
capture detection must evaluate

# Board state

`DotsGame.board` is a two-dimensional NumPy array

```python
board[row, col]
```

Each cell contains one of these integer values

| Constant | Value | Meaning |
| --- | ---: | --- |
| `EMPTY` | `0` | No dot is present |
| `PLAYER_1` | `1` | Player 1 has a dot here |
| `PLAYER_2` | `-1` | Player 2 has a dot here |

The board answers one question

> Who has a dot on this cell

For example

```python
board = np.array([
    [ 1,  1,  0],
    [ 1, -1,  0],
    [ 0,  0,  0],
])
```

The renderer displays that state approximately as

```text
● ● ·
● ○ ·
· · ·
```

The symbols are only presentation
Internally, the algorithms use the integer values

The board is also the source of truth for enclosure geometry
During a capture check for `player`, a cell for which
`board[cell] == player` is a wall
Empty cells and opponent dots are traversable by flood fill

# Territory state

`DotsGame.territory` is another two-dimensional NumPy array with the same
shape as `board`

```python
territory[row, col]
```

Its values mean

| Value | Meaning |
| ---: | --- |
| `EMPTY`, or `0` | The cell has not been captured |
| `PLAYER_1`, or `1` | The cell was captured by Player 1 |
| `PLAYER_2`, or `-1` | The cell was captured by Player 2 |

Territory answers a different question from the board

> Has this cell already become captured game state, and by whom

A territory cell may be empty on the board
It may also contain a dot that remains visible after capture

For example, suppose Player 1 captures a Player 2 dot at `(2, 2)`

```python
board[2, 2] == PLAYER_2
territory[2, 2] == PLAYER_1
```

The `PLAYER_2` value is not removed from `board`
The territory value records that Player 1 owns the captured cell
Rendering still shows the dot's original board symbol
Only a captured cell that is empty on `board` is rendered as `×`

Territory is used for game-state decisions

- A move is legal only when both `board[cell]` and `territory[cell]` are empty

- A captured opponent dot is not scored again

- Existing territory keeps its current owner if a later captured region
  overlaps it

- Active UnionFind groups exclude every cell whose territory is nonempty

- Captured empty cells render as `×`

Territory is not used as a flood-fill wall
`flood_fill_region()` does not receive a territory array at all
Flood fill can traverse a cell even when that cell has previously captured
territory, provided the board cell is not a current-player dot

The resulting distinction is

```text
board
    determines enclosure geometry

territory
    stores capture ownership, move availability, scoring state,
    and whether dots are active
```

One subtle consequence follows directly from the current code
A current-player dot that remains visible on `board` satisfies
`board[cell] == player` and is therefore still a geometric wall, even if its
territory is nonempty
It is inactive for UnionFind, but the flood-fill function does not inspect that
active state

# Active vs inactive dots

A dot is active when both conditions are true

```python
board[cell] in PLAYERS
territory[cell] == EMPTY
```

An inactive dot is still present on `board`, but its territory is nonempty
This happens when that dot lies in a captured region

Example

```text
Cell A
board[A] = PLAYER_1
territory[A] = EMPTY
→ active Player 1 dot

Cell B
board[B] = PLAYER_1
territory[B] = PLAYER_2
→ visible Player 1 dot captured by Player 2
→ inactive dot
```

Only active dots are registered in the active UnionFind structure
When `_could_have_closed_loop()` examines the neighbors of `last_move`, it uses
the same rule

```python
board[neighbor] == player
and territory[neighbor] == EMPTY
```

Therefore, a path that looks connected on `board` does not count as an active
graph path if it depends on captured dots

```text
active ● ── inactive ● ── active ●

Visible on board: connected
Active UnionFind graph: separated
```

After a capture, `place_dot()` rebuilds all groups from `board` and `territory`
This removes newly inactive dots from connectivity without deleting their board
values

# Coordinates and neighbors

Coordinates are written as

```python
(row, col)
```

The row comes first and the column comes second
Row numbers increase downward and column numbers increase to the right

```text
             columns
          0   1   2   3
       0  ·   ·   ·   ·
rows   1  ·   ·   X   ·
       2  ·   ·   ·   ·

X is at (1, 2)
```

`get_neighbors()` calculates nearby coordinates using small offsets

- `dr` means delta row, or the change in the row index

- `dc` means delta column, or the change in the column index

- `nr` means new row and is calculated as `row + dr`

- `nc` means new column and is calculated as `col + dc`

For example

```python
dr = -1
dc = 0
nr = row + dr
nc = col + dc
```

This selects the cell immediately above the current cell
Before returning a neighbor, `get_neighbors()` verifies that `nr` and `nc` are
inside the board

## Eight-connected player dots

Player connectivity includes horizontal, vertical, and diagonal neighbors

```text
↖  ↑  ↗
←  X  →
↙  ↓  ↘
```

`place_dot()`, `rebuild_groups()`, and `_could_have_closed_loop()` all request
neighbors with `include_diagonals=True`
Two active same-player dots touching at a corner can therefore belong to the
same UnionFind group

## Four-connected flood fill

Flood fill moves only horizontally and vertically

```text
   ↑
←  X  →
   ↓
```

The code implements these moves with `ORTHOGONAL`

The two connectivity models answer different questions

- Eight-connected dot connectivity asks whether same-player dots form one
  active graph, including diagonal connections

- Four-connected flood fill asks which non-wall board cells form one region

As a result, a diagonal touch can connect the player's graph without allowing
the flood fill to move diagonally through a corner

# UnionFind

UnionFind, also called a disjoint-set structure, keeps track of which active
dots belong to the same connected group

It answers connectivity questions quickly without searching the whole board on
every move

## Internal dictionaries

Each `UnionFind` object contains two dictionaries

```python
self._parent = {}
self._rank = {}
```

`_parent[cell]` points toward the representative of the cell's group
A cell that points to itself is a root

`_rank[root]` is a small value used to keep the internal parent tree shallow
It is an implementation aid, not a game score and not a board distance

A root is only the technical representative of a connected group
It is not the first dot placed, the physical beginning of a line, or a special
location on the board

## `add(cell)`

`add()` registers a coordinate as a one-element group

```python
groups.add((2, 3))
```

After this call, the dictionaries contain approximately

```python
_parent = {
    (2, 3): (2, 3),
}

_rank = {
    (2, 3): 0,
}
```

Calling `add()` again for the same cell does nothing

## `find(cell)`

`find()` follows parent links until it reaches a root and returns that root
It raises `KeyError` if the cell was never added

Suppose the structure contains

```python
_parent = {
    (2, 3): (2, 3),
    (2, 4): (2, 3),
}
```

Then

```python
groups.find((2, 4)) == (2, 3)
```

The method also performs path compression
If a cell reaches its root through several intermediate parents, `find()`
rewrites those parent links to point directly to the root

```text
Before find
A → B → C → C

After find(A)
A → C
B → C
C → C
```

Path compression makes later searches faster

## `union(a, b)`

`union()` merges the groups containing two registered cells

```python
groups = UnionFind()
groups.add((2, 3))
groups.add((2, 4))
groups.union((2, 3), (2, 4))
```

Before the union

```python
_parent = {
    (2, 3): (2, 3),
    (2, 4): (2, 4),
}

_rank = {
    (2, 3): 0,
    (2, 4): 0,
}
```

After the union, one possible and current-order result is

```python
_parent = {
    (2, 3): (2, 3),
    (2, 4): (2, 3),
}

_rank = {
    (2, 3): 1,
    (2, 4): 0,
}
```

The implementation uses rank to attach the shallower tree below the deeper
tree
When both roots have the same rank, the selected new root has its rank
increased

`union()` returns `False` when both cells already have the same root
Otherwise it merges the groups and returns `True`

## `connected(a, b)`

`connected()` is a convenient comparison

```python
groups.connected(a, b)
```

is equivalent to

```python
groups.find(a) == groups.find(b)
```

Both cells must already be registered

## What UnionFind does not know

UnionFind does not know any of the following

- Board dimensions

- Cell ownership

- Whether two coordinates are neighbors

- Whether a dot is captured

- Which side of a cycle is inside

For example, this call would connect the coordinates if both had already been
added

```python
groups.union((2, 3), (2, 6))
```

UnionFind itself does not reject the call merely because the cells are far
apart
Geometry and active-state validation happen before `union()` is called

# Rebuilding active groups

`rebuild_groups(board, territory)` creates a fresh UnionFind object

It first registers every cell satisfying

```python
board[cell] in PLAYERS
and territory[cell] == EMPTY
```

It then checks all eight neighbors of each registered cell and unions neighbors
that contain the same player value

The function is used after a capture because multiple dots may have become
inactive at once
Rebuilding ensures that no captured dot remains in the active connectivity
graph

When no territory array is supplied, `rebuild_groups()` creates an all-empty
territory array and therefore treats every board dot as active

# `place_dot()`

`DotsGame.place_dot(row, col, player)` owns the full state-changing move
operation
It returns a list of opponent coordinates newly captured by the move

The method executes these steps in this exact order

## 1  Validate the move

The method verifies that `player` is either `PLAYER_1` or `PLAYER_2`
It then calls `is_legal_move()`

A legal move must be inside the board and satisfy

```python
board[row, col] == EMPTY
and territory[row, col] == EMPTY
```

An occupied cell and a captured empty cell are both illegal

## 2  Write the move to `board`

```python
last_move = (row, col)
self.board[last_move] = player
```

The new dot must be visible to the geometry check because it may complete a
wall

At this moment

```text
board  = state after last_move
groups = active connectivity before last_move
```

## 3  Detect capture using pre-move groups

The method calls

```python
capture = detect_capture_info(
    self.board,
    last_move,
    player,
    territory=self.territory,
    groups=self.groups,
)
```

The new dot has deliberately not been added to UnionFind yet
`_could_have_closed_loop()` must compare the new dot's neighbors using the
connectivity that existed before the move

If the new dot were unioned first, all same-player neighbors joined through the
new dot could appear to have the same root
That would lose the information needed to distinguish a genuinely pre-existing
path from a path created only by `last_move`

## 4  Add the new dot to active connectivity

After capture information has been calculated, `place_dot()` adds `last_move`
to `self.groups`

It visits all eight board neighbors and unions a neighbor only when

```python
territory[neighbor] == EMPTY
board[neighbor] == player
neighbor in self.groups
```

The checks ensure that the neighbor is active, belongs to the moving player,
and is registered in the current active graph

## 5  Apply a capture

If `capture.happened` is true, the method applies the returned information

For each cell in every captured region

```python
if self.territory[cell] == EMPTY:
    self.territory[cell] = player
```

Previously owned territory is not overwritten

The score increases by the number of newly captured opponent dots

```python
self.score[player] += len(capture.captured_dots)
```

Finally, all groups are rebuilt from `board` and the updated `territory`
This removes newly captured dots from active connectivity

If no capture happened, the territory and score are unchanged and the
incrementally updated UnionFind remains in use

## 6  Return captured dots

The return value is

```python
list(capture.captured_dots)
```

The complete method flow is

```text
place_dot(row, col, player)
    ↓
valid player and legal cell
    ↓
board[last_move] = player
    ↓
detect_capture_info(board after move, groups before move)
    ↓
groups.add(last_move)
    ↓
union active same-player neighbors
    ↓
capture happened?
    ├── no  → return []
    └── yes
          ↓
        mark new territory cells
          ↓
        increase score
          ↓
        rebuild active groups
          ↓
        return captured opponent coordinates
```

# `_could_have_closed_loop()`

This function asks one narrow question

> Could the newly placed dot have created a graph cycle using active player
> dots

Connected dots alone are not enough

```text
● ● ● ● ●
```

This is a connected line, but it has no cycle

A cycle looks conceptually like

```text
● ● ● ●
●     ●
●     ●
● ● ● ●
```

The function starts with the eight neighboring cells around `last_move`
It keeps only neighbors satisfying

```python
board[neighbor] == player
and territory[neighbor] == EMPTY
```

These are active same-player neighbors
With fewer than two such neighbors, the move can only extend a line or tree, so
the function returns `False`

When the caller supplies `groups`, the function uses that object directly as
the authoritative pre-move active connectivity state
Visible captured dots elsewhere on the board do not cause it to rebuild the
provided object

For a standalone call with `groups=None`, the function reconstructs pre-move
connectivity

```text
copy board
    ↓
set pre_move_board[last_move] to EMPTY
    ↓
rebuild_groups(pre_move_board, territory)
```

Using `territory` during this rebuild prevents inactive dots from re-entering
the graph

The core check compares roots of active neighbors

```text
neighbor A ─────── existing active path ─────── neighbor B
       \                                      /
        \-------------- last_move -----------/
```

If Neighbor A and Neighbor B already had the same root before the move, an
existing path already connected them
Adding both edges through `last_move` creates another path and therefore a graph
cycle candidate

The implementation detects this with a set

```python
roots = set()

for neighbor in same_player_neighbors:
    root = groups.find(neighbor)

    if root in roots:
        return True

    roots.add(root)
```

Different roots mean that `last_move` is joining previously separate groups,
not closing a cycle between already connected neighbors

This function is intentionally only a fast pre-check
It does not determine what is inside the graph cycle, whether a candidate region
reaches the board edge, or whether the region contains an opponent dot

```text
False
    → no cycle candidate
    → skip geometric flood fill

True
    → a cycle candidate exists
    → continue with geometric analysis
```

# `find_candidate_regions()`

After a cycle candidate is detected, capture detection needs starting cells for
the geometric search
Each starting cell is called a `seed`

The function inspects only the four orthogonal neighbors of `last_move`

```text
              seed
                ↓
seed  →  last_move  ←  seed
                ↑
              seed
```

Board edges may reduce the number of available neighbors

If a neighboring cell contains the current player's dot, it is a wall and is
not returned as a seed
Every other orthogonal neighbor is a candidate, including empty cells,
opponent dots, and cells with existing territory

Although `find_candidate_regions()` retains a `territory` parameter for API
compatibility, it does not use territory to filter the seeds

The seeds are local to `last_move` because the capture pipeline is looking for
regions whose boundary may include that newly placed dot

# `flood_fill_region()`

`flood_fill_region(board, seed, player, visited)` explores one connected region
of cells that are not current-player dots

It returns three values

```python
region, reaches_edge, opponent_cells
```

## Important variables

| Variable | Meaning |
| --- | --- |
| `stack` | Cells discovered but not yet processed |
| `visited` | Boolean NumPy array shared between candidate searches |
| `region` | Every processed cell in this connected region |
| `opponent_cells` | Opponent dots found while processing the region |
| `reaches_edge` | Whether the explored region includes a board-edge cell |

The search begins with

```python
stack = [seed]
visited[seed] = True
```

The main loop is

```python
while stack:
    row, col = stack.pop()
```

`while stack` means continue until no discovered cells remain to inspect
`pop()` removes the last list element, so the list is being used as a
last-in-first-out depth-first-search stack

A cell is marked visited when it is added to the stack, not when it is later
popped
This prevents the same cell from being added several times

If the seed is already visited, the function immediately returns empty result
lists and `False` for `reaches_edge`
It does the same when the seed itself is a current-player wall

For every popped cell, the function

1. Adds `(row, col)` to `region`
2. Adds it to `opponent_cells` if `board[row, col]` contains the opponent
3. Sets `reaches_edge = True` if the cell is on any board edge
4. Examines its four orthogonal neighbors
5. Adds each valid, unvisited, non-wall neighbor to `stack`

Flood fill cannot continue through

- A coordinate outside the board

- A coordinate already marked in `visited`

- A cell where `board[cell] == player`

Opponent dots are not walls
They are added to both `region` and `opponent_cells`, and the search may continue
through them

Empty cells are also traversable
Territory is not checked and is therefore not a wall

## Walkthrough

Consider this board after Player 1 has completed the boundary at `(2, 3)`

```text
      col 0  1  2  3
row 0     ●  ●  ●  ●
row 1     ●  ·  ·  ●
row 2     ●  ·  ○  ●
row 3     ●  ●  ●  ●
```

Use `(2, 2)` as the seed
Player 1 dots are walls and Player 2's dot is traversable

Initial state

```text
stack = [(2, 2)]
region = []
opponent_cells = []
reaches_edge = False
```

Iteration 1

```text
popped cell = (2, 2)
region = [(2, 2)]
opponent_cells = [(2, 2)]

newly discovered = (1, 2), (2, 1)
stack = [(1, 2), (2, 1)]
```

The down and right neighbors are Player 1 walls
Because this is a stack, `(2, 1)` is processed next

Iteration 2

```text
popped cell = (2, 1)
region = [(2, 2), (2, 1)]

newly discovered = (1, 1)
stack = [(1, 2), (1, 1)]
```

Iteration 3

```text
popped cell = (1, 1)
region = [(2, 2), (2, 1), (1, 1)]

newly discovered = none
stack = [(1, 2)]
```

`(1, 2)` was already marked visited when Iteration 1 discovered it, so it is
not added again

Iteration 4

```text
popped cell = (1, 2)
region = [(2, 2), (2, 1), (1, 1), (1, 2)]

newly discovered = none
stack = []
```

The loop ends because the stack is empty
No processed coordinate lies on row `0`, row `3`, column `0`, or column `3`, so
`reaches_edge` remains `False`

## Meaning of `reaches_edge`

`reaches_edge` describes the explored region, not the graph history of the
move

```text
reaches_edge = True
    → the region has an open path to a board edge through non-player cells

reaches_edge = False
    → the region is geometrically separated from every edge by player-dot walls
```

`reaches_edge == False` is not proof that `last_move` created a graph cycle
That question was handled earlier by `_could_have_closed_loop()` using pre-move
active connectivity

# `find_enclosed_regions()`

This function coordinates candidate seeds and flood fill

```text
create one all-False visited array
    ↓
find_candidate_regions()
    ↓
for each seed
    ↓
already visited?
    ├── yes → skip duplicate search
    └── no
          ↓
        flood_fill_region()
          ↓
        region exists, does not reach edge,
        and contains opponent dots?
          ├── no  → reject region
          └── yes → append region and opponent cells
```

All candidate seeds share the same `visited` array
If two seeds lead into the same connected region, the first flood fill marks
the region and the later seed is skipped

A returned entry has this shape

```python
(region, opponent_cells)
```

The function returns only regions satisfying all three checks

- `region` is not empty

- `reaches_edge` is `False`

- `opponent_cells` is not empty

An empty enclosed area is therefore not a capture under the current rules

`find_enclosed_regions()` performs geometry only
It does not independently prove that `last_move` created the cycle
That protection comes from calling it through `detect_capture_info()`, after the
UnionFind pre-check succeeds

# `detect_capture_info()`

`detect_capture_info()` connects the graph pre-check to the geometric check
The caller must provide a board that already contains `last_move`
When `groups` is supplied, it must describe active connectivity before that
move was added

If `territory` is omitted, the function creates an all-empty territory array
for the check

## Input validation

Before capture detection, the function verifies

- `last_move` lies inside `board`

- `player` is `PLAYER_1` or `PLAYER_2`

- `board[last_move]` contains the moving player's dot

- `territory[last_move]` is empty

Invalid input raises `ValueError`

## Capture pipeline

```text
board already contains last_move
groups still represent the pre-move active graph
    ↓
validate input
    ↓
_could_have_closed_loop()
    ↓
False ──→ CaptureInfo(captured_dots=(), captured_regions=())
    ↓ True
find_enclosed_regions()
    ↓
for each geometrically enclosed region
    ↓
remove opponent dots whose territory is already nonempty
    ↓
no newly capturable opponent dot remains?
    ├── yes → skip this region
    └── no  → store region and new opponent dots
    ↓
deduplicate and sort captured dot coordinates
    ↓
return CaptureInfo
```

The post-flood-fill territory filter prevents the same opponent dot from being
captured and scored again
Territory is used here as game state after geometry has already been decided

The returned value has the form

```python
CaptureInfo(
    captured_dots=((2, 2),),
    captured_regions=(((1, 1), (1, 2), (2, 1), (2, 2)),),
)
```

The exact order within a region follows flood-fill traversal order
`captured_dots` is deduplicated and sorted before it is stored

# `CaptureInfo`

`CaptureInfo` is a frozen dataclass with two tuple fields

```python
@dataclass(frozen=True)
class CaptureInfo:
    captured_dots: tuple
    captured_regions: tuple
```

`captured_dots` contains newly captured opponent coordinates
`captured_regions` contains the full traversed board regions associated with
those captures

The convenience property is

```python
@property
def happened(self):
    return bool(self.captured_dots)
```

Python considers an empty tuple false

```python
captured_dots = ()
bool(captured_dots) is False
```

A nonempty tuple is true

```python
captured_dots = ((2, 3),)
bool(captured_dots) is True
```

Therefore, a capture happens only when at least one new opponent dot is present
An enclosed empty region alone does not make `happened` true

# Complete example move

Consider a `4 × 4` board
Player 1 has almost completed the boundary, Player 2 has a dot inside, and all
territory cells are initially empty

Before the move

```text
      col 0  1  2  3
row 0     ●  ●  ●  ●
row 1     ●  ·  ·  ●
row 2     ●  ·  ○  ·  ← last opening
row 3     ●  ●  ●  ●
```

Player 1 calls

```python
game.place_dot(2, 3, PLAYER_1)
```

## 1  Validate and update the board

`(2, 3)` is inside the board, empty on `board`, and empty in `territory`, so the
move is legal

The method writes Player 1's value at `last_move`

```text
      col 0  1  2  3
row 0     ●  ●  ●  ●
row 1     ●  ·  ·  ●
row 2     ●  ·  ○  ●  ← last_move
row 3     ●  ●  ●  ●
```

## 2  Check whether the move created a graph cycle candidate

The new dot's active Player 1 neighbors include dots on the right side and
bottom side of the existing path
Before `last_move`, those neighbors were already connected by the long active
path around the top, left, and bottom edges

```text
neighbor A ── top ── left ── bottom ── neighbor B
       \                                      /
        \-------------- last_move -----------/
```

At least two neighbors therefore have the same pre-move UnionFind root
`_could_have_closed_loop()` returns `True`

## 3  Find candidate seeds

The orthogonal neighbors of `(2, 3)` are checked
The cells above and below are Player 1 walls
The coordinate to the right is outside the board
The cell to the left, `(2, 2)`, is the Player 2 dot and becomes the flood-fill
seed

## 4  Flood-fill the region

Starting at `(2, 2)`, flood fill reaches

```python
{
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
}
```

The Player 2 dot at `(2, 2)` is traversable and is collected in
`opponent_cells`
Every route out of the region meets a Player 1 dot, so no visited cell lies on
the edge and `reaches_edge` is `False`

## 5  Build capture information

The region is nonempty, does not reach the edge, and contains an opponent dot
The opponent dot has empty territory, so it has not been scored before

The result is equivalent to

```python
CaptureInfo(
    captured_dots=((2, 2),),
    captured_regions=(
        ((2, 2), (2, 1), (1, 1), (1, 2)),
    ),
)
```

The region order shown here follows the current depth-first traversal for this
example

## 6  Update active connectivity

After detection, `place_dot()` adds `(2, 3)` to UnionFind and unions it with its
active same-player neighbors

## 7  Apply territory, score, and group changes

Because `capture.happened` is true, every previously uncaptured cell in the
returned region gets

```python
territory[cell] = PLAYER_1
```

This includes the captured opponent cell and the enclosed empty cells
The Player 2 dot remains present on `board`, but it is now inactive

Player 1's score increases by one

```python
score[PLAYER_1] += 1
```

Finally, UnionFind is rebuilt so the newly captured dot and all other cells with
nonempty territory are absent from active connectivity

`place_dot()` returns

```python
[(2, 2)]
```

# Why this architecture exists

UnionFind and flood fill solve different parts of capture detection

```text
UnionFind
    → cheap graph connectivity and cycle pre-check

flood fill
    → geometric region exploration and opponent collection
```

Most moves cannot close a graph cycle
For those moves, `_could_have_closed_loop()` returns `False` and the code avoids
allocating a `visited` array and exploring board regions

This split is useful whenever many moves are evaluated
It is especially suitable for later search systems such as Monte Carlo Tree
Search, where simulated moves may be applied repeatedly
The current modules implement the game and capture logic, not MCTS itself

# Important invariants

- `board` contains the physical dots that were placed

- `territory` is separate captured ownership and game-state information

- A legal move requires an empty board cell and empty territory

- Captured dots may remain visible on `board`

- Only active dots participate in active UnionFind groups

- An active dot has a player value on `board` and empty `territory`

- Player-dot connectivity is eight-directional

- Flood-fill connectivity is four-directional

- During flood fill, `board[cell] == player` is the only wall condition

- Territory does not act as a flood-fill wall

- Opponent dots are traversable and are collected during flood fill

- UnionFind does not validate ownership, adjacency, or board geometry

- `_could_have_closed_loop()` is a fast pre-check, not final capture detection

- `reaches_edge` describes region geometry and does not prove a new graph cycle

- `last_move` is central to cycle checking and candidate seed selection

- Empty enclosed regions are not captures under the current rules

- Already captured opponent dots are not scored again

- Provided groups for capture detection represent active connectivity before
  `last_move`

# Function reference

This table is a compact reference after the conceptual explanation above

| Function | Responsibility | Main input | Output |
| --- | --- | --- | --- |
| `get_neighbors(row, col, shape, include_diagonals=True)` | Return valid neighboring coordinates | A coordinate, board shape, and connectivity choice | `list` of `(row, col)` tuples |
| `UnionFind.add(cell)` | Register one coordinate as its own group when absent | A cell coordinate | `None` |
| `UnionFind.find(cell)` | Find and compress the path to the group's representative | A registered cell | Root coordinate, or `KeyError` for an unknown cell |
| `UnionFind.union(a, b)` | Merge two registered groups using rank | Two registered cells | `True` if merged, `False` if already connected |
| `UnionFind.connected(a, b)` | Compare the roots of two registered cells | Two registered cells | `bool` |
| `rebuild_groups(board, territory=None)` | Recreate eight-connected same-player groups from active dots | Board and optional territory | New `UnionFind` object |
| `DotsGame.is_legal_move(row, col)` | Check bounds, board emptiness, and territory emptiness | Row and column | `bool` |
| `DotsGame.legal_moves()` | Find every cell that is empty in both arrays | Current game state | `list` of coordinate tuples |
| `DotsGame.place_dot(row, col, player)` | Apply one move, detect and apply capture, update groups and score | Coordinate and player | `list` of newly captured opponent coordinates |
| `_could_have_closed_loop(board, territory, last_move, player, groups=None)` | Check whether `last_move` created an active graph cycle candidate | Post-move board, territory, move, player, optional pre-move groups | `bool` |
| `find_candidate_regions(board, territory, last_move, player)` | Find orthogonal non-player flood-fill seeds around `last_move` | Board state and move context | `list` of seed coordinates |
| `flood_fill_region(board, seed, player, visited)` | Explore one four-connected non-player region | Board, seed, moving player, shared visited array | `(region, reaches_edge, opponent_cells)` |
| `find_enclosed_regions(board, territory, last_move, player)` | Flood-fill candidate regions and retain enclosed regions containing opponents | Board state and move context | `list` of `(region, opponent_cells)` pairs |
| `detect_capture_info(board, last_move, player, territory=None, groups=None)` | Run validation, cycle pre-check, geometry, and duplicate-score filtering | Post-move board and optional game state | `CaptureInfo` |
| `detect_capture(board, last_move, player, territory=None, groups=None)` | Provide the simpler compatibility wrapper | Same capture inputs | `list` of captured opponent coordinates |
| `render_board(board, territory=None, colorize=False)` | Render dots and captured empty cells without mutating state | Board, optional territory, color choice | Multiline string |
