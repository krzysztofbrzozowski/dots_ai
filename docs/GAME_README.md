# UnionFind
- structure:
    ._parent = {}
    ._rank = {}


```python
# --- Step 1: the dot becomes its own root
groups = UnionFind()

groups.add((2, 3))

# After add():
#
# _parent = {
#     (2, 3): (2, 3)
# }
# _rank = {
#     (2, 3): 0
# }
#
# (2, 3) points to itself, so it is the root of its own group

# --- Step 2: another dot becomes its root
groups.add((2, 4))
# After add():
#
# _parent = {
#     (2, 3): (2, 3),
#     (2, 4): (2, 4)
# }
#
# _rank = {
#     (2, 3): 0,
#     (2, 4): 0
# }
#

# First call of union
groups.union((2, 3), (2, 4))
    # Inside there is call find
    groups.find((2, 4))

    # Flow:
    # root = (2, 4)
    # no while loop will work here
    # -> return 2,4

# After union():
#
# _parent = {
#     (2, 3): (2, 3),
#     (2, 4): (2, 3)
# }

# Later call of find()
groups.find((2, 4))

    # Flow:
    # root = (2, 4)
    # parent = _parent[(2, 4)] = (2, 3)
    # parent != root
    # -> move to parent
    #
    # root = (2, 3)
    # parent = _parent[(2, 3)] = (2, 3)
    # parent == root
    # -> stop
    #
    # return (2, 3)
```

# place_dot()
```python
self.board[last_move] = player

# last_move = (2, 4)
# player = PLAYER_1  # 1


#    board  = state AFTER the move
#    groups = state BEFORE the move
capture = detect_capture_info(
    self.board,
    last_move,
    player,
    territory=self.territory,
    groups=self.groups,
)

# Example 1:
# roots = [(1, 1), (1, 1)]
#
# len(roots) = 2
# len(set(roots)) = 1
#
# 1 < 2 -> True
# At least two neighboring dots have the same root,
# so they already belong to the same connected group


# Example 2:
# roots = [(1, 1), (3, 4)]
#
# len(roots) = 2
# len(set(roots)) = 2
#
# 2 < 2 -> False
# The neighboring dots have different roots,
# so they belong to different connected groups
capture = _could_have_closed_loop()
# If _could_have_closed_loop(...) == False
#   -> Return empty CaptureInfo class
#   CaptureInfo(captured_dots=(), captured_regions=()

# Candidate regions around the newly placed dot:
#
#      seed seed seed
#         ↓    ↓    ↓
#      .    .    .
#      ●    ●    ●
#           ↑
#       last_move
#      .    .    .
#         ↑    ↑    ↑
#      seed seed seed
#
# Player dots form the wall
# The surrounding cells become flood-fill seeds
# used to check whether they belong to an enclosed region
find_enclosed_regions()


```