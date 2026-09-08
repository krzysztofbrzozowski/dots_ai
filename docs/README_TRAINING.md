```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)

# Value head: reduce channel depth but keep all 10x10 cell locations.
x = layers.Conv2D(filters=8, kernel_size=1, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(3, activation="softmax")(x)
```

### 1893 training data -> 1 layer, batch_size=default(32),
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
```
![alt text](imgs/training_history_1_layer_8depth_1893.png)

### 553 training data -> 1 layer, batch_size=default(32),
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
```
![alt text](imgs/training_history_1_layer_8depth_5553.png)

### 64322 training data -> 1 layer, batch_size=4096, epochs=20
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)
```
![alt text](imgs/training_history_1_layer_32depth_64322.png)
Test loss: 0.792
Test accuracy: 0.596

### 64322 training data -> 2 layers, batch_size=4096, epochs=20
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)
```
![alt text](imgs/training_history_2_layers_32_64_depth_64322.png)
Test loss: 0.789
Test accuracy: 0.601

### 64322 training data -> 3 layers, batch_size=4096, epochs=20
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)
```
![alt text](imgs/training_history_3_layers_32_64_128_depth_64322.png)
Test loss: 0.793
Test accuracy: 0.596

### 64322 training data -> 2 layers, batch_size=4096, epochs=20
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)
```
![alt text](imgs/training_history_2_layers_32_64depth_64322_augumentation.png)
Test loss: 0.778
Test accuracy: 0.606

### 64322 training data -> many layers, batch_size=4096, epochs=30
```python
# Input channels: my dots, opponent dots, my territory, opponent territory,
# legal moves, my score, and opponent score. Three residual blocks give the
# network a board-wide receptive field without reducing the 10x10 grid.
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# Residual block 1
residual = x
x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Activation("relu")(x)

# Residual block 2
residual = x
x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Activation("relu")(x)

# Residual block 3
residual = x
x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(
    filters=32,
    kernel_size=3,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Add()([x, residual])
x = layers.Activation("relu")(x)

# Value head: reduce channel depth but keep all 10x10 cell locations.
x = layers.Conv2D(
    filters=8,
    kernel_size=1,
    padding="same",
    use_bias=False,
)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
```
![alt text](imgs/training_history_10x10_64322_residual_d4.png)
Test loss: 0.829
Test accuracy: 0.574

### 64322 training data -> many layers, batch_size=4096, epochs=30
```python
inputs = keras.Input(shape=(10, 10, 7))

x = layers.Conv2D(filters=32, kernel_size=2, use_bias=False)(inputs)

# We apply a series of convolutional blocks with increasing feature
# depth. Each block consists of two batch-normalized depthwise
# separable convolution layers and a max pooling layer, with a residual
# connection around the entire block.
for size in [8, 16]:
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(
        size, 1, strides=2, padding="same", use_bias=False
    )(residual)
    x = layers.add([x, residual])

# In the original model, we used a Flatten layer before the Dense
# layer. Here, we go with a GlobalAveragePooling2D layer.
x = layers.GlobalAveragePooling2D()(x)
# Like in the original model, we add a dropout layer for
# regularization.
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation="softmax")(x)
```
![alt text](imgs/training_history_10x10_64322_new_model.png)
Test loss: 0.781
Test accuracy: 0.603

## Value-model data relabeling plan

### Why the next experiment focuses on data

The tested model architectures consistently plateau near 60% test accuracy.
Increasing convolutional depth, adding residual connections, and applying D4
augmentation produced only small changes. This suggests that the primary
limitation is the training target and data-generation process rather than model
capacity.

The legacy polarized collection is not evidence that the model learned a
general board-value function. It contains 326 games in which Player -1 had a
much larger MCTS budget and won every game. That setup makes it possible to
identify the stronger player from move parity, score, or playing style.

The current 64,322-game collection is more balanced, but its MCTS searches are
very weak: a sample of 2,000 games had a median of approximately eight rollouts
per searched move and three rollouts during the first 20 moves. Each position
then receives the final result of one noisy continuation. The collection also
mixes equal, Player +1 advantage, and Player -1 advantage matchups without
providing future search strength as a model input. Consequently, identical
positions can legitimately have conflicting labels.

### Terminology

- **D0**: the current collection of approximately 6.31 million positions with
  their original terminal-game labels.
- **V0**: the current trained value model. It is retained as a baseline but is
  not used to create new labels.
- **D0_clean**: a smaller, diverse subset of D0 whose positions are reevaluated
  by a stronger classical MCTS.
- **V1**: a new value model trained from scratch on D0_clean.
- **D1**: an optional later dataset generated by self-play using V1 together
  with MCTS.

### Stage 1: preserve V0 as the baseline

Do not delete V0, but do not use it as a teacher. Record its checkpoint,
configuration, overall metrics, per-game-phase metrics, and draw recall. This
provides the baseline needed to determine whether better labels improve the
model.

Training V1 from scratch is preferred for the first comparison. Warm-starting
from V0 could make training faster, but it would make the effect of relabeling
harder to isolate.

### Stage 2: select positions from D0

Use D0 as a pool of valid board states rather than accepting all original
labels. Build a substantially smaller candidate set by:

1. grouping repetitions of the same `collection-gNNNNNN` plan so correlated
   trajectories cannot cross dataset splits;
2. selecting only a small number of positions from each game;
3. sampling evenly across early, middle, and late game phases;
4. covering different score differences and board situations;
5. removing exact duplicate positions;
6. canonicalizing the eight D4 rotations and reflections so equivalent square
   positions are evaluated only once.

Random opening positions remain valid value-model inputs. The important change
is that their value will come from a common teacher rather than from the single
weak game in which they originally appeared.

Create training and validation partitions by group. Reserve a completely
independent test set generated with a different seed or collection run.

### Stage 3: relabel the selected positions with classical MCTS

Evaluate every selected position with the same stronger classical MCTS
configuration. Prefer a fixed simulation count over a wall-clock time budget so
that label quality does not depend on machine load or game phase.

V0 must not participate in this search. The objective is to construct an
independent teacher target rather than allow the current model to reproduce its
own errors.

Before implementation, choose one target definition and use it consistently:

- a scalar value such as expected outcome in `[-1, 1]`; or
- soft loss/draw/win probabilities estimated from multiple continuations.

For example, if ten continuations produce three losses, one draw, and six wins,
the soft target is:

```text
[loss=0.3, draw=0.1, win=0.6]
```

The equivalent scalar value is `0.6 - 0.3 = 0.3`. Existing `q_values` and
`visit_counts` may help define a scalar search estimate, but their current
3-8-rollout searches are too weak to serve as clean labels.

The result of this stage is D0_clean: existing board states paired with stronger
and consistent teacher evaluations. Merely filtering D0 while retaining its
original terminal labels would not address the main source of noise.

### Stage 4: train and evaluate V1

Train V1 from random initialization on D0_clean. Initially keep the model
architecture and optimization settings fixed so the experiment measures the
effect of label quality rather than combining several changes.

Compare V1 with V0 using:

- loss and accuracy on the independent test set;
- metrics split by game phase;
- loss, draw, and win recall;
- calibration or Brier score for probabilistic targets;
- correlation or mean error against the strong MCTS teacher;
- inference speed compared with running the teacher MCTS directly.

Top-1 outcome accuracy is not the only success criterion. A calibrated value
estimate can be useful to MCTS even when a single stochastic game result is
not predictable.

### Stage 5: optionally generate D1 with V1 and MCTS

Only after V1 demonstrates a clear improvement should it be used to guide new
self-play. In that later stage, MCTS uses V1 to evaluate leaves more cheaply,
plays complete games, and records the positions that a stronger agent actually
visits.

```text
D0 + noisy terminal labels
        |
        v
select and deduplicate positions
        |
        v
strong classical MCTS relabeling
        |
        v
D0_clean
        |
        v
train from scratch
        |
        v
V1
        |
        v
V1-guided MCTS self-play
        |
        v
D1
        |
        v
strong relabeling and training on D0_clean + D1_clean
        |
        v
V2
```

D1 addresses a limitation that relabeling alone cannot solve: D0 positions were
generated by weak play and may not match the state distribution encountered by
a stronger agent. Retain exploration, independent evaluation, and older clean
data to reduce the risk of V1 reinforcing its own mistakes.

### Decision point: direct MCTS or a value model

Strong MCTS should remain the reference baseline. If it already meets the
required playing strength and latency, using it directly is simpler than
maintaining a neural model. The value model is useful when the teacher MCTS is
too expensive at runtime: expensive search is performed offline once, and the
trained model amortizes that cost by providing cheap leaf evaluations during
future searches.

### Immediate next experiment

The next experiment is therefore:

```text
D0
-> select, group, stratify, and D4-deduplicate positions
-> relabel them with stronger fixed-count classical MCTS
-> create D0_clean
-> train V1 from scratch
-> compare V1 with the preserved V0 baseline
```

Do not generate D1 and do not use V0 as a teacher until this experiment shows
that stronger, consistent labels materially improve value prediction.
