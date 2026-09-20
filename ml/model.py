"""Build the project's residual dual-head policy/value network."""

import keras
from keras import layers


def build_dual_head_model(
    board_shape,
    *,
    name="dots_dual_head_human_sgf_v1",
):
    """Build the existing four-block model for a specific board shape."""

    rows, columns = (int(value) for value in board_shape)
    if rows <= 0 or columns <= 0:
        raise ValueError("board dimensions must be positive")

    board_inputs = keras.Input(shape=(rows, columns, 5), name="board")
    score_inputs = keras.Input(shape=(2,), name="scores")

    score_planes = layers.Reshape((1, 1, 2), name="score_reshape")(score_inputs)
    score_planes = layers.UpSampling2D(size=(rows, columns), interpolation="nearest", name="score_planes")(score_planes)
    x = layers.Concatenate(name="board_with_scores")([board_inputs, score_planes])

    x = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_batch_norm")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    for block_index in range(4):
        residual = x

        y = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name=f"trunk_{block_index}_conv_1")(x)
        y = layers.BatchNormalization(name=f"trunk_{block_index}_batch_norm_1")(y)
        y = layers.Activation("relu", name=f"trunk_{block_index}_relu_1")(y)

        y = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name=f"trunk_{block_index}_conv_2")(y)
        y = layers.BatchNormalization(name=f"trunk_{block_index}_batch_norm_2")(y)

        x = layers.Add(name=f"trunk_{block_index}_residual")([residual, y])
        x = layers.Activation("relu", name=f"trunk_{block_index}_relu_2")(x)

    policy = layers.Conv2D(filters=32, kernel_size=1, use_bias=False, name="policy_conv")(x)
    policy = layers.BatchNormalization(name="policy_batch_norm")(policy)
    policy = layers.Activation("relu", name="policy_relu")(policy)
    policy = layers.Conv2D(filters=1, kernel_size=1, name="policy_map")(policy)
    policy_logits = layers.Reshape((rows * columns,), name="policy")(policy)

    value = layers.Conv2D(filters=8, kernel_size=1, use_bias=False, name="value_conv")(x)
    value = layers.BatchNormalization(name="value_batch_norm")(value)
    value = layers.Activation("relu", name="value_relu")(value)
    value = layers.GlobalAveragePooling2D(name="value_pool")(value)
    value = layers.Dense(64, activation="relu", name="value_dense")(value)
    value = layers.Dropout(0.2, name="value_dropout")(value)
    value_probabilities = layers.Dense(3, activation="softmax", name="value")(value)

    return keras.Model(
        inputs=(board_inputs, score_inputs),
        outputs={
            "policy": policy_logits,
            "value": value_probabilities,
        },
        name=name,
    )


def compile_dual_head_model(model, *, learning_rate=0.0003):
    """Compile a dual-head model with the existing losses and metrics."""

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss={
            "policy": keras.losses.CategoricalCrossentropy(from_logits=True),
            "value": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "policy": 1.0,
            "value": 1.0,
        },
        weighted_metrics={
            "policy": [
                keras.metrics.CategoricalAccuracy(name="top1"),
                keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "value": [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        },
    )
    return model
