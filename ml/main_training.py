import pathlib
import sys

import matplotlib.pyplot as plt

# --- START DATA LOADING
# Add temporary directory above current ml directory to the Path
# Possible to run main_training.py as regular file
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "10x10_083769_dual_head_v1.keras"
)
HISTORY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "imgs"
    / "training_history_10x10_083769_dual_head_v1.png"
)
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_loader import load_dual_head_training_data
from ml.training_pipeline import (
    batched_array_dataset,
    random_dual_head_d4_augmentation,
)


DATA_DIRECTORY = (
    PROJECT_ROOT / "training_data" / "_game_arena_data_collection" / "10x10_083769"
)
VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.1
DATA_SEED = 42

(
    (train_inputs, train_targets, train_sample_weights),
    (validation_inputs, validation_targets, validation_sample_weights),
    (test_inputs, test_targets, test_sample_weights),
) = load_dual_head_training_data(
    DATA_DIRECTORY,
    validation_fraction=VALIDATION_FRACTION,
    test_fraction=TEST_FRACTION,
    seed=DATA_SEED,
)

# --- ADD DATA AUGMENTATION
import tensorflow as tf

BATCH_SIZE = 2048
EPOCHS = 50

# Keep array-to-tensor conversion batch-sized so the large NumPy dataset is not
# duplicated in memory. Shuffling happens again whenever a new epoch starts.
train_dataset = batched_array_dataset(
    train_inputs,
    train_targets,
    BATCH_SIZE,
    sample_weights=train_sample_weights,
    shuffle=True,
    seed=DATA_SEED,
)

# A square Dots board has exactly eight valid spatial symmetries: four
# rotations, with and without a reflection. Unlike arbitrary image rotation or
# zoom, these operations preserve discrete cells and always produce legal
# board representations. The policy map is transformed by the exact same
# symmetry as the board; the value target and scalar scores stay unchanged.
# TODO Skip augumentation for now
# augmented_train_dataset = train_dataset.map(
#     random_dual_head_d4_augmentation,
#     num_parallel_calls=tf.data.AUTOTUNE,
# ).prefetch(tf.data.AUTOTUNE)
training_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

validation_dataset = batched_array_dataset(
    validation_inputs,
    validation_targets,
    BATCH_SIZE,
    sample_weights=validation_sample_weights,
).prefetch(tf.data.AUTOTUNE)

test_dataset = batched_array_dataset(
    test_inputs,
    test_targets,
    BATCH_SIZE,
    sample_weights=test_sample_weights,
).prefetch(tf.data.AUTOTUNE)
# --- END DATA AUGMENTATION
# --- END DATA LOADING
# --- START MODEL DEFINITION
import keras
from keras import layers

board_inputs = keras.Input(shape=(10, 10, 5), name="board")
score_inputs = keras.Input(shape=(2,), name="scores")

# Broadcast the two normalized score scalars over the whole board so both the
# policy and value heads can condition their predictions on the current score.
score_planes = layers.Reshape((1, 1, 2), name="score_reshape")(score_inputs)
score_planes = layers.UpSampling2D(
    size=(10, 10),
    interpolation="nearest",
    name="score_planes",
)(score_planes)
x = layers.Concatenate(name="board_with_scores")(
    [board_inputs, score_planes]
)

# Keep the 10x10 spatial resolution throughout the shared trunk. With one stem
# convolution and four two-convolution residual blocks, every output cell has a
# receptive field large enough to use the whole board.
x = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name="stem_conv",)(x)
x = layers.BatchNormalization(name="stem_batch_norm")(x)
x = layers.Activation("relu", name="stem_relu")(x)

for block_index in range(4):
    residual = x

    y = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name=f"trunk_{block_index}_conv_1",)(x)
    y = layers.BatchNormalization(name=f"trunk_{block_index}_batch_norm_1")(y)
    y = layers.Activation("relu", name=f"trunk_{block_index}_relu_1",)(y)

    y = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, name=f"trunk_{block_index}_conv_2",)(y)
    y = layers.BatchNormalization(name=f"trunk_{block_index}_batch_norm_2")(y)

    x = layers.Add(name=f"trunk_{block_index}_residual")([residual, y])
    x = layers.Activation("relu", name=f"trunk_{block_index}_relu_2",)(x)

# Policy head: one raw logit per board cell. Illegal actions are deliberately
# not masked inside the model; callers must apply the exact legal-action mask
# before softmax. Raw logits keep the training loss numerically stable.
policy = layers.Conv2D(filters=32, kernel_size=1, use_bias=False, name="policy_conv",)(x)
policy = layers.BatchNormalization(name="policy_batch_norm")(policy)
policy = layers.Activation("relu", name="policy_relu")(policy)
policy = layers.Conv2D(filters=1, kernel_size=1, name="policy_map",)(policy)
policy_logits = layers.Reshape((100,), name="policy")(policy)

# Value head: loss/draw/win probabilities from the current player's
# perspective, matching the existing labels produced by data_loader.py.
value = layers.Conv2D(filters=8, kernel_size=1, use_bias=False, name="value_conv",)(x)
value = layers.BatchNormalization(name="value_batch_norm")(value)
value = layers.Activation("relu", name="value_relu")(value)
value = layers.GlobalAveragePooling2D(name="value_pool")(value)
value = layers.Dense(64, activation="relu", name="value_dense")(value)
value = layers.Dropout(0.2, name="value_dropout")(value)
value_probabilities = layers.Dense(3, activation="softmax", name="value",)(value)

model = keras.Model(
    inputs=(board_inputs, score_inputs),
    outputs={
        "policy": policy_logits,
        "value": value_probabilities,
    },
    name="dots_dual_head_v1",
)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0003,
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
    # Weighted metrics ignore the zero-weight policy targets associated with
    # random opening positions that did not run MCTS.
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

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
history = model.fit(
    # augmented_train_dataset,
    #TODO: for now not using augumented dataset
    training_dataset,
    epochs=EPOCHS,
    # The generator already reshuffles training indices at every epoch.
    shuffle=False,
    validation_data=validation_dataset,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ],
)
# --- END MODEL DEFINITION

# --- START MODEL VERIFICATION
value_accuracy = history.history["value_accuracy"]
val_value_accuracy = history.history["val_value_accuracy"]
policy_top1 = history.history["policy_top1"]
val_policy_top1 = history.history["val_policy_top1"]
policy_top5 = history.history["policy_top5"]
val_policy_top5 = history.history["val_policy_top5"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
value_loss = history.history["value_loss"]
val_value_loss = history.history["val_value_loss"]
epochs = range(1, len(value_accuracy) + 1)

fig, (value_ax, policy_ax, loss_ax, value_loss_ax) = plt.subplots(
    1,
    4,
    figsize=(24, 5),
)
value_ax.plot(epochs, value_accuracy, "r--", label="Training")
value_ax.plot(epochs, val_value_accuracy, "b", label="Validation")
value_ax.set_title("Value accuracy")
value_ax.set_xlabel("Epoch")
value_ax.set_ylabel("Accuracy")
value_ax.legend()

policy_ax.plot(epochs, policy_top1, "r--", label="Training top-1")
policy_ax.plot(epochs, val_policy_top1, "r", label="Validation top-1")
policy_ax.plot(epochs, policy_top5, "b--", label="Training top-5")
policy_ax.plot(epochs, val_policy_top5, "b", label="Validation top-5")
policy_ax.set_title("Policy accuracy")
policy_ax.set_xlabel("Epoch")
policy_ax.set_ylabel("Accuracy")
policy_ax.legend()

loss_ax.plot(epochs, loss, "r--", label="Training loss")
loss_ax.plot(epochs, val_loss, "b", label="Validation loss")
loss_ax.set_title("Training and validation loss")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Loss")
loss_ax.legend()

value_loss_ax.plot(epochs, value_loss, "r--", label="Training value loss")
value_loss_ax.plot(epochs, val_value_loss, "b", label="Validation value loss")
value_loss_ax.set_title("Value loss")
value_loss_ax.set_xlabel("Epoch")
value_loss_ax.set_ylabel("Loss")
value_loss_ax.legend()
fig.tight_layout()
fig.savefig(HISTORY_PATH)
# --- END MODEL VERIFICATION

# --- START TEST EVALUATION
test_model = keras.models.load_model(MODEL_PATH)
test_metrics = test_model.evaluate(test_dataset, return_dict=True)
for metric_name, metric_value in sorted(test_metrics.items()):
    print(f"Test {metric_name}: {metric_value:.3f}")
# --- END TEST EVALUATION

# Display after evaluation so closing the plot is not required to see test results.
plt.show()

pass
