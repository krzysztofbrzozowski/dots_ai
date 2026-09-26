"""Train the 30x30 dual-head model from filtered new_data games."""

from pathlib import Path
import sys

import keras
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_loader import count_game_positions, split_game_paths
from ml.model import build_dual_head_model, compile_dual_head_model
from ml.training_pipeline import streaming_dual_head_npz_dataset


DATA_DIRECTORY = (
    PROJECT_ROOT
    / "training_data"
    / "new_data"
    / "npz"
    / "30x30_filtered"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "30x30_090507_new_data_dual_head_v1_test.keras"
)
HISTORY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "imgs"
    / "training_history_30x30_090507_new_data_dual_head_v1_test.png"
)

BOARD_SHAPE = (30, 30)
VALIDATION_FRACTION = 0.05
TEST_FRACTION = 0.05
DATA_SEED = 42
BATCH_SIZE = 2048
EPOCHS = 7
FILE_WORKERS = 10

# The machine has 48 GB of unified memory. About 30 GiB of training tensors,
# together with the model, TensorFlow, and macOS allocations, should place the
# observed total near 42-45 GB without changing the optimizer batch size.
SHUFFLE_BUFFER_MEMORY_GIB = 32
BYTES_PER_POSITION = (
    # Board input: 30 x 30 cells x 5 channels x 4 bytes (float32).
    # The channels are my dots, opponent dots, my territory,
    # opponent territory, and the legal-move mask.
    BOARD_SHAPE[0] * BOARD_SHAPE[1] * 5 * 4
    # Score input: my score and the opponent's score as two float32 values.
    + 2 * 4
    # Policy target: one float32 value for each of the 30 x 30 actions.
    + BOARD_SHAPE[0] * BOARD_SHAPE[1] * 4
    # Value target: one int64 class (0=loss, 1=draw, 2=win).
    + 8
    # Sample weights: one float32 weight for policy and one for value.
    + 2 * 4
)
SHUFFLE_BUFFER_SIZE = int(
    SHUFFLE_BUFFER_MEMORY_GIB * 1024**3 / BYTES_PER_POSITION
)


def create_datasets():
    """Split by complete games and create bounded-memory input pipelines."""
    # Shuffle and split the data paths -> training, validataion and test ones
    training_paths, validation_paths, test_paths = split_game_paths(
        DATA_DIRECTORY,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=DATA_SEED,
    )
    print(
        "Game split: "
        f"{len(training_paths):,} training, "
        f"{len(validation_paths):,} validation, "
        f"{len(test_paths):,} test"
    )
    print(
        "Training shuffle buffer: "
        f"{SHUFFLE_BUFFER_SIZE:,} positions "
        f"(~{SHUFFLE_BUFFER_MEMORY_GIB} GiB tensor payload)"
    )

    # Calculate amount of training positions/sets stored in all *.npz files
    # Done parallel using FILE_WORKERS to know how long the batch will be processed during training (used later in streaming_dual_head_npz_dataset)
    # e.g. calcuation will look like this
    #   training_position_count -> 1 000 000
    #   BATCH_SIZE              -> 2048
    #   batch count             -> ceil(1 000 000 / 2048) = 489
    #
    #   during training     -> 123/489 batches
    #                       -> time needed for one epoch can be calculateds
    #                       -> (in streaming_dual_head_npz_dataset) 12/5613 ━━━━━━━━━━━━━━━━━━━━ 2:19:58 1s/step - ...
    print("Counting positions for exact epoch progress...")
    training_position_count = count_game_positions(
        training_paths,
        worker_count=FILE_WORKERS,
    )
    validation_position_count = count_game_positions(
        validation_paths,
        worker_count=FILE_WORKERS,
    )
    test_position_count = count_game_positions(
        test_paths,
        worker_count=FILE_WORKERS,
    )
    print(
        "Position split: "
        f"{training_position_count:,} training, "
        f"{validation_position_count:,} validation, "
        f"{test_position_count:,} test"
    )

    training_dataset = streaming_dual_head_npz_dataset(
        training_paths,
        BOARD_SHAPE,
        BATCH_SIZE,
        position_count=training_position_count,
        shuffle=True,
        seed=DATA_SEED,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        file_workers=FILE_WORKERS,
    )
    validation_dataset = streaming_dual_head_npz_dataset(
        validation_paths,
        BOARD_SHAPE,
        BATCH_SIZE,
        position_count=validation_position_count,
        file_workers=FILE_WORKERS,
    )
    test_dataset = streaming_dual_head_npz_dataset(
        test_paths,
        BOARD_SHAPE,
        BATCH_SIZE,
        position_count=test_position_count,
        file_workers=FILE_WORKERS,
    )
    return training_dataset, validation_dataset, test_dataset


def save_training_history(history):
    """Save value, policy, and loss curves from one training run."""

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

    figure, (value_ax, policy_ax, loss_ax, value_loss_ax) = plt.subplots(
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

    value_loss_ax.plot(epochs, value_loss, "r--", label="Training")
    value_loss_ax.plot(epochs, val_value_loss, "b", label="Validation")
    value_loss_ax.set_title("Value loss")
    value_loss_ax.set_xlabel("Epoch")
    value_loss_ax.set_ylabel("Loss")
    value_loss_ax.legend()

    figure.tight_layout()
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(HISTORY_PATH)
    return figure


def main():
    training_dataset, validation_dataset, test_dataset = create_datasets()

    model = build_dual_head_model(BOARD_SHAPE)
    compile_dual_head_model(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        training_dataset,
        epochs=EPOCHS,
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

    figure = save_training_history(history)
    test_model = keras.models.load_model(MODEL_PATH)
    test_metrics = test_model.evaluate(test_dataset, return_dict=True)
    for metric_name, metric_value in sorted(test_metrics.items()):
        print(f"Test {metric_name}: {metric_value:.3f}")

    plt.show()
    return figure


if __name__ == "__main__":
    main()
