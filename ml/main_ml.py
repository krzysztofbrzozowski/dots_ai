import pathlib
import sys

import matplotlib.pyplot as plt

# --- START DATA LOADING
# Add temporary directory above current ml directory to the Path
# Possible to run main_ml.py as regular file
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "10x10_64322_5conv_d4_normalized.keras"
)
HISTORY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "imgs"
    / "training_history_10x10_64322_5conv_d4_normalized.png"
)
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_loader import load_training_data
from ml.training_pipeline import batched_array_dataset, random_d4_augmentation

(
    (train_inputs, train_labels),
    (val_inputs, val_labels),
    (test_inputs, test_labels),
) = load_training_data(
    PROJECT_ROOT / "training_data" / "_game_arena_data_collection" / "10x10_64322",
    validation_fraction=0.2,
    test_fraction=0.1,
)

# --- ADD DATA AUGMENTATION
import tensorflow as tf

BATCH_SIZE = 4096
EPOCHS = 30

# Keep array-to-tensor conversion batch-sized so the large NumPy dataset is not
# duplicated in memory. Shuffling happens again whenever a new epoch starts.
train_dataset = batched_array_dataset(
    train_inputs,
    train_labels,
    BATCH_SIZE,
    shuffle=True,
    seed=42,
)

# A square Dots board has exactly eight valid spatial symmetries: four
# rotations, with and without a reflection. Unlike arbitrary image rotation or
# zoom, these operations preserve discrete cells and always produce legal
# board representations. Targets do not change under these symmetries.
augmented_train_dataset = train_dataset.map(
    random_d4_augmentation,
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

validation_dataset = batched_array_dataset(
    val_inputs,
    val_labels,
    BATCH_SIZE,
).prefetch(tf.data.AUTOTUNE)

test_dataset = batched_array_dataset(
    test_inputs,
    test_labels,
    BATCH_SIZE,
).prefetch(tf.data.AUTOTUNE)
# --- END DATA AUGMENTATION
# --- END DATA LOADING
# --- START MODEL DEFINITION
import keras
from keras import layers

# ### START PREV MODEL
# inputs = keras.Input(shape=(10, 10, 7))
# # Rescales inputs to the [0, 1] range by dividing them by 255
# # TODO - probably not needed
# # x = layers.Rescaling(1.0 / 255)(inputs)

# x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
# x = layers.MaxPooling2D(pool_size=2)(x)

# # x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
# # x = layers.MaxPooling2D(pool_size=2)(x)

# # x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
# # x = layers.MaxPooling2D(pool_size=2)(x)

# # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
# # # x = layers.MaxPooling2D(pool_size=2)(x)

# # x = layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)

# # Converts the 3D activations with shape (7, 7, 512) into a 1D vector
# # with shape (512,) by averaging each feature map over its spatial dimensions

# x = layers.GlobalAveragePooling2D()(x)
# outputs = layers.Dense(3, activation="softmax")(x)
### END PREV MODEL

board_inputs = keras.Input(shape=(10, 10, 5), name="board")
score_inputs = keras.Input(shape=(2,), name="scores")

x = layers.Conv2D(filters=32, kernel_size=2, use_bias=False)(board_inputs)

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
x = layers.Concatenate()([x, score_inputs])
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=(board_inputs, score_inputs), outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0003,
        clipnorm=1.0,
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
history = model.fit(
    augmented_train_dataset,
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
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

fig, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 5))
accuracy_ax.plot(epochs, accuracy, "r--", label="Training accuracy")
accuracy_ax.plot(epochs, val_accuracy, "b", label="Validation accuracy")
accuracy_ax.set_title("Training and validation accuracy")
accuracy_ax.set_xlabel("Epoch")
accuracy_ax.set_ylabel("Accuracy")
accuracy_ax.legend()

loss_ax.plot(epochs, loss, "r--", label="Training loss")
loss_ax.plot(epochs, val_loss, "b", label="Validation loss")
loss_ax.set_title("Training and validation loss")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Loss")
loss_ax.legend()
fig.tight_layout()
fig.savefig(HISTORY_PATH)
# --- END MODEL VERIFICATION

# --- START TEST EVALUATION
test_model = keras.models.load_model(MODEL_PATH)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test loss: {test_loss:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
# --- END TEST EVALUATION

# Display after evaluation so closing the plot is not required to see test results.
plt.show()

pass
