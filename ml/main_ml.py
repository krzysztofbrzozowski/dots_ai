import pathlib
import sys

import matplotlib.pyplot as plt

# --- START DATA LOADING
# Add temporary directory above current ml directory to the Path
# Possible to run main_ml.py as regular file
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "value_model.keras"
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_loader import load_training_data

(
    (train_samples, train_labels),
    (val_samples, val_labels),
    (test_samples, test_labels),
) = load_training_data(
    PROJECT_ROOT / "training_data" / "_game_arena" / "10x10",
    validation_fraction=0.2,
    test_fraction=0.1,
)
# --- END DATA LOADING
# --- START MODEL DEFINITION
import keras
from keras import layers
# Input data:
#   -> my dots (current player)
#   -> opponet dots
#   -> my territory (current player)
#   -> opponent territory
#   -> legal mask
#   -> my socre
#   -> opponet score
inputs = keras.Input(shape=(10, 10, 7))
# Rescales inputs to the [0, 1] range by dividing them by 255
# TODO - probably not needed
# x = layers.Rescaling(1.0 / 255)(inputs)

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# x = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
# # x = layers.MaxPooling2D(pool_size=2)(x)

# x = layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same")(x)

# Converts the 3D activations with shape (7, 7, 512) into a 1D vector
# with shape (512,) by averaging each feature map over its spatial dimensions

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
history = model.fit(
    train_samples,
    train_labels,
    epochs=50,
    validation_data=(val_samples, val_labels),
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        )
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
fig.savefig(MODEL_PATH.parent / "training_history.png")
# --- END MODEL VERIFICATION

# --- START TEST EVALUATION
test_model = keras.models.load_model(MODEL_PATH)
test_loss, test_acc = test_model.evaluate(test_samples, test_labels)
print(f"Test loss: {test_loss:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
# --- END TEST EVALUATION

# Display after evaluation so closing the plot is not required to see test results.
plt.show()

pass