import os, shutil, pathlib, sys

# --- START DATA LOADING
# Add temporary directory above current ml directory to the Path
# Possible to run main_ml.py as regular file
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_loader import load_training_data

(train_samples, train_labels), (val_samples, val_labels) = load_training_data(
    "training_data/_game_arena/5x5"
)
pass
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
inputs = keras.Input(shape=(5, 5, 7))
# Rescales inputs to the [0, 1] range by dividing them by 255
# TODO - probably not needed
# x = layers.Rescaling(1.0 / 255)(inputs)

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
# x = layers.MaxPooling2D(pool_size=2)(x)

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

history = model.fit(
    train_samples,
    train_labels,
    epochs=50,
    validation_data=(val_samples, val_labels),
)

pass
