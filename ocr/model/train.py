"""
Sudoku digit CNN — training script.

Setup
-----
1. Populate ocr/model/dataset/ with one subfolder per class:
       dataset/0/   ← images of blank cells
       dataset/1/   ← images of the digit 1
       ...
       dataset/9/   ← images of the digit 9

   Each image should be a grayscale PNG/JPEG of a single cell.
   Aim for at least ~200 images per class.

2. Install dependencies (if not already):
       pip install tensorflow opencv-python numpy pillow

3. From the project root, run:
       python -m ocr.model.train
"""

from __future__ import annotations

import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "sudoku_digit_cnn.h5")
IMG_SIZE    = 28
BATCH_SIZE  = 32
EPOCHS      = 20


def build_model() -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset():
    X, y = [], []
    for label in range(10):
        folder = os.path.join(DATASET_DIR, str(label))
        if not os.path.isdir(folder):
            print(f"  Warning: {folder} not found — skipping class {label}.")
            continue
        found = 0
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
            found += 1
        print(f"  Class {label}: {found} images")
    return np.array(X, dtype="float32") / 255.0, np.array(y, dtype="int32")


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    if len(X) == 0:
        print("\nNo images found. Populate ocr/model/dataset/0 through ocr/model/dataset/9 first.")
        return

    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print(f"\nTotal samples: {len(X)}  |  Classes present: {sorted(set(y.tolist()))}")

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    datagen = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.10,
        brightness_range=[0.8, 1.2],
    )

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
    )

    model.save(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
