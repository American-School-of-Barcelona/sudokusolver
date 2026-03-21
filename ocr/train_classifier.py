"""
Train a handwritten-digit classifier on the client's handwriting samples.

Saves the trained model to ocr/digit_model.pkl.

Run once from the project root:
    python -m ocr.train_classifier
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = pathlib.Path(__file__).parent
TRAINING_DIR  = _HERE / "training_data"
MODEL_PATH    = _HERE / "digit_model.pkl"

# ── Preprocessing ─────────────────────────────────────────────────────────────
IMG_SIZE = 28

# HOG descriptor — must match classifier.py exactly.
# 28×28 window, 14×14 blocks, 7px stride, 7×7 cells, 9 orientation bins → 324 features.
_HOG_DESC = cv2.HOGDescriptor(
    _winSize=(IMG_SIZE, IMG_SIZE),
    _blockSize=(14, 14),
    _blockStride=(7, 7),
    _cellSize=(7, 7),
    _nbins=9,
)

# Number of random augmented copies generated per original sample.
N_AUGMENTS = 7


def _to_binary(img: np.ndarray) -> np.ndarray:
    """
    Grayscale image → clean 28×28 binary image (digit=white, bg=black).

    Must stay in exact sync with _to_binary in classifier.py.
    Pipeline: CLAHE → fixed threshold (180) → dilate → keep largest connected
              component → bounding-box crop → resize to 28×28.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean = np.zeros_like(binary)
        clean[labels == largest] = 255
        binary = clean

    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(w, h) // 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        binary = binary[y1:y2, x1:x2]

    return cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def _hog(binary: np.ndarray) -> np.ndarray:
    """28×28 binary image → HOG feature vector (324 floats)."""
    return _HOG_DESC.compute(binary).flatten().astype(np.float32)


def _augment(binary: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply one random geometric transformation to a 28×28 binary image.

    Augments in image space (before HOG extraction) so the HOG features reflect
    genuinely varied digit shapes rather than artifacts of the augmentation.

    Transformations:
      - Rotation    : ±15°
      - Scale       : 88–112%
      - Translation : ±2 px
    """
    cx, cy = IMG_SIZE / 2, IMG_SIZE / 2
    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.88, 1.12)
    tx    = rng.uniform(-2.0, 2.0)
    ty    = rng.uniform(-2.0, 2.0)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(binary, M, (IMG_SIZE, IMG_SIZE), borderValue=0)


def load_dataset() -> tuple[list[np.ndarray], np.ndarray]:
    """
    Walk training_data/{1..9}/ and return (binary_images, labels).
    Each element of binary_images is a preprocessed 28×28 uint8 array.
    """
    X_binary: list[np.ndarray] = []
    y: list[int]               = []

    for label in range(1, 10):
        folder = TRAINING_DIR / str(label)
        if not folder.is_dir():
            print(f"  [warn] missing folder: {folder}", file=sys.stderr)
            continue

        images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        for img_path in images:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Cannot read: {img_path}")
                X_binary.append(_to_binary(img))
                y.append(label)
            except Exception as exc:
                print(f"  [warn] skipping {img_path.name}: {exc}", file=sys.stderr)

    return X_binary, np.array(y)


def augment_dataset(
    X_binary: list[np.ndarray],
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment binary images, then extract HOG features from all of them.

    Augmentation happens in image space so HOG features are computed on the
    varied shapes — correctly reflects what real geometric variation looks like
    to the model.

    Returns (X_hog, y_all) ready to pass to the classifier.
    """
    rng = np.random.default_rng(seed=42)

    all_binary = list(X_binary)
    all_labels = list(y)
    for img, label in zip(X_binary, y):
        for _ in range(N_AUGMENTS):
            all_binary.append(_augment(img, rng))
            all_labels.append(label)

    X_hog = np.array([_hog(b) for b in all_binary])
    return X_hog, np.array(all_labels)


def _build_model() -> Pipeline:
    """
    Build a Pipeline: StandardScaler → MLPClassifier.

    Trained on HOG features (324-dim).  Hidden layers are deliberately large
    so the model fully memorises this specific client's handwriting.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=1000,
            random_state=42,
        )),
    ])


def train() -> None:
    print("Loading training data…")
    X_binary, y = load_dataset()
    print(f"  {len(X_binary)} original samples | {len(set(y))} classes")

    # CV on HOG features from original (un-augmented) data — honest accuracy estimate.
    print("Running cross-validation on original data…")
    X_hog_orig = np.array([_hog(b) for b in X_binary])
    scores = cross_val_score(_build_model(), X_hog_orig, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy : {scores.mean():.1%} ± {scores.std():.1%}")

    # Augment in image space, extract HOG, train the final model.
    X_all, y_all = augment_dataset(X_binary, y)
    print(f"  Augmented to {len(X_all)} samples ({N_AUGMENTS}× per original)")

    final_model = _build_model()
    final_model.fit(X_all, y_all)
    print(f"  Training accuracy  : {final_model.score(X_all, y_all):.1%}")

    joblib.dump(final_model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train()
