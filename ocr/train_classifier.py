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
# 28×28 matches the standard MNIST size; keeps the feature vector short (784
# floats) and training fast while retaining enough detail for digit recognition.
IMG_SIZE = 28

# Number of random augmented copies generated per original sample.
# 509 samples × 7 augmentations = ~3563 extra examples, bringing the effective
# training set to ~4072 total.  More variety helps the model handle slight
# differences in how the client writes on a given day.
N_AUGMENTS = 7


def _preprocess(img_path: pathlib.Path) -> np.ndarray:
    """
    Load one sample image and convert it to a flat, normalised feature vector.

    Steps:
      1. Read as grayscale — colour is irrelevant for digit identity.
      2. Binarise with Otsu's threshold so brightness differences don't affect
         features.  The image is inverted so digit pixels are white (foreground).
      3. Crop tightly to the digit's bounding box so position in the photo is
         invariant — a "1" centred vs. off-centre must look the same to the model.
      4. Resize to IMG_SIZE × IMG_SIZE.
      5. Flatten and normalise to [0, 1] for the MLP's gradient descent.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Binarise: digit pixels become 255, background becomes 0.
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Crop to bounding box of the digit content, with a small proportional pad.
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(w, h) // 6          # ~17% border keeps the digit from touching edges
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        binary = binary[y1:y2, x1:x2]

    binary = cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return binary.flatten().astype(np.float32) / 255.0


def _augment(flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply one random geometric transformation to a preprocessed image vector.

    Transformations applied (all chosen randomly within realistic bounds):
      - Rotation    : ±15° — handwritten digits are naturally slightly tilted
      - Scale       : 88–112% — the client may write larger or smaller
      - Translation : ±2 px — slight shifts in pen-lift position

    The digit pixels are white (1.0) on a black background (0.0), so the
    border fill for areas exposed by the warp is 0 (black = background).
    Returns a new flat float32 vector of the same length.
    """
    img = (flat.reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
    cx, cy = IMG_SIZE / 2, IMG_SIZE / 2

    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.88, 1.12)
    tx    = rng.uniform(-2.0, 2.0)
    ty    = rng.uniform(-2.0, 2.0)

    # Build a combined rotation + scale + translation matrix.
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=0)
    return warped.flatten().astype(np.float32) / 255.0


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Walk training_data/{1..9}/ and build X (features) and y (labels) arrays.
    Each row of X is the preprocessed feature vector for one image.
    Labels are the digit values 1–9.
    """
    X: list[np.ndarray] = []
    y: list[int]        = []

    for label in range(1, 10):
        folder = TRAINING_DIR / str(label)
        if not folder.is_dir():
            print(f"  [warn] missing folder: {folder}", file=sys.stderr)
            continue

        images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        for img_path in images:
            try:
                X.append(_preprocess(img_path))
                y.append(label)
            except Exception as exc:
                print(f"  [warn] skipping {img_path.name}: {exc}", file=sys.stderr)

    return np.array(X), np.array(y)


def augment_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the dataset by adding N_AUGMENTS randomly-transformed copies of each
    original sample.

    Why augment only for the final training run (not during cross-validation)?
    Cross-validation splits the original samples into folds to give an honest
    accuracy estimate on unseen data.  Augmenting before the split would let
    near-identical copies of a test sample appear in training, inflating the
    score.  Instead, CV runs on the original data (honest estimate) and the
    final saved model trains on original + augmented data (maximum capacity).
    """
    rng = np.random.default_rng(seed=42)
    X_aug = [_augment(x, rng) for x in X for _ in range(N_AUGMENTS)]
    y_aug = [label for label in y for _ in range(N_AUGMENTS)]
    X_all = np.vstack([X, np.array(X_aug)])
    y_all = np.concatenate([y, np.array(y_aug)])
    return X_all, y_all


def _build_model() -> Pipeline:
    """
    Build a Pipeline that standardises features then trains an MLP.

    StandardScaler (zero mean, unit variance) is applied first because raw
    pixel intensities have unequal variance across positions, which slows
    gradient descent and can prevent convergence.

    The large hidden layers (256 → 128) are intentionally oversized so the
    model fully memorises this specific client's handwriting style.
    Generalisation to other people's writing is not a goal.
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
    X, y = load_dataset()
    print(f"  {len(X)} original samples | {len(set(y))} classes")

    # CV on original (un-augmented) data gives an honest accuracy estimate.
    print("Running cross-validation on original data…")
    scores = cross_val_score(_build_model(), X, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy : {scores.mean():.1%} ± {scores.std():.1%}")

    # Augment and train the final model on all available data.
    X_all, y_all = augment_dataset(X, y)
    print(f"  Augmented to {len(X_all)} samples ({N_AUGMENTS}× per original)")

    final_model = _build_model()
    final_model.fit(X_all, y_all)
    print(f"  Training accuracy  : {final_model.score(X_all, y_all):.1%}")

    joblib.dump(final_model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train()
