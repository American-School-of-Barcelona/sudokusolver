from __future__ import annotations

import pathlib
from typing import Dict, List

import cv2
import joblib
import numpy as np
import pytesseract

# ── Thresholds ────────────────────────────────────────────────────────────────
# Blank detection: any pixel below this level is considered non-background ink.
# Must be high enough to catch both:
#   - Pure black printed digits  (grayscale ≈ 0–50)
#   - Blue handwritten digits    (grayscale ≈ 80–150 after BGR→gray conversion)
# White background is ≈240–255, so 180 gives a clear gap above any ink.
NONWHITE_LEVEL = 180

# A cell is blank if it has fewer than this many sub-NONWHITE_LEVEL pixels
# (measured after CLAHE enhancement).
# 80 gives a wide margin: real digits have 200–600 dark pixels in a 60×60 cell;
# shadow/noise in genuinely blank cells rarely exceeds 80.
BLANK_PIXEL_THRESHOLD = 80

# Pixels darker than this are considered pure black ink (printed givens).
# Used only in _is_given to distinguish printing-press black from other ink.
INK_THRESHOLD = 85


_TESS_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"

# ── Trained handwriting model ─────────────────────────────────────────────────
# Must match the value used in train_classifier.py.
_IMG_SIZE   = 28
_MODEL_PATH = pathlib.Path(__file__).parent / "digit_model.pkl"

# HOG descriptor — must match train_classifier.py exactly.
# 28×28 window, 14×14 blocks, 7px stride, 7×7 cells, 9 orientation bins → 324 features.
_HOG_DESC = cv2.HOGDescriptor(
    _winSize=(_IMG_SIZE, _IMG_SIZE),
    _blockSize=(14, 14),
    _blockStride=(7, 7),
    _cellSize=(7, 7),
    _nbins=9,
)


def _load_model():
    """Load the trained MLP classifier at module startup, or None if not trained yet."""
    if _MODEL_PATH.exists():
        return joblib.load(_MODEL_PATH)
    return None


# Loaded once when the module is first imported so inference has no startup cost.
_MODEL = _load_model()


# ── Helper functions ──────────────────────────────────────────────────────────

def _is_blank(gray_cell: np.ndarray) -> bool:
    """
    True if the cell contains no digit.

    Applies CLAHE first so shadow noise in blank cells is suppressed and the
    threshold of NONWHITE_LEVEL (180) reliably separates ink from background.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_cell)
    return int(np.sum(enhanced < NONWHITE_LEVEL)) < BLANK_PIXEL_THRESHOLD


def _to_binary(gray_cell: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline → clean 28×28 binary image (digit=white, bg=black).

    Pipeline: CLAHE → fixed threshold → dilate → keep largest connected component
              → bounding-box crop → resize to 28×28.

    Must stay in exact sync with _to_binary in train_classifier.py.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_cell)

    _, binary = cv2.threshold(enhanced, NONWHITE_LEVEL, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Keep only the largest connected component — eliminates shadow noise and
    # grid-line residue that confuse the feature extraction.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean = np.zeros_like(binary)
        clean[labels == largest] = 255
        binary = clean

    # Crop tightly so position within the cell is irrelevant.
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(w, h) // 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        binary = binary[y1:y2, x1:x2]

    return cv2.resize(binary, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)


def _preprocess_for_model(gray_cell: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale cell to the HOG feature vector the MLP expects.

    HOG (Histogram of Oriented Gradients) captures digit shape via local gradient
    directions rather than raw pixel values, making it robust to lighting variation,
    slight positional shifts, and pen-pressure differences.
    """
    binary = _to_binary(gray_cell)
    return _HOG_DESC.compute(binary).flatten().astype(np.float32)


_TTA_RNG = np.random.default_rng(seed=0)  # fixed seed → reproducible across calls


def _augment_cell(gray_cell: np.ndarray) -> np.ndarray:
    """One random geometric perturbation of a raw grayscale cell for TTA."""
    cx, cy = gray_cell.shape[1] / 2, gray_cell.shape[0] / 2
    angle = float(_TTA_RNG.uniform(-10.0, 10.0))
    scale = float(_TTA_RNG.uniform(0.90, 1.10))
    tx    = float(_TTA_RNG.uniform(-3.0, 3.0))
    ty    = float(_TTA_RNG.uniform(-3.0, 3.0))
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(
        gray_cell, M, (gray_cell.shape[1], gray_cell.shape[0]),
        borderValue=255,  # white background fill
    )


def _predict_digit_model(gray_cell: np.ndarray) -> int:
    """
    MLP with 7-way test-time augmentation (TTA) and low-confidence fallback.

    Runs the original cell plus 6 randomly-perturbed versions through the MLP
    and takes the majority vote.  This handles off-centre digits, size variation,
    and lighting asymmetry without any model change.

    If fewer than 4 of 7 votes agree AND predict_proba max < 0.55, the model is
    genuinely uncertain — Tesseract is tried as a second opinion.
    """
    if _MODEL is None:
        return 0

    N_TTA = 7
    cells = [gray_cell] + [_augment_cell(gray_cell) for _ in range(N_TTA - 1)]
    votes: Dict[int, int] = {}
    for cell in cells:
        pred = int(_MODEL.predict(_preprocess_for_model(cell).reshape(1, -1))[0])
        votes[pred] = votes.get(pred, 0) + 1

    winner    = max(votes, key=lambda k: votes[k])
    top_count = votes[winner]

    if top_count <= 3:
        proba = _MODEL.predict_proba(
            _preprocess_for_model(gray_cell).reshape(1, -1)
        )[0]
        if float(proba.max()) < 0.55:
            tess = _predict_digit_tesseract(gray_cell)
            if tess > 0:
                return tess

    return winner


def _predict_digit_tesseract(gray_cell: np.ndarray) -> int:
    """
    Use Tesseract OCR (single-character mode) to read a printed/typed digit.
    Returns 1–9, or 0 if unrecognised.

    This is only called after _is_blank confirms the cell has a digit, so the
    cell always contains both dark ink and light background — two well-separated
    pixel clusters.  Otsu reliably finds the split between them regardless of
    per-cell brightness variation from scan lighting or warp interpolation.
    """
    img = cv2.resize(gray_cell, (160, 160), interpolation=cv2.INTER_CUBIC)

    # Otsu is reliable here: the blank check guarantees two pixel clusters
    # (dark ink ≈ 0–80, white background ≈ 200–255) are present.
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Generous padding so Tesseract's character bounding-box logic has room.
    padded = cv2.copyMakeBorder(thresh, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    text   = pytesseract.image_to_string(padded, config=_TESS_CONFIG).strip()
    return int(text[0]) if text and text[0] in "123456789" else 0



def _is_given(color_cell: np.ndarray, gray_cell: np.ndarray) -> bool:
    """
    Decide whether a digit is a puzzle given (printed/typed) or a user entry.

    The key insight: printing-press black ink is simultaneously very dark
    (HSV value ≈ 0–60) AND achromatic (HSV saturation ≈ 0–30).
    Handwritten digits — even dark navy or dark blue ones — always have
    notable saturation or are noticeably brighter than print.
    We measure both channels on the ink pixels and require BOTH conditions
    to declare a given; anything that fails either check is a user entry.
    """
    hsv = cv2.cvtColor(color_cell, cv2.COLOR_BGR2HSV)

    # Identify ink pixels as anything noticeably non-white.
    ink_mask = gray_cell < NONWHITE_LEVEL
    n_ink = int(np.count_nonzero(ink_mask))
    if n_ink < 10:
        return False   # almost no ink — shouldn't happen after blank check

    ink_sat = hsv[:, :, 1][ink_mask]   # S channel: 0 = grey/black, 255 = vivid colour
    ink_val = hsv[:, :, 2][ink_mask]   # V channel: 0 = black, 255 = bright

    median_sat = float(np.median(ink_sat))
    median_val = float(np.median(ink_val))

    # Printed black: dark (V < 80) AND achromatic (S < 40).
    # Handwritten ink fails at least one: it's either coloured (S ≥ 40) or
    # brighter than pure print (V ≥ 80), even for dark navy or grey pencil.
    return median_val < 80 and median_sat < 40


def classify_cells(
    gray_cells: List[np.ndarray],
    color_cells: List[np.ndarray],
) -> List[Dict]:
    """
    Classify all 81 cells.

    The given/user decision is made first because it determines which digit
    recogniser to use:
      - Given cells  → Tesseract (highly accurate on uniform printed fonts).
      - User entries → trained MLP (specialised to this client's handwriting).

    Falls back to Tesseract for user cells if the model has not been trained.

    Returns a list of {"digit": int, "isGiven": bool} in row-major order.
    """
    results = []
    for gray, color in zip(gray_cells, color_cells):
        if _is_blank(gray):
            results.append({"digit": 0, "isGiven": False})
            continue

        # Determine given/user FIRST so the correct recogniser is chosen.
        is_given = _is_given(color, gray)

        if is_given:
            digit = _predict_digit_tesseract(gray)
        else:
            digit = _predict_digit_model(gray)
            if digit == 0:
                # Model not available yet — fall back to Tesseract.
                digit = _predict_digit_tesseract(gray)

        results.append({"digit": digit, "isGiven": is_given if digit > 0 else False})

    return results
