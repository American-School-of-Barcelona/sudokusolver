from __future__ import annotations

import pathlib
from typing import Dict, List

import cv2
import joblib
import numpy as np
import pytesseract

# ── Thresholds ────────────────────────────────────────────────────────────────
# A cell is blank if it has fewer than this many dark pixels.
# Counting dark pixels directly is more reliable than a white-pixel ratio:
# blank cells have near-zero dark pixels, while even the thinnest digit (1, 7)
# produces hundreds of dark pixels at 60×60 resolution.
BLANK_DARK_PIXEL_THRESHOLD = 60

# Pixels darker than this are considered ink (digit strokes).
# 85 works well for clean black-on-white prints; anything above the digit
# but below background (~240) is treated as background.
INK_THRESHOLD = 85

# Blue HSV pixels found → digit was typed in blue (user entry in digital apps).
BLUE_PIXEL_THRESHOLD = 8

# Stroke-width coefficient of variation above this → handwritten (user entry).
HANDWRITTEN_CV_THRESHOLD = 0.45

_TESS_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"

# ── Trained handwriting model ─────────────────────────────────────────────────
# Must match the value used in train_classifier.py.
_IMG_SIZE   = 28
_MODEL_PATH = pathlib.Path(__file__).parent / "digit_model.pkl"


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

    Counts pixels that are unambiguously dark ink rather than measuring how
    much of the cell is white.  Blank cells have virtually zero dark pixels;
    even the thinnest printed digit (1, 7) has several hundred.
    """
    return int(np.sum(gray_cell < INK_THRESHOLD)) < BLANK_DARK_PIXEL_THRESHOLD


def _preprocess_for_model(gray_cell: np.ndarray) -> np.ndarray:
    """
    Apply the same preprocessing used at training time so inference features
    match what the model was trained on.

    Returns a flat float32 vector of length _IMG_SIZE².
    """
    # Binarise and invert: digit becomes white, background becomes black.
    _, binary = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Crop tightly to the digit bounding box — same as the training pipeline.
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(w, h) // 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)
        binary = binary[y1:y2, x1:x2]

    binary = cv2.resize(binary, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
    return binary.flatten().astype(np.float32) / 255.0


def _predict_digit_model(gray_cell: np.ndarray) -> int:
    """
    Use the trained MLP to recognise a handwritten digit.
    Returns 1–9, or 0 if the model has not been trained yet.

    This is used for user-entry (handwritten) cells because Tesseract is
    trained on printed text and performs poorly on handwriting.
    """
    if _MODEL is None:
        return 0
    features = _preprocess_for_model(gray_cell).reshape(1, -1)
    return int(_MODEL.predict(features)[0])


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


def _stroke_width_cv(gray_cell: np.ndarray) -> float:
    """
    Coefficient of variation of stroke widths using the distance transform.

    How it works:
      - Binarise and invert so digit pixels are white.
      - The distance transform assigns each foreground pixel its distance to
        the nearest background pixel — i.e. the local half-width of the stroke.
      - CV = std / mean over all foreground pixels.

    Printed / computer-rendered digits have very uniform stroke widths → low CV.
    Handwritten digits have variable stroke widths → high CV.
    """
    _, binary = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.sum() == 0:
        return 0.0
    dist      = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    fg_widths = dist[binary > 0]
    if len(fg_widths) < 5:
        return 0.0
    return float(fg_widths.std() / (fg_widths.mean() + 1e-6))


def _is_given(color_cell: np.ndarray, gray_cell: np.ndarray) -> bool:
    """
    Decide whether a digit is a puzzle given (printed/typed) or a user entry.

    Three-stage check in order of reliability:

    1. Blue pixels → user entry typed in blue (digital Sudoku apps).

    2. Ink brightness → the most reliable signal for physical puzzles.
       Printed Sudoku ink is literally pure black (HSV value ≈ 0–50).
       Handwritten pencil strokes are noticeably brighter (value ≈ 100–200).
       This is more robust than stroke-width CV, which breaks down on bold
       printed fonts because their thick strokes have naturally high variance.

    3. Stroke-width CV → fallback for edge cases (e.g. very light printing).
    """
    hsv = cv2.cvtColor(color_cell, cv2.COLOR_BGR2HSV)

    # Stage 1: blue ink → user.
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    if int(np.count_nonzero(blue_mask)) >= BLUE_PIXEL_THRESHOLD:
        return False

    # Stage 2: ink darkness check.
    # Take only pixels that are clearly ink (dark in grayscale), then inspect
    # their HSV brightness.  Printed black ink stays very dark (V < 60) even
    # after perspective-warp interpolation; handwritten strokes are brighter.
    ink_mask = gray_cell < INK_THRESHOLD
    n_ink = int(np.count_nonzero(ink_mask))
    if n_ink > 20:
        ink_brightness = hsv[:, :, 2][ink_mask]   # V channel of ink pixels only
        if float(np.percentile(ink_brightness, 25)) < 60:
            return True   # very dark ink → printed given

    # Stage 3: stroke-width CV fallback.
    return _stroke_width_cv(gray_cell) < HANDWRITTEN_CV_THRESHOLD


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
