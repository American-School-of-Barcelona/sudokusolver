from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import pytesseract

# Cell is blank if this fraction of pixels are light (> 200 brightness).
BLANK_WHITE_RATIO = 0.85

# Blue HSV pixels found → digit was typed in blue (user entry in digital apps).
BLUE_PIXEL_THRESHOLD = 8

# Stroke-width coefficient of variation above this → handwritten (user entry).
HANDWRITTEN_CV_THRESHOLD = 0.45

_TESS_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"


def _is_blank(gray_cell: np.ndarray) -> bool:
    """True if the cell is empty (mostly light pixels)."""
    return (np.sum(gray_cell > 200) / gray_cell.size) >= BLANK_WHITE_RATIO


def _predict_digit(gray_cell: np.ndarray) -> int:
    """
    Use Tesseract OCR (single-character mode) to read a digit from a cell.
    Returns 1–9, or 0 if unrecognised.
    No training required — Tesseract handles printed and cleanly rendered digits.
    """
    # Scale up so Tesseract has enough pixels to work with.
    img = cv2.resize(gray_cell, (84, 84), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tesseract needs a little surrounding whitespace.
    padded = cv2.copyMakeBorder(thresh, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    text = pytesseract.image_to_string(padded, config=_TESS_CONFIG).strip()
    return int(text[0]) if text and text[0] in "123456789" else 0


def _stroke_width_cv(gray_cell: np.ndarray) -> float:
    """
    Coefficient of variation of stroke widths using the distance transform.

    How it works:
      - Binarise and invert so digit pixels are white.
      - The distance transform assigns each foreground pixel its distance to
        the nearest background pixel — i.e. the half-width of the stroke at
        that point.
      - CV = std / mean over all foreground pixels.

    Printed / computer-rendered digits have very uniform stroke widths → low CV.
    Handwritten digits have variable stroke widths → high CV.
    """
    _, binary = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    fg_widths = dist[binary > 0]
    if len(fg_widths) < 5:
        return 0.0
    return float(fg_widths.std() / (fg_widths.mean() + 1e-6))


def _is_given(color_cell: np.ndarray, gray_cell: np.ndarray) -> bool:
    """
    Decide whether a digit is a puzzle given (printed/typed) or a user entry
    (handwritten or typed in a different colour).

    Two-stage check:
      1. Colour — blue HSV pixels indicate a user entry typed in blue
         (works for digital Sudoku screenshots, including this web app).
      2. Stroke uniformity — computer fonts have very uniform stroke widths
         (low CV); handwritten strokes vary (high CV).
         This works for photos of physical Sudoku books.

    Returns True → given (machine-printed/typed font).
    Returns False → user entry (blue-typed or handwritten).
    """
    # Stage 1: colour check.
    hsv = cv2.cvtColor(color_cell, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    if int(np.count_nonzero(blue_mask)) >= BLUE_PIXEL_THRESHOLD:
        return False  # blue digit → user entry

    # Stage 2: stroke-width uniformity.
    cv = _stroke_width_cv(gray_cell)
    if cv >= HANDWRITTEN_CV_THRESHOLD:
        return False  # irregular strokes → handwritten user entry

    return True  # uniform strokes → given (printed/typed font)


def classify_cells(
    gray_cells: List[np.ndarray],
    color_cells: List[np.ndarray],
) -> List[Dict]:
    """
    Classify all 81 cells.
    Returns a list of {"digit": int, "isGiven": bool} in row-major order.
    """
    results = []
    for gray, color in zip(gray_cells, color_cells):
        if _is_blank(gray):
            results.append({"digit": 0, "isGiven": False})
            continue

        digit = _predict_digit(gray)
        is_given = _is_given(color, gray) if digit > 0 else False
        results.append({"digit": digit, "isGiven": is_given})

    return results
