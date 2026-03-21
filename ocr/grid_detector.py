from __future__ import annotations

import cv2
import numpy as np

# Larger warp size = more pixels per cell = better OCR accuracy.
# 630 px → 70 px per cell (vs 50 px at 450), which gives Tesseract
# substantially more detail, especially for thin digits like 1 and 7.
TARGET_SIZE = 630


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return corners in order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  = smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right = largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right = smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left = largest y-x
    return rect


def detect_and_warp(thresh: np.ndarray, color: np.ndarray, size: int = TARGET_SIZE):
    """
    Find the largest 4-sided contour (the grid border), apply perspective warp.
    Returns (warped_gray, warped_color) both at size×size pixels.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No grid found in image. Make sure the full Sudoku grid is visible.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    for cnt in contours[:8]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            grid_contour = approx
            break

    if grid_contour is None:
        raise ValueError(
            "Could not detect a rectangular grid border. "
            "Try cropping the image so only the Sudoku board is visible."
        )

    pts = grid_contour.reshape(4, 2).astype("float32")
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.warpPerspective(gray, M, (size, size))
    warped_color = cv2.warpPerspective(color, M, (size, size))

    return warped_gray, warped_color
