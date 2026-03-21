from __future__ import annotations

import cv2
import numpy as np


def preprocess(image_bytes: bytes):
    """
    Decode image bytes, apply blur + adaptive threshold.
    Returns (thresh, color_bgr) — both needed downstream.
    thresh is BINARY_INV so digits are white on black.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("Could not decode image. Upload a valid PNG or JPEG.")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Check contrast — reject images that are too flat
    if blurred.std() < 10:
        raise ValueError("Image is too low-contrast or blank. Try a clearer screenshot.")

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    return thresh, color
