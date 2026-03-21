from __future__ import annotations

from typing import List, Tuple

import numpy as np

GRID_SIZE  = 450
CELL_COUNT = 9
MARGIN     = 5   # pixels trimmed from each edge to remove grid lines


def extract_cells(
    warped_gray: np.ndarray,
    warped_color: np.ndarray,
    grid_size: int = GRID_SIZE,
    cell_count: int = CELL_COUNT,
    margin: int = MARGIN,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Slice both warped images into 81 cells.
    Each cell is trimmed by `margin` pixels on every edge to remove grid lines.
    Returns (gray_cells, color_cells) — lists of 81 arrays, row-major order.
    """
    cell_px = grid_size // cell_count

    gray_cells:  List[np.ndarray] = []
    color_cells: List[np.ndarray] = []

    for row in range(cell_count):
        for col in range(cell_count):
            y0 = row * cell_px + margin
            y1 = (row + 1) * cell_px - margin
            x0 = col * cell_px + margin
            x1 = (col + 1) * cell_px - margin

            gray_cells.append(warped_gray[y0:y1, x0:x1].copy())
            color_cells.append(warped_color[y0:y1, x0:x1].copy())

    return gray_cells, color_cells
