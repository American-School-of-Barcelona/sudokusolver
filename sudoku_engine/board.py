from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from sudoku_engine.models import ValidationResult, ConflictType

RC = Tuple[int, int]


def parse_81(s: str) -> List[List[int]]:
    s = "".join(ch for ch in s if not ch.isspace())
    if len(s) != 81:
        raise ValueError(f"Expected 81 characters after removing whitespace, got {len(s)}")
    grid: List[List[int]] = []
    for r in range(9):
        row: List[int] = []
        for c in range(9):
            ch = s[r * 9 + c]
            if ch in ".0":
                row.append(0)
            elif ch.isdigit() and ch != "0":
                row.append(int(ch))
            else:
                raise ValueError(f"Invalid char '{ch}' in grid.")
        grid.append(row)
    return grid


class Board:
    """
    Minimal, reliable board:
    - grid: current entries (givens + user entries)
    - given_mask: True where the original givens are
    """

    def __init__(self, grid: List[List[int]], given_mask: List[List[bool]]):
        self.grid = grid
        self.given_mask = given_mask

        self.rows = [[(r, c) for c in range(9)] for r in range(9)]
        self.cols = [[(r, c) for r in range(9)] for c in range(9)]
        self.boxes = [
            [(r, c)
             for r in range(br * 3, br * 3 + 3)
             for c in range(bc * 3, bc * 3 + 3)]
            for br in range(3) for bc in range(3)
        ]

    @staticmethod
    def from_strings(givens_81: str, current_81: Optional[str] = None) -> "Board":
        givens = parse_81(givens_81)
        current = parse_81(current_81) if current_81 is not None else [row[:] for row in givens]
        given_mask = [[givens[r][c] != 0 for c in range(9)] for r in range(9)]
        return Board(current, given_mask)

    def pretty(self) -> str:
        lines = []
        for r in range(9):
            if r in (3, 6):
                lines.append("-" * 21)
            row = []
            for c in range(9):
                if c in (3, 6):
                    row.append("|")
                v = self.grid[r][c]
                row.append(str(v) if v != 0 else ".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def validate_rules(self) -> ValidationResult:
        for r in range(9):
            cells = self._find_duplicate_in_unit(self.rows[r])
            if cells:
                return ValidationResult(False, ConflictType.ROW, cells)
        for c in range(9):
            cells = self._find_duplicate_in_unit(self.cols[c])
            if cells:
                return ValidationResult(False, ConflictType.COL, cells)
        for b in range(9):
            cells = self._find_duplicate_in_unit(self.boxes[b])
            if cells:
                return ValidationResult(False, ConflictType.BOX, cells)
        return ValidationResult(True, ConflictType.NONE, [])

    def _find_duplicate_in_unit(self, unit: List[RC]) -> List[RC]:
        seen: Dict[int, List[RC]] = {}
        for (r, c) in unit:
            v = self.grid[r][c]
            if v == 0:
                continue
            seen.setdefault(v, []).append((r, c))
        for d in range(1, 10):
            if d in seen and len(seen[d]) > 1:
                return seen[d]
        return []
