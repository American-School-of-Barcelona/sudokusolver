from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional

from sudoku_engine.models import ValidationResult, ConflictType

RC = Tuple[int, int]

FULL_MASK = (1 << 9) - 1  # 0b111111111


def bit(d: int) -> int:
    return 1 << (d - 1)


def popcount(mask: int) -> int:
    return mask.bit_count()


def mask_to_digits(mask: int) -> List[int]:
    return [d for d in range(1, 10) if mask & bit(d)]


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
    Human-technique solver board:
    - grid[r][c] = 0..9
    - cand[r][c] = bitmask of candidates for empty cells
    - contradiction = True if any empty cell has 0 candidates
    """

    def __init__(self, grid: List[List[int]], given_mask: List[List[bool]]):
        self.grid = [row[:] for row in grid]
        self.given_mask = [row[:] for row in given_mask]

        self.rows = [[(r, c) for c in range(9)] for r in range(9)]
        self.cols = [[(r, c) for r in range(9)] for c in range(9)]
        self.boxes = [
            [(r, c)
             for r in range(br * 3, br * 3 + 3)
             for c in range(bc * 3, bc * 3 + 3)]
            for br in range(3) for bc in range(3)
        ]
        self.units = self.rows + self.cols + self.boxes

        self.peers_of: Dict[RC, Set[RC]] = {}
        self._precompute_peers()

        self.cand: List[List[int]] = [[FULL_MASK for _ in range(9)] for _ in range(9)]
        self.contradiction = False
        self._init_candidates_from_grid()

    @staticmethod
    def from_strings(givens_81: str, current_81: Optional[str] = None) -> "Board":
        givens = parse_81(givens_81)
        current = parse_81(current_81) if current_81 is not None else [row[:] for row in givens]
        given_mask = [[givens[r][c] != 0 for c in range(9)] for r in range(9)]
        return Board(current, given_mask)

    def clone(self) -> "Board":
        b = Board(self.grid, self.given_mask)
        b.cand = [row[:] for row in self.cand]
        b.contradiction = self.contradiction
        return b

    def _precompute_peers(self) -> None:
        for r in range(9):
            for c in range(9):
                peers: Set[RC] = set()
                # row + col + box
                peers.update(self.rows[r])
                peers.update(self.cols[c])
                box_i = (r // 3) * 3 + (c // 3)
                peers.update(self.boxes[box_i])
                peers.discard((r, c))
                self.peers_of[(r, c)] = peers

    def _init_candidates_from_grid(self) -> None:
        # Start with full candidates for empties; singletons for filled
        for r in range(9):
            for c in range(9):
                v = self.grid[r][c]
                self.cand[r][c] = bit(v) if 1 <= v <= 9 else FULL_MASK

        # Basic elimination based on existing filled digits
        for r in range(9):
            for c in range(9):
                v = self.grid[r][c]
                if 1 <= v <= 9:
                    self._eliminate_from_peers(r, c, v)

        self._update_contradiction_flag()

    def _update_contradiction_flag(self) -> None:
        self.contradiction = False
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0 and self.cand[r][c] == 0:
                    self.contradiction = True
                    return

    def _eliminate_from_peers(self, r: int, c: int, d: int) -> None:
        bd = bit(d)
        for (pr, pc) in self.peers_of[(r, c)]:
            if self.grid[pr][pc] == 0:
                self.cand[pr][pc] &= ~bd

    def eliminate_candidate(self, r: int, c: int, d: int) -> bool:
        """Return True if change occurred."""
        if self.grid[r][c] != 0:
            return False
        before = self.cand[r][c]
        self.cand[r][c] &= ~bit(d)
        changed = self.cand[r][c] != before
        if changed and self.cand[r][c] == 0:
            self.contradiction = True
        return changed

    def restrict_candidates(self, r: int, c: int, allowed_mask: int) -> bool:
        """Keep only allowed candidates. Return True if change occurred."""
        if self.grid[r][c] != 0:
            return False
        before = self.cand[r][c]
        self.cand[r][c] &= allowed_mask
        changed = self.cand[r][c] != before
        if changed and self.cand[r][c] == 0:
            self.contradiction = True
        return changed

    def place_digit(self, r: int, c: int, d: int) -> None:
        """Place digit and perform basic elimination on peers."""
        self.grid[r][c] = d
        self.cand[r][c] = bit(d)
        self._eliminate_from_peers(r, c, d)
        self._update_contradiction_flag()

    def is_solved(self) -> bool:
        return all(self.grid[r][c] != 0 for r in range(9) for c in range(9))

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
            dup = self._find_duplicate(self.rows[r])
            if dup:
                return ValidationResult(False, ConflictType.ROW, dup)
        for c in range(9):
            dup = self._find_duplicate(self.cols[c])
            if dup:
                return ValidationResult(False, ConflictType.COL, dup)
        for b in range(9):
            dup = self._find_duplicate(self.boxes[b])
            if dup:
                return ValidationResult(False, ConflictType.BOX, dup)
        return ValidationResult(True, ConflictType.NONE, [])

    def _find_duplicate(self, unit: List[RC]) -> List[RC]:
        seen: Dict[int, List[RC]] = {}
        for (r, c) in unit:
            v = self.grid[r][c]
            if v == 0:
                continue
            seen.setdefault(v, []).append((r, c))
        for d, cells in seen.items():
            if len(cells) > 1:
                return cells
        return []
