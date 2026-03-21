from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from sudoku_engine.models import ValidationResult, ConflictType

RC = Tuple[int, int]

# Bitmask with bits 0–8 set: represents all 9 digits as candidates.
# Digit d is encoded at bit (d-1), so digit 1 → bit 0, digit 9 → bit 8.
# Example: 0b000000101 means digits 1 and 3 are the only candidates.
FULL_MASK = (1 << 9) - 1  # 0b111111111 = 511


def bit(d: int) -> int:
    """
    Return the single-bit mask for digit d.
    Digit 1 maps to bit 0 (value 1), digit 9 maps to bit 8 (value 256).
    Using (d-1) ensures the 9 digits fit neatly into a 9-bit integer.
    """
    return 1 << (d - 1)


def popcount(mask: int) -> int:
    """Return the number of candidate digits encoded in a bitmask."""
    return mask.bit_count()


def mask_to_digits(mask: int) -> List[int]:
    """Convert a candidate bitmask to an ordered list of digit values (1–9)."""
    return [d for d in range(1, 10) if mask & bit(d)]


def parse_81(s: str) -> List[List[int]]:
    """
    Parse an 81-character Sudoku string into a 9×9 integer grid.
    Accepts '0' or '.' for empty cells; raises ValueError on malformed input.
    """
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
                raise ValueError(f"Invalid character '{ch}' in grid string.")
        grid.append(row)
    return grid


class Board:
    """
    Central representation of a Sudoku puzzle state.

    Grid representation
    -------------------
    grid[r][c]      — integer 0–9: 0 means empty, 1–9 means placed digit.
    given_mask[r][c]— bool: True if the cell is a fixed given clue that the
                      player must not change.

    Candidate bitmask encoding
    --------------------------
    cand[r][c] is a 9-bit integer where bit (d-1) is set if digit d is still
    a legal candidate for that cell.  Example:
      0b000000001 (1)   → only digit 1 is possible
      0b000000110 (6)   → digits 2 and 3 are possible
      0b111111111 (511) → all nine digits remain possible (FULL_MASK)

    Bitmasks make candidate elimination a single bitwise AND-NOT operation
    (O(1)) rather than a list search or set removal (O(n)).

    For a filled cell, cand[r][c] is set to bit(d) — a sentinel that records
    which digit was placed, so solvers can read candidates uniformly.

    Pre-computed structure
    ----------------------
    rows[r]   — list of (r, c) pairs for row r (used by units iteration).
    cols[c]   — list of (r, c) pairs for column c.
    boxes[b]  — list of (r, c) pairs for the b-th 3×3 box (row-major order).
    units     — all 27 units (9 rows + 9 cols + 9 boxes) in one flat list,
                used by solving techniques that must scan every unit.
    peers_of[(r,c)] — the set of all 20 cells that share a row, column, or
                box with (r, c).  Pre-computed once at construction because
                the peer relationship is fixed by the grid shape — paying
                the cost once avoids rebuilding the set inside hot loops.

    contradiction — True if any empty cell has had all candidates eliminated
                    (the board can no longer be legally completed).
    """

    def __init__(self, grid: List[List[int]], given_mask: List[List[bool]]):
        self.grid = [row[:] for row in grid]
        self.given_mask = [row[:] for row in given_mask]

        # Build unit membership lists — used directly by solver techniques.
        self.rows  = [[(r, c) for c in range(9)] for r in range(9)]
        self.cols  = [[(r, c) for r in range(9)] for c in range(9)]
        self.boxes = [
            [(r, c)
             for r in range(br * 3, br * 3 + 3)
             for c in range(bc * 3, bc * 3 + 3)]
            for br in range(3) for bc in range(3)
        ]
        # units = all 27 units in one list for techniques that iterate them all.
        self.units = self.rows + self.cols + self.boxes

        self.peers_of: Dict[RC, Set[RC]] = {}
        self._precompute_peers()

        self.cand: List[List[int]] = [[FULL_MASK] * 9 for _ in range(9)]
        self.contradiction = False
        self._init_candidates_from_grid()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_strings(givens_81: str, current_81: Optional[str] = None) -> "Board":
        """
        Build a Board from one or two 81-character strings.
        givens_81  — the puzzle's fixed clues (determines given_mask).
        current_81 — the player's current state; defaults to givens_81.
        """
        givens  = parse_81(givens_81)
        current = parse_81(current_81) if current_81 is not None else [row[:] for row in givens]
        given_mask = [[givens[r][c] != 0 for c in range(9)] for r in range(9)]
        return Board(current, given_mask)

    def clone(self) -> "Board":
        """Return a deep copy with identical grid, candidates, and contradiction flag."""
        cloned = Board(self.grid, self.given_mask)
        cloned.cand = [row[:] for row in self.cand]
        cloned.contradiction = self.contradiction
        return cloned

    def _precompute_peers(self) -> None:
        """
        Build peers_of[(r,c)] for every cell — the 20 other cells that share
        a row, column, or 3×3 box with (r, c).
        Computed once at construction because the peer relationship is fixed by
        the grid topology; recomputing it on every elimination would be wasteful.
        """
        for r in range(9):
            for c in range(9):
                peers: Set[RC] = set()
                peers.update(self.rows[r])
                peers.update(self.cols[c])
                box_index = (r // 3) * 3 + (c // 3)
                peers.update(self.boxes[box_index])
                peers.discard((r, c))   # a cell is not its own peer
                self.peers_of[(r, c)] = peers

    def _init_candidates_from_grid(self) -> None:
        """
        Initialise cand[][] from the current grid values using two passes:
          Pass 1 — assign initial masks (FULL_MASK for empty, bit(d) for filled).
          Pass 2 — eliminate candidates for every already-placed digit.
        Two passes are required: eliminating during pass 1 would process cells in
        row-major order, meaning a later peer in the same unit might not yet have
        its singleton mask set, so the elimination could be skipped.
        """
        # Pass 1: set initial bitmasks.
        for r in range(9):
            for c in range(9):
                v = self.grid[r][c]
                self.cand[r][c] = bit(v) if 1 <= v <= 9 else FULL_MASK

        # Pass 2: propagate existing digits into all peers.
        for r in range(9):
            for c in range(9):
                v = self.grid[r][c]
                if 1 <= v <= 9:
                    self._eliminate_from_peers(r, c, v)

        self._update_contradiction_flag()

    def _update_contradiction_flag(self) -> None:
        """Set contradiction=True if any empty cell has no remaining candidates."""
        self.contradiction = any(
            self.grid[r][c] == 0 and self.cand[r][c] == 0
            for r in range(9) for c in range(9)
        )

    def _eliminate_from_peers(self, r: int, c: int, d: int) -> None:
        """
        Remove digit d from the candidate masks of all 20 peers of (r, c).
        Only acts on empty cells — filled cells keep their singleton mask.
        """
        bd = bit(d)
        for (pr, pc) in self.peers_of[(r, c)]:
            if self.grid[pr][pc] == 0:
                self.cand[pr][pc] &= ~bd

    # ------------------------------------------------------------------
    # Public API — candidate and validity queries
    # ------------------------------------------------------------------

    def get_candidates(self, r: int, c: int) -> List[int]:
        """
        Return the list of legal candidate digits for cell (r, c).
        For a filled cell, returns [d] where d is the placed digit.
        For an empty cell, returns all digits whose bits are set in cand[r][c].
        """
        return mask_to_digits(self.cand[r][c])

    def is_valid_move(self, r: int, c: int, d: int) -> bool:
        """
        Return True if placing digit d at (r, c) is legal under Sudoku rules:
          - The cell must be empty (not a given and not already filled).
          - d must not already appear in the same row, column, or 3×3 box,
            which is equivalent to checking that bit(d) is set in cand[r][c]
            (because _eliminate_from_peers already cleared it if d is present
            in any peer).
        """
        if self.given_mask[r][c] or self.grid[r][c] != 0:
            return False
        return bool(self.cand[r][c] & bit(d))

    # ------------------------------------------------------------------
    # Public API — mutation
    # ------------------------------------------------------------------

    def place_digit(self, r: int, c: int, d: int) -> None:
        """
        Place digit d at (r, c) and propagate eliminations to all 20 peers.

        Raises ValueError if:
          - (r, c) is a fixed given clue (given cells are immutable).
          - d is outside the valid range 1–9.
          - (r, c) is outside the 0–8 grid bounds.
        """
        if not (0 <= r <= 8 and 0 <= c <= 8):
            raise ValueError(f"place_digit: coordinates must be 0–8, got ({r}, {c}).")
        if not (1 <= d <= 9):
            raise ValueError(f"place_digit: digit must be 1–9, got {d}.")
        if self.given_mask[r][c]:
            raise ValueError(f"place_digit: cannot overwrite fixed given at ({r}, {c}).")

        self.grid[r][c] = d
        self.cand[r][c] = bit(d)
        self._eliminate_from_peers(r, c, d)
        self._update_contradiction_flag()

    def clear_digit(self, r: int, c: int) -> None:
        """
        Erase a user-placed digit from (r, c) and restore its candidate mask.

        Given cells cannot be cleared — they are fixed puzzle clues.
        After clearing, the cell's candidates are recomputed from scratch by
        scanning all peers, because incrementally un-eliminating candidates
        would require knowing which peer caused each elimination (intractable).
        """
        if self.given_mask[r][c]:
            raise ValueError(f"clear_digit: cannot clear fixed given at ({r}, {c}).")
        if self.grid[r][c] == 0:
            return  # already empty; nothing to do

        self.grid[r][c] = 0
        # Rebuild this cell's candidates by checking all peers.
        used = {self.grid[pr][pc] for (pr, pc) in self.peers_of[(r, c)] if self.grid[pr][pc] != 0}
        self.cand[r][c] = FULL_MASK & ~sum(bit(d) for d in used)
        self._update_contradiction_flag()

    def eliminate_candidate(self, r: int, c: int, d: int) -> bool:
        """
        Remove digit d from the candidate list of empty cell (r, c).
        Returns True if the mask changed (i.e. d was previously a candidate).
        Raises ValueError if d is outside 1–9.
        """
        if not (1 <= d <= 9):
            raise ValueError(f"eliminate_candidate: digit must be 1–9, got {d}.")
        if self.grid[r][c] != 0:
            return False
        before = self.cand[r][c]
        self.cand[r][c] &= ~bit(d)
        changed = self.cand[r][c] != before
        if changed and self.cand[r][c] == 0:
            self.contradiction = True
        return changed

    def restrict_candidates(self, r: int, c: int, allowed_mask: int) -> bool:
        """
        Keep only the candidates in allowed_mask for empty cell (r, c).
        Returns True if the mask changed (i.e. some candidates were removed).
        Raises ValueError if allowed_mask is 0 — that would unconditionally
        produce a contradiction and is almost certainly a caller bug.
        """
        if allowed_mask == 0:
            raise ValueError("restrict_candidates: allowed_mask of 0 would force an immediate contradiction.")
        if self.grid[r][c] != 0:
            return False
        before = self.cand[r][c]
        self.cand[r][c] &= allowed_mask
        changed = self.cand[r][c] != before
        if changed and self.cand[r][c] == 0:
            self.contradiction = True
        return changed

    # ------------------------------------------------------------------
    # Public API — board-level queries
    # ------------------------------------------------------------------

    def is_solved(self) -> bool:
        """Return True if every cell has a placed digit (no zeros remain)."""
        return all(self.grid[r][c] != 0 for r in range(9) for c in range(9))

    def get_all_conflict_cells(self) -> List[Dict]:
        """
        Scan all 27 units and return every cell involved in a duplicate-digit
        rule violation, formatted as {"r": 1-indexed, "c": 1-indexed, "digit": d}.

        Returns an empty list if the board has no rule violations.

        Unlike validate_rules() (which stops at the first conflict), this method
        collects ALL conflicting cells so the UI can highlight every problem at
        once rather than fixing them one at a time.
        """
        # Map (r, c) → offending digit so each cell appears at most once.
        bad: Dict[RC, int] = {}
        for unit in self.units:
            seen: Dict[int, RC] = {}          # digit → first cell that placed it
            for (r, c) in unit:
                v = self.grid[r][c]
                if v == 0:
                    continue
                if v in seen:
                    bad[seen[v]] = v          # mark both the original cell …
                    bad[(r, c)]  = v          # … and the duplicate
                else:
                    seen[v] = (r, c)
        return [{"r": r + 1, "c": c + 1, "digit": d} for (r, c), d in bad.items()]

    def validate_rules(self) -> ValidationResult:
        """
        Check all 27 units for duplicate digits.
        Returns after the first violation found (use get_all_conflict_cells
        if you need every conflict at once).
        """
        for r in range(9):
            dup = self._find_duplicate(self.rows[r])
            if dup:
                return ValidationResult(False, ConflictType.ROW, dup)
        for c in range(9):
            dup = self._find_duplicate(self.cols[c])
            if dup:
                return ValidationResult(False, ConflictType.COL, dup)
        for box_i in range(9):
            dup = self._find_duplicate(self.boxes[box_i])
            if dup:
                return ValidationResult(False, ConflictType.BOX, dup)
        return ValidationResult(True, ConflictType.NONE, [])

    def _find_duplicate(self, unit: List[RC]) -> List[RC]:
        """Return the two (or more) cells in unit that share the same digit, or []."""
        seen: Dict[int, List[RC]] = {}
        for (r, c) in unit:
            v = self.grid[r][c]
            if v == 0:
                continue
            seen.setdefault(v, []).append((r, c))
        for cells in seen.values():
            if len(cells) > 1:
                return cells
        return []

    def pretty(self) -> str:
        """Return a human-readable ASCII representation of the board."""
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
