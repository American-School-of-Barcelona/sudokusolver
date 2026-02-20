from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sudoku_engine.board import Board
from sudoku_engine.reports import MistakeReport, MistakeItem, generate_violation_report

RC = Tuple[int, int]
ReasonMap = Dict[RC, Dict[str, str]]

ALL = (1 << 9) - 1  # bits for digits 1..9


# ------------------ bitmask helpers ------------------
def bit(d: int) -> int:
    return 1 << (d - 1)


def popcount(x: int) -> int:
    return int(x).bit_count()


def mask_to_digits(mask: int) -> List[int]:
    return [d for d in range(1, 10) if mask & bit(d)]


def box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)  # 0..8


def cells_in_box(bi: int) -> List[RC]:
    br = (bi // 3) * 3
    bc = (bi % 3) * 3
    return [(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)]


def init_candidates(grid) -> Optional[List[List[int]]]:
    """
    Returns cand[r][c] bitmask for empty cells, 0 for filled.
    Returns None if contradiction (empty cell has no candidates) or duplicate constraint detected.
    """
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9

    # build used masks + detect duplicates
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            b = bit(v)
            bi = box_index(r, c)
            if (row_used[r] & b) or (col_used[c] & b) or (box_used[bi] & b):
                return None
            row_used[r] |= b
            col_used[c] |= b
            box_used[bi] |= b

    cand = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                cand[r][c] = 0
                continue
            bi = box_index(r, c)
            allowed = ALL & ~(row_used[r] | col_used[c] | box_used[bi])
            if allowed == 0:
                return None
            cand[r][c] = allowed

    return cand


# ------------------ Hint dataclass ------------------
@dataclass(frozen=True)
class HintResult:
    has_hint: bool
    technique: str = ""
    action: str = ""  # "PLACE" | "ELIMINATE" | "RESTRICT" | "FIX_MISTAKE" | "ERROR"
    message: str = ""


def _fmt_cell(r: int, c: int) -> str:
    return f"(r{r+1}, c{c+1})"


# ------------------ Technique hint finders ------------------
def hint_naked_single(grid, cand) -> Optional[HintResult]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and popcount(cand[r][c]) == 1:
                d = mask_to_digits(cand[r][c])[0]
                msg = (
                    f"Naked Single found at {_fmt_cell(r,c)}.\n"
                    f"Only one candidate is possible: {d}.\n"
                    f"Hint: place {d} in {_fmt_cell(r,c)}."
                )
                return HintResult(True, "Naked Single", "PLACE", msg)
    return None


def hint_hidden_single(grid, cand) -> Optional[HintResult]:
    # rows
    for r in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for c in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) == 1:
                rr, cc = spots[0]
                msg = (
                    f"Hidden Single (Row) in row {r+1}.\n"
                    f"Digit {d} can only go in {_fmt_cell(rr,cc)}.\n"
                    f"Hint: place {d} in {_fmt_cell(rr,cc)}."
                )
                return HintResult(True, "Hidden Single (Row)", "PLACE", msg)

    # cols
    for c in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for r in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) == 1:
                rr, cc = spots[0]
                msg = (
                    f"Hidden Single (Column) in column {c+1}.\n"
                    f"Digit {d} can only go in {_fmt_cell(rr,cc)}.\n"
                    f"Hint: place {d} in {_fmt_cell(rr,cc)}."
                )
                return HintResult(True, "Hidden Single (Column)", "PLACE", msg)

    # boxes
    for bi in range(9):
        box_cells = cells_in_box(bi)
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for (r, c) in box_cells if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) == 1:
                rr, cc = spots[0]
                msg = (
                    f"Hidden Single (Box) in box {bi+1}.\n"
                    f"Digit {d} can only go in {_fmt_cell(rr,cc)}.\n"
                    f"Hint: place {d} in {_fmt_cell(rr,cc)}."
                )
                return HintResult(True, "Hidden Single (Box)", "PLACE", msg)

    return None


def hint_naked_pair(grid, cand) -> Optional[HintResult]:
    # units: rows, cols, boxes
    units = []
    for r in range(9):
        units.append(("row", r, [(r, c) for c in range(9)]))
    for c in range(9):
        units.append(("col", c, [(r, c) for r in range(9)]))
    for bi in range(9):
        units.append(("box", bi, cells_in_box(bi)))

    for kind, idx, cells in units:
        pairs: Dict[int, List[RC]] = {}
        for (r, c) in cells:
            if grid[r][c] == 0 and popcount(cand[r][c]) == 2:
                pairs.setdefault(cand[r][c], []).append((r, c))

        for mask, pair_cells in pairs.items():
            if len(pair_cells) == 2:
                digits = mask_to_digits(mask)
                affected = []
                for (r, c) in cells:
                    if (r, c) in pair_cells or grid[r][c] != 0:
                        continue
                    if cand[r][c] & mask:
                        affected.append((r, c))

                if affected:
                    a1, a2 = pair_cells
                    unit_name = f"{kind} {idx+1}" if kind != "box" else f"box {idx+1}"
                    msg = (
                        f"Naked Pair in {unit_name}.\n"
                        f"Cells {_fmt_cell(*a1)} and {_fmt_cell(*a2)} share the same two candidates {digits}.\n"
                        f"Therefore {digits} can be eliminated from other cells in {unit_name}.\n"
                        f"Example elimination target: {_fmt_cell(*affected[0])}."
                    )
                    return HintResult(True, "Naked Pair", "ELIMINATE", msg)
    return None


def hint_hidden_pair(grid, cand) -> Optional[HintResult]:
    units = []
    for r in range(9):
        units.append(("row", r, [(r, c) for c in range(9)]))
    for c in range(9):
        units.append(("col", c, [(r, c) for r in range(9)]))
    for bi in range(9):
        units.append(("box", bi, cells_in_box(bi)))

    for kind, idx, cells in units:
        digit_cells: Dict[int, List[RC]] = {d: [] for d in range(1, 10)}
        for (r, c) in cells:
            if grid[r][c] != 0:
                continue
            m = cand[r][c]
            for d in range(1, 10):
                if m & bit(d):
                    digit_cells[d].append((r, c))

        digits = list(range(1, 10))
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                d1, d2 = digits[i], digits[j]
                loc1, loc2 = digit_cells[d1], digit_cells[d2]
                if len(loc1) == 2 and loc1 == loc2:
                    c1, c2 = loc1
                    unit_name = f"{kind} {idx+1}" if kind != "box" else f"box {idx+1}"
                    msg = (
                        f"Hidden Pair in {unit_name}.\n"
                        f"Digits {d1} and {d2} can only occur in {_fmt_cell(*c1)} and {_fmt_cell(*c2)}.\n"
                        f"Therefore those two cells must be restricted to {{{d1}, {d2}}} (remove other candidates there)."
                    )
                    return HintResult(True, "Hidden Pair", "RESTRICT", msg)
    return None


def hint_pointing_pair_triple(grid, cand) -> Optional[HintResult]:
    # In a box, digit candidates confined to one row/col => eliminate from that row/col outside the box
    for bi in range(9):
        box_cells = cells_in_box(bi)
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for (r, c) in box_cells if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) < 2:
                continue
            rows = {r for (r, _) in spots}
            cols = {c for (_, c) in spots}

            if len(rows) == 1:
                rr = next(iter(rows))
                targets = [(rr, c) for c in range(9) if (rr, c) not in box_cells and grid[rr][c] == 0 and (cand[rr][c] & b)]
                if targets:
                    msg = (
                        f"Pointing Pair/Triple in box {bi+1} for digit {d}.\n"
                        f"All candidates for {d} in this box are in row {rr+1}.\n"
                        f"Therefore eliminate {d} from row {rr+1} outside box {bi+1}.\n"
                        f"Example elimination target: {_fmt_cell(*targets[0])}."
                    )
                    return HintResult(True, "Pointing Pair/Triple", "ELIMINATE", msg)

            if len(cols) == 1:
                cc = next(iter(cols))
                targets = [(r, cc) for r in range(9) if (r, cc) not in box_cells and grid[r][cc] == 0 and (cand[r][cc] & b)]
                if targets:
                    msg = (
                        f"Pointing Pair/Triple in box {bi+1} for digit {d}.\n"
                        f"All candidates for {d} in this box are in column {cc+1}.\n"
                        f"Therefore eliminate {d} from column {cc+1} outside box {bi+1}.\n"
                        f"Example elimination target: {_fmt_cell(*targets[0])}."
                    )
                    return HintResult(True, "Pointing Pair/Triple", "ELIMINATE", msg)

    return None


def hint_claiming_box_line(grid, cand) -> Optional[HintResult]:
    # Row claiming: in a row, if all digit candidates fall inside one box => eliminate from rest of that box
    for r in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for c in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) < 2:
                continue
            boxes = {box_index(rr, cc) for (rr, cc) in spots}
            if len(boxes) == 1:
                bi = next(iter(boxes))
                targets = [(rr, cc) for (rr, cc) in cells_in_box(bi) if rr != r and grid[rr][cc] == 0 and (cand[rr][cc] & b)]
                if targets:
                    msg = (
                        f"Claiming (Box-Line) in row {r+1} for digit {d}.\n"
                        f"All candidates for {d} in row {r+1} are inside box {bi+1}.\n"
                        f"Therefore eliminate {d} from the rest of box {bi+1} (outside row {r+1}).\n"
                        f"Example elimination target: {_fmt_cell(*targets[0])}."
                    )
                    return HintResult(True, "Claiming (Box-Line)", "ELIMINATE", msg)

    # Column claiming
    for c in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for r in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) < 2:
                continue
            boxes = {box_index(rr, cc) for (rr, cc) in spots}
            if len(boxes) == 1:
                bi = next(iter(boxes))
                targets = [(rr, cc) for (rr, cc) in cells_in_box(bi) if cc != c and grid[rr][cc] == 0 and (cand[rr][cc] & b)]
                if targets:
                    msg = (
                        f"Claiming (Box-Line) in column {c+1} for digit {d}.\n"
                        f"All candidates for {d} in column {c+1} are inside box {bi+1}.\n"
                        f"Therefore eliminate {d} from the rest of box {bi+1} (outside column {c+1}).\n"
                        f"Example elimination target: {_fmt_cell(*targets[0])}."
                    )
                    return HintResult(True, "Claiming (Box-Line)", "ELIMINATE", msg)

    return None


def next_hint_from_grid(grid) -> HintResult:
    cand = init_candidates(grid)
    if cand is None:
        return HintResult(False, "—", "ERROR", "Cannot generate hint: grid is invalid or has a contradiction in candidates.")

    finders = [
        hint_naked_single,
        hint_hidden_single,
        hint_naked_pair,
        hint_hidden_pair,
        hint_pointing_pair_triple,
        hint_claiming_box_line,
    ]

    for f in finders:
        h = f(grid, cand)
        if h is not None and h.has_hint:
            return h

    return HintResult(False, "—", "ERROR", "No hint available: none of the 6 techniques produce a step from the current grid.")


def generate_hint(
    givens_board: Board,
    user_board: Board,
    mistake_report: Optional[MistakeReport] = None,
    reasons_map: Optional[ReasonMap] = None,
) -> HintResult:
    """
    Priority order:
      1) If validation fails -> return ERROR hint.
      2) If mistake exists -> return FIX_MISTAKE hint with proof when possible.
      3) Otherwise -> return next technique hint from current grid.
    """
    violation = generate_violation_report(givens_board, user_board)
    if violation.has_violation:
        return HintResult(True, "Validation", "ERROR", violation.explanation)

    if mistake_report is not None and mistake_report.has_mistake and mistake_report.items:
        item: MistakeItem = mistake_report.items[0]
        r, c = item.cell

        proof = ""
        if reasons_map is not None:
            p = reasons_map.get((r, c))
            if p:
                proof = f"\nProof technique: {p.get('technique','Unknown')}\nProof statement: {p.get('explanation','').strip()}"

        msg = (
            f"Fix the mistake first.\n"
            f"Cell {_fmt_cell(r,c)} is inconsistent.\n"
            f"Entered: {item.entered} | Expected: {item.expected}\n"
            f"Why: This value contradicts the forced deductions from the givens, so the puzzle cannot be completed correctly."
            f"{proof}\n"
            f"Hint: Replace {item.entered} with {item.expected} in {_fmt_cell(r,c)} (or clear the cell and re-solve)."
        )
        return HintResult(True, "Fix Mistake", "FIX_MISTAKE", msg)

    # No mistakes -> provide next technique hint
    return next_hint_from_grid(user_board.grid)
