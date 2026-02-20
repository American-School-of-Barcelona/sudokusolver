from __future__ import annotations
from typing import Dict, Tuple, Optional, List

from sudoku_engine.board import Board, popcount, mask_to_digits, bit
from sudoku_engine.models import SolutionResult

RC = Tuple[int, int]
ReasonMap = Dict[RC, Dict[str, str]]  # (r,c) -> {"technique":..., "explanation":...}


def solve_from_givens_only(board: Board) -> SolutionResult:
    # Build givens-only board
    givens_grid = [
        [board.grid[r][c] if board.given_mask[r][c] else 0 for c in range(9)]
        for r in range(9)
    ]
    givens_mask = [row[:] for row in board.given_mask]
    b = Board(givens_grid, givens_mask)

    if not b.validate_rules().is_valid or b.contradiction:
        return SolutionResult(False, None)

    solved = solve_using_6_techniques(b, reasons=None)
    if not solved:
        return SolutionResult(False, None)

    return SolutionResult(True, b.grid)


def solve_from_givens_only_with_reasons(board: Board) -> tuple[SolutionResult, ReasonMap]:
    """
    Same solver, but returns a reason map explaining WHY each placement was forced.
    """
    givens_grid = [
        [board.grid[r][c] if board.given_mask[r][c] else 0 for c in range(9)]
        for r in range(9)
    ]
    givens_mask = [row[:] for row in board.given_mask]
    b = Board(givens_grid, givens_mask)

    reasons: ReasonMap = {}

    if not b.validate_rules().is_valid or b.contradiction:
        return SolutionResult(False, None), reasons

    solved = solve_using_6_techniques(b, reasons=reasons)
    if not solved:
        return SolutionResult(False, None), reasons

    return SolutionResult(True, b.grid), reasons


def solve_using_6_techniques(b: Board, reasons: Optional[ReasonMap]) -> bool:
    techniques = [
        naked_single,
        hidden_single,
        naked_pair,
        hidden_pair,
        pointing_pair_triple,
        claiming_box_line,
    ]

    while True:
        if b.contradiction or (not b.validate_rules().is_valid):
            return False
        if b.is_solved():
            return True

        progress = False
        for tech in techniques:
            if tech(b, reasons):
                progress = True
                break  # restart from technique 1
        if not progress:
            return False


def _box_number_1_to_9(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3) + 1


def _present_in_row(b: Board, r: int) -> List[int]:
    return sorted({b.grid[r][c] for c in range(9) if b.grid[r][c] != 0})


def _present_in_col(b: Board, c: int) -> List[int]:
    return sorted({b.grid[r][c] for r in range(9) if b.grid[r][c] != 0})


def _present_in_box(b: Board, r: int, c: int) -> List[int]:
    br = (r // 3) * 3
    bc = (c // 3) * 3
    vals = set()
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            v = b.grid[rr][cc]
            if v != 0:
                vals.add(v)
    return sorted(vals)


# ---------- Technique 1: Naked Single ----------
def naked_single(b: Board, reasons: Optional[ReasonMap]) -> bool:
    for r in range(9):
        for c in range(9):
            if b.grid[r][c] == 0 and popcount(b.cand[r][c]) == 1:
                d = mask_to_digits(b.cand[r][c])[0]

                if reasons is not None:
                    row_vals = _present_in_row(b, r)
                    col_vals = _present_in_col(b, c)
                    box_vals = _present_in_box(b, r, c)
                    reasons[(r, c)] = {
                        "technique": "Naked Single",
                        "explanation": (
                            f"After eliminating digits already present in row {r+1} {row_vals}, "
                            f"column {c+1} {col_vals}, and box {_box_number_1_to_9(r,c)} {box_vals}, "
                            f"this cell has only one candidate left: {d}."
                        )
                    }

                b.place_digit(r, c, d)
                return True
    return False


# ---------- Technique 2: Hidden Single ----------
def hidden_single(b: Board, reasons: Optional[ReasonMap]) -> bool:
    # rows
    for r in range(9):
        positions = {d: [] for d in range(1, 10)}
        for c in range(9):
            if b.grid[r][c] != 0:
                continue
            mask = b.cand[r][c]
            for d in range(1, 10):
                if mask & bit(d):
                    positions[d].append((r, c))
        for d in range(1, 10):
            if len(positions[d]) == 1:
                (rr, cc) = positions[d][0]
                if reasons is not None:
                    reasons[(rr, cc)] = {
                        "technique": "Hidden Single (Row)",
                        "explanation": (
                            f"In row {r+1}, digit {d} can only go in cell (r{rr+1}, c{cc+1}). "
                            f"All other empty cells in the row cannot take {d} based on candidates."
                        )
                    }
                b.place_digit(rr, cc, d)
                return True

    # cols
    for c in range(9):
        positions = {d: [] for d in range(1, 10)}
        for r in range(9):
            if b.grid[r][c] != 0:
                continue
            mask = b.cand[r][c]
            for d in range(1, 10):
                if mask & bit(d):
                    positions[d].append((r, c))
        for d in range(1, 10):
            if len(positions[d]) == 1:
                (rr, cc) = positions[d][0]
                if reasons is not None:
                    reasons[(rr, cc)] = {
                        "technique": "Hidden Single (Column)",
                        "explanation": (
                            f"In column {c+1}, digit {d} can only go in cell (r{rr+1}, c{cc+1}). "
                            f"All other empty cells in the column cannot take {d} based on candidates."
                        )
                    }
                b.place_digit(rr, cc, d)
                return True

    # boxes
    for br in range(3):
        for bc in range(3):
            box_cells = [(r, c) for r in range(br*3, br*3+3) for c in range(bc*3, bc*3+3)]
            positions = {d: [] for d in range(1, 10)}
            for (r, c) in box_cells:
                if b.grid[r][c] != 0:
                    continue
                mask = b.cand[r][c]
                for d in range(1, 10):
                    if mask & bit(d):
                        positions[d].append((r, c))
            box_num = br * 3 + bc + 1
            for d in range(1, 10):
                if len(positions[d]) == 1:
                    (rr, cc) = positions[d][0]
                    if reasons is not None:
                        reasons[(rr, cc)] = {
                            "technique": "Hidden Single (Box)",
                            "explanation": (
                                f"In box {box_num}, digit {d} can only go in cell (r{rr+1}, c{cc+1}). "
                                f"All other empty cells in the box cannot take {d} based on candidates."
                            )
                        }
                    b.place_digit(rr, cc, d)
                    return True

    return False


# ---------- Technique 3: Naked Pair ----------
def naked_pair(b: Board, reasons: Optional[ReasonMap]) -> bool:
    for unit in b.units:
        pairs = {}
        for (r, c) in unit:
            if b.grid[r][c] == 0 and popcount(b.cand[r][c]) == 2:
                pairs.setdefault(b.cand[r][c], []).append((r, c))

        for mask, cells in pairs.items():
            if len(cells) == 2:
                changed = False
                for (r, c) in unit:
                    if (r, c) in cells or b.grid[r][c] != 0:
                        continue
                    before = b.cand[r][c]
                    b.cand[r][c] &= ~mask
                    if b.cand[r][c] != before:
                        changed = True
                        if b.cand[r][c] == 0:
                            b.contradiction = True
                            return True
                if changed:
                    b._update_contradiction_flag()
                    return True
    return False


# ---------- Technique 4: Hidden Pair ----------
def hidden_pair(b: Board, reasons: Optional[ReasonMap]) -> bool:
    for unit in b.units:
        digit_cells = {d: [] for d in range(1, 10)}
        for (r, c) in unit:
            if b.grid[r][c] != 0:
                continue
            mask = b.cand[r][c]
            for d in range(1, 10):
                if mask & bit(d):
                    digit_cells[d].append((r, c))

        digits = list(range(1, 10))
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                d1, d2 = digits[i], digits[j]
                cells1, cells2 = digit_cells[d1], digit_cells[d2]
                if len(cells1) == 2 and cells1 == cells2:
                    allowed = bit(d1) | bit(d2)
                    changed = False
                    for (r, c) in cells1:
                        changed |= b.restrict_candidates(r, c, allowed)
                    if changed:
                        return True
    return False


# ---------- Technique 5: Pointing Pair/Triple ----------
def pointing_pair_triple(b: Board, reasons: Optional[ReasonMap]) -> bool:
    for box in b.boxes:
        for d in range(1, 10):
            bd = bit(d)
            locs = [(r, c) for (r, c) in box if b.grid[r][c] == 0 and (b.cand[r][c] & bd)]
            if len(locs) < 2:
                continue
            rows = {r for (r, _) in locs}
            cols = {c for (_, c) in locs}

            if len(rows) == 1:
                target_row = next(iter(rows))
                changed = False
                for (rr, cc) in b.rows[target_row]:
                    if (rr, cc) not in box:
                        changed |= b.eliminate_candidate(rr, cc, d)
                if changed:
                    return True

            if len(cols) == 1:
                target_col = next(iter(cols))
                changed = False
                for (rr, cc) in b.cols[target_col]:
                    if (rr, cc) not in box:
                        changed |= b.eliminate_candidate(rr, cc, d)
                if changed:
                    return True
    return False


# ---------- Technique 6: Claiming / Box-Line Reduction ----------
def claiming_box_line(b: Board, reasons: Optional[ReasonMap]) -> bool:
    for r in range(9):
        for d in range(1, 10):
            bd = bit(d)
            locs = [(rr, cc) for (rr, cc) in b.rows[r] if b.grid[rr][cc] == 0 and (b.cand[rr][cc] & bd)]
            if len(locs) < 2:
                continue
            boxes = {(rr // 3) * 3 + (cc // 3) for (rr, cc) in locs}
            if len(boxes) == 1:
                box_i = next(iter(boxes))
                changed = False
                for (rr, cc) in b.boxes[box_i]:
                    if rr != r:
                        changed |= b.eliminate_candidate(rr, cc, d)
                if changed:
                    return True

    for c in range(9):
        for d in range(1, 10):
            bd = bit(d)
            locs = [(rr, cc) for (rr, cc) in b.cols[c] if b.grid[rr][cc] == 0 and (b.cand[rr][cc] & bd)]
            if len(locs) < 2:
                continue
            boxes = {(rr // 3) * 3 + (cc // 3) for (rr, cc) in locs}
            if len(boxes) == 1:
                box_i = next(iter(boxes))
                changed = False
                for (rr, cc) in b.boxes[box_i]:
                    if cc != c:
                        changed |= b.eliminate_candidate(rr, cc, d)
                if changed:
                    return True
    return False
