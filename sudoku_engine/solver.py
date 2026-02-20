from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from sudoku_engine.board import Board
from sudoku_engine.models import SolutionResult

RC = Tuple[int, int]
ReasonMap = Dict[RC, Dict[str, str]]  # (r,c) -> {"technique": "...", "explanation": "..."}

ALL = (1 << 9) - 1  # bits for digits 1..9


# ------------------ bitmask helpers ------------------
def bit(d: int) -> int:
    return 1 << (d - 1)


def popcount(x: int) -> int:
    return int(x).bit_count()


def mask_to_digits(mask: int) -> List[int]:
    return [d for d in range(1, 10) if mask & bit(d)]


# ------------------ grid helpers ------------------
def box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)


def row_vals(grid, r: int) -> List[int]:
    return sorted({grid[r][c] for c in range(9) if grid[r][c] != 0})


def col_vals(grid, c: int) -> List[int]:
    return sorted({grid[r][c] for r in range(9) if grid[r][c] != 0})


def box_vals(grid, r: int, c: int) -> List[int]:
    br = (r // 3) * 3
    bc = (c // 3) * 3
    vals = set()
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            v = grid[rr][cc]
            if v != 0:
                vals.add(v)
    return sorted(vals)


def is_solved(grid) -> bool:
    return all(grid[r][c] != 0 for r in range(9) for c in range(9))


def init_state(grid):
    """
    Build:
      - used masks for rows/cols/boxes
      - candidate mask grid for empty cells
    Returns (ok, row_used, col_used, box_used, cand)
    """
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9

    # validate + build used masks
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            b = bit(v)
            bi = box_index(r, c)
            if (row_used[r] & b) or (col_used[c] & b) or (box_used[bi] & b):
                return False, row_used, col_used, box_used, None
            row_used[r] |= b
            col_used[c] |= b
            box_used[bi] |= b

    # init candidates
    cand = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                cand[r][c] = 0
                continue
            bi = box_index(r, c)
            allowed = ALL & ~(row_used[r] | col_used[c] | box_used[bi])
            cand[r][c] = allowed
            if allowed == 0:
                return False, row_used, col_used, box_used, cand

    return True, row_used, col_used, box_used, cand


def place(grid, cand, row_used, col_used, box_used, r: int, c: int, d: int) -> bool:
    """
    Place digit d into grid[r][c] and propagate candidate eliminations.
    Returns False if contradiction occurs.
    """
    if grid[r][c] != 0:
        return grid[r][c] == d

    b = bit(d)
    bi = box_index(r, c)

    # must be allowed
    if (row_used[r] & b) or (col_used[c] & b) or (box_used[bi] & b):
        return False

    # commit
    grid[r][c] = d
    cand[r][c] = 0
    row_used[r] |= b
    col_used[c] |= b
    box_used[bi] |= b

    # remove from peers
    for cc in range(9):
        if grid[r][cc] == 0:
            before = cand[r][cc]
            cand[r][cc] &= ~b
            if cand[r][cc] == 0 and before != 0:
                return False
    for rr in range(9):
        if grid[rr][c] == 0:
            before = cand[rr][c]
            cand[rr][c] &= ~b
            if cand[rr][c] == 0 and before != 0:
                return False
    br = (r // 3) * 3
    bc = (c // 3) * 3
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            if grid[rr][cc] == 0:
                before = cand[rr][cc]
                cand[rr][cc] &= ~b
                if cand[rr][cc] == 0 and before != 0:
                    return False

    return True


# ------------------ 6 techniques ------------------
def naked_single(grid, cand, row_used, col_used, box_used, reasons: Optional[ReasonMap]) -> bool:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and popcount(cand[r][c]) == 1:
                d = mask_to_digits(cand[r][c])[0]
                if reasons is not None:
                    reasons[(r, c)] = {
                        "technique": "Naked Single",
                        "explanation": (
                            f"Only one candidate remained.\n"
                            f"- Row {r+1} has {row_vals(grid, r)}\n"
                            f"- Column {c+1} has {col_vals(grid, c)}\n"
                            f"- Box {box_index(r,c)+1} has {box_vals(grid, r, c)}\n"
                            f"Therefore cell (r{r+1}, c{c+1}) must be {d}."
                        )
                    }
                return place(grid, cand, row_used, col_used, box_used, r, c, d)
    return False


def hidden_single(grid, cand, row_used, col_used, box_used, reasons: Optional[ReasonMap]) -> bool:
    # rows
    for r in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for c in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) == 1:
                rr, cc = spots[0]
                if reasons is not None:
                    reasons[(rr, cc)] = {
                        "technique": "Hidden Single (Row)",
                        "explanation": (
                            f"In row {r+1}, digit {d} can only go in cell (r{rr+1}, c{cc+1}). "
                            f"All other empty cells in the row cannot take {d}."
                        )
                    }
                return place(grid, cand, row_used, col_used, box_used, rr, cc, d)

    # cols
    for c in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for r in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) == 1:
                rr, cc = spots[0]
                if reasons is not None:
                    reasons[(rr, cc)] = {
                        "technique": "Hidden Single (Column)",
                        "explanation": (
                            f"In column {c+1}, digit {d} can only go in cell (r{rr+1}, c{cc+1}). "
                            f"All other empty cells in the column cannot take {d}."
                        )
                    }
                return place(grid, cand, row_used, col_used, box_used, rr, cc, d)

    # boxes
    for br in range(3):
        for bc in range(3):
            box_cells = [(r, c) for r in range(br*3, br*3+3) for c in range(bc*3, bc*3+3)]
            box_num = br*3 + bc + 1
            for d in range(1, 10):
                b = bit(d)
                spots = [(r, c) for (r, c) in box_cells if grid[r][c] == 0 and (cand[r][c] & b)]
                if len(spots) == 1:
                    rr, cc = spots[0]
                    if reasons is not None:
                        reasons[(rr, cc)] = {
                            "technique": "Hidden Single (Box)",
                            "explanation": (
                                f"In box {box_num}, digit {d} can only go in cell (r{rr+1}, c{cc+1}). "
                                f"All other empty cells in the box cannot take {d}."
                            )
                        }
                    return place(grid, cand, row_used, col_used, box_used, rr, cc, d)

    return False


def naked_pair(grid, cand, *_args) -> bool:
    # returns True if any candidate elimination happened
    changed = False

    # units: rows, cols, boxes
    units = []
    for r in range(9):
        units.append([(r, c) for c in range(9)])
    for c in range(9):
        units.append([(r, c) for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(r, c) for r in range(br*3, br*3+3) for c in range(bc*3, bc*3+3)])

    for unit in units:
        pair_map: Dict[int, List[RC]] = {}
        for (r, c) in unit:
            if grid[r][c] == 0 and popcount(cand[r][c]) == 2:
                pair_map.setdefault(cand[r][c], []).append((r, c))

        for mask, cells in pair_map.items():
            if len(cells) == 2:
                for (r, c) in unit:
                    if (r, c) in cells or grid[r][c] != 0:
                        continue
                    before = cand[r][c]
                    cand[r][c] &= ~mask
                    if cand[r][c] != before:
                        changed = True
                        if cand[r][c] == 0:
                            return True
    return changed


def hidden_pair(grid, cand, *_args) -> bool:
    changed = False

    units = []
    for r in range(9):
        units.append([(r, c) for c in range(9)])
    for c in range(9):
        units.append([(r, c) for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(r, c) for r in range(br*3, br*3+3) for c in range(bc*3, bc*3+3)])

    for unit in units:
        digit_cells: Dict[int, List[RC]] = {d: [] for d in range(1, 10)}
        for (r, c) in unit:
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
                cells1, cells2 = digit_cells[d1], digit_cells[d2]
                if len(cells1) == 2 and cells1 == cells2:
                    allowed = bit(d1) | bit(d2)
                    for (r, c) in cells1:
                        before = cand[r][c]
                        cand[r][c] &= allowed
                        if cand[r][c] != before:
                            changed = True
                            if cand[r][c] == 0:
                                return True
    return changed


def pointing_pair_triple(grid, cand, *_args) -> bool:
    changed = False
    # For each box and digit: if all candidate spots lie in one row/col, eliminate outside box in that row/col
    for br in range(3):
        for bc in range(3):
            box_cells = [(r, c) for r in range(br*3, br*3+3) for c in range(bc*3, bc*3+3)]
            for d in range(1, 10):
                b = bit(d)
                spots = [(r, c) for (r, c) in box_cells if grid[r][c] == 0 and (cand[r][c] & b)]
                if len(spots) < 2:
                    continue
                rows = {r for (r, _) in spots}
                cols = {c for (_, c) in spots}

                if len(rows) == 1:
                    rr = next(iter(rows))
                    for c in range(9):
                        if (rr, c) not in box_cells and grid[rr][c] == 0:
                            before = cand[rr][c]
                            cand[rr][c] &= ~b
                            if cand[rr][c] != before:
                                changed = True
                                if cand[rr][c] == 0:
                                    return True

                if len(cols) == 1:
                    cc = next(iter(cols))
                    for r in range(9):
                        if (r, cc) not in box_cells and grid[r][cc] == 0:
                            before = cand[r][cc]
                            cand[r][cc] &= ~b
                            if cand[r][cc] != before:
                                changed = True
                                if cand[r][cc] == 0:
                                    return True
    return changed


def claiming_box_line(grid, cand, *_args) -> bool:
    changed = False
    # Row claiming: if in a row, digit spots all in one box -> eliminate from rest of box
    for r in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for c in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) < 2:
                continue
            boxes = {box_index(rr, cc) for (rr, cc) in spots}
            if len(boxes) == 1:
                bi = next(iter(boxes))
                br = (bi // 3) * 3
                bc = (bi % 3) * 3
                for rr in range(br, br + 3):
                    for cc in range(bc, bc + 3):
                        if rr == r:
                            continue
                        if grid[rr][cc] == 0:
                            before = cand[rr][cc]
                            cand[rr][cc] &= ~b
                            if cand[rr][cc] != before:
                                changed = True
                                if cand[rr][cc] == 0:
                                    return True

    # Col claiming: if in a col, digit spots all in one box -> eliminate from rest of box
    for c in range(9):
        for d in range(1, 10):
            b = bit(d)
            spots = [(r, c) for r in range(9) if grid[r][c] == 0 and (cand[r][c] & b)]
            if len(spots) < 2:
                continue
            boxes = {box_index(rr, cc) for (rr, cc) in spots}
            if len(boxes) == 1:
                bi = next(iter(boxes))
                br = (bi // 3) * 3
                bc = (bi % 3) * 3
                for rr in range(br, br + 3):
                    for cc in range(bc, bc + 3):
                        if cc == c:
                            continue
                        if grid[rr][cc] == 0:
                            before = cand[rr][cc]
                            cand[rr][cc] &= ~b
                            if cand[rr][cc] != before:
                                changed = True
                                if cand[rr][cc] == 0:
                                    return True

    return changed


# ------------------ public API ------------------
def solve_from_givens_only_with_reasons(board: Board) -> tuple[SolutionResult, ReasonMap]:
    # keep only original givens
    grid = [
        [board.grid[r][c] if board.given_mask[r][c] else 0 for c in range(9)]
        for r in range(9)
    ]

    reasons: ReasonMap = {}
    ok, row_used, col_used, box_used, cand = init_state(grid)
    if not ok or cand is None:
        return SolutionResult(False, None), reasons

    techniques = [
        naked_single,
        hidden_single,
        naked_pair,
        hidden_pair,
        pointing_pair_triple,
        claiming_box_line,
    ]

    while True:
        if is_solved(grid):
            return SolutionResult(True, grid), reasons

        progressed = False
        for tech in techniques:
            # For eliminations, we don't need row/col/box masks, so pass anyway safely
            if tech(grid, cand, row_used, col_used, box_used, reasons):
                progressed = True
                break  # restart from first technique (your spec)
        if not progressed:
            return SolutionResult(False, None), reasons
