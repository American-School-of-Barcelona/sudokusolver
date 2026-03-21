from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sudoku_engine.board import Board, bit, mask_to_digits, popcount

RC = Tuple[int, int]


def _naked_single(grid, cand, board) -> Optional[Dict[str, Any]]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and popcount(cand[r][c]) == 1:
                d = mask_to_digits(cand[r][c])[0]
                msg = (
                    f"Cell (r{r+1}, c{c+1}) has only one possible value.\n"
                    f"In its row, column, and 3×3 box, all other digits already appear, "
                    f"so the only candidate is {d}.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {"has_hint": True, "technique": "Naked Single",
                        "r": r+1, "c": c+1, "digit": d, "cell": [r, c], "message": msg}
    return None


def _hidden_single_row(grid, cand, board) -> Optional[Dict[str, Any]]:
    for r in range(9):
        for d in range(1, 10):
            cols = [c for c in range(9) if grid[r][c] == 0 and (cand[r][c] & bit(d))]
            if len(cols) == 1:
                c = cols[0]
                msg = (
                    f"In row {r+1}, digit {d} can only go in one place.\n"
                    f"All other empty cells in row {r+1} cannot take {d} due to column/box constraints.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {"has_hint": True, "technique": "Hidden Single (Row)",
                        "r": r+1, "c": c+1, "digit": d, "cell": [r, c], "message": msg}
    return None


def _hidden_single_col(grid, cand, board) -> Optional[Dict[str, Any]]:
    for c in range(9):
        for d in range(1, 10):
            rows = [r for r in range(9) if grid[r][c] == 0 and (cand[r][c] & bit(d))]
            if len(rows) == 1:
                r = rows[0]
                msg = (
                    f"In column {c+1}, digit {d} can only go in one place.\n"
                    f"All other empty cells in column {c+1} cannot take {d} due to row/box constraints.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {"has_hint": True, "technique": "Hidden Single (Column)",
                        "r": r+1, "c": c+1, "digit": d, "cell": [r, c], "message": msg}
    return None


def _hidden_single_box(grid, cand, board) -> Optional[Dict[str, Any]]:
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            for d in range(1, 10):
                cells = [
                    (r, c)
                    for r in range(br, br + 3)
                    for c in range(bc, bc + 3)
                    if grid[r][c] == 0 and (cand[r][c] & bit(d))
                ]
                if len(cells) == 1:
                    r, c = cells[0]
                    msg = (
                        f"In the 3×3 box (rows {br+1}–{br+3}, cols {bc+1}–{bc+3}), "
                        f"digit {d} fits only one cell.\n"
                        f"All other empty cells in that box cannot take {d} due to row/column constraints.\n"
                        f"Therefore place {d} at (r{r+1}, c{c+1})."
                    )
                    return {"has_hint": True, "technique": "Hidden Single (Box)",
                            "r": r+1, "c": c+1, "digit": d, "cell": [r, c], "message": msg}
    return None


def _naked_pair(grid, cand, board) -> Optional[Dict[str, Any]]:
    for unit in board.units:
        pairs: Dict[int, List[RC]] = {}
        for (r, c) in unit:
            if grid[r][c] == 0 and popcount(cand[r][c]) == 2:
                pairs.setdefault(cand[r][c], []).append((r, c))
        for mask, cells in pairs.items():
            if len(cells) == 2:
                d1, d2 = mask_to_digits(mask)
                (r1, c1), (r2, c2) = cells
                affected = [
                    (r, c) for (r, c) in unit
                    if (r, c) not in cells and grid[r][c] == 0 and (cand[r][c] & mask)
                ]
                if affected:
                    msg = (
                        f"Naked Pair: cells (r{r1+1},c{c1+1}) and (r{r2+1},c{c2+1}) "
                        f"each have only candidates {d1} and {d2}.\n"
                        f"Since both must be placed in those two cells, "
                        f"{d1} and {d2} can be removed from all other cells in their shared unit."
                    )
                    return {"has_hint": True, "technique": "Naked Pair",
                            "r": r1+1, "c": c1+1, "digit": d1, "cell": [r1, c1], "message": msg}
    return None


def _hidden_pair(grid, cand, board) -> Optional[Dict[str, Any]]:
    for unit in board.units:
        digit_cells: Dict[int, List[RC]] = {d: [] for d in range(1, 10)}
        for (r, c) in unit:
            if grid[r][c] != 0:
                continue
            for d in range(1, 10):
                if cand[r][c] & bit(d):
                    digit_cells[d].append((r, c))
        for i in range(1, 10):
            for j in range(i + 1, 10):
                if len(digit_cells[i]) == 2 and digit_cells[i] == digit_cells[j]:
                    (r1, c1), (r2, c2) = digit_cells[i]
                    allowed = bit(i) | bit(j)
                    if (cand[r1][c1] & ~allowed) or (cand[r2][c2] & ~allowed):
                        msg = (
                            f"Hidden Pair: digits {i} and {j} can only appear in "
                            f"(r{r1+1},c{c1+1}) and (r{r2+1},c{c2+1}) within their unit.\n"
                            f"All other candidates can be removed from those two cells."
                        )
                        return {"has_hint": True, "technique": "Hidden Pair",
                                "r": r1+1, "c": c1+1, "digit": i, "cell": [r1, c1], "message": msg}
    return None


def _pointing_pair(grid, cand, board) -> Optional[Dict[str, Any]]:
    for box in board.boxes:
        box_set = set(box)
        for d in range(1, 10):
            bd = bit(d)
            locs = [(r, c) for (r, c) in box if grid[r][c] == 0 and (cand[r][c] & bd)]
            if len(locs) < 2:
                continue
            rows = {r for (r, _) in locs}
            cols = {c for (_, c) in locs}
            if len(rows) == 1:
                target_row = next(iter(rows))
                affected = [
                    (r, c) for (r, c) in board.rows[target_row]
                    if (r, c) not in box_set and grid[r][c] == 0 and (cand[r][c] & bd)
                ]
                if affected:
                    r0, c0 = locs[0]
                    msg = (
                        f"Pointing Pair: digit {d} in the 3×3 box is confined to row {target_row+1}.\n"
                        f"Therefore {d} can be eliminated from all other cells in row {target_row+1} outside the box."
                    )
                    return {"has_hint": True, "technique": "Pointing Pair",
                            "r": r0+1, "c": c0+1, "digit": d, "cell": [r0, c0], "message": msg}
            if len(cols) == 1:
                target_col = next(iter(cols))
                affected = [
                    (r, c) for (r, c) in board.cols[target_col]
                    if (r, c) not in box_set and grid[r][c] == 0 and (cand[r][c] & bd)
                ]
                if affected:
                    r0, c0 = locs[0]
                    msg = (
                        f"Pointing Pair: digit {d} in the 3×3 box is confined to column {target_col+1}.\n"
                        f"Therefore {d} can be eliminated from all other cells in column {target_col+1} outside the box."
                    )
                    return {"has_hint": True, "technique": "Pointing Pair",
                            "r": r0+1, "c": c0+1, "digit": d, "cell": [r0, c0], "message": msg}
    return None


def _box_line_reduction(grid, cand, board) -> Optional[Dict[str, Any]]:
    for r in range(9):
        for d in range(1, 10):
            bd = bit(d)
            locs = [(rr, cc) for (rr, cc) in board.rows[r] if grid[rr][cc] == 0 and (cand[rr][cc] & bd)]
            if len(locs) < 2:
                continue
            box_ids = {(rr // 3) * 3 + (cc // 3) for (rr, cc) in locs}
            if len(box_ids) == 1:
                box_i = next(iter(box_ids))
                affected = [
                    (rr, cc) for (rr, cc) in board.boxes[box_i]
                    if rr != r and grid[rr][cc] == 0 and (cand[rr][cc] & bd)
                ]
                if affected:
                    r0, c0 = locs[0]
                    msg = (
                        f"Box-Line Reduction: digit {d} in row {r+1} is confined to one 3×3 box.\n"
                        f"Therefore {d} can be eliminated from all other cells in that box."
                    )
                    return {"has_hint": True, "technique": "Box-Line Reduction",
                            "r": r0+1, "c": c0+1, "digit": d, "cell": [r0, c0], "message": msg}

    for c in range(9):
        for d in range(1, 10):
            bd = bit(d)
            locs = [(rr, cc) for (rr, cc) in board.cols[c] if grid[rr][cc] == 0 and (cand[rr][cc] & bd)]
            if len(locs) < 2:
                continue
            box_ids = {(rr // 3) * 3 + (cc // 3) for (rr, cc) in locs}
            if len(box_ids) == 1:
                box_i = next(iter(box_ids))
                affected = [
                    (rr, cc) for (rr, cc) in board.boxes[box_i]
                    if cc != c and grid[rr][cc] == 0 and (cand[rr][cc] & bd)
                ]
                if affected:
                    r0, c0 = locs[0]
                    msg = (
                        f"Box-Line Reduction: digit {d} in column {c+1} is confined to one 3×3 box.\n"
                        f"Therefore {d} can be eliminated from all other cells in that box."
                    )
                    return {"has_hint": True, "technique": "Box-Line Reduction",
                            "r": r0+1, "c": c0+1, "digit": d, "cell": [r0, c0], "message": msg}

    return None


def generate_hint(board: Board) -> Dict[str, Any]:
    grid = board.grid
    cand = board.cand

    for fn in (
        _naked_single,
        _hidden_single_row, _hidden_single_col, _hidden_single_box,
        _naked_pair,
        _hidden_pair,
        _pointing_pair,
        _box_line_reduction,
    ):
        res = fn(grid, cand, board)
        if res is not None:
            return res

    return {"has_hint": False, "technique": None, "message": "No hint available using current techniques."}
