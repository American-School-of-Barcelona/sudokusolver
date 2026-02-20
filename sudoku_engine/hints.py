from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any

from sudoku_engine.board import Board

RC = Tuple[int, int]


def _row_vals(grid: List[List[int]], r: int) -> Set[int]:
    return {v for v in grid[r] if v != 0}


def _col_vals(grid: List[List[int]], c: int) -> Set[int]:
    return {grid[r][c] for r in range(9) if grid[r][c] != 0}


def _box_vals(grid: List[List[int]], r: int, c: int) -> Set[int]:
    br = (r // 3) * 3
    bc = (c // 3) * 3
    s = set()
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            v = grid[rr][cc]
            if v != 0:
                s.add(v)
    return s


def _candidates(grid: List[List[int]], r: int, c: int) -> Set[int]:
    if grid[r][c] != 0:
        return set()
    used = _row_vals(grid, r) | _col_vals(grid, c) | _box_vals(grid, r, c)
    return {d for d in range(1, 10) if d not in used}


def _naked_single(grid: List[List[int]]) -> Optional[Dict[str, Any]]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                cand = _candidates(grid, r, c)
                if len(cand) == 1:
                    d = next(iter(cand))
                    br = (r // 3) * 3
                    bc = (c // 3) * 3
                    msg = (
                        f"Cell (r{r+1}, c{c+1}) has only one possible value.\n"
                        f"In its row/col/box, all other digits already appear, so the only candidate is {d}.\n"
                        f"Therefore place {d} at (r{r+1}, c{c+1})."
                    )
                    return {"has_hint": True, "technique": "Naked Single", "r": r+1, "c": c+1, "digit": d, "message": msg}
    return None


def _hidden_single_row(grid: List[List[int]]) -> Optional[Dict[str, Any]]:
    for r in range(9):
        missing = {d for d in range(1, 10)} - _row_vals(grid, r)
        if not missing:
            continue
        cand_map: Dict[int, List[int]] = {d: [] for d in missing}
        for c in range(9):
            if grid[r][c] == 0:
                cand = _candidates(grid, r, c)
                for d in cand & missing:
                    cand_map[d].append(c)
        for d, cols in cand_map.items():
            if len(cols) == 1:
                c = cols[0]
                msg = (
                    f"In row {r+1}, digit {d} can only go in one place.\n"
                    f"All other empty cells in row {r+1} cannot take {d} due to column/box constraints.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {"has_hint": True, "technique": "Hidden Single (Row)", "r": r+1, "c": c+1, "digit": d, "message": msg}
    return None


def _hidden_single_col(grid: List[List[int]]) -> Optional[Dict[str, Any]]:
    for c in range(9):
        missing = {d for d in range(1, 10)} - _col_vals(grid, c)
        if not missing:
            continue
        cand_map: Dict[int, List[int]] = {d: [] for d in missing}
        for r in range(9):
            if grid[r][c] == 0:
                cand = _candidates(grid, r, c)
                for d in cand & missing:
                    cand_map[d].append(r)
        for d, rows in cand_map.items():
            if len(rows) == 1:
                r = rows[0]
                msg = (
                    f"In column {c+1}, digit {d} can only go in one place.\n"
                    f"All other empty cells in column {c+1} cannot take {d} due to row/box constraints.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {"has_hint": True, "technique": "Hidden Single (Column)", "r": r+1, "c": c+1, "digit": d, "message": msg}
    return None


def _hidden_single_box(grid: List[List[int]]) -> Optional[Dict[str, Any]]:
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            present = set()
            empties: List[RC] = []
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    v = grid[r][c]
                    if v != 0:
                        present.add(v)
                    else:
                        empties.append((r, c))
            missing = {d for d in range(1, 10)} - present
            if not missing:
                continue

            cand_map: Dict[int, List[RC]] = {d: [] for d in missing}
            for (r, c) in empties:
                cand = _candidates(grid, r, c)
                for d in cand & missing:
                    cand_map[d].append((r, c))

            for d, cells in cand_map.items():
                if len(cells) == 1:
                    r, c = cells[0]
                    msg = (
                        f"In the 3×3 box (rows {br+1}-{br+3}, cols {bc+1}-{bc+3}), digit {d} fits only one cell.\n"
                        f"All other empty cells in that box cannot take {d} due to row/column constraints.\n"
                        f"Therefore place {d} at (r{r+1}, c{c+1})."
                    )
                    return {"has_hint": True, "technique": "Hidden Single (Box)", "r": r+1, "c": c+1, "digit": d, "message": msg}
    return None


def generate_hint(*args) -> Dict[str, Any]:
    """
    Flexible signature to avoid more import breakage.
    Supported calls:
      - generate_hint(board)
      - generate_hint(givens_board, user_board)
    Returns:
      dict with keys: has_hint, technique, message, r, c, digit (when available)
    """
    if len(args) == 1:
        board: Board = args[0]
    elif len(args) >= 2:
        board = args[-1]  # assume last is user_board
    else:
        return {"has_hint": False, "technique": None, "message": ""}

    grid = board.grid

    # Technique order (expand later if you want your full “6 techniques” list)
    for fn in (_naked_single, _hidden_single_row, _hidden_single_col, _hidden_single_box):
        res = fn(grid)
        if res is not None:
            return res

    return {
        "has_hint": False,
        "technique": None,
        "message": "No hint available using current techniques."
    }
