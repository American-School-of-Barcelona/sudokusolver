"""
Hint-generation engine for the Sudoku solver.

Each private function detects one solving technique and returns a hint dict
(or None if the technique does not apply to the current board state).

All functions operate on an up-to-date Board object whose candidate bitmasks
(board.cand) have already been propagated.  They never mutate the board —
they only read it to identify the next logical step for the player.

Technique order in generate_hint() reflects increasing complexity, so the
player always receives the simplest applicable hint first.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sudoku_engine.board import Board, bit, mask_to_digits, popcount


def _naked_single(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Naked Single: an empty cell whose candidate bitmask has exactly
    one bit set, meaning all other digits have been eliminated by peer constraints.
    The remaining candidate is forced — no other digit can legally go there.
    """
    grid = board.grid
    cand = board.cand
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and popcount(cand[r][c]) == 1:
                d = mask_to_digits(cand[r][c])[0]
                msg = (
                    f"Cell (r{r+1}, c{c+1}) has only one possible value.\n"
                    f"In its row, column, and 3×3 box, all other digits already appear, "
                    f"so the only remaining candidate is {d}.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {
                    "has_hint": True, "technique": "Naked Single",
                    "r": r + 1, "c": c + 1, "digit": d,
                    "cell": [r, c], "message": msg,
                }
    return None


def _hidden_single_row(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Hidden Single in a row: a digit that can only go in one cell of
    its row (even though that cell may have multiple candidates).
    'Hidden' because the forcing constraint comes from the row as a whole, not
    just from the cell's own candidate list.
    """
    grid = board.grid
    cand = board.cand
    for r in range(9):
        for d in range(1, 10):
            cols = [c for c in range(9) if grid[r][c] == 0 and (cand[r][c] & bit(d))]
            if len(cols) == 1:
                c = cols[0]
                msg = (
                    f"In row {r+1}, digit {d} can only go in one place.\n"
                    f"All other empty cells in row {r+1} are blocked from taking {d} "
                    f"by column or box constraints.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {
                    "has_hint": True, "technique": "Hidden Single (Row)",
                    "r": r + 1, "c": c + 1, "digit": d,
                    "cell": [r, c], "message": msg,
                }
    return None


def _hidden_single_col(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Hidden Single in a column: a digit that can only go in one cell
    of its column after accounting for row and box constraints.
    """
    grid = board.grid
    cand = board.cand
    for c in range(9):
        for d in range(1, 10):
            rows = [r for r in range(9) if grid[r][c] == 0 and (cand[r][c] & bit(d))]
            if len(rows) == 1:
                r = rows[0]
                msg = (
                    f"In column {c+1}, digit {d} can only go in one place.\n"
                    f"All other empty cells in column {c+1} are blocked from taking {d} "
                    f"by row or box constraints.\n"
                    f"Therefore place {d} at (r{r+1}, c{c+1})."
                )
                return {
                    "has_hint": True, "technique": "Hidden Single (Column)",
                    "r": r + 1, "c": c + 1, "digit": d,
                    "cell": [r, c], "message": msg,
                }
    return None


def _hidden_single_box(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Hidden Single in a 3×3 box: a digit that can fit in exactly one
    cell of a box after eliminating cells blocked by row/column constraints.
    """
    grid = board.grid
    cand = board.cand
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
                        f"digit {d} fits in only one cell.\n"
                        f"All other empty cells in that box are blocked from taking {d} "
                        f"by row or column constraints.\n"
                        f"Therefore place {d} at (r{r+1}, c{c+1})."
                    )
                    return {
                        "has_hint": True, "technique": "Hidden Single (Box)",
                        "r": r + 1, "c": c + 1, "digit": d,
                        "cell": [r, c], "message": msg,
                    }
    return None


def _naked_pair(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Naked Pair: two cells in the same unit that each have exactly the
    same two candidates.  Because one cell must take one of those digits and the
    other cell must take the other, both digits can be eliminated from every
    other cell in that shared unit.

    Only reported if at least one elimination would actually occur (otherwise
    the pattern exists but has no consequence yet).
    """
    grid = board.grid
    cand = board.cand
    for unit in board.units:
        # Collect cells that have exactly 2 candidates, grouped by their mask.
        pairs: Dict[int, list] = {}
        for (r, c) in unit:
            if grid[r][c] == 0 and popcount(cand[r][c]) == 2:
                pairs.setdefault(cand[r][c], []).append((r, c))

        for mask, cells in pairs.items():
            if len(cells) == 2:
                d1, d2 = mask_to_digits(mask)
                (r1, c1), (r2, c2) = cells
                # Only hint if this pair would actually eliminate something.
                affected = [
                    (r, c) for (r, c) in unit
                    if (r, c) not in cells and grid[r][c] == 0 and (cand[r][c] & mask)
                ]
                if affected:
                    msg = (
                        f"Naked Pair: cells (r{r1+1},c{c1+1}) and (r{r2+1},c{c2+1}) "
                        f"each have only candidates {d1} and {d2}.\n"
                        f"Since both digits must be placed in those two cells, "
                        f"{d1} and {d2} can be removed from all other cells in their shared unit."
                    )
                    return {
                        "has_hint": True, "technique": "Naked Pair",
                        "r": r1 + 1, "c": c1 + 1, "digit": d1,
                        "cell": [r1, c1], "message": msg,
                    }
    return None


def _hidden_pair(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Hidden Pair: two digits that can only appear in exactly the same
    two cells within a unit.  Even if those cells have other candidates, the
    two digits lock them in — so every other candidate can be removed from
    those two cells.
    """
    grid = board.grid
    cand = board.cand
    for unit in board.units:
        # Map each digit to the list of empty cells in this unit where it is a candidate.
        digit_cells: Dict[int, list] = {d: [] for d in range(1, 10)}
        for (r, c) in unit:
            if grid[r][c] != 0:
                continue
            for d in range(1, 10):
                if cand[r][c] & bit(d):
                    digit_cells[d].append((r, c))

        # Check every pair of digits (i, j) to see if they share exactly the same two cells.
        for i in range(1, 10):
            for j in range(i + 1, 10):
                if len(digit_cells[i]) == 2 and digit_cells[i] == digit_cells[j]:
                    (r1, c1), (r2, c2) = digit_cells[i]
                    allowed = bit(i) | bit(j)
                    # Only hint if there are extra candidates to remove.
                    if (cand[r1][c1] & ~allowed) or (cand[r2][c2] & ~allowed):
                        msg = (
                            f"Hidden Pair: digits {i} and {j} can only appear in "
                            f"(r{r1+1},c{c1+1}) and (r{r2+1},c{c2+1}) within their unit.\n"
                            f"All other candidates can be removed from those two cells."
                        )
                        return {
                            "has_hint": True, "technique": "Hidden Pair",
                            "r": r1 + 1, "c": c1 + 1, "digit": i,
                            "cell": [r1, c1], "message": msg,
                        }
    return None


def _pointing_pair(board: Board) -> Optional[Dict[str, Any]]:
    """
    Detect a Pointing Pair (or Triple): when all candidates for a digit within
    a 3×3 box lie on the same row or column, that digit can be eliminated from
    the rest of that row/column outside the box.

    The name 'pointing' reflects that the confined candidates 'point to' the
    row or column they share, allowing eliminations beyond the box boundary.
    """
    grid = board.grid
    cand = board.cand
    for box in board.boxes:
        box_set = set(box)
        for d in range(1, 10):
            bd = bit(d)
            locs = [(r, c) for (r, c) in box if grid[r][c] == 0 and (cand[r][c] & bd)]
            if len(locs) < 2:
                continue  # need at least two positions to form a pair/triple

            rows = {r for (r, _) in locs}
            cols = {c for (_, c) in locs}

            if len(rows) == 1:   # all candidates for d in this box share one row
                target_row = next(iter(rows))
                affected = [
                    (r, c) for (r, c) in board.rows[target_row]
                    if (r, c) not in box_set and grid[r][c] == 0 and (cand[r][c] & bd)
                ]
                if affected:
                    r0, c0 = locs[0]
                    msg = (
                        f"Pointing Pair: digit {d} in the 3×3 box is confined to row {target_row+1}.\n"
                        f"Therefore {d} can be eliminated from all other cells in "
                        f"row {target_row+1} that lie outside the box."
                    )
                    return {
                        "has_hint": True, "technique": "Pointing Pair",
                        "r": r0 + 1, "c": c0 + 1, "digit": d,
                        "cell": [r0, c0], "message": msg,
                    }

            if len(cols) == 1:   # all candidates for d in this box share one column
                target_col = next(iter(cols))
                affected = [
                    (r, c) for (r, c) in board.cols[target_col]
                    if (r, c) not in box_set and grid[r][c] == 0 and (cand[r][c] & bd)
                ]
                if affected:
                    r0, c0 = locs[0]
                    msg = (
                        f"Pointing Pair: digit {d} in the 3×3 box is confined to column {target_col+1}.\n"
                        f"Therefore {d} can be eliminated from all other cells in "
                        f"column {target_col+1} that lie outside the box."
                    )
                    return {
                        "has_hint": True, "technique": "Pointing Pair",
                        "r": r0 + 1, "c": c0 + 1, "digit": d,
                        "cell": [r0, c0], "message": msg,
                    }
    return None


def _box_line_reduction_for(board: Board, use_rows: bool) -> Optional[Dict[str, Any]]:
    """
    Shared logic for Box-Line Reduction across rows (use_rows=True) and
    columns (use_rows=False).

    Box-Line Reduction (also called 'claiming'): when all candidates for a digit
    in a row or column lie inside a single 3×3 box, that digit can be eliminated
    from all other cells in that box — because the row/col forces the digit to
    stay within the box.
    """
    grid = board.grid
    cand = board.cand
    lines = board.rows if use_rows else board.cols
    axis_label = "row" if use_rows else "column"

    for line_idx, line in enumerate(lines):
        for d in range(1, 10):
            bd = bit(d)
            locs = [(r, c) for (r, c) in line if grid[r][c] == 0 and (cand[r][c] & bd)]
            if len(locs) < 2:
                continue
            box_ids = {(r // 3) * 3 + (c // 3) for (r, c) in locs}
            if len(box_ids) != 1:
                continue  # candidates spread across multiple boxes — no reduction possible

            box_i = next(iter(box_ids))
            # Eliminate from the other cells in this box that aren't in our line.
            affected = [
                (r, c) for (r, c) in board.boxes[box_i]
                if (use_rows and r != line_idx) or (not use_rows and c != line_idx)
                if grid[r][c] == 0 and (cand[r][c] & bd)
            ]
            if affected:
                r0, c0 = locs[0]
                msg = (
                    f"Box-Line Reduction: digit {d} in {axis_label} {line_idx+1} "
                    f"is confined to one 3×3 box.\n"
                    f"Therefore {d} can be eliminated from all other cells in that box."
                )
                return {
                    "has_hint": True, "technique": "Box-Line Reduction",
                    "r": r0 + 1, "c": c0 + 1, "digit": d,
                    "cell": [r0, c0], "message": msg,
                }
    return None


def generate_hint(board: Board) -> Dict[str, Any]:
    """
    Return the simplest applicable hint for the current board state by applying
    solving techniques in order of increasing complexity.

    Simpler techniques (e.g. Naked Single) are returned first so the player
    can learn incrementally.  Only one hint is returned per call — just enough
    information to make one logical deduction.

    Returns a dict with at least:
      has_hint  — bool
      technique — name string (or None)
      message   — human-readable explanation (or empty string)
      cell      — [row, col] 0-indexed of the target cell (when has_hint=True)
      digit     — the digit involved (when has_hint=True)
    """
    # A contradicted board has no legal candidates — no hint makes sense.
    if board.contradiction:
        return {
            "has_hint": False, "technique": None,
            "message": "The board is in a contradicted state; no hint is possible.",
        }

    for technique in (
        _naked_single,
        _hidden_single_row, _hidden_single_col, _hidden_single_box,
        _naked_pair,
        _hidden_pair,
        _pointing_pair,
        lambda b: _box_line_reduction_for(b, use_rows=True),
        lambda b: _box_line_reduction_for(b, use_rows=False),
    ):
        result = technique(board)
        if result is not None:
            return result

    return {
        "has_hint": False, "technique": None,
        "message": "No hint available — the puzzle may require techniques beyond those implemented.",
    }
