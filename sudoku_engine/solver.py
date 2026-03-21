"""
Sudoku solving engine.

Two solvers are provided:

  solve_using_6_techniques(board)
      Human-technique solver that applies the same logical steps a person would
      use: naked singles, hidden singles, naked pairs, hidden pairs, pointing
      pairs, and box-line reduction.  It mutates the Board in place as it places
      digits, recording the reason for each placement in an optional ReasonMap.
      Because it mirrors human reasoning, it is the primary solver — its reasons
      map is used to generate meaningful mistake explanations.

  solve_exact_from_givens(board)
      Exact backtracking solver used only as a fallback when the human-technique
      solver gets stuck (the puzzle requires techniques beyond the 6 implemented).
      It finds the unique solution (if one exists) but cannot produce a reasons
      map, so its result is used only for mistake-checking and hint generation,
      not for explanations.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, List

from sudoku_engine.board import Board, popcount, mask_to_digits, bit
from sudoku_engine.models import SolutionResult

RC = Tuple[int, int]
ReasonMap = Dict[RC, Dict[str, str]]  # (r, c) → {"technique": ..., "explanation": ...}


# ------------------------------------------------------------------
# Private helper — shared board construction
# ------------------------------------------------------------------

def _build_givens_board(board: Board) -> Board:
    """
    Return a fresh Board containing only the puzzle's given clues (all
    user-entered digits cleared).  Starting fresh from givens guarantees
    that the solution found reflects the puzzle's intended answer rather
    than a state possibly corrupted by incorrect user entries.
    """
    givens_grid = [
        [board.grid[r][c] if board.given_mask[r][c] else 0 for c in range(9)]
        for r in range(9)
    ]
    return Board(givens_grid, [row[:] for row in board.given_mask])


# ------------------------------------------------------------------
# Public — human-technique solver (primary)
# ------------------------------------------------------------------

def solve_from_givens_only(board: Board) -> SolutionResult:
    """
    Solve the puzzle from its given clues using only the 6 human techniques.
    Returns a SolutionResult with is_solvable=False if the techniques get stuck.
    """
    working = _build_givens_board(board)
    if not working.validate_rules().is_valid or working.contradiction:
        return SolutionResult(False, None)
    solved = solve_using_6_techniques(working, reasons=None)
    return SolutionResult(True, working.grid) if solved else SolutionResult(False, None)


def solve_from_givens_only_with_reasons(board: Board) -> tuple[SolutionResult, ReasonMap]:
    """
    Same as solve_from_givens_only, but also populates a ReasonMap that records
    the technique and explanation for each digit placement.  The reasons map is
    used by generate_mistake_report to give the player a logical proof of why
    their entry is wrong.
    """
    working  = _build_givens_board(board)
    reasons: ReasonMap = {}
    if not working.validate_rules().is_valid or working.contradiction:
        return SolutionResult(False, None), reasons
    solved = solve_using_6_techniques(working, reasons=reasons)
    if not solved:
        return SolutionResult(False, None), reasons
    return SolutionResult(True, working.grid), reasons


# ------------------------------------------------------------------
# Public — exact backtracking solver (fallback)
# ------------------------------------------------------------------

def solve_exact_from_givens(board: Board, max_solutions: int = 2) -> tuple[int, Optional[List[List[int]]]]:
    """
    Exact backtracking solver used only as a fallback when the human-technique
    solver cannot complete the puzzle.

    Uses a Minimum Remaining Values (MRV) heuristic: always branch on the empty
    cell with the fewest legal digits first.  MRV dramatically prunes the search
    tree because cells with fewer options are more likely to cause a conflict
    early, cutting off dead branches before they grow large.

    Returns (solution_count, first_solution_grid) where solution_count is capped
    at max_solutions (default 2) — we only need to know whether the puzzle has
    0, 1, or multiple solutions, not the exact count.
    """
    # Work on a grid copy derived from givens only.
    grid = [
        [board.grid[r][c] if board.given_mask[r][c] else 0 for c in range(9)]
        for r in range(9)
    ]

    first_solution: Optional[List[List[int]]] = None
    solution_count = 0

    def search() -> None:
        nonlocal solution_count, first_solution
        # nonlocal is required because `search` is a recursive closure.
        # Without it Python would treat these as read-only references to the
        # enclosing scope, and assignment inside search() would create new
        # local variables instead of updating the outer ones.

        if solution_count >= max_solutions:
            return

        # MRV selection: find the empty cell with the fewest legal digits.
        best_rc:    Optional[RC]        = None
        best_cands: Optional[List[int]] = None

        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    legal = _legal_digits(grid, r, c)
                    if not legal:
                        return  # dead end — this branch has no valid completion
                    if best_cands is None or len(legal) < len(best_cands):
                        best_rc    = (r, c)
                        best_cands = legal

        if best_rc is None:
            # Every cell is filled — a complete valid solution was found.
            solution_count += 1
            if first_solution is None:
                first_solution = [row[:] for row in grid]
            return

        r, c = best_rc
        for d in best_cands:
            grid[r][c] = d
            search()
            grid[r][c] = 0   # undo — backtrack and try the next digit
            if solution_count >= max_solutions:
                return

    search()
    return solution_count, first_solution


def _legal_digits(grid: List[List[int]], r: int, c: int) -> List[int]:
    """
    Return all digits 1–9 that are not already present in the same row,
    column, or 3×3 box as (r, c).  Used only by the backtracking solver.
    """
    if grid[r][c] != 0:
        return []

    used = set(grid[r])                            # digits in the same row
    used |= {grid[rr][c] for rr in range(9)}       # digits in the same column

    box_row_start = (r // 3) * 3   # top-left corner of the 3×3 box containing (r, c)
    box_col_start = (c // 3) * 3
    used |= {
        grid[rr][cc]
        for rr in range(box_row_start, box_row_start + 3)
        for cc in range(box_col_start, box_col_start + 3)
    }

    used.discard(0)
    return [d for d in range(1, 10) if d not in used]


# ------------------------------------------------------------------
# Human-technique loop
# ------------------------------------------------------------------

def solve_using_6_techniques(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Apply the six human techniques in priority order, restarting from technique 1
    after every successful step.

    Restarting from the simplest technique after each step mirrors how a human
    solves: after placing a digit, simpler deductions often become available
    that were previously blocked.  Always applying the simplest available
    technique first also ensures a minimal, pedagogically meaningful reasons map.

    Returns True if the board is fully solved, False if no technique makes
    progress (the puzzle requires harder techniques not implemented here).
    """
    techniques = [
        naked_single,
        hidden_single,
        naked_pair,
        hidden_pair,
        pointing_pair_triple,
        claiming_box_line,
    ]

    while True:
        if board.contradiction or not board.validate_rules().is_valid:
            # Contradiction or rule violation — puzzle is unsolvable or internally corrupt.
            return False
        if board.is_solved():
            return True

        progress = False
        for technique in techniques:
            if technique(board, reasons):
                progress = True
                break   # restart from technique 1 (simplest first)
        if not progress:
            return False   # no technique made progress — solver is stuck


# ------------------------------------------------------------------
# Helper functions for the reasons map (explanation generation only)
# ------------------------------------------------------------------

def _box_number(r: int, c: int) -> int:
    """Return the 1-indexed box number (1–9, row-major) for cell (r, c)."""
    return (r // 3) * 3 + (c // 3) + 1


def _present_in_row(board: Board, r: int) -> List[int]:
    """Return sorted placed digits in row r — used only for reason text."""
    return sorted({board.grid[r][c] for c in range(9) if board.grid[r][c] != 0})


def _present_in_col(board: Board, c: int) -> List[int]:
    """Return sorted placed digits in column c — used only for reason text."""
    return sorted({board.grid[r][c] for r in range(9) if board.grid[r][c] != 0})


def _present_in_box(board: Board, r: int, c: int) -> List[int]:
    """Return sorted placed digits in the 3×3 box containing (r, c) — for reason text."""
    box_row_start = (r // 3) * 3
    box_col_start = (c // 3) * 3
    vals = {
        board.grid[rr][cc]
        for rr in range(box_row_start, box_row_start + 3)
        for cc in range(box_col_start, box_col_start + 3)
        if board.grid[rr][cc] != 0
    }
    return sorted(vals)


# ------------------------------------------------------------------
# Technique 1: Naked Single
# ------------------------------------------------------------------
def naked_single(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Find an empty cell with exactly one remaining candidate and place it.
    A Naked Single arises when every other digit has been eliminated from a cell
    by the presence of those digits in the cell's row, column, or box peers.
    """
    for r in range(9):
        for c in range(9):
            if board.grid[r][c] == 0 and popcount(board.cand[r][c]) == 1:
                d = mask_to_digits(board.cand[r][c])[0]

                if reasons is not None:
                    row_vals = _present_in_row(board, r)
                    col_vals = _present_in_col(board, c)
                    box_vals = _present_in_box(board, r, c)
                    reasons[(r, c)] = {
                        "technique": "Naked Single",
                        "explanation": (
                            f"After eliminating digits already present in row {r+1} {row_vals}, "
                            f"column {c+1} {col_vals}, and box {_box_number(r, c)} {box_vals}, "
                            f"this cell has only one candidate left: {d}."
                        )
                    }

                board.place_digit(r, c, d)
                return True
    return False


# ------------------------------------------------------------------
# Technique 2: Hidden Single
# ------------------------------------------------------------------
def hidden_single(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Find a digit that can go in only one cell within some row, column, or box,
    and place it.  'Hidden' because the cell may have other candidates; the
    forcing logic comes from the unit as a whole, not from the cell alone.
    """
    # --- Rows ---
    for r in range(9):
        positions: Dict[int, list] = {d: [] for d in range(1, 10)}
        for c in range(9):
            if board.grid[r][c] != 0:
                continue
            for d in range(1, 10):
                if board.cand[r][c] & bit(d):
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
                board.place_digit(rr, cc, d)
                return True

    # --- Columns ---
    for c in range(9):
        positions = {d: [] for d in range(1, 10)}
        for r in range(9):
            if board.grid[r][c] != 0:
                continue
            for d in range(1, 10):
                if board.cand[r][c] & bit(d):
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
                board.place_digit(rr, cc, d)
                return True

    # --- Boxes (use board.boxes to avoid recomputing box cell lists) ---
    for box_i, box_cells in enumerate(board.boxes):
        positions = {d: [] for d in range(1, 10)}
        for (r, c) in box_cells:
            if board.grid[r][c] != 0:
                continue
            for d in range(1, 10):
                if board.cand[r][c] & bit(d):
                    positions[d].append((r, c))
        box_num = box_i + 1   # 1-indexed for human-readable messages
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
                board.place_digit(rr, cc, d)
                return True

    return False


# ------------------------------------------------------------------
# Technique 3: Naked Pair
# ------------------------------------------------------------------
def naked_pair(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Find two cells in the same unit that share the exact same two candidates,
    then eliminate those two digits from every other cell in that unit.

    Because one of the two cells must take one digit and the other must take
    the second, no other cell in the unit can hold either digit.
    """
    for unit in board.units:
        pairs: Dict[int, list] = {}
        for (r, c) in unit:
            if board.grid[r][c] == 0 and popcount(board.cand[r][c]) == 2:
                pairs.setdefault(board.cand[r][c], []).append((r, c))

        for mask, cells in pairs.items():
            if len(cells) == 2:
                d1, d2 = mask_to_digits(mask)
                changed = False
                for (r, c) in unit:
                    if (r, c) in cells or board.grid[r][c] != 0:
                        continue
                    # Use the public eliminate_candidate API rather than
                    # mutating board.cand directly — this keeps contradiction
                    # tracking and future invariants encapsulated in Board.
                    changed |= board.eliminate_candidate(r, c, d1)
                    changed |= board.eliminate_candidate(r, c, d2)
                if changed:
                    return True
    return False


# ------------------------------------------------------------------
# Technique 4: Hidden Pair
# ------------------------------------------------------------------
def hidden_pair(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Find two digits that can only appear in exactly the same two cells within a
    unit, then restrict those cells to only those two digits (removing all other
    candidates from them).
    """
    for unit in board.units:
        digit_cells: Dict[int, list] = {d: [] for d in range(1, 10)}
        for (r, c) in unit:
            if board.grid[r][c] != 0:
                continue
            for d in range(1, 10):
                if board.cand[r][c] & bit(d):
                    digit_cells[d].append((r, c))

        for i in range(1, 10):
            for j in range(i + 1, 10):
                if len(digit_cells[i]) == 2 and digit_cells[i] == digit_cells[j]:
                    allowed = bit(i) | bit(j)
                    changed = False
                    for (r, c) in digit_cells[i]:
                        changed |= board.restrict_candidates(r, c, allowed)
                    if changed:
                        return True
    return False


# ------------------------------------------------------------------
# Technique 5: Pointing Pair / Triple
# ------------------------------------------------------------------
def pointing_pair_triple(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Find a digit confined to a single row or column within a 3×3 box, then
    eliminate it from the rest of that row/column outside the box.

    If all candidates for digit d in a box lie in one row, then d must be
    placed in that row (somewhere in the box), so it cannot appear elsewhere
    in that row.  The column case is symmetric.
    """
    for box in board.boxes:
        for d in range(1, 10):
            digit_bit = bit(d)
            locs = [(r, c) for (r, c) in box if board.grid[r][c] == 0 and (board.cand[r][c] & digit_bit)]
            if len(locs) < 2:
                continue

            rows = {r for (r, _) in locs}
            cols = {c for (_, c) in locs}

            if len(rows) == 1:
                target_row = next(iter(rows))
                changed = False
                for (rr, cc) in board.rows[target_row]:
                    if (rr, cc) not in box:
                        changed |= board.eliminate_candidate(rr, cc, d)
                if changed:
                    return True

            if len(cols) == 1:
                target_col = next(iter(cols))
                changed = False
                for (rr, cc) in board.cols[target_col]:
                    if (rr, cc) not in box:
                        changed |= board.eliminate_candidate(rr, cc, d)
                if changed:
                    return True
    return False


# ------------------------------------------------------------------
# Technique 6: Box-Line Reduction (Claiming)
# ------------------------------------------------------------------
def claiming_box_line(board: Board, reasons: Optional[ReasonMap]) -> bool:
    """
    Find a digit confined to one 3×3 box within a row or column, then
    eliminate it from all other cells in that box.

    If all candidates for digit d in a row lie in one box, then d must be
    placed in that row's intersection with the box, so it cannot appear in
    other rows of the same box.  The column case is symmetric.
    """
    # Check rows.
    for r in range(9):
        for d in range(1, 10):
            digit_bit = bit(d)
            locs = [(rr, cc) for (rr, cc) in board.rows[r]
                    if board.grid[rr][cc] == 0 and (board.cand[rr][cc] & digit_bit)]
            if len(locs) < 2:
                continue
            box_ids = {(rr // 3) * 3 + (cc // 3) for (rr, cc) in locs}
            if len(box_ids) == 1:
                box_i = next(iter(box_ids))
                changed = False
                for (rr, cc) in board.boxes[box_i]:
                    if rr != r:
                        changed |= board.eliminate_candidate(rr, cc, d)
                if changed:
                    return True

    # Check columns.
    for c in range(9):
        for d in range(1, 10):
            digit_bit = bit(d)
            locs = [(rr, cc) for (rr, cc) in board.cols[c]
                    if board.grid[rr][cc] == 0 and (board.cand[rr][cc] & digit_bit)]
            if len(locs) < 2:
                continue
            box_ids = {(rr // 3) * 3 + (cc // 3) for (rr, cc) in locs}
            if len(box_ids) == 1:
                box_i = next(iter(box_ids))
                changed = False
                for (rr, cc) in board.boxes[box_i]:
                    if cc != c:
                        changed |= board.eliminate_candidate(rr, cc, d)
                if changed:
                    return True

    return False
