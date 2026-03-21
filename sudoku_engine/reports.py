from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

from sudoku_engine.board import Board

RC = Tuple[int, int]
ReasonMap = Dict[RC, Dict[str, str]]  # (r,c) → {"technique": "...", "explanation": "..."}


@dataclass(frozen=True)
class ViolationReport:
    """
    Result of checking the board for given-tampering or rule violations.
    has_violation=True means the board is in an illegal state.
    """
    has_violation: bool
    violation_type: str = ""      # "ROW" | "COL" | "BOX" | "GIVEN_TAMPERING"
    unit_index: int = -1          # 1-indexed row/col/box number (if applicable)
    digit: int = -1               # the offending digit (if applicable)
    conflict_cells: List[RC] = None
    explanation: str = ""


@dataclass(frozen=True)
class MistakeItem:
    """One cell where the player's entry differs from the forced solution."""
    cell: RC
    entered: int
    expected: int
    explanation: str
    row_values: List[int]
    col_values: List[int]
    box_values: List[int]


@dataclass(frozen=True)
class MistakeReport:
    """Summary of all incorrect non-given entries on the current board."""
    has_mistake: bool
    items: List[MistakeItem]
    summary: str


def _cells_1_indexed(cells: List[RC]) -> List[Tuple[int, int]]:
    return [(r + 1, c + 1) for (r, c) in cells]


def generate_violation_report(givens_board: Board, user_board: Board) -> ViolationReport:
    """
    Check the user's board for two categories of illegal state and return a
    report on the first violation found.

    Given tampering is checked first: if a fixed clue was changed, further
    duplicate checks are meaningless because the grid's structural validity
    depends on the givens being intact.
    Only if no tampering is found are row/column/box duplicates checked.
    """

    # A) Given tampering — the player changed a cell that was a fixed clue.
    edited = [
        (r, c) for r in range(9) for c in range(9)
        if user_board.given_mask[r][c] and user_board.grid[r][c] != givens_board.grid[r][c]
    ]
    if edited:
        return ViolationReport(
            has_violation=True,
            violation_type="GIVEN_TAMPERING",
            conflict_cells=edited,
            explanation=(
                "Given tampering detected: a given is a fixed clue and must never be changed.\n"
                f"Edited given cells (1-indexed): {_cells_1_indexed(edited)}."
            )
        )

    # B) Sudoku rule violations (duplicate digits in a unit).
    # Only reached if no given tampering was detected above.
    user_grid = user_board.grid

    for r in range(9):
        positions: Dict[int, List[RC]] = {}
        for c in range(9):
            v = user_grid[r][c]
            if v == 0:
                continue
            positions.setdefault(v, []).append((r, c))
        for d, cells in positions.items():
            if len(cells) > 1:
                return ViolationReport(
                    has_violation=True,
                    violation_type="ROW",
                    unit_index=r + 1,
                    digit=d,
                    conflict_cells=cells,
                    explanation=(
                        f"Row rule violation: digit {d} appears more than once in row {r+1}.\n"
                        f"Conflict cells (1-indexed): {_cells_1_indexed(cells)}.\n"
                        "Sudoku rule: each digit 1–9 may appear at most once per row."
                    )
                )

    for c in range(9):
        positions = {}
        for r in range(9):
            v = user_grid[r][c]
            if v == 0:
                continue
            positions.setdefault(v, []).append((r, c))
        for d, cells in positions.items():
            if len(cells) > 1:
                return ViolationReport(
                    has_violation=True,
                    violation_type="COL",
                    unit_index=c + 1,
                    digit=d,
                    conflict_cells=cells,
                    explanation=(
                        f"Column rule violation: digit {d} appears more than once in column {c+1}.\n"
                        f"Conflict cells (1-indexed): {_cells_1_indexed(cells)}.\n"
                        "Sudoku rule: each digit 1–9 may appear at most once per column."
                    )
                )

    for box_i, box_cells in enumerate(user_board.boxes):
        positions = {}
        for (r, c) in box_cells:
            v = user_grid[r][c]
            if v == 0:
                continue
            positions.setdefault(v, []).append((r, c))
        box_num = box_i + 1
        for d, cells in positions.items():
            if len(cells) > 1:
                return ViolationReport(
                    has_violation=True,
                    violation_type="BOX",
                    unit_index=box_num,
                    digit=d,
                    conflict_cells=cells,
                    explanation=(
                        f"Box rule violation: digit {d} appears more than once in box {box_num}.\n"
                        f"Conflict cells (1-indexed): {_cells_1_indexed(cells)}.\n"
                        "Sudoku rule: each digit 1–9 may appear at most once per 3×3 box."
                    )
                )

    return ViolationReport(
        has_violation=False,
        conflict_cells=[],
        explanation="No violations detected (no given tampering, no row/col/box duplicates)."
    )


def generate_mistake_report(
    user_board: Board,
    solution_grid: List[List[int]],
    reasons_map: Optional[ReasonMap] = None
) -> MistakeReport:
    """
    Identify cells where the player entered a digit that contradicts the
    unique solution, and explain why each such entry is wrong.

    The explanation is NOT just "the solution says X."  It states that the
    puzzle's own constraints force the correct value — keeping the wrong digit
    would make the puzzle impossible to complete.  When a reasons_map is
    provided by the human-technique solver, the explanation also cites the
    specific logical step that forced the correct value.
    """
    user_grid = user_board.grid
    items: List[MistakeItem] = []

    for r in range(9):
        for c in range(9):
            if user_board.given_mask[r][c]:
                continue   # never accuse given clues
            entered = user_grid[r][c]
            if entered == 0:
                continue   # empty cells are not yet mistakes
            expected = solution_grid[r][c]
            if entered == expected:
                continue   # correct entry

            # Gather contextual values for the explanation.
            row_vals = sorted(v for v in user_grid[r] if v != 0)
            col_vals = sorted(user_grid[rr][c] for rr in range(9) if user_grid[rr][c] != 0)
            box_i    = (r // 3) * 3 + (c // 3)
            box_vals = sorted(
                user_grid[rr][cc]
                for (rr, cc) in user_board.boxes[box_i]
                if user_grid[rr][cc] != 0
            )

            # Build the explanation, optionally citing the solver's deduction proof.
            explanation = (
                f"You entered {entered} in (r{r+1}, c{c+1}), but this cell must be {expected}. "
                f"The puzzle's logic forces {expected} here — keeping {entered} makes it "
                f"impossible to complete the grid correctly."
            )
            if reasons_map is not None:
                proof = reasons_map.get((r, c))
                if proof:
                    explanation += (
                        f"\nProof — {proof.get('technique', 'Unknown')}: "
                        f"{proof.get('explanation', '').strip()}"
                    )

            items.append(MistakeItem(
                cell=(r, c),
                entered=entered,
                expected=expected,
                explanation=explanation,
                row_values=row_vals,
                col_values=col_vals,
                box_values=box_vals,
            ))

    if items:
        count = len(items)
        return MistakeReport(
            has_mistake=True,
            items=items,
            summary=f"{count} incorrect user entr{'y' if count == 1 else 'ies'} detected "
                    f"(valid so far, but logically inconsistent with the solution)."
        )

    return MistakeReport(
        has_mistake=False,
        items=[],
        summary="No mistakes detected: all user entries are consistent with the solved grid."
    )
