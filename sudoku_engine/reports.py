from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

from sudoku_engine.board import Board

RC = Tuple[int, int]
ReasonMap = Dict[RC, Dict[str, str]]  # (r,c) -> {"technique": "...", "explanation": "..."}


@dataclass(frozen=True)
class ViolationReport:
    has_violation: bool
    violation_type: str = ""          # "ROW" | "COL" | "BOX" | "GIVEN_TAMPERING"
    unit_index: int = -1              # 1..9 (if applicable)
    digit: int = -1
    conflict_cells: List[RC] = None
    explanation: str = ""


@dataclass(frozen=True)
class MistakeItem:
    cell: RC
    entered: int
    expected: int
    explanation: str
    row_values: List[int]
    col_values: List[int]
    box_values: List[int]


@dataclass(frozen=True)
class MistakeReport:
    has_mistake: bool
    items: List[MistakeItem]
    summary: str


def _cells_1_indexed(cells: List[RC]) -> List[Tuple[int, int]]:
    return [(r + 1, c + 1) for (r, c) in cells]


def _box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)  # 0..8


def _box_cells(box_i: int) -> List[RC]:
    br = (box_i // 3) * 3
    bc = (box_i % 3) * 3
    return [(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)]


def _values_in_cells(grid, cells: List[RC]) -> List[int]:
    vals = [grid[r][c] for (r, c) in cells if grid[r][c] != 0]
    return sorted(vals)


def generate_violation_report(givens_board: Board, user_board: Board) -> ViolationReport:
    """
    1) Given tampering (user changed a fixed clue)
    2) Sudoku rule violations (duplicate digit in row/col/box)
    Returns the FIRST detected violation with a clear explanation.
    """

    # A) Given tampering
    edited = []
    for r in range(9):
        for c in range(9):
            if user_board.given_mask[r][c]:
                expected = givens_board.grid[r][c]
                got = user_board.grid[r][c]
                if got != expected:
                    edited.append((r, c))
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

    g = user_board.grid

    # B) Row duplicates
    for r in range(9):
        positions: Dict[int, List[RC]] = {}
        for c in range(9):
            v = g[r][c]
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

    # C) Column duplicates
    for c in range(9):
        positions: Dict[int, List[RC]] = {}
        for r in range(9):
            v = g[r][c]
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

    # D) Box duplicates
    for br in range(3):
        for bc in range(3):
            positions: Dict[int, List[RC]] = {}
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    v = g[r][c]
                    if v == 0:
                        continue
                    positions.setdefault(v, []).append((r, c))

            box_index = br * 3 + bc + 1
            for d, cells in positions.items():
                if len(cells) > 1:
                    return ViolationReport(
                        has_violation=True,
                        violation_type="BOX",
                        unit_index=box_index,
                        digit=d,
                        conflict_cells=cells,
                        explanation=(
                            f"Box rule violation: digit {d} appears more than once in box {box_index}.\n"
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
    Mistake = a non-given cell where the user entered a digit that differs from the solved grid.

    The explanation is NOT 'because solution says so'.
    It is:
      - the correct value is forced by Sudoku constraints derived from the givens
      - we attach a proof from the human-technique solver when available (reasons_map)
      - therefore the user's value blocks any valid completion
    """
    g = user_board.grid
    items: List[MistakeItem] = []

    for r in range(9):
        for c in range(9):
            if user_board.given_mask[r][c]:
                continue  # never accuse givens
            entered = g[r][c]
            if entered == 0:
                continue
            expected = solution_grid[r][c]
            if entered == expected:
                continue

            # context values (useful for explanation screenshots)
            row_vals = _values_in_cells(g, [(r, x) for x in range(9)])
            col_vals = _values_in_cells(g, [(x, c) for x in range(9)])
            box_i = _box_index(r, c)
            box_vals = _values_in_cells(g, _box_cells(box_i))

            # proof from solver, if recorded
            proof_lines: List[str] = []
            if reasons_map is not None:
                proof = reasons_map.get((r, c))
                if proof:
                    proof_lines.append(f"Proof technique: {proof.get('technique', 'Unknown')}")
                    proof_lines.append(f"Proof statement: {proof.get('explanation', '').strip()}")
                else:
                    proof_lines.append("Proof technique: (not recorded for this exact cell)")
                    proof_lines.append(
                        "Proof statement: The solver reaches this value via earlier eliminations; "
                        "this cell becomes forced later in the deduction chain."
                    )
            else:
                proof_lines.append("Proof technique: (not available)")
                proof_lines.append("Proof statement: No reasons_map provided by solver.")

            explanation = (
                f"Logical reason this is a mistake:\n"
                f"- Your entry: {entered}\n"
                f"- Forced value: {expected}\n"
                f"- Why forced: the puzzle deductions from the original givens require {expected} at (r{r+1}, c{c+1}).\n"
                f"- {'; '.join(proof_lines)}\n"
                f"- Consequence: Keeping {entered} does not necessarily break Sudoku rules immediately, "
                f"but it makes the board inconsistent with the forced deduction chain, so the puzzle cannot be completed correctly."
            )

            items.append(
                MistakeItem(
                    cell=(r, c),
                    entered=entered,
                    expected=expected,
                    explanation=explanation,
                    row_values=row_vals,
                    col_values=col_vals,
                    box_values=box_vals,
                )
            )

    if items:
        return MistakeReport(
            has_mistake=True,
            items=items,
            summary=f"{len(items)} incorrect user entr{'y' if len(items)==1 else 'ies'} detected (valid so far, but logically inconsistent)."
        )

    return MistakeReport(
        has_mistake=False,
        items=[],
        summary="No mistakes detected: all user entries match the forced deductions / solution."
    )
