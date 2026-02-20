from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

from sudoku_engine.board import Board

RC = Tuple[int, int]


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
                "A given is a fixed clue and must never be changed by the user. "
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
                        f"Digit {d} appears more than once in row {r+1} at "
                        f"{_cells_1_indexed(cells)}. This violates the Sudoku row rule (each digit 1–9 once per row)."
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
                        f"Digit {d} appears more than once in column {c+1} at "
                        f"{_cells_1_indexed(cells)}. This violates the Sudoku column rule (each digit 1–9 once per column)."
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
                            f"Digit {d} appears more than once in box {box_index} at "
                            f"{_cells_1_indexed(cells)}. This violates the Sudoku box rule (each digit 1–9 once per 3×3 box)."
                        )
                    )

    return ViolationReport(
        has_violation=False,
        conflict_cells=[],
        explanation="No violations detected (no given tampering, no row/col/box duplicates)."
    )


def generate_mistake_report(user_board: Board, solution_grid: List[List[int]], reasons_map=None) -> MistakeReport:
    """
    Mistake = non-given user entry differs from solved grid.
    Explanation includes a logical 'proof' from the human-technique solver when available.
    """
    g = user_board.grid
    items: List[MistakeItem] = []

    for r in range(9):
        for c in range(9):
            if user_board.given_mask[r][c]:
                continue
            entered = g[r][c]
            if entered == 0:
                continue
            expected = solution_grid[r][c]
            if entered == expected:
                continue

            row_vals = _values_in_cells(g, [(r, x) for x in range(9)])
            col_vals = _values_in_cells(g, [(x, c) for x in range(9)])
            box_i = _box_index(r, c)
            box_vals = _values_in_cells(g, _box_cells(box_i))

            proof = None
            if reasons_map is not None:
                proof = reasons_map.get((r, c))

            if proof:
                tech = proof.get("technique", "Unknown technique")
                proof_stmt = proof.get("explanation", "").strip()
                explanation = (
                    f"Logical reason this is a mistake:\n"
                    f"- Your entry: {entered}\n"
                    f"- Forced value: {expected}\n"
                    f"- Proof technique: {tech}\n"
                    f"- Proof statement: {proof_stmt}\n"
                    f"- Consequence: Even if {entered} does not create a duplicate yet, it contradicts a forced deduction, "
                    f"so the puzzle cannot be completed correctly."
                )
            else:
                explanation = (
                    f"Logical reason this is a mistake:\n"
                    f"- Your entry: {entered}\n"
                    f"- Forced value: {expected}\n"
                    f"- Proof: This cell becomes forced later in the deduction chain after eliminations.\n"
                    f"- Consequence: Keeping {entered} blocks any valid completion consistent with the givens."
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

    return MistakeReport(
        has_mistake=False,
        items=[],
        summary="No mistakes detected: all user entries match the solution."
    )
