"""
Shared immutable data-transfer objects (frozen dataclasses) and enums used
across the sudoku_engine package.

All classes here are pure value objects — they hold data only, contain no
logic, and cannot be mutated after construction (frozen=True).  They form the
contract between the engine layers (board, solver, reports) so that each layer
can depend on a stable, typed interface.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

RC = Tuple[int, int]  # (row, col), always 0-indexed internally


class ConflictType(str, Enum):
    """Identifies which type of Sudoku unit contains a rule violation."""
    ROW = "ROW"
    COL = "COL"
    BOX = "BOX"
    NONE = "NONE"


@dataclass(frozen=True)
class ValidationResult:
    """
    Returned by Board.validate_rules().

    is_valid       — False if any unit contains a duplicate digit.
    conflict_type  — which kind of unit the violation was found in.
    conflict_cells — the (r, c) pairs that are in conflict (0-indexed).
    """
    is_valid: bool
    conflict_type: ConflictType = ConflictType.NONE
    conflict_cells: List[RC] = field(default_factory=list)


@dataclass(frozen=True)
class SolutionResult:
    """
    Returned by the solver functions.

    is_solvable    — True if the puzzle has at least one valid solution.
    solution_grid  — the completed 9×9 grid, or None if unsolvable.
    """
    is_solvable: bool
    solution_grid: Optional[List[List[int]]] = None
