from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

RC = Tuple[int, int]  # (row, col)


class ConflictType(str, Enum):
    ROW = "ROW"
    COL = "COL"
    BOX = "BOX"
    NONE = "NONE"


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    conflict_type: ConflictType = ConflictType.NONE
    conflict_cells: List[RC] = None


class MistakeType(str, Enum):
    RULE_VIOLATION = "RULE_VIOLATION"
    CLIENT_ENTRY_MISMATCH = "CLIENT_ENTRY_MISMATCH"


@dataclass(frozen=True)
class MistakeReport:
    is_mistake: bool
    mistake_type: Optional[MistakeType] = None
    conflicting_cells: List[RC] = None
    hint_text: str = ""


@dataclass(frozen=True)
class SolutionResult:
    is_solvable: bool
    solution_grid: Optional[List[List[int]]] = None


@dataclass(frozen=True)
class HintReport:
    technique: str
    target_cells: List[RC]
    support_cells: List[RC]
    explanation: str
