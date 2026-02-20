from __future__ import annotations

from typing import List
from flask import Flask, request, jsonify
from flask_cors import CORS

from sudoku_engine.board import Board
from sudoku_engine.reports import generate_violation_report, generate_mistake_report
from sudoku_engine.solver import solve_from_givens_only_with_reasons
from sudoku_engine.hints import generate_hint

app = Flask(__name__)
CORS(app)  # allow requests from your Swift app during development


def normalize_81(s: str) -> str:
    """
    Normalize an 81-character Sudoku string.
    Accepts digits + '.'; converts '.' -> '0'; removes whitespace.
    """
    if s is None:
        raise ValueError("Missing grid string.")
    s2 = "".join(ch for ch in s.strip() if ch not in [" ", "\n", "\t", "\r"])
    s2 = s2.replace(".", "0")
    if len(s2) != 81:
        raise ValueError(f"Expected 81 chars after stripping whitespace, got {len(s2)}")
    if not all(ch.isdigit() for ch in s2):
        raise ValueError("Grid string must contain only digits or '.'")
    return s2


def grid_to_81(grid: List[List[int]]) -> str:
    """Convert a 9x9 int grid into an 81-char string using '0' for blanks."""
    out = []
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            out.append(str(v) if 1 <= v <= 9 else "0")
    return "".join(out)


@app.post("/analyze")
def analyze():
    """
    Request JSON:
      {
        "givens":  "81 chars (digits/0/.)",
        "current": "81 chars (digits/0/.)"   # optional; if omitted uses givens
      }

    Response JSON:
      {
        "validation": {...},
        "solver": {...},
        "mistakes": {...},
        "hint": {...}
      }
    """
    try:
        data = request.get_json(force=True) or {}

        givens81 = normalize_81(data.get("givens", ""))
        current81 = normalize_81(data.get("current", givens81))

        givens_board = Board.from_strings(givens81)
        user_board = Board.from_strings(givens81, current81)

        # 1) VALIDATION
        violation = generate_violation_report(givens_board, user_board)
        if violation.has_violation:
            hint = generate_hint(givens_board, user_board, None, None)
            return jsonify({
                "validation": {"ok": False, "explanation": violation.explanation},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "summary": "Skipped (validation failed).", "items": []},
                "hint": {
                    "has_hint": hint.has_hint,
                    "technique": hint.technique,
                    "action": hint.action,
                    "message": hint.message,
                }
            })

        # 2) SOLVE (givens-only using your 6-technique engine)
        sol, reasons_map = solve_from_givens_only_with_reasons(user_board)
        if not sol.is_solvable or sol.solution_grid is None:
            hint = generate_hint(givens_board, user_board, None, reasons_map)
            return jsonify({
                "validation": {"ok": True, "explanation": "PASS: no tampering, no duplicates."},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "summary": "Skipped (solver stuck).", "items": []},
                "hint": {
                    "has_hint": hint.has_hint,
                    "technique": hint.technique,
                    "action": hint.action,
                    "message": hint.message,
                }
            })

        solution81 = grid_to_81(sol.solution_grid)

        # 3) MISTAKES (valid grid but inconsistent with solution)
        mr = generate_mistake_report(user_board, sol.solution_grid, reasons_map)
        items = []
        for it in mr.items:
            r, c = it.cell
            items.append({
                "r": r + 1,
                "c": c + 1,
                "entered": it.entered,
                "expected": it.expected,
                "explanation": it.explanation,
                "row_values": it.row_values,
                "col_values": it.col_values,
                "box_values": it.box_values
            })

        # 4) HINT (fix mistake first, otherwise next technique step)
        hint = generate_hint(givens_board, user_board, mr, reasons_map)

        return jsonify({
            "validation": {"ok": True, "explanation": "PASS: no tampering, no duplicates."},
            "solver": {"ok": True, "solution81": solution81},
            "mistakes": {"has_mistake": mr.has_mistake, "summary": mr.summary, "items": items},
            "hint": {
                "has_hint": hint.has_hint,
                "technique": hint.technique,
                "action": hint.action,
                "message": hint.message,
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # iPhone access on same Wi-Fi needs 0.0.0.0
    app.run(host="0.0.0.0", port=8000, debug=True)
