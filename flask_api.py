from __future__ import annotations

from flask import Flask, request, jsonify
from flask_cors import CORS

from sudoku_engine.board import Board
from sudoku_engine.reports import generate_mistake_report

try:
    from sudoku_engine.solver import solve_from_givens_only
except ImportError:
    from sudoku_engine.solver import solve_from_givens_only_with_reasons as solve_from_givens_only

from sudoku_engine.hints import generate_hint

app = Flask(__name__)
CORS(app)


def _norm81(s: str) -> str:
    s = (s or "").replace(".", "0")
    s = "".join(s.split())
    if len(s) != 81:
        raise ValueError(f"Expected 81 characters, got {len(s)}")
    if any(ch not in "0123456789" for ch in s):
        raise ValueError("Only digits, 0, '.' and whitespace are allowed")
    return s


def _grid_to_81(grid) -> str:
    return "".join(str(grid[r][c]) for r in range(9) for c in range(9))


def _extract_solution_grid(sol):
    """
    Your solver result type can vary. Extract (ok, solution_grid_or_None).
    """
    if hasattr(sol, "is_solvable") and hasattr(sol, "solution_grid"):
        ok = bool(sol.is_solvable) and sol.solution_grid is not None
        return ok, sol.solution_grid

    if hasattr(sol, "ok") and hasattr(sol, "grid"):
        ok = bool(sol.ok) and sol.grid is not None
        return ok, sol.grid

    if isinstance(sol, dict):
        g = sol.get("solution_grid") or sol.get("grid")
        ok = bool(sol.get("is_solvable") or sol.get("ok")) and g is not None
        return ok, g

    return False, None


def _duplicate_violation(grid):
    """
    STRICT rule violation: duplicates in any row/col/box.
    Does NOT do 'unsolvable' or 'contradiction' checks.
    Returns (has_violation, explanation, cells)
    cells: list of {"r": int, "c": int, "digit": int}
    """
    bad = []

    def mark_group(cells):
        seen = {}
        for (r, c) in cells:
            v = grid[r][c]
            if v == 0:
                continue
            if v in seen:
                pr, pc = seen[v]
                bad.append({"r": pr + 1, "c": pc + 1, "digit": v})
                bad.append({"r": r + 1, "c": c + 1, "digit": v})
            else:
                seen[v] = (r, c)

    # rows
    for r in range(9):
        mark_group([(r, c) for c in range(9)])

    # cols
    for c in range(9):
        mark_group([(r, c) for r in range(9)])

    # boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = [(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)]
            mark_group(cells)

    if not bad:
        return False, "", []

    # De-dup by cell (same cell might be added multiple times)
    uniq = {(x["r"], x["c"]): x for x in bad}
    bad = list(uniq.values())

    explanation = (
        "Sudoku rule violation: there are duplicate digits in a row, column, or 3×3 box. "
        "Remove the duplicates before continuing."
    )
    return True, explanation, bad


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/analyze")
def analyze():
    try:
        data = request.get_json(force=True) or {}

        givens81 = _norm81(data.get("givens", ""))
        current_raw = data.get("current", None)
        current81 = (
            _norm81(current_raw)
            if current_raw is not None and str(current_raw).strip() != ""
            else givens81
        )

        givens_board = Board.from_strings(givens81)
        user_board = Board.from_strings(givens81, current81)

        # 1) GIVEN TAMPERING (must come first)
        tampered = []
        for r in range(9):
            for c in range(9):
                if givens_board.grid[r][c] != 0 and user_board.grid[r][c] != givens_board.grid[r][c]:
                    tampered.append({
                        "r": r + 1,
                        "c": c + 1,
                        "expected": givens_board.grid[r][c],
                        "got": user_board.grid[r][c],
                    })

        if tampered:
            return jsonify({
                "validation": {"ok": False, "explanation": "Given tampering: a fixed clue was changed."},
                "tampered": tampered,
                "duplicates": [],
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        # 2) RULE VIOLATION (duplicates only)
        has_v, expl, dup_cells = _duplicate_violation(user_board.grid)
        if has_v:
            return jsonify({
                "validation": {"ok": False, "explanation": expl},
                "tampered": [],
                "duplicates": dup_cells,
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        # 3) SOLVE TRUE SOLUTION FROM GIVENS ONLY (critical for mistake detection)
        sol = solve_from_givens_only(givens_board)
        ok, solution_grid = _extract_solution_grid(sol)

        if not ok or solution_grid is None:
            # If solver can't solve, we still return a hint attempt
            hint = generate_hint(user_board)
            return jsonify({
                "validation": {"ok": True, "explanation": ""},
                "tampered": [],
                "duplicates": [],
                "hint": hint,
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        solution81 = _grid_to_81(solution_grid)

        # 4) MISTAKE REPORT (valid grid but inconsistent with canonical solution)
        mistakes = generate_mistake_report(user_board, solution_grid)

        if mistakes.has_mistake:
            # Mistakes MUST override hint highlighting
            return jsonify({
                "validation": {"ok": True, "explanation": ""},
                "tampered": [],
                "duplicates": [],
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": True, "solution81": solution81},
                "mistakes": {
                    "has_mistake": True,
                    "items": [
                        {
                            "r": it.cell[0] + 1,
                            "c": it.cell[1] + 1,
                            "entered": it.entered,
                            "expected": it.expected,
                            "explanation": it.explanation,
                        }
                        for it in mistakes.items
                    ],
                },
            })

        # 5) ONLY NOW generate hint (because board is valid AND mistake-free)
        hint = generate_hint(user_board)

        return jsonify({
            "validation": {"ok": True, "explanation": ""},
            "tampered": [],
            "duplicates": [],
            "hint": hint,
            "solver": {"ok": True, "solution81": solution81},
            "mistakes": {"has_mistake": False, "items": []},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)