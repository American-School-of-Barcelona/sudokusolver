from __future__ import annotations

from flask import Flask, request, jsonify
from flask_cors import CORS

from sudoku_engine.board import Board
from sudoku_engine.reports import generate_mistake_report
from sudoku_engine.solver import solve_from_givens_only, solve_exact_from_givens
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


def _duplicate_violation(grid):
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

    for r in range(9):
        mark_group([(r, c) for c in range(9)])
    for c in range(9):
        mark_group([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            mark_group([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])

    if not bad:
        return False, "", []

    uniq = {(x["r"], x["c"]): x for x in bad}
    explanation = (
        "Sudoku rule violation: there are duplicate digits in a row, column, or 3×3 box. "
        "Remove the duplicates before continuing."
    )
    return True, explanation, list(uniq.values())


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/ocr")
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    image_bytes = request.files["image"].read()
    try:
        from ocr.preprocess import preprocess
        from ocr.grid_detector import detect_and_warp
        from ocr.cell_extractor import extract_cells
        from ocr.classifier import classify_cells
        thresh, color = preprocess(image_bytes)
        warped_gray, warped_color = detect_and_warp(thresh, color)
        gray_cells, color_cells = extract_cells(warped_gray, warped_color)
        board = classify_cells(gray_cells, color_cells)
        return jsonify({"board": board})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


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

        # 1) Given tampering
        tampered = [
            {"r": r + 1, "c": c + 1,
             "expected": givens_board.grid[r][c], "got": user_board.grid[r][c]}
            for r in range(9) for c in range(9)
            if givens_board.grid[r][c] != 0
            and user_board.grid[r][c] != givens_board.grid[r][c]
        ]
        if tampered:
            return jsonify({
                "validation": {"ok": False, "explanation": "Given tampering: a fixed clue was changed."},
                "tampered": tampered, "duplicates": [],
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        # 2) Rule violation (duplicates)
        has_v, expl, dup_cells = _duplicate_violation(user_board.grid)
        if has_v:
            return jsonify({
                "validation": {"ok": False, "explanation": expl},
                "tampered": [], "duplicates": dup_cells,
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        # 3) Solve from givens; fall back to exact solver if human techniques get stuck
        sol = solve_from_givens_only(givens_board)
        solution_grid = sol.solution_grid if (sol.is_solvable and sol.solution_grid) else None

        if solution_grid is None:
            solution_count, exact_grid = solve_exact_from_givens(givens_board, max_solutions=2)

            if solution_count == 0 or exact_grid is None:
                return jsonify({
                    "validation": {"ok": False, "explanation": "This puzzle has no valid solution."},
                    "tampered": [], "duplicates": [],
                    "hint": {"has_hint": False, "technique": None, "message": ""},
                    "solver": {"ok": False, "solution81": None, "reason": "no_solution"},
                    "ambiguity": {"has_multiple_solutions": False},
                    "mistakes": {"has_mistake": False, "items": []},
                })

            if solution_count > 1:
                return jsonify({
                    "validation": {"ok": True, "explanation": ""},
                    "tampered": [], "duplicates": [],
                    "hint": {"has_hint": False, "technique": None, "message": ""},
                    "solver": {"ok": False, "solution81": None, "reason": "multiple_solutions"},
                    "ambiguity": {
                        "has_multiple_solutions": True,
                        "explanation": "This puzzle has more than one valid solution, so no single forced hint exists.",
                    },
                    "mistakes": {"has_mistake": False, "items": []},
                })

            solution_grid = exact_grid

        solution81 = _grid_to_81(solution_grid)

        # 4) Mistake report
        mistakes = generate_mistake_report(user_board, solution_grid)
        if mistakes.has_mistake:
            return jsonify({
                "validation": {"ok": True, "explanation": ""},
                "tampered": [], "duplicates": [],
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": True, "solution81": solution81},
                "mistakes": {
                    "has_mistake": True,
                    "items": [
                        {"r": it.cell[0] + 1, "c": it.cell[1] + 1,
                         "entered": it.entered, "expected": it.expected,
                         "explanation": it.explanation}
                        for it in mistakes.items
                    ],
                },
            })

        # 5) Generate hint
        hint = generate_hint(user_board)

        return jsonify({
            "validation": {"ok": True, "explanation": ""},
            "tampered": [], "duplicates": [],
            "hint": hint,
            "solver": {"ok": True, "solution81": solution81},
            "mistakes": {"has_mistake": False, "items": []},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
