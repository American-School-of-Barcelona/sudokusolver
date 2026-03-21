from __future__ import annotations

from flask import Flask, request, jsonify
from flask_cors import CORS

from sudoku_engine.board import Board
from sudoku_engine.reports import generate_mistake_report, generate_violation_report
from sudoku_engine.solver import solve_from_givens_only, solve_exact_from_givens
from sudoku_engine.hints import generate_hint

app = Flask(__name__)
CORS(app)


def _norm81(raw: str) -> str:
    """
    Normalise a raw puzzle string to a clean 81-digit string.
    Accepts '.' as a synonym for '0' (empty cell) and ignores whitespace.
    Raises ValueError with a descriptive message on any malformed input.
    """
    cleaned = (raw or "").replace(".", "0")
    cleaned = "".join(cleaned.split())     # strip all whitespace
    if len(cleaned) != 81:
        raise ValueError(f"Expected 81 characters, got {len(cleaned)}")
    if any(ch not in "0123456789" for ch in cleaned):
        raise ValueError("Only digits, '0', '.' and whitespace are allowed")
    return cleaned


def _grid_to_81(grid) -> str:
    """
    Flatten a 9×9 grid to an 81-character string, row by row left-to-right.
    This is the compact Sudoku string representation used throughout the API
    (9×9 = 81 cells, each encoded as a single digit character).
    """
    return "".join(str(grid[r][c]) for r in range(9) for c in range(9))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/ocr")
def ocr():
    """Receive an image file and return an 81-cell board via the OCR pipeline."""
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
    except Exception as exc:
        app.logger.exception("OCR pipeline error")
        return jsonify({"error": str(exc)}), 400


@app.post("/analyze")
def analyze():
    """
    Core analysis endpoint.  Receives the puzzle givens and the player's
    current grid, and returns validation results, a hint, and the solution.

    Steps are ordered from cheapest/most critical to most expensive:
      1. Given tampering   — cheapest; if a fixed clue was changed, nothing else matters.
      2. Rule violations   — cheap;   if there's a duplicate, hints are meaningless.
      3. Solve the puzzle  — moderate; needed for mistake-checking and hint generation.
      4. Mistake report    — cheap once the solution is known.
      5. Generate hint     — only reached when the board is correct so far.
    """
    try:
        data = request.get_json(force=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        givens81 = _norm81(data.get("givens", ""))
        current_raw = data.get("current", None)
        current81 = (
            _norm81(current_raw)
            if current_raw is not None and str(current_raw).strip() != ""
            else givens81
        )

        givens_board = Board.from_strings(givens81)
        user_board   = Board.from_strings(givens81, current81)

        # Step 1 & 2: given tampering, then rule violations.
        # generate_violation_report checks tampering first (structural integrity),
        # then duplicates (rule correctness) — returning at the first problem found.
        violation = generate_violation_report(givens_board, user_board)
        if violation.has_violation and violation.violation_type == "GIVEN_TAMPERING":
            tampered = [
                {"r": r + 1, "c": c + 1,
                 "expected": givens_board.grid[r][c],
                 "got": user_board.grid[r][c]}
                for (r, c) in (violation.conflict_cells or [])
            ]
            return jsonify({
                "validation": {"ok": False, "explanation": violation.explanation},
                "tampered": tampered, "duplicates": [],
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        if violation.has_violation:
            # Rule violation — use get_all_conflict_cells so the UI can highlight
            # every conflicting cell at once rather than just the first pair found.
            all_conflicts = user_board.get_all_conflict_cells()
            return jsonify({
                "validation": {"ok": False, "explanation": violation.explanation},
                "tampered": [], "duplicates": all_conflicts,
                "hint": {"has_hint": False, "technique": None, "message": ""},
                "solver": {"ok": False, "solution81": None},
                "mistakes": {"has_mistake": False, "items": []},
            })

        # Step 3: solve the puzzle.
        # Human-technique solver is preferred because it mirrors how a person solves,
        # producing a reasons map that powers meaningful mistake explanations.
        # If it gets stuck (the puzzle requires harder techniques), fall back to exact
        # backtracking, which can verify solvability and retrieve the unique solution.
        solution_result = solve_from_givens_only(givens_board)
        solution_grid = (
            solution_result.solution_grid
            if (solution_result.is_solvable and solution_result.solution_grid)
            else None
        )

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
                        "explanation": (
                            "This puzzle has more than one valid solution, "
                            "so no single forced hint exists."
                        ),
                    },
                    "mistakes": {"has_mistake": False, "items": []},
                })

            solution_grid = exact_grid

        solution81 = _grid_to_81(solution_grid)

        # Step 4: check for incorrect entries.
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

        # Step 5: generate a hint (only reached when the board has no mistakes).
        hint = generate_hint(user_board)

        return jsonify({
            "validation": {"ok": True, "explanation": ""},
            "tampered": [], "duplicates": [],
            "hint": hint,
            "solver": {"ok": True, "solution81": solution81},
            "mistakes": {"has_mistake": False, "items": []},
        })

    except Exception as exc:
        app.logger.exception("Unhandled error in /analyze")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
