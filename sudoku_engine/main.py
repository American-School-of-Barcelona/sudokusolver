import argparse
from typing import Tuple

from sudoku_engine.board import Board
from sudoku_engine.solver import solve_from_givens_only_with_reasons
from sudoku_engine.reports import generate_violation_report, generate_mistake_report
from sudoku_engine.hints import generate_hint

RC = Tuple[int, int]


def print_grid(grid):
    for r in range(9):
        if r in (3, 6):
            print("-" * 21)
        row = []
        for c in range(9):
            if c in (3, 6):
                row.append("|")
            v = grid[r][c]
            row.append(str(v) if v != 0 else ".")
        print(" ".join(row))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--givens", required=True, help="81-char givens (digits + . or 0)")
    p.add_argument("--current", required=False, help="81-char user grid (digits + . or 0)")
    p.add_argument("--hint", action="store_true", help="Print one next-step hint")
    args = p.parse_args()

    current_str = args.current if args.current is not None else args.givens

    givens_board = Board.from_strings(args.givens)
    user_board = Board.from_strings(args.givens, current_str)

    print("\nGIVENS:\n")
    print_grid(givens_board.grid)

    print("\nUSER GRID:\n")
    print_grid(user_board.grid)
    print()

    print("RUN REPORT")
    print("=" * 60)

    # 1) VALIDATION
    violation = generate_violation_report(givens_board, user_board)
    print("VALIDATION REPORT")
    print("-" * 60)
    if violation.has_violation:
        print("Status: FAIL")
        print(violation.explanation)
        print("=" * 60)
        return
    print("Status: PASS")
    print("No given tampering detected.")
    print("No Sudoku rule violations detected (no row/col/box duplicates).")
    print("-" * 60)

    # 2) SOLVE (givens only, technique loop)
    sol, reasons_map = solve_from_givens_only_with_reasons(user_board)
    print("SOLVER REPORT")
    print("-" * 60)
    if not sol.is_solvable or sol.solution_grid is None:
        print("Status: FAIL (Technique-only solver stuck)")
        print("Explanation: This puzzle cannot be completed using only the 6 techniques implemented.")
        print("=" * 60)
        return
    print("Status: PASS (Solved from givens only)")
    print("\nSOLUTION (from givens only):\n")
    print_grid(sol.solution_grid)
    print("-" * 60)

    # 3) MISTAKE REPORT
    mistake_report = generate_mistake_report(user_board, sol.solution_grid, reasons_map)
    print("MISTAKE REPORT")
    print("-" * 60)
    print(mistake_report.summary)

    if mistake_report.has_mistake:
        for item in mistake_report.items:
            r, c = item.cell
            print(f"\n- Cell (r{r+1}, c{c+1})")
            print(f"  Entered: {item.entered} | Expected: {item.expected}")
            print("  Why it's a mistake:")
            print("  " + item.explanation.replace("\n", "\n  "))
            print(f"  Row values: {item.row_values}")
            print(f"  Col values: {item.col_values}")
            print(f"  Box values: {item.box_values}")
    else:
        print("No incorrect entries detected so far. Your entries are consistent with the solved grid.")

    print("-" * 60)

    # 4) HINT REPORT (optional)
    if args.hint:
        hint = generate_hint(givens_board, user_board, mistake_report, reasons_map)
        print("HINT REPORT")
        print("-" * 60)
        if hint.has_hint:
            print(f"Technique: {hint.technique}")
            print(f"Action: {hint.action}")
            print(hint.message)
        else:
            print("No hint available.")
        print("-" * 60)

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
