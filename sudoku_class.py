class SudokuBoard:
    # Simple 9x9 Sudoku board
    # Uses 0 for empty cells

    def __init__(self, grid=None):
        if grid is None:
            self.grid = []
            for _ in range(9):
                self.grid.append([0] * 9)
        else:
            self.grid = grid  # assume it's a valid 9x9 list of lists

    # ---------- Easy access ----------
    def row(self, r):
        return self.grid[r]  # returns the actual row list

    def col(self, c):
        column = []
        for r in range(9):
            column.append(self.grid[r][c])
        return column

    def box(self, box_r, box_c):
        """
        box_r and box_c are 0..2
        (0,0) = top-left box, (2,2) = bottom-right box
        Returns a list of 9 values.
        """
        start_r = box_r * 3
        start_c = box_c * 3
        values = []
        for r in range(start_r, start_r + 3):
            for c in range(start_c, start_c + 3):
                values.append(self.grid[r][c])
        return values

    def box_by_cell(self, r, c):
        """Get the 3x3 box that contains (r, c)."""
        return self.box(r // 3, c // 3)

    # ---------- Easy editing ----------
    def set(self, r, c, value):
        # value should be 0..9 (0 means empty)
        self.grid[r][c] = value

    def clear(self, r, c):
        self.grid[r][c] = 0

    # ---------- Easy to read printing ----------
    def print_board(self):
        for r in range(9):
            line = ""
            for c in range(9):
                v = self.grid[r][c]
                if v == 0:
                    line += ". "
                else:
                    line += str(v) + " "

                if c == 2 or c == 5:
                    line += "| "

            print(line.strip())

            if r == 2 or r == 5:
                print("-" * 21)


# ---- Example usage ----
if __name__ == "__main__":
    b = SudokuBoard()
    b.set(0, 0, 5)
    b.set(0, 1, 3)
    b.set(0, 4, 7)

    b.print_board()

    print("Row 0:", b.row(0))
    print("Col 0:", b.col(0))
    print("Box (0,0):", b.box(0, 0))
    print("Box containing (0,4):", b.box_by_cell(0, 4))
