class SudokuBoard:
    # Uses 0 for empty

    def __init__(self, grid=None):
        self.grid = []
        self.load(grid)

    def load(self, grid=None):
        # If no grid provided: make a blank board
        if grid is None:
            self.grid = []
            for _ in range(9):
                self.grid.append([0] * 9)
            return

        # Basic validation (kept simple for IA)
        if len(grid) != 9:
            raise ValueError("Grid must have 9 rows.")

        for row in grid:
            if len(row) != 9:
                raise ValueError("Each row must have 9 columns.")
            for v in row:
                if type(v) != int or v < 0 or v > 9:
                    raise ValueError("Each cell must be an int from 0 to 9.")

        # Copy to avoid accidental external edits
        self.grid = []
        for row in grid:
            self.grid.append(row[:])

    # ---------- Easy access ----------
    def row(self, r):
        return self.grid[r]

    def col(self, c):
        column = []
        for r in range(9):
            column.append(self.grid[r][c])
        return column

    def box(self, box_r, box_c):
        start_r = box_r * 3
        start_c = box_c * 3
        values = []
        for r in range(start_r, start_r + 3):
            for c in range(start_c, start_c + 3):
                values.append(self.grid[r][c])
        return values

    def box_by_cell(self, r, c):
        return self.box(r // 3, c // 3)

    # ---------- Editing ----------
    def set(self, r, c, value):
        self.grid[r][c] = value

    def clear(self, r, c):
        self.grid[r][c] = 0

    # ---------- Printing ----------
    def print_board(self):
        for r in range(9):
            line = ""
            for c in range(9):
                v = self.grid[r][c]
                line += (". " if v == 0 else str(v) + " ")
                if c == 2 or c == 5:
                    line += "| "
            print(line.strip())
            if r == 2 or r == 5:
                print("-" * 21)

detected_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],

    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],

    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

board = SudokuBoard(detected_grid)
board.print_board()