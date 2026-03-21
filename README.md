<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]

<br />
<div align="center">
<h3 align="center">Sudoku Solver</h3>
  <p align="center">
    A web-based sudoku solver with step-by-step hints and OCR puzzle import.
  </p>
</div>

---

## Table of Contents
- [About The Project](#about-the-project)
  - [About the Client](#about-the-client-ia-a01-client-scenario)
  - [Built With](#built-with-ia-a03-justification)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Prototype Status](#prototype-status)
- [Roadmap](#roadmap)

---

## About The Project

A browser-based sudoku assistant built for a specific client. The user can:
- Import a puzzle from a photo using OCR (works with both printed and handwritten grids)
- Review and correct the OCR reading before loading
- Get step-by-step solving hints explaining the technique used (Naked Single, Hidden Single, etc.)
- Reveal the full solution at any time

The back-end is a Python/Flask API. The front-end is plain HTML/CSS/JS — no framework.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### About the Client (IA A01 Client Scenario)

The client is a student who regularly solves sudoku puzzles by hand and wanted a digital tool to help when stuck. Key requirements:
- Import puzzles directly from a photo (the client writes puzzles by hand into a printed grid)
- Receive hints that explain *why* a digit belongs in a cell, not just what the digit is
- Be able to mark their own progress (given vs. user-entered cells shown distinctly)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### Built With (IA A03 Justification)

| Technology | Version | Role |
|---|---|---|
| Python | 3.10.7 | Back-end language |
| Flask | 3.1.2 | REST API server |
| flask-cors | — | Cross-origin requests from the browser |
| OpenCV | 4.13.0 | Image processing, grid detection, OCR preprocessing |
| scikit-learn | 1.7.2 | MLP classifier for handwritten digit recognition |
| NumPy | 2.2.6 | Array operations throughout the OCR pipeline |
| Tesseract / pytesseract | system | Printed digit recognition |
| Pillow | — | Image format support for pytesseract |
| joblib | — | Saving/loading the trained digit model |
| HTML / CSS / JS | — | Front-end (no framework) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

- Python 3.10+
- Tesseract OCR binary

Install Tesseract (macOS):
```sh
brew install tesseract
```

Install Tesseract (Linux):
```sh
sudo apt install tesseract-ocr
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/American-School-of-Barcelona/sudokusolver.git
   cd sudokusolver
   ```

2. Install Python dependencies
   ```sh
   pip install -r requirements.txt
   ```

### Running the App

1. Start the Flask API (in one terminal):
   ```sh
   python3 flask_api.py
   ```

2. Open `sudoku_web/index.html` in a browser (or serve it with any static file server).

The API runs on `http://127.0.0.1:8000` by default.

> **Retraining the digit model** (only needed if new training data is added):
> ```sh
> python3 -m ocr.train_classifier
> ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Prototype Status

### Completed features

- **OCR import** — photograph a sudoku grid (printed or handwritten), the app detects the grid with perspective correction, classifies each cell as blank/given/user-entry, and reads the digit
  - Printed digits: Tesseract with CLAHE preprocessing (~95% accuracy)
  - Handwritten digits: HOG + MLP with 7-way test-time augmentation (98% cross-validation accuracy)
  - Review modal: user can correct any misread cell before loading
  - Loading overlay shown during processing

- **Hint engine** — analyses the current board state and explains the next logical step:
  - Naked Single: only one candidate remains in a cell
  - Hidden Single: a digit can only go in one cell within a row, column, or box

- **Solve** — reveals the complete solution

- **Manual entry** — paste or type an 81-digit string to load a puzzle directly

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [x] Grid detection with perspective warp
- [x] Cell classification (blank / given / user-entry)
- [x] Printed digit OCR (Tesseract)
- [x] Handwritten digit OCR (HOG + MLP)
- [x] OCR review modal
- [x] Naked Single hints
- [x] Hidden Single hints
- [x] Hidden Single visual highlighting (show excluded cells and blocker houses)
- [ ] Pointing Pairs / Box-Line Reduction hints
- [ ] Pencil marks (candidate tracking)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/American-School-of-Barcelona/sudokusolver.svg?style=for-the-badge
[contributors-url]: https://github.com/American-School-of-Barcelona/sudokusolver/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/American-School-of-Barcelona/sudokusolver.svg?style=for-the-badge
[issues-url]: https://github.com/American-School-of-Barcelona/sudokusolver/issues
