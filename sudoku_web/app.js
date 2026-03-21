// ─── Configuration ────────────────────────────────────────────────────────────
// Base URL of the Flask back-end.  All fetch() calls use this prefix so the
// URL only needs to change in one place if the server address changes.
const API_BASE = "http://127.0.0.1:8000";

// ─── DOM references ───────────────────────────────────────────────────────────
const boardEl   = document.getElementById("board");
const statusEl  = document.getElementById("status");

const line1 = document.getElementById("line1");
const line2 = document.getElementById("line2");
const line3 = document.getElementById("line3");

const modal      = document.getElementById("modal");
const givensBox  = document.getElementById("givensBox");
const currentBox = document.getElementById("currentBox");

const btnImport  = document.getElementById("btnImport");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnReveal  = document.getElementById("btnReveal");
const btnLoad    = document.getElementById("btnLoad");
const btnExample = document.getElementById("btnExample");
const btnOCR     = document.getElementById("btnOCR");
const ocrInput   = document.getElementById("ocrInput");

const ocrModal      = document.getElementById("ocrModal");
const ocrPreviewImg = document.getElementById("ocrPreviewImg");
const ocrBoardEl    = document.getElementById("ocrBoard");
const btnOcrCancel  = document.getElementById("btnOcrCancel");
const btnOcrLoad    = document.getElementById("btnOcrLoad");

// ─── Application state ────────────────────────────────────────────────────────
// All mutable UI state lives here in one object, making it easy to reset and
// to understand at a glance what the app is tracking at any point in time.
let state = {
  givens81:  "0".repeat(81),   // 81-char string: fixed clues ('0' = not a given)
  current81: "0".repeat(81),   // 81-char string: current grid including user entries
  givenMask: Array(81).fill(false), // true at index i if cell i is a fixed given

  // Hint highlight state — set by applyResponseVisualState after each /analyze call.
  highlightCell:   null,        // {r, c} of the target hint cell, or null
  highlightScope:  "none",      // "none" | "rowcolbox" | "box" | "hidden-row" | "hidden-col" | "hidden-box"
  hintDigit:       null,        // digit string ("1"–"9") involved in the current hint
  excludedCells:   new Set(),   // flat indices of excluded cells (can't take hintDigit)
  constraintCells: new Set(),   // flat indices of blocker cells (hold the blocking digit)
  blockerRows:     new Set(),   // row indices that contain blocker digits (for house tinting)
  blockerCols:     new Set(),   // col indices that contain blocker digits
  blockerBoxes:    new Set(),   // box indices (0–8) that contain blocker digits

  // Error highlight state — set when the server reports violations or mistakes.
  badCells:   new Set(),        // flat indices of cells with an error (red fill)
  badOutline: new Set(),        // flat indices to also receive a red outline (first error)
};

// ─── Example puzzle ───────────────────────────────────────────────────────────
// A known solvable puzzle pre-loaded when the user clicks "Load Example".
// givens is the 81-char fixed clue string; current starts empty.
const EXAMPLE = {
  givens:  "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
  current: ""
};

// ─── Utility functions ────────────────────────────────────────────────────────

/** Update the status bar text at the bottom of the sidebar. */
function setStatus(msg) {
  statusEl.textContent = msg || "";
}

/**
 * Update the three hint-card text lines.
 * line1 is the technique name (large), line2 and line3 are explanatory sentences.
 */
function setExplain(titleText, descText, detailText) {
  line1.textContent = titleText  || "";
  line2.textContent = descText   || "";
  line3.textContent = detailText || "";
}

/** Remove all whitespace characters from a string. */
function stripWS(s) {
  return (s || "").replace(/\s+/g, "");
}

/**
 * Normalise a raw puzzle string to a clean 81-digit string.
 * Accepts '.' as a synonym for '0' (empty cell) and ignores whitespace.
 * Throws a descriptive Error on malformed input so the UI can show it.
 */
function normalize81(s) {
  const cleaned = stripWS(s).replaceAll(".", "0");
  if (cleaned.length !== 81) throw new Error(`Expected 81 characters, got ${cleaned.length}`);
  if (!/^[0-9]{81}$/.test(cleaned)) throw new Error("Only digits / 0 / '.' are allowed");
  return cleaned;
}

/**
 * Build a boolean mask from a givens string.
 * Index i is true if the character at position i is not '0' (i.e. it is a given).
 */
function buildGivenMask(givens81) {
  return Array.from({ length: 81 }, (_, i) => givens81[i] !== "0");
}

/** Convert (row, col) — both 0-indexed — to a flat index in the 81-element arrays. */
function indexRC(r, c) {
  return r * 9 + c;
}

/** Convert a flat index back to {r, c}. */
function rcFromIndex(i) {
  return { r: Math.floor(i / 9), c: i % 9 };
}

/**
 * Return the top-left corner of the 3×3 box containing (r, c).
 * br = first row of the box, bc = first column of the box.
 */
function boxOf(r, c) {
  return { br: Math.floor(r / 3) * 3, bc: Math.floor(c / 3) * 3 };
}

// ─── Hint parsing helpers ─────────────────────────────────────────────────────

/**
 * Extract the target cell {r, c} (0-indexed) from a hint response.
 * Prefers the structured `hint.cell` array the server provides; falls back to
 * parsing the human-readable message string as a safety net.
 */
function extractHintCell(resp) {
  if (resp?.hint?.cell && Array.isArray(resp.hint.cell) && resp.hint.cell.length === 2) {
    const r = Number(resp.hint.cell[0]);
    const c = Number(resp.hint.cell[1]);
    if (Number.isInteger(r) && Number.isInteger(c) && r >= 0 && r <= 8 && c >= 0 && c <= 8) {
      return { r, c };
    }
  }
  // Fallback: parse "(rX, cY)" from the message text.
  const msg = String(resp?.hint?.message || "");
  const match = msg.match(/\(r(\d+),\s*c(\d+)\)/i);
  if (!match) return null;
  const r = parseInt(match[1], 10) - 1;   // server uses 1-indexed; convert to 0-indexed
  const c = parseInt(match[2], 10) - 1;
  if (r < 0 || r > 8 || c < 0 || c > 8) return null;
  return { r, c };
}

/**
 * Format a technique name for display.
 * "Hidden Single (Row)" → "Hidden Single · Row" so it reads better in the hint card.
 */
function prettyTechniqueName(name) {
  if (!name) return "Hint";
  return name.replace(/^Hidden Single\s*\((Row|Column|Box)\)$/i, "Hidden Single · $1");
}

/** Return the digit string ("1"–"9") currently placed at cell (r, c), or "0" if empty. */
function getCellDigit(r, c) {
  return state.current81[indexRC(r, c)];
}

/**
 * Extract the digit involved in a hint response.
 * Prefers the structured `hint.digit` field; falls back to keyword parsing of the message.
 */
function extractHintDigit(resp) {
  const direct = resp?.hint?.digit ?? resp?.hint?.value;
  if (direct != null && /^[1-9]$/.test(String(direct))) return String(direct);

  const msg = String(resp?.hint?.message || "");
  let match = msg.match(/\bdigit\s+([1-9])\b/i);
  if (match) return match[1];

  match = msg.match(/\bvalue\s+(?:is|must be)\s+([1-9])\b/i);
  if (match) return match[1];

  return null;
}

// ─── Hidden-single visualisation ─────────────────────────────────────────────

/**
 * Find every cell in the same house (row / col / box) as the target that
 * contains a digit which blocks `digit` from being placed there.
 * Returns a Set of flat cell indices.
 *
 * A "blocking digit" is any placed digit that matches `digit` and appears in
 * a shared row, column, or box — preventing `digit` from being a candidate
 * for the cell at (r, c).
 */
function collectBlockingDigits(r, c, digit) {
  const blockers = new Set();

  // Same column: any other cell in column c that holds `digit`.
  for (let rr = 0; rr < 9; rr++) {
    if (rr !== r && getCellDigit(rr, c) === digit) blockers.add(indexRC(rr, c));
  }

  // Same row: any other cell in row r that holds `digit`.
  for (let cc = 0; cc < 9; cc++) {
    if (cc !== c && getCellDigit(r, cc) === digit) blockers.add(indexRC(r, cc));
  }

  // Same 3×3 box: any other cell in the box that holds `digit`.
  const b = boxOf(r, c);
  for (let rr = b.br; rr < b.br + 3; rr++) {
    for (let cc = b.bc; cc < b.bc + 3; cc++) {
      if ((rr !== r || cc !== c) && getCellDigit(rr, cc) === digit) {
        blockers.add(indexRC(rr, cc));
      }
    }
  }

  return blockers;
}

/**
 * Compute the visual data needed to render a Hidden Single hint:
 *   - excluded  — cells in the active house that can't take `digit` (shown purple)
 *   - constraints — cells outside the house that hold the blocking digits (inset border)
 *   - blockerRows/Cols/Boxes — which rows/cols/boxes contain a blocking digit
 *                             (those houses receive a light tint outside the active house)
 *
 * Works by scanning every non-target cell in the active house and calling
 * collectBlockingDigits on each one.  A cell is "excluded" if at least one
 * blocker exists; the blockers are classified by whether they share a row,
 * column, or box with the excluded cell.
 */
function buildHiddenSingleVisuals(resp) {
  const tech  = String(resp?.hint?.technique || "").toLowerCase();
  const cell  = extractHintCell(resp);
  const digit = extractHintDigit(resp);

  const excluded     = new Set();
  const constraints  = new Set();
  const blockerRows  = new Set();
  const blockerCols  = new Set();
  const blockerBoxes = new Set();

  if (!cell || !digit || !tech.includes("hidden single")) {
    return { digit: null, scope: "none", excluded, constraints, blockerRows, blockerCols, blockerBoxes };
  }

  /**
   * Examine one candidate cell in the active house (not the target).
   * If `digit` is blocked from going there, record the cell as excluded and
   * classify each blocker by which house type it belongs to relative to
   * the candidate — so the correct house can be tinted.
   */
  function processCandidate(candidateR, candidateC) {
    const idx = indexRC(candidateR, candidateC);
    if (state.current81[idx] !== "0") return;  // filled cells are not candidates

    const blockers = collectBlockingDigits(candidateR, candidateC, digit);
    if (blockers.size === 0) return;  // nothing blocking it here — not excluded

    excluded.add(idx);
    blockers.forEach(blockerIdx => {
      constraints.add(blockerIdx);
      const blockerRow = Math.floor(blockerIdx / 9);
      const blockerCol = blockerIdx % 9;

      // Classify the blocker's house type relative to the candidate cell.
      if (blockerCol === candidateC) {
        blockerCols.add(blockerCol);             // blocker is in the same column
      } else if (blockerRow === candidateR) {
        blockerRows.add(blockerRow);             // blocker is in the same row
      } else {
        blockerBoxes.add(Math.floor(blockerRow / 3) * 3 + Math.floor(blockerCol / 3));
      }
    });
  }

  if (tech.includes("row")) {
    for (let c = 0; c < 9; c++) {
      if (c !== cell.c) processCandidate(cell.r, c);
    }
    return { digit, scope: "hidden-row", excluded, constraints, blockerRows, blockerCols, blockerBoxes };
  }

  if (tech.includes("column")) {
    for (let r = 0; r < 9; r++) {
      if (r !== cell.r) processCandidate(r, cell.c);
    }
    return { digit, scope: "hidden-col", excluded, constraints, blockerRows, blockerCols, blockerBoxes };
  }

  // Box single — scan all cells in the target's 3×3 box except the target itself.
  const b = boxOf(cell.r, cell.c);
  for (let r = b.br; r < b.br + 3; r++) {
    for (let c = b.bc; c < b.bc + 3; c++) {
      if (r !== cell.r || c !== cell.c) processCandidate(r, c);
    }
  }
  return { digit, scope: "hidden-box", excluded, constraints, blockerRows, blockerCols, blockerBoxes };
}

// ─── Board rendering ──────────────────────────────────────────────────────────

/**
 * Reset all highlight / error state back to neutral.
 * Called before applying a new server response so stale visuals don't persist.
 */
function resetHighlights() {
  state.highlightCell   = null;
  state.badCells        = new Set();
  state.badOutline      = new Set();
  state.highlightScope  = "none";
  state.hintDigit       = null;
  state.excludedCells   = new Set();
  state.constraintCells = new Set();
  state.blockerRows     = new Set();
  state.blockerCols     = new Set();
  state.blockerBoxes    = new Set();
}

/**
 * Rebuild the entire board DOM from the current state.
 *
 * CSS class layering for hint visuals (later class wins within equal specificity):
 *   blocker-house → house-row/col/box → excluded-cell (compound) → hint-blue (!important)
 *
 * Each cell gets an <input> for user entry.  Given cells are disabled; user
 * cells are styled blue immediately via inline !important to prevent the
 * browser's default black-text flash before CSS kicks in.
 */
function renderBoard() {
  boardEl.innerHTML = "";

  for (let i = 0; i < 81; i++) {
    const { r, c } = rcFromIndex(i);
    const cell = document.createElement("div");
    cell.className = "cell";

    // Thick borders mark the 3×3 box boundaries.
    if (r % 3 === 0) cell.classList.add("thick-top");
    if (c % 3 === 0) cell.classList.add("thick-left");
    if (r === 8)     cell.classList.add("thick-bottom");
    if (c === 8)     cell.classList.add("thick-right");

    const isGiven = state.givenMask[i];
    cell.classList.add(isGiven ? "given" : "user");

    // Apply hint highlight classes if a hint is active.
    if (state.highlightCell) {
      const hr = state.highlightCell.r;
      const hc = state.highlightCell.c;
      const isTarget = (r === hr && c === hc);

      // LAYER 2 — active house (yellow).
      if (state.highlightScope === "hidden-row" && r === hr) {
        cell.classList.add("house-row");
      } else if (state.highlightScope === "hidden-col" && c === hc) {
        cell.classList.add("house-col");
      } else if (state.highlightScope === "hidden-box") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) cell.classList.add("house-box");
      } else if (state.highlightScope === "rowcolbox") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) cell.classList.add("box-yellow");
        if (r === hr || c === hc) cell.classList.add("rc-yellow");
      } else if (state.highlightScope === "box") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) cell.classList.add("box-yellow");
      }

      // LAYER 1 — blocker-house tint for cells outside the active house whose
      // row, col, or box contains a blocking digit.
      const inActiveHouse = (
        (state.highlightScope === "hidden-row" && r === hr) ||
        (state.highlightScope === "hidden-col" && c === hc) ||
        (state.highlightScope === "hidden-box" && (() => {
          const b = boxOf(hr, hc);
          return r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3;
        })())
      );

      if (state.highlightScope.startsWith("hidden-") && !inActiveHouse) {
        const cellBoxId = Math.floor(r / 3) * 3 + Math.floor(c / 3);
        if (state.blockerRows.has(r) || state.blockerCols.has(c) || state.blockerBoxes.has(cellBoxId)) {
          cell.classList.add("blocker-house");
        }
      }

      // LAYER 3 — excluded cell (purple fill inside the active house).
      if (state.excludedCells.has(i))   cell.classList.add("excluded-cell");
      // LAYER 4 — constraint cell (inset purple border on the blocking digit cell).
      if (state.constraintCells.has(i)) cell.classList.add("constraint-cell");
      // LAYER 5 — target cell (blue fill, always wins via !important).
      if (isTarget) cell.classList.add("hint-blue");
    }

    // Error classes — applied independently of hint state.
    if (state.badCells.has(i))   cell.classList.add("bad");
    if (state.badOutline.has(i)) cell.classList.add("bad-outline");

    // Build the input element.
    const input = document.createElement("input");
    input.type      = "text";
    input.inputMode = "numeric";
    input.maxLength = 1;
    input.value = state.current81[i] === "0" ? "" : state.current81[i];

    if (isGiven) {
      input.disabled = true;
    } else {
      // Force blue immediately via inline !important to avoid the browser's
      // default black text flash before the CSS rule has a chance to apply.
      input.style.setProperty("color", "#2f6fe4", "important");
      input.style.setProperty("-webkit-text-fill-color", "#2f6fe4", "important");

      input.addEventListener("input", (e) => {
        // Re-apply inline colour on every keystroke — some browsers reset inline
        // styles after the input event fires before CSS can override them.
        e.target.style.setProperty("color", "#2f6fe4", "important");
        e.target.style.setProperty("-webkit-text-fill-color", "#2f6fe4", "important");

        // Accept only one digit 1–9; strip anything else.
        const digit = (e.target.value || "").replace(/[^1-9]/g, "");
        e.target.value = digit;

        // Keep state.current81 in sync with what the user typed.
        const arr = state.current81.split("");
        arr[i] = digit === "" ? "0" : digit;
        state.current81 = arr.join("");
      });
    }

    cell.appendChild(input);

    // Show an "×N" badge on excluded cells so the player can see which digit
    // is being blocked and by which constraint.
    if (state.excludedCells.has(i) && state.hintDigit) {
      const badge = document.createElement("div");
      badge.className   = "excluded-badge";
      badge.textContent = `×${state.hintDigit}`;
      cell.appendChild(badge);
    }

    boardEl.appendChild(cell);
  }
}

// ─── Modal (import dialog) ────────────────────────────────────────────────────

function openModal() {
  modal.classList.add("show");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeModal() {
  modal.classList.remove("show");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}

// Close modal when the backdrop or the × button is clicked.
modal.addEventListener("click", (e) => {
  if (e.target?.dataset?.close === "1") closeModal();
});

btnImport.addEventListener("click", openModal);

// Pre-fill the import text areas with a known solvable example puzzle.
btnExample.addEventListener("click", () => {
  givensBox.value  = EXAMPLE.givens;
  currentBox.value = EXAMPLE.current;
});

/**
 * Parse the two import text areas, update app state, and re-render the board.
 * Only loads the board locally — no server call is made at this point.
 */
function loadFromTextboxes() {
  const givens81  = normalize81(givensBox.value);
  const currentRaw = stripWS(currentBox.value);
  const current81 = currentRaw.length ? normalize81(currentBox.value) : givens81;

  state.givens81  = givens81;
  state.current81 = current81;
  state.givenMask = buildGivenMask(givens81);

  resetHighlights();
  renderBoard();

  setExplain(
    "Preview loaded",
    "Edit the board on the left if you want.",
    "Then click Analyze / Get Hint."
  );
  setStatus("Board preview ready.");
}

btnLoad.addEventListener("click", () => {
  try {
    loadFromTextboxes();
    closeModal();
  } catch (err) {
    alert(err.message || String(err));
  }
});

// ─── Server communication ─────────────────────────────────────────────────────

/**
 * POST the current givens + user grid to the /analyze endpoint.
 * Returns the parsed JSON response, or an object with an `error` key on failure.
 */
async function callAnalyze() {
  const res = await fetch(`${API_BASE}/analyze`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ givens: state.givens81, current: state.current81 })
  });
  const data = await res.json();
  if (!res.ok) return { error: data.error || "Request failed" };
  return data;
}

// ─── Client-side duplicate detection ─────────────────────────────────────────

/**
 * Scan an 81-char grid string and return a Set of flat indices that are
 * involved in any row / column / box duplicate violation.
 *
 * This runs entirely in the browser for instant visual feedback as the user
 * types — it mirrors the server-side validation in board.get_all_conflict_cells()
 * but operates on the string representation rather than a Board object.
 */
function computeDuplicateCells(current81) {
  const bad = new Set();

  function markGroup(indices) {
    const seen = new Map();  // digit → first index that held it
    for (const idx of indices) {
      const d = current81[idx];
      if (d === "0") continue;
      if (seen.has(d)) {
        bad.add(idx);
        bad.add(seen.get(d));
      } else {
        seen.set(d, idx);
      }
    }
  }

  for (let r = 0; r < 9; r++) {
    markGroup(Array.from({ length: 9 }, (_, c) => indexRC(r, c)));
  }
  for (let c = 0; c < 9; c++) {
    markGroup(Array.from({ length: 9 }, (_, r) => indexRC(r, c)));
  }
  for (let br = 0; br < 9; br += 3) {
    for (let bc = 0; bc < 9; bc += 3) {
      const inds = [];
      for (let r = br; r < br + 3; r++)
        for (let c = bc; c < bc + 3; c++)
          inds.push(indexRC(r, c));
      markGroup(inds);
    }
  }
  return bad;
}

// ─── Response → UI mapping ────────────────────────────────────────────────────

/**
 * Convert a /analyze server response to three display strings for the hint card.
 * Priority order mirrors the server's step order:
 *   error → validation failure → incorrect entry → ambiguous → no hint → hint.
 */
function toExplainLines(resp) {
  if (resp.error) {
    return ["Error", resp.error, ""];
  }

  if (resp.validation?.ok === false) {
    return [
      "Validation failed",
      resp.validation.explanation || "Invalid board.",
      "Fix the highlighted cells, then analyze again."
    ];
  }

  if (resp.mistakes?.has_mistake && resp.mistakes.items?.length) {
    const it = resp.mistakes.items[0];
    const entered  = it.entered ?? "?";
    const expected = it.expected ?? "?";
    const row = it.r ?? (it.cell ? it.cell[0] + 1 : "?");
    const col = it.c ?? (it.cell ? it.cell[1] + 1 : "?");
    const cellLabel = `(r${row}, c${col})`;
    return [
      "Incorrect entry",
      `You entered ${entered} in ${cellLabel}, but this cell must be ${expected}.`,
      `${cellLabel} may not cause an immediate conflict, but the puzzle's logic already forces ${expected} here. Keeping ${entered} blocks the correct solution.`
    ];
  }

  if (resp.ambiguity?.has_multiple_solutions) {
    return [
      "Puzzle is ambiguous",
      "This grid has more than one valid solution.",
      "Because no single solution is forced, the app cannot give a reliable hint."
    ];
  }

  if (!resp.hint?.has_hint) {
    return [
      "No hint available",
      "None of the enabled techniques produce a step.",
      "Either enter more values, use Reveal Solution, or try a different puzzle."
    ];
  }

  const techniqueName = prettyTechniqueName(resp.hint.technique || "Hint");
  const messageParts  = (resp.hint.message || "").trim()
                          .split(/\n+/).map(s => s.trim()).filter(Boolean);

  return [
    techniqueName,
    messageParts[0] || "A logical step is available.",
    messageParts[1] || "Apply the step, then re-analyze."
  ];
}

/**
 * Apply all visual highlight / error state from a /analyze response.
 * Must be called before renderBoard() so the new state is reflected in the DOM.
 */
function applyResponseVisualState(resp) {
  resetHighlights();

  // Validation failure: highlight all duplicate cells, outline the first.
  if (resp.validation && !resp.validation.ok) {
    const duplicates = computeDuplicateCells(state.current81);
    state.badCells   = duplicates;
    const firstDuplicate = duplicates.values().next().value;
    if (firstDuplicate !== undefined) state.badOutline.add(firstDuplicate);
    return;
  }

  // Incorrect entries: highlight all wrong cells, outline the first one.
  if (resp.mistakes?.has_mistake && resp.mistakes.items?.length) {
    for (const it of resp.mistakes.items) {
      const row = it.r ?? (it.cell ? it.cell[0] + 1 : null);
      const col = it.c ?? (it.cell ? it.cell[1] + 1 : null);
      if (row != null && col != null) state.badCells.add(indexRC(row - 1, col - 1));
    }
    const first = resp.mistakes.items[0];
    const row0  = first.r ?? (first.cell ? first.cell[0] + 1 : null);
    const col0  = first.c ?? (first.cell ? first.cell[1] + 1 : null);
    if (row0 != null && col0 != null) state.badOutline.add(indexRC(row0 - 1, col0 - 1));
    return;
  }

  if (resp.ambiguity?.has_multiple_solutions) return;

  // Hint available: set highlight cell and scope so renderBoard() can apply classes.
  if (resp.hint?.has_hint) {
    state.highlightCell = extractHintCell(resp);

    const tech = String(resp?.hint?.technique || "").toLowerCase();

    if (tech.includes("hidden single") && state.highlightCell) {
      // Build the full hidden-single visual overlay (excluded cells, blocker houses, etc.)
      const visuals = buildHiddenSingleVisuals(resp);
      state.hintDigit       = visuals.digit;
      state.highlightScope  = visuals.scope;
      state.excludedCells   = visuals.excluded;
      state.constraintCells = visuals.constraints;
      state.blockerRows     = visuals.blockerRows;
      state.blockerCols     = visuals.blockerCols;
      state.blockerBoxes    = visuals.blockerBoxes;

      // The target cell is the answer — don't mark it as excluded or a blocker.
      const targetIdx = indexRC(state.highlightCell.r, state.highlightCell.c);
      state.excludedCells.delete(targetIdx);
      state.constraintCells.delete(targetIdx);
    } else if (tech.includes("naked single")) {
      // Naked single: highlight the target's row, col, and box.
      state.highlightScope = "rowcolbox";
    } else {
      // Other techniques: highlight the target's box only.
      state.highlightScope = "box";
    }
  }
}

/** Return true if the user has loaded any non-blank puzzle. */
function hasLoadedBoard() {
  return state.givens81 && state.givens81 !== "0".repeat(81);
}

// ─── Button handlers ──────────────────────────────────────────────────────────

btnAnalyze.addEventListener("click", async () => {
  if (!hasLoadedBoard()) { openModal(); return; }

  try {
    setStatus("Analyzing…");
    const resp = await callAnalyze();

    applyResponseVisualState(resp);
    const [titleText, descText, detailText] = toExplainLines(resp);
    setExplain(titleText, descText, detailText);

    renderBoard();
    setStatus("Done.");
  } catch (err) {
    alert(err.message || String(err));
    setStatus("Analysis failed.");
  }
});

btnReveal.addEventListener("click", async () => {
  if (!hasLoadedBoard()) { openModal(); return; }

  try {
    setStatus("Solving…");
    const resp   = await callAnalyze();
    const sol81  = resp?.solver?.solution81;

    if (resp?.solver?.ok && typeof sol81 === "string" && sol81.length === 81) {
      // Fill every non-given cell with the solution digit.
      const arr = state.current81.split("");
      for (let i = 0; i < 81; i++) {
        if (!state.givenMask[i]) arr[i] = sol81[i];
      }
      state.current81 = arr.join("");
      resetHighlights();
      renderBoard();
      setExplain("Solution applied", "The board has been filled in.", "");
      setStatus("Solution applied.");
      return;
    }

    if (resp?.ambiguity?.has_multiple_solutions) {
      resetHighlights();
      renderBoard();
      setExplain(
        "Puzzle is ambiguous",
        "This grid has more than one valid solution.",
        "There is no single correct solution to reveal."
      );
      setStatus("No single solution exists.");
      return;
    }

    if (resp?.solver?.reason === "no_solution") {
      resetHighlights();
      renderBoard();
      setExplain(
        "No solution exists",
        "This grid cannot be completed as a valid Sudoku.",
        "Check the givens or remove conflicting entries."
      );
      setStatus("Puzzle is unsolvable.");
      return;
    }

    applyResponseVisualState(resp);
    const [titleText, descText, detailText] = toExplainLines(resp);
    setExplain(titleText, descText, detailText);
    renderBoard();
    setStatus("No solution returned.");
  } catch (err) {
    setStatus(String(err.message || err));
  }
});

// ─── OCR import ───────────────────────────────────────────────────────────────

/**
 * Working copy of the 81 cells shown in the OCR review modal.
 * Each element is {digit: number (0 = empty), isGiven: boolean}.
 * Populated by openOcrModal() and read by btnOcrLoad.
 * @type {Array<{digit: number, isGiven: boolean}>}
 */
let ocrReviewCells = [];

/**
 * Open the OCR review modal with the image and initial cell data from the server.
 * The user can edit digits and toggle each cell's given/user classification before
 * confirming, so mistakes in the automated OCR pass can be corrected manually.
 *
 * @param {string} imageUrl - object URL of the uploaded image for the preview panel
 * @param {Array<{digit: number, isGiven: boolean}>} cells - 81-element OCR result
 */
function openOcrModal(imageUrl, cells) {
  ocrReviewCells = cells.map(c => ({ digit: c.digit, isGiven: c.isGiven }));
  ocrPreviewImg.src = imageUrl;
  renderOcrBoard();

  ocrModal.classList.add("show");
  ocrModal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeOcrModal() {
  ocrModal.classList.remove("show");
  ocrModal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");

  // Release the object URL to free memory.
  if (ocrPreviewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(ocrPreviewImg.src);
  }
  ocrPreviewImg.src = "";
}

/**
 * Render (or re-render) the 9×9 board inside the OCR review modal.
 * Each cell shows an editable input and a small "G/U" toggle badge.
 * Clicking the badge switches a cell between given (dark) and user (blue).
 */
function renderOcrBoard() {
  ocrBoardEl.innerHTML = "";

  for (let i = 0; i < 81; i++) {
    const r = Math.floor(i / 9);
    const c = i % 9;
    const cellData = ocrReviewCells[i];

    const cell = document.createElement("div");
    cell.className = "ocr-cell";

    // Thick borders for 3×3 box boundaries (same logic as the main board).
    if (r % 3 === 0) cell.classList.add("thick-top");
    if (c % 3 === 0) cell.classList.add("thick-left");
    if (r === 8)     cell.classList.add("thick-bottom");
    if (c === 8)     cell.classList.add("thick-right");

    cell.classList.add(cellData.isGiven ? "ocr-given" : "ocr-user");

    // Editable digit input.
    const input = document.createElement("input");
    input.type      = "text";
    input.inputMode = "numeric";
    input.maxLength = 1;
    input.value = cellData.digit > 0 ? String(cellData.digit) : "";

    input.addEventListener("input", (e) => {
      const val = (e.target.value || "").replace(/[^1-9]/g, "");
      e.target.value = val;
      ocrReviewCells[i].digit = val === "" ? 0 : parseInt(val, 10);
    });

    // Small corner badge — click to toggle between given and user entry.
    const badge = document.createElement("div");
    badge.className   = "ocr-toggle";
    badge.textContent = cellData.isGiven ? "G" : "U";
    badge.title       = cellData.isGiven ? "Click to mark as user entry" : "Click to mark as given";

    badge.addEventListener("click", () => {
      ocrReviewCells[i].isGiven = !ocrReviewCells[i].isGiven;
      // Update the cell's appearance without rebuilding the whole board,
      // so the user's focused input is not disrupted.
      cell.classList.toggle("ocr-given", ocrReviewCells[i].isGiven);
      cell.classList.toggle("ocr-user",  !ocrReviewCells[i].isGiven);
      badge.textContent = ocrReviewCells[i].isGiven ? "G" : "U";
      badge.title = ocrReviewCells[i].isGiven ? "Click to mark as user entry" : "Click to mark as given";
    });

    cell.appendChild(input);
    cell.appendChild(badge);
    ocrBoardEl.appendChild(cell);
  }
}

/**
 * Commit the OCR review cells to the main board state and close the modal.
 * Cells marked isGiven=true with a non-zero digit form the givens string;
 * all non-zero digits (regardless of type) form the current grid string.
 */
function confirmOcrLoad() {
  const givens81  = ocrReviewCells.map(c => c.isGiven && c.digit > 0 ? String(c.digit) : "0").join("");
  const current81 = ocrReviewCells.map(c => c.digit > 0 ? String(c.digit) : "0").join("");

  state.givens81  = givens81;
  state.current81 = current81;
  state.givenMask = buildGivenMask(givens81);

  resetHighlights();
  renderBoard();

  setExplain(
    "Board loaded from image",
    "Review the grid for any OCR errors.",
    "Then click Analyze / Get Hint."
  );
  setStatus("OCR import complete.");
  closeOcrModal();
}

// Close OCR modal via backdrop or × button.
ocrModal.addEventListener("click", (e) => {
  if (e.target?.dataset?.ocrClose === "1") closeOcrModal();
});

btnOcrCancel.addEventListener("click", closeOcrModal);
btnOcrLoad.addEventListener("click", confirmOcrLoad);

// Clicking the visible button triggers the hidden file input.
btnOCR.addEventListener("click", () => ocrInput.click());

ocrInput.addEventListener("change", async () => {
  const file = ocrInput.files[0];
  if (!file) return;

  // Create an object URL early — used for the image preview in the review modal.
  const imageUrl = URL.createObjectURL(file);

  setStatus("Reading image…");
  const formData = new FormData();
  formData.append("image", file);

  try {
    const res  = await fetch(`${API_BASE}/ocr`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.error) {
      URL.revokeObjectURL(imageUrl);
      alert(data.error || "OCR failed.");
      setStatus("OCR failed.");
      return;
    }
    setStatus("OCR complete — review the board.");
    openOcrModal(imageUrl, data.board);
  } catch (err) {
    URL.revokeObjectURL(imageUrl);
    alert(err.message || String(err));
    setStatus("OCR failed.");
  } finally {
    ocrInput.value = "";  // reset so the same file can be selected again
  }
});

// ─── Initialisation ───────────────────────────────────────────────────────────
// Render an empty board immediately and open the import modal so the user is
// prompted to load a puzzle without needing to find the Import button.
(function init() {
  renderBoard();
  setExplain(
    "No board loaded",
    "Import a puzzle using one of the buttons below.",
    "You can type an 81-character string or import from a screenshot."
  );
  setStatus("Import a grid to begin.");
})();
