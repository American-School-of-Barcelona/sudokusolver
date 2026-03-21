console.log("NEW APP JS LOADED");

const API_BASE = "http://127.0.0.1:8000";

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");

const line1 = document.getElementById("line1");
const line2 = document.getElementById("line2");
const line3 = document.getElementById("line3");

const modal = document.getElementById("modal");
const givensBox = document.getElementById("givensBox");
const currentBox = document.getElementById("currentBox");

const btnImport  = document.getElementById("btnImport");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnReveal  = document.getElementById("btnReveal");
const btnLoad    = document.getElementById("btnLoad");
const btnExample = document.getElementById("btnExample");
const btnOCR     = document.getElementById("btnOCR");
const ocrInput   = document.getElementById("ocrInput");

let state = {
  givens81: "0".repeat(81),
  current81: "0".repeat(81),
  givenMask: Array(81).fill(false),

  highlightCell: null,
  badCells: new Set(),
  badOutline: new Set(),

  highlightScope: "none",

  hintDigit: null,
  excludedCells: new Set(),
  constraintCells: new Set(),
  blockerRows: new Set(),
  blockerCols: new Set(),
  blockerBoxes: new Set(),
};

const EXAMPLE = {
  givens: "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
  current: ""
};

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function setExplain(a, b, c) {
  line1.textContent = a || "";
  line2.textContent = b || "";
  line3.textContent = c || "";
}

function stripWS(s) {
  return (s || "").replace(/\s+/g, "");
}

function normalize81(s) {
  const t = stripWS(s).replaceAll(".", "0");
  if (t.length !== 81) throw new Error(`Expected 81 characters, got ${t.length}`);
  if (!/^[0-9]{81}$/.test(t)) throw new Error("Only digits / 0 / '.' are allowed");
  return t;
}

function buildGivenMask(givens81) {
  const m = Array(81).fill(false);
  for (let i = 0; i < 81; i++) m[i] = givens81[i] !== "0";
  return m;
}

function indexRC(r, c) {
  return r * 9 + c;
}

function rcFromIndex(i) {
  return { r: Math.floor(i / 9), c: i % 9 };
}

function boxOf(r, c) {
  return { br: Math.floor(r / 3) * 3, bc: Math.floor(c / 3) * 3 };
}

function extractHintCell(resp) {
  if (resp?.hint?.cell && Array.isArray(resp.hint.cell) && resp.hint.cell.length === 2) {
    const r = Number(resp.hint.cell[0]);
    const c = Number(resp.hint.cell[1]);
    if (Number.isInteger(r) && Number.isInteger(c) && r >= 0 && r <= 8 && c >= 0 && c <= 8) {
      return { r, c };
    }
  }

  const msg = String(resp?.hint?.message || "");
  const m = msg.match(/\(r(\d+),\s*c(\d+)\)/i);
  if (!m) return null;

  const r = parseInt(m[1], 10) - 1;
  const c = parseInt(m[2], 10) - 1;
  if (r < 0 || r > 8 || c < 0 || c > 8) return null;

  return { r, c };
}

function prettyTechniqueName(name) {
  if (!name) return "Hint";
  return name.replace(/^Hidden Single\s*\((Row|Column|Box)\)$/i, "Hidden Single · $1");
}

function getCellDigit(r, c) {
  return state.current81[indexRC(r, c)];
}

function extractHintDigit(resp) {
  const direct = resp?.hint?.digit ?? resp?.hint?.value;
  if (direct != null && /^[1-9]$/.test(String(direct))) return String(direct);

  const msg = String(resp?.hint?.message || "");

  let m = msg.match(/\bdigit\s+([1-9])\b/i);
  if (m) return m[1];

  m = msg.match(/\bvalue\s+(?:is|must be)\s+([1-9])\b/i);
  if (m) return m[1];

  return null;
}

function collectBlockingDigits(r, c, digit) {
  const blockers = new Set();

  for (let rr = 0; rr < 9; rr++) {
    if (rr !== r && getCellDigit(rr, c) === digit) {
      blockers.add(indexRC(rr, c));
    }
  }

  for (let cc = 0; cc < 9; cc++) {
    if (cc !== c && getCellDigit(r, cc) === digit) {
      blockers.add(indexRC(r, cc));
    }
  }

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

function buildHiddenSingleVisuals(resp) {
  const tech = String(resp?.hint?.technique || "").toLowerCase();
  const cell = extractHintCell(resp);
  const digit = extractHintDigit(resp);

  const excluded = new Set();
  const constraints = new Set();
  const blockerRows = new Set();
  const blockerCols = new Set();
  const blockerBoxes = new Set();

  if (!cell || !digit || !tech.includes("hidden single")) {
    return { digit: null, scope: "none", excluded, constraints, blockerRows, blockerCols, blockerBoxes };
  }

  function processCandidate(candidateR, candidateC) {
    const idx = indexRC(candidateR, candidateC);
    if (state.current81[idx] !== "0") return;
    const blockers = collectBlockingDigits(candidateR, candidateC, digit);
    if (blockers.size === 0) return;
    excluded.add(idx);
    blockers.forEach(bIdx => {
      constraints.add(bIdx);
      const br = Math.floor(bIdx / 9);
      const bc = bIdx % 9;
      if (bc === candidateC) {
        blockerCols.add(bc);
      } else if (br === candidateR) {
        blockerRows.add(br);
      } else {
        blockerBoxes.add(Math.floor(br / 3) * 3 + Math.floor(bc / 3));
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

  const b = boxOf(cell.r, cell.c);
  for (let r = b.br; r < b.br + 3; r++) {
    for (let c = b.bc; c < b.bc + 3; c++) {
      if (r !== cell.r || c !== cell.c) processCandidate(r, c);
    }
  }
  return { digit, scope: "hidden-box", excluded, constraints, blockerRows, blockerCols, blockerBoxes };
}

function resetHighlights() {
  state.highlightCell = null;
  state.badCells = new Set();
  state.badOutline = new Set();
  state.highlightScope = "none";
  state.hintDigit = null;
  state.excludedCells = new Set();
  state.constraintCells = new Set();
  state.blockerRows = new Set();
  state.blockerCols = new Set();
  state.blockerBoxes = new Set();
}

function renderBoard() {
  boardEl.innerHTML = "";

  for (let i = 0; i < 81; i++) {
    const { r, c } = rcFromIndex(i);
    const cell = document.createElement("div");
    cell.className = "cell";

    if (r % 3 === 0) cell.classList.add("thick-top");
    if (c % 3 === 0) cell.classList.add("thick-left");
    if (r === 8) cell.classList.add("thick-bottom");
    if (c === 8) cell.classList.add("thick-right");

    const given = state.givenMask[i];
    cell.classList.add(given ? "given" : "user");

    if (state.highlightCell) {
      const hr = state.highlightCell.r;
      const hc = state.highlightCell.c;
      const isTarget = r === hr && c === hc;

      if (state.highlightScope === "hidden-row") {
        if (r === hr) cell.classList.add("house-row");
      } else if (state.highlightScope === "hidden-col") {
        if (c === hc) cell.classList.add("house-col");
      } else if (state.highlightScope === "hidden-box") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) {
          cell.classList.add("house-box");
        }
      } else if (state.highlightScope === "rowcolbox") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) {
          cell.classList.add("box-yellow");
        }
        if (r === hr || c === hc) {
          cell.classList.add("rc-yellow");
        }
      } else if (state.highlightScope === "box") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) {
          cell.classList.add("box-yellow");
        }
      }

      let inActiveHouse = false;
      if (state.highlightScope === "hidden-row" && r === hr) inActiveHouse = true;
      else if (state.highlightScope === "hidden-col" && c === hc) inActiveHouse = true;
      else if (state.highlightScope === "hidden-box") {
        const b = boxOf(hr, hc);
        if (r >= b.br && r < b.br + 3 && c >= b.bc && c < b.bc + 3) inActiveHouse = true;
      }

      if (state.highlightScope.startsWith("hidden-") && !inActiveHouse) {
        const cellBoxId = Math.floor(r / 3) * 3 + Math.floor(c / 3);
        if (state.blockerRows.has(r) || state.blockerCols.has(c) || state.blockerBoxes.has(cellBoxId)) {
          cell.classList.add("blocker-house");
        }
      }

      if (state.excludedCells.has(i)) cell.classList.add("excluded-cell");
      if (state.constraintCells.has(i)) cell.classList.add("constraint-cell");
      if (isTarget) cell.classList.add("hint-blue");
    }

    if (state.badCells.has(i)) cell.classList.add("bad");
    if (state.badOutline.has(i)) cell.classList.add("bad-outline");

    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "numeric";
    input.maxLength = 1;

    const ch = state.current81[i];
    input.value = ch === "0" ? "" : ch;

    if (given) {
      input.disabled = true;
    } else {
      input.style.setProperty("color", "#2f6fe4", "important");
      input.style.setProperty("-webkit-text-fill-color", "#2f6fe4", "important");
      input.addEventListener("input", (e) => {
        e.target.style.setProperty("color", "#2f6fe4", "important");
        e.target.style.setProperty("-webkit-text-fill-color", "#2f6fe4", "important");
        const v = (e.target.value || "").replace(/[^1-9]/g, "");
        e.target.value = v;

        const arr = state.current81.split("");
        arr[i] = v === "" ? "0" : v;
        state.current81 = arr.join("");
      });
    }

    cell.appendChild(input);

    if (state.excludedCells.has(i) && state.hintDigit) {
      const badge = document.createElement("div");
      badge.className = "excluded-badge";
      badge.textContent = `×${state.hintDigit}`;
      cell.appendChild(badge);
    }

    boardEl.appendChild(cell);
  }
}

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

modal.addEventListener("click", (e) => {
  if (e.target?.dataset?.close === "1") closeModal();
});

btnImport.addEventListener("click", openModal);

btnExample.addEventListener("click", () => {
  givensBox.value = EXAMPLE.givens;
  currentBox.value = EXAMPLE.current;
});

function loadFromTextboxesPreviewOnly() {
  const givens81 = normalize81(givensBox.value);
  const currentRaw = stripWS(currentBox.value);
  const current81 = currentRaw.length ? normalize81(currentBox.value) : givens81;

  state.givens81 = givens81;
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
    loadFromTextboxesPreviewOnly();
    closeModal();
  } catch (err) {
    alert(err.message || String(err));
  }
});

async function callAnalyze() {
  const res = await fetch(API_BASE + "/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      givens: state.givens81,
      current: state.current81
    })
  });

  const data = await res.json();
  if (!res.ok) return { error: data.error || "Request failed" };
  return data;
}

function computeDuplicateCells(current81) {
  const bad = new Set();

  function markGroup(indices) {
    const seen = new Map();
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
    const inds = [];
    for (let c = 0; c < 9; c++) inds.push(indexRC(r, c));
    markGroup(inds);
  }

  for (let c = 0; c < 9; c++) {
    const inds = [];
    for (let r = 0; r < 9; r++) inds.push(indexRC(r, c));
    markGroup(inds);
  }

  for (let br = 0; br < 9; br += 3) {
    for (let bc = 0; bc < 9; bc += 3) {
      const inds = [];
      for (let r = br; r < br + 3; r++) {
        for (let c = bc; c < bc + 3; c++) {
          inds.push(indexRC(r, c));
        }
      }
      markGroup(inds);
    }
  }

  return bad;
}

function toExplainLines(resp) {
  if (resp.error) {
    return ["Error", resp.error, ""];
  }

  if (resp.validation && resp.validation.ok === false) {
    return [
      "Validation failed",
      resp.validation.explanation || "Invalid board.",
      "Fix the highlighted cells, then analyze again."
    ];
  }

  if (resp.mistakes && resp.mistakes.has_mistake && Array.isArray(resp.mistakes.items) && resp.mistakes.items.length) {
    const it = resp.mistakes.items[0];
    const entered = it.entered ?? "?";
    const expected = it.expected ?? "?";
    const r = it.r ?? (it.cell ? it.cell[0] + 1 : "?");
    const c = it.c ?? (it.cell ? it.cell[1] + 1 : "?");
    const cell = `(r${r}, c${c})`;

    return [
      "Incorrect entry",
      `You entered ${entered} in ${cell}, but this cell must be ${expected}.`,
      `${cell} may not cause an immediate row, column, or box conflict, but the puzzle’s logic already forces ${expected} here. Keeping ${entered} blocks the correct solution.`
    ];
  }

  if (resp.ambiguity && resp.ambiguity.has_multiple_solutions) {
    return [
      "Puzzle is ambiguous",
      "This grid has more than one valid solution.",
      "Because no single solution is forced, the app cannot give a reliable hint."
    ];
  }

  if (!resp.hint || !resp.hint.has_hint) {
    return [
      "No hint available",
      "None of the enabled techniques produce a step.",
      "Either enter more values, use Reveal Solution, or try a different puzzle."
    ];
  }

  const tech = prettyTechniqueName(resp.hint.technique || "Hint");
  const msg = (resp.hint.message || "").trim();
  const parts = msg.split(/\n+/).map(s => s.trim()).filter(Boolean);

  if (/hidden single/i.test(tech)) {
    return [
      tech,
      parts[0] || "Only one cell in the highlighted house can take this digit.",
      parts[1] || "The marked cell is the only valid location."
    ];
  }

  return [
    tech,
    parts[0] || "A logical step is available.",
    parts[1] || "Apply the step, then re-analyze."
  ];
}

function hasLoadedBoard() {
  return state.givens81 && state.givens81 !== "0".repeat(81);
}

function applyResponseVisualState(resp) {
  resetHighlights();

  if (resp.validation && !resp.validation.ok) {
    const bad = computeDuplicateCells(state.current81);
    state.badCells = bad;
    const first = bad.values().next().value;
    if (first !== undefined) state.badOutline.add(first);
    return;
  }

  if (resp.mistakes?.has_mistake && Array.isArray(resp.mistakes.items) && resp.mistakes.items.length) {
    for (const it of resp.mistakes.items) {
      const r = it.r ?? (it.cell ? it.cell[0] + 1 : null);
      const c = it.c ?? (it.cell ? it.cell[1] + 1 : null);
      if (r == null || c == null) continue;
      state.badCells.add(indexRC(r - 1, c - 1));
    }

    const it0 = resp.mistakes.items[0];
    const r0 = it0.r ?? (it0.cell ? it0.cell[0] + 1 : null);
    const c0 = it0.c ?? (it0.cell ? it0.cell[1] + 1 : null);
    if (r0 != null && c0 != null) {
      state.badOutline.add(indexRC(r0 - 1, c0 - 1));
    }
    return;
  }

  if (resp.ambiguity?.has_multiple_solutions) {
    return;
  }

  if (resp.hint?.has_hint) {
    state.highlightCell = extractHintCell(resp);

    const tech = String(resp?.hint?.technique || "").toLowerCase();

    if (tech.includes("hidden single") && state.highlightCell) {
      const visuals = buildHiddenSingleVisuals(resp);
      state.hintDigit = visuals.digit;
      state.highlightScope = visuals.scope;
      state.excludedCells = visuals.excluded;
      state.constraintCells = visuals.constraints;
      state.blockerRows = visuals.blockerRows;
      state.blockerCols = visuals.blockerCols;
      state.blockerBoxes = visuals.blockerBoxes;

      const targetIdx = indexRC(state.highlightCell.r, state.highlightCell.c);
      state.excludedCells.delete(targetIdx);
      state.constraintCells.delete(targetIdx);

      console.log("[hint]", {
        technique: resp?.hint?.technique,
        target: state.highlightCell,
        digit: state.hintDigit,
        excludedCells: state.excludedCells.size,
        constraintCells: state.constraintCells.size,
        blockerRows: [...state.blockerRows],
        blockerCols: [...state.blockerCols],
        blockerBoxes: [...state.blockerBoxes],
      });
    } else if (tech.includes("naked single")) {
      state.highlightScope = "rowcolbox";
    } else {
      state.highlightScope = "box";
    }
  }
}

btnAnalyze.addEventListener("click", async () => {
  if (!hasLoadedBoard()) {
    openModal();
    return;
  }

  try {
    setStatus("Analyzing…");
    const resp = await callAnalyze();

    applyResponseVisualState(resp);

    const [a, b, c] = toExplainLines(resp);
    setExplain(a, b, c);

    renderBoard();
    setStatus("Done.");
  } catch (err) {
    alert(err.message || String(err));
    setStatus("Analysis failed.");
  }
});

btnReveal.addEventListener("click", async () => {
  if (!hasLoadedBoard()) {
    openModal();
    return;
  }

  try {
    setStatus("Solving…");
    const resp = await callAnalyze();
    const sol81 = resp?.solver?.solution81;

    if (resp?.solver?.ok && typeof sol81 === "string" && sol81.length === 81) {
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
    const [a, b, c] = toExplainLines(resp);
    setExplain(a, b, c);
    renderBoard();
    setStatus("No solution returned.");
  } catch (err) {
    setStatus(String(err.message || err));
  }
});

function loadFromOCR(cells) {
  const givens81  = cells.map(c => c.isGiven && c.digit > 0 ? String(c.digit) : "0").join("");
  const current81 = cells.map(c => c.digit > 0 ? String(c.digit) : "0").join("");

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
}

btnOCR.addEventListener("click", () => {
  ocrInput.click();
});

ocrInput.addEventListener("change", async () => {
  const file = ocrInput.files[0];
  if (!file) return;

  setStatus("Reading image…");
  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch(API_BASE + "/ocr", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.error) {
      alert(data.error || "OCR failed.");
      setStatus("OCR failed.");
      return;
    }
    loadFromOCR(data.board);
  } catch (err) {
    alert(err.message || String(err));
    setStatus("OCR failed.");
  } finally {
    ocrInput.value = "";
  }
});

(function init() {
  renderBoard();
  setStatus("Import a grid to begin.");
  openModal();
})();