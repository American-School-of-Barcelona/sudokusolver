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

let state = {
  givens81: "0".repeat(81),
  current81: "0".repeat(81),
  givenMask: Array(81).fill(false),

  highlightCell: null,
  badCells: new Set(),
  badOutline: new Set(),
  highlightScope: "box", // "box" or "rowcolbox"
};

const EXAMPLE = {
  givens: "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
  current: ""
};

function setStatus(msg){ statusEl.textContent = msg || ""; }
function setExplain(a,b,c){ line1.textContent=a||""; line2.textContent=b||""; line3.textContent=c||""; }

function stripWS(s){ return (s||"").replace(/\s+/g,""); }
function normalize81(s){
  const t = stripWS(s).replaceAll(".", "0");
  if (t.length !== 81) throw new Error(`Expected 81 characters, got ${t.length}`);
  if (!/^[0-9]{81}$/.test(t)) throw new Error("Only digits / 0 / '.' are allowed");
  return t;
}
function buildGivenMask(givens81){
  const m = Array(81).fill(false);
  for (let i=0;i<81;i++) m[i] = givens81[i] !== "0";
  return m;
}
function indexRC(r,c){ return r*9+c; }
function rcFromIndex(i){ return {r: Math.floor(i/9), c: i%9}; }
function boxOf(r,c){ return {br: Math.floor(r/3)*3, bc: Math.floor(c/3)*3}; }

function renderBoard(){
  boardEl.innerHTML = "";
  for (let i=0;i<81;i++){
    const {r,c} = rcFromIndex(i);
    const cell = document.createElement("div");
    cell.className = "cell";

    if (r % 3 === 0) cell.classList.add("thick-top");
    if (c % 3 === 0) cell.classList.add("thick-left");
    if (r === 8) cell.classList.add("thick-bottom");
    if (c === 8) cell.classList.add("thick-right");

    const given = state.givenMask[i];
    cell.classList.add(given ? "given" : "user");

    // hint highlight
    if (state.highlightCell){
      const hr = state.highlightCell.r;
      const hc = state.highlightCell.c;

      // Always highlight the 3×3 box (yellow)
      const b = boxOf(hr, hc);
      if (r>=b.br && r<b.br+3 && c>=b.bc && c<b.bc+3) cell.classList.add("box-yellow");

      // For Naked Singles: also highlight row + column (yellow tint)
      if (state.highlightScope === "rowcolbox"){
        if (r === hr || c === hc) cell.classList.add("rc-yellow");
      }

      // Target cell (blue)
      if (r===hr && c===hc) cell.classList.add("hint-blue");
    }

    // error highlight
    if (state.badCells.has(i)) cell.classList.add("bad");
    if (state.badOutline.has(i)) cell.classList.add("bad-outline");

    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "numeric";
    input.maxLength = 1;

    const ch = state.current81[i];
    input.value = (ch === "0" ? "" : ch);

    if (given){
      input.disabled = true;
    } else {
      input.addEventListener("input", (e) => {
        const v = (e.target.value || "").replace(/[^1-9]/g,"");
        e.target.value = v;

        const arr = state.current81.split("");
        arr[i] = v === "" ? "0" : v;
        state.current81 = arr.join("");
      });
    }

    cell.appendChild(input);
    boardEl.appendChild(cell);
  }
}

function openModal(){
  modal.classList.add("show");
  modal.setAttribute("aria-hidden","false");
  document.body.classList.add("modal-open");
}
function closeModal(){
  modal.classList.remove("show");
  modal.setAttribute("aria-hidden","true");
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

function loadFromTextboxesPreviewOnly(){
  const givens81 = normalize81(givensBox.value);
  const currentRaw = stripWS(currentBox.value);
  const current81 = currentRaw.length ? normalize81(currentBox.value) : givens81;

  state.givens81 = givens81;
  state.current81 = current81;
  state.givenMask = buildGivenMask(givens81);

  // preview resets
  state.highlightCell = null;
  state.badCells = new Set();
  state.badOutline = new Set();

  renderBoard();
  setExplain("Preview loaded",
             "Edit the board on the left if you want",
             "Then click Analyze / Get Hint");
  setStatus("Board preview ready.");
}

btnLoad.addEventListener("click", () => {
  try{
    loadFromTextboxesPreviewOnly();
    closeModal();
  }catch(err){
    alert(err.message || String(err));
  }
});

async function callAnalyze(){
  const res = await fetch(API_BASE + "/analyze", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ givens: state.givens81, current: state.current81 })
  });
  const data = await res.json();
  if (!res.ok) return {error: data.error || "Request failed"};
  return data;
}

function extractHintCell(resp){
  const msg = resp?.hint?.message || "";
  const m = msg.match(/\(r(\d+),\s*c(\d+)\)/);
  if (!m) return null;
  const r = parseInt(m[1],10)-1;
  const c = parseInt(m[2],10)-1;
  if (r<0||r>8||c<0||c>8) return null;
  return {r,c};
}

function computeDuplicateCells(current81){
  const bad = new Set();
  function markGroup(indices){
    const seen = new Map();
    for (const idx of indices){
      const d = current81[idx];
      if (d === "0") continue;
      if (seen.has(d)){
        bad.add(idx);
        bad.add(seen.get(d));
      } else {
        seen.set(d, idx);
      }
    }
  }

  for (let r=0;r<9;r++){
    const inds=[];
    for (let c=0;c<9;c++) inds.push(indexRC(r,c));
    markGroup(inds);
  }
  for (let c=0;c<9;c++){
    const inds=[];
    for (let r=0;r<9;r++) inds.push(indexRC(r,c));
    markGroup(inds);
  }
  for (let br=0;br<9;br+=3){
    for (let bc=0;bc<9;bc+=3){
      const inds=[];
      for (let r=br;r<br+3;r++){
        for (let c=bc;c<bc+3;c++) inds.push(indexRC(r,c));
      }
      markGroup(inds);
    }
  }
  return bad;
}

function toExplainLines(resp){
  if (resp.error) return ["Error", resp.error, ""];

  // 1) Validation failures (tampering / duplicates)
  if (resp.validation && resp.validation.ok === false){
    return ["Validation failed", resp.validation.explanation || "Invalid board.", "Fix the highlighted cells, then analyze again."];
  }

  // 2) Mistakes must override hint text
  if (resp.mistakes && resp.mistakes.has_mistake && Array.isArray(resp.mistakes.items) && resp.mistakes.items.length){
    const it = resp.mistakes.items[0];
    const entered = it.entered ?? "?";
    const expected = it.expected ?? "?";
    const why = it.explanation || "This entry doesn’t break Sudoku rules yet, but it contradicts the unique solution from the original givens, so the puzzle cannot be completed correctly.";
    return [
      "Mistake detected",
      `Cell (r${it.r}, c${it.c}) is inconsistent: entered ${entered}, but the correct value is ${expected}.`,
      why
    ];
  }

  // 3) Only then show hint or 'no hint'
  if (!resp.hint || !resp.hint.has_hint){
    return ["No hint available", "None of the enabled techniques produce a step.", "Either enter more values, or use Reveal Solution."];
  }

  const tech = resp.hint.technique || "Hint";
  const msg = (resp.hint.message || "").trim();
  const parts = msg.split(/\n+/).map(s=>s.trim()).filter(Boolean);

  return [
    tech,
    parts[0] || "A logical step is available.",
    parts[1] || "Apply the step, then re-analyze."
  ];
}
function hasLoadedBoard(){
  // if givens are all 0, user hasn't imported anything meaningful
  return state.givens81 && state.givens81 !== "0".repeat(81);
}

btnAnalyze.addEventListener("click", async () => {
  if (!hasLoadedBoard()){
    openModal();
    return;
  }

  try{
    setStatus("Analyzing…");

    const resp = await callAnalyze();

    // IMPORTANT: do NOT reload from textboxes here.
    // We analyze the *current board state*, including edits made on the grid.

    state.highlightCell = null;
    state.badCells = new Set();
    state.badOutline = new Set();

    if (resp.validation && !resp.validation.ok){
      const bad = computeDuplicateCells(state.current81);
      state.badCells = bad;
      const first = bad.values().next().value;
      if (first !== undefined) state.badOutline.add(first);
    } else if (resp.mistakes?.has_mistake && Array.isArray(resp.mistakes.items) && resp.mistakes.items.length){
      for (const it of resp.mistakes.items){
        const idx = indexRC(it.r-1, it.c-1);
        state.badCells.add(idx);
      }
      const it0 = resp.mistakes.items[0];
      state.badOutline.add(indexRC(it0.r-1, it0.c-1));
    } else {
      state.highlightCell = extractHintCell(resp);

      const tech = (resp?.hint?.technique || "").toLowerCase();
      state.highlightScope = tech.includes("naked single") ? "rowcolbox" : "box";
    }

    const [a,b,c] = toExplainLines(resp);
    setExplain(a,b,c);

    renderBoard();
    setStatus("Done.");
  }catch(err){
    alert(err.message || String(err));
  }
});

btnReveal.addEventListener("click", async () => {
  if (!hasLoadedBoard()){
    openModal();
    return;
  }

  try{
    setStatus("Solving…");
    const resp = await callAnalyze();

    const sol81 = resp?.solver?.solution81;
    if (resp?.solver?.ok && typeof sol81 === "string" && sol81.length === 81){
      // apply solution only to non-given cells
      const arr = state.current81.split("");
      for (let i=0;i<81;i++){
        if (!state.givenMask[i]) arr[i] = sol81[i];
      }
      state.current81 = arr.join("");

      state.highlightCell = null;
      state.badCells = new Set();
      state.badOutline = new Set();

      const [a,b,c] = toExplainLines(resp);
      setExplain(a,b,c);

      renderBoard();
      setStatus("Solution applied.");
    } else {
      setStatus("No solution returned (solver stuck or validation failed).");
    }
  }catch(err){
    setStatus(String(err.message || err));
  }
});

// Init
(function init(){
  renderBoard();
  setStatus("Import a grid to begin.");
  openModal();
})();
