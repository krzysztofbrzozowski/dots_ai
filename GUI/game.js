"use strict";

const PLAYER_1 = 1;
const PLAYER_2 = -1;

const canvas = document.querySelector("#game-board");
const canvasWrap = document.querySelector("#canvas-wrap");
const context = canvas.getContext("2d");

const elements = {
  boardTitle: document.querySelector("#board-title"),
  loading: document.querySelector("#loading-state"),
  turnPill: document.querySelector("#turn-pill"),
  scorePlayer1: document.querySelector("#score-player-1"),
  scorePlayer2: document.querySelector("#score-player-2"),
  moveNumber: document.querySelector("#move-number"),
  lastMove: document.querySelector("#last-move"),
  legalMoves: document.querySelector("#legal-moves"),
  capturedDots: document.querySelector("#captured-dots"),
  captureHappened: document.querySelector("#capture-happened"),
  couldCloseLoop: document.querySelector("#could-close-loop"),
  candidateCount: document.querySelector("#candidate-count"),
  regionCount: document.querySelector("#region-count"),
  opponentCells: document.querySelector("#opponent-cells"),
  candidateRegions: document.querySelector("#candidate-regions"),
  capturedRegions: document.querySelector("#captured-regions"),
  rawBoard: document.querySelector("#raw-board"),
  rawTerritory: document.querySelector("#raw-territory"),
  status: document.querySelector("#status"),
  newGame: document.querySelector("#new-game"),
  resetGame: document.querySelector("#reset-game"),
  toggleDebug: document.querySelector("#toggle-debug"),
  debugDetails: document.querySelector("#debug-details"),
  rawDetails: document.querySelector("#raw-details"),
};

const view = {
  game: null,
  legalMoves: new Set(),
  hoverCell: null,
  keyboardCell: null,
  keyboardSelectionActive: false,
  layout: null,
  submitting: false,
};

function coordinateKey(row, col) {
  return `${row},${col}`;
}

function formatCoordinate(cell) {
  return cell ? `(${cell[0]}, ${cell[1]})` : "—";
}

function formatCoordinates(cells) {
  return cells.length ? cells.map(formatCoordinate).join(", ") : "None";
}

function playerName(player) {
  return player === PLAYER_1 ? "Player 1" : "Player 2";
}

function showStatus(message, isError = false) {
  elements.status.querySelector("p").textContent = message;
  elements.status.classList.toggle("is-error", isError);
}

function setControlsDisabled(disabled) {
  view.submitting = disabled;
  elements.newGame.disabled = disabled;
  elements.resetGame.disabled = disabled;
  canvas.setAttribute("aria-busy", String(disabled));
}

async function requestJSON(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();

  if (!response.ok) {
    const error = new Error(payload.error || "The game engine rejected the request.");
    error.state = payload.state;
    throw error;
  }

  return payload;
}

async function loadGame() {
  try {
    setGameState(await requestJSON("/api/state"));
  } catch (error) {
    elements.loading.textContent = "Could not connect to the game engine.";
    showStatus("Could not load the game. Check that the FastAPI server is running.", true);
  }
}

async function resetGame(label) {
  if (view.submitting) return;
  setControlsDisabled(true);
  showStatus(`${label}…`);

  try {
    setGameState(await requestJSON("/api/reset", { method: "POST", body: "{}" }));
  } catch (error) {
    showStatus("The game could not be reset.", true);
  } finally {
    setControlsDisabled(false);
  }
}

async function submitMove(row, col) {
  if (view.submitting) return;
  setControlsDisabled(true);
  showStatus(`Placing a dot at (${row}, ${col})…`);

  try {
    const state = await requestJSON("/api/move", {
      method: "POST",
      body: JSON.stringify({ row, col }),
    });
    setGameState(state);
  } catch (error) {
    if (error.state) setGameState(error.state);
    showStatus(error.message || "Illegal move.", true);
  } finally {
    setControlsDisabled(false);
  }
}

function setGameState(game) {
  view.game = game;
  view.legalMoves = new Set(
    game.legal_moves.map(([row, col]) => coordinateKey(row, col)),
  );
  view.hoverCell = null;

  if (
    !view.keyboardCell ||
    !view.legalMoves.has(coordinateKey(view.keyboardCell[0], view.keyboardCell[1]))
  ) {
    view.keyboardCell = game.legal_moves[0] || null;
  }

  elements.loading.hidden = true;
  updateInformationPanel();
  resizeAndDrawBoard();
  showStatus(game.message);
}

function updateInformationPanel() {
  const game = view.game;
  const debug = game.debug;
  const name = playerName(game.current_player);

  elements.boardTitle.textContent = `${game.rows} × ${game.cols} board`;
  elements.turnPill.textContent = `${name} to move`;
  elements.turnPill.className = `turn-pill ${game.current_player === PLAYER_1 ? "player-one" : "player-two"}`;
  elements.scorePlayer1.textContent = game.score.player_1;
  elements.scorePlayer2.textContent = game.score.player_2;
  elements.moveNumber.textContent = game.move_number;
  elements.lastMove.textContent = formatCoordinate(game.last_move);
  elements.legalMoves.textContent = game.legal_move_count;
  elements.capturedDots.textContent = formatCoordinates(game.last_captured_dots);
  elements.captureHappened.textContent = game.capture_happened ? "Yes" : "No";
  elements.captureHappened.classList.toggle("capture-yes", game.capture_happened);

  elements.couldCloseLoop.textContent =
    debug.could_have_closed_loop === null
      ? "Not checked"
      : debug.could_have_closed_loop
        ? "Yes"
        : "No";
  elements.candidateCount.textContent = debug.candidate_region_count;
  elements.regionCount.textContent = debug.enclosed_region_count;
  elements.opponentCells.textContent = formatCoordinates(debug.opponent_cells_found);
  elements.candidateRegions.textContent = JSON.stringify(debug.candidate_regions, null, 2);
  elements.capturedRegions.textContent = JSON.stringify(debug.detected_enclosed_regions, null, 2);
  elements.rawBoard.textContent = formatMatrix(game.board);
  elements.rawTerritory.textContent = formatMatrix(game.territory);

  canvas.setAttribute(
    "aria-label",
    `${game.rows} by ${game.cols} Dots board. ${name} to move. ` +
      `${game.legal_move_count} legal moves remain. Use arrow keys and Enter to place a dot.`,
  );
}

function formatMatrix(matrix) {
  const columnCount = matrix[0]?.length || 0;
  const header = `     ${Array.from({ length: columnCount }, (_, index) =>
    String(index).padStart(3, " "),
  ).join("")}`;
  const rows = matrix.map(
    (row, index) =>
      `${String(index).padStart(3, " ")}  ${row
        .map((value) => String(value).padStart(3, " "))
        .join("")}`,
  );
  return [header, ...rows].join("\n");
}

function cssColor(variable) {
  return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
}

function resizeAndDrawBoard() {
  if (!view.game) return;

  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  const { rows, cols } = view.game;
  const labelSpace = 36;
  const edgeSpace = 18;
  const availableWidth = Math.max(1, width - labelSpace - edgeSpace);
  const availableHeight = Math.max(1, height - labelSpace - edgeSpace);
  const step = Math.min(
    availableWidth / Math.max(cols - 1, 1),
    availableHeight / Math.max(rows - 1, 1),
  );
  const gridWidth = step * Math.max(cols - 1, 0);
  const gridHeight = step * Math.max(rows - 1, 0);
  const originX = labelSpace + (availableWidth - gridWidth) / 2;
  const originY = labelSpace + (availableHeight - gridHeight) / 2;

  view.layout = { width, height, step, originX, originY };
  drawBoard();
}

function drawBoard() {
  if (!view.game || !view.layout) return;

  const { width, height, step, originX, originY } = view.layout;
  const { rows, cols, board, territory, last_move: lastMove } = view.game;
  const surface = cssColor("--surface-subtle");
  const grid = cssColor("--grid");
  const textSecondary = cssColor("--text-secondary");
  const player1 = cssColor("--player-1");
  const player2 = cssColor("--player-2");

  context.clearRect(0, 0, width, height);
  context.fillStyle = surface;
  context.fillRect(0, 0, width, height);

  // Territory belongs to intersections in the engine, so each captured cell
  // is shown as a soft tile centered on its corresponding grid point.
  const territorySize = Math.max(10, step * 0.76);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const owner = territory[row][col];
      if (owner === 0) continue;
      const x = originX + col * step;
      const y = originY + row * step;
      context.fillStyle = cssColor(
        owner === PLAYER_1 ? "--territory-player-1" : "--territory-player-2",
      );
      context.beginPath();
      context.roundRect(
        x - territorySize / 2,
        y - territorySize / 2,
        territorySize,
        territorySize,
        Math.max(3, step * 0.12),
      );
      context.fill();
    }
  }

  context.strokeStyle = grid;
  context.lineWidth = 1;
  context.beginPath();
  for (let col = 0; col < cols; col += 1) {
    const x = Math.round(originX + col * step) + 0.5;
    context.moveTo(x, originY);
    context.lineTo(x, originY + (rows - 1) * step);
  }
  for (let row = 0; row < rows; row += 1) {
    const y = Math.round(originY + row * step) + 0.5;
    context.moveTo(originX, y);
    context.lineTo(originX + (cols - 1) * step, y);
  }
  context.stroke();

  context.fillStyle = textSecondary;
  context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  const columnLabelStride = step < 20 ? 2 : 1;
  for (let col = 0; col < cols; col += 1) {
    if (col % columnLabelStride !== 0) continue;
    context.fillText(String(col), originX + col * step, originY - 18);
  }
  context.textAlign = "right";
  for (let row = 0; row < rows; row += 1) {
    context.fillText(String(row), originX - 15, originY + row * step);
  }

  const dotRadius = Math.max(5, Math.min(9, step * 0.24));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const player = board[row][col];
      if (player === 0) continue;
      const x = originX + col * step;
      const y = originY + row * step;
      const isInactive = territory[row][col] !== 0;

      context.save();
      if (isInactive) context.globalAlpha = 0.42;
      context.fillStyle = player === PLAYER_1 ? player1 : player2;
      context.beginPath();
      context.arc(x, y, dotRadius, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = surface;
      context.lineWidth = 2;
      context.stroke();
      context.restore();
    }
  }

  if (lastMove) {
    const [row, col] = lastMove;
    const x = originX + col * step;
    const y = originY + row * step;
    context.strokeStyle = cssColor("--text-primary");
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, dotRadius + 5, 0, Math.PI * 2);
    context.stroke();
  }

  const selectedCell =
    view.hoverCell ||
    (view.keyboardSelectionActive && document.activeElement === canvas
      ? view.keyboardCell
      : null);
  if (selectedCell) {
    const [row, col] = selectedCell;
    const x = originX + col * step;
    const y = originY + row * step;
    context.fillStyle =
      view.game.current_player === PLAYER_1
        ? "rgba(182, 93, 80, 0.16)"
        : "rgba(82, 115, 163, 0.17)";
    context.strokeStyle = view.game.current_player === PLAYER_1 ? player1 : player2;
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(x, y, dotRadius + 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
}

// Canvas coordinates are mapped to the nearest grid intersection; the legal
// move list still comes exclusively from the Python engine.
function eventToCell(event) {
  if (!view.layout || !view.game) return null;
  const bounds = canvas.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  const { originX, originY, step } = view.layout;
  const col = Math.round((x - originX) / step);
  const row = Math.round((y - originY) / step);

  if (row < 0 || row >= view.game.rows || col < 0 || col >= view.game.cols) {
    return null;
  }

  const intersectionX = originX + col * step;
  const intersectionY = originY + row * step;
  const hitRadius = Math.max(11, Math.min(20, step * 0.42));
  if (Math.hypot(x - intersectionX, y - intersectionY) > hitRadius) return null;
  return [row, col];
}

canvas.addEventListener("pointermove", (event) => {
  view.keyboardSelectionActive = false;
  const cell = eventToCell(event);
  const isLegal = cell && view.legalMoves.has(coordinateKey(cell[0], cell[1]));
  view.hoverCell = isLegal ? cell : null;
  canvas.classList.toggle("is-actionable", Boolean(isLegal));
  drawBoard();
});

canvas.addEventListener("pointerdown", () => {
  view.keyboardSelectionActive = false;
  drawBoard();
});

canvas.addEventListener("pointerleave", () => {
  view.hoverCell = null;
  canvas.classList.remove("is-actionable");
  drawBoard();
});

canvas.addEventListener("click", (event) => {
  const cell = eventToCell(event);
  if (!cell || !view.legalMoves.has(coordinateKey(cell[0], cell[1]))) return;
  submitMove(cell[0], cell[1]);
});

canvas.addEventListener("keydown", (event) => {
  if (!view.game || !view.keyboardCell) return;
  const directions = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  };

  if (event.key in directions) {
    event.preventDefault();
    view.keyboardSelectionActive = true;
    const [rowDelta, colDelta] = directions[event.key];
    const row = Math.max(0, Math.min(view.game.rows - 1, view.keyboardCell[0] + rowDelta));
    const col = Math.max(0, Math.min(view.game.cols - 1, view.keyboardCell[1] + colDelta));
    view.keyboardCell = [row, col];
    drawBoard();
  }

  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    view.keyboardSelectionActive = true;
    const [row, col] = view.keyboardCell;
    if (view.legalMoves.has(coordinateKey(row, col))) {
      submitMove(row, col);
    } else {
      showStatus(`Intersection (${row}, ${col}) is not available.`, true);
    }
  }
});

canvas.addEventListener("focus", drawBoard);
canvas.addEventListener("blur", () => {
  view.keyboardSelectionActive = false;
  drawBoard();
});

elements.newGame.addEventListener("click", () => resetGame("Starting a new game"));
elements.resetGame.addEventListener("click", () => resetGame("Resetting the game"));

elements.toggleDebug.addEventListener("click", () => {
  const panels = [elements.debugDetails, elements.rawDetails];
  const shouldOpen = !panels.some((panel) => panel.open);
  panels.forEach((panel) => {
    panel.open = shouldOpen;
  });
  elements.toggleDebug.textContent = shouldOpen ? "Hide debug" : "Show debug";
  elements.toggleDebug.setAttribute("aria-pressed", String(shouldOpen));
});

[elements.debugDetails, elements.rawDetails].forEach((panel) => {
  panel.addEventListener("toggle", () => {
    const anyOpen = elements.debugDetails.open || elements.rawDetails.open;
    elements.toggleDebug.textContent = anyOpen ? "Hide debug" : "Show debug";
    elements.toggleDebug.setAttribute("aria-pressed", String(anyOpen));
  });
});

const resizeObserver = new ResizeObserver(resizeAndDrawBoard);
resizeObserver.observe(canvasWrap);
window.addEventListener("resize", resizeAndDrawBoard);

loadGame();
