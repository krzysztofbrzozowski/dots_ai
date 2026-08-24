"use strict";

const PLAYER_1 = 1;
const PLAYER_2 = -1;
const POLL_INTERVAL_MS = 300;

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
  gameResult: document.querySelector("#game-result"),
  rawBoard: document.querySelector("#raw-board"),
  rawTerritory: document.querySelector("#raw-territory"),
  status: document.querySelector("#status"),
};

const view = {
  game: null,
  version: null,
  layout: null,
  polling: false,
};

function formatCoordinate(cell) {
  return cell ? `(${cell[0]}, ${cell[1]})` : "—";
}

function formatCoordinates(cells) {
  return cells.length ? cells.map(formatCoordinate).join(", ") : "None";
}

function playerName(player) {
  return player === PLAYER_1 ? "Player 1" : "Player 2";
}

function resultName(winner) {
  if (winner === PLAYER_1) return "Player 1 wins";
  if (winner === PLAYER_2) return "Player 2 wins";
  return "Draw";
}

function showStatus(message, isError = false) {
  elements.status.querySelector("p").textContent = message;
  elements.status.classList.toggle("is-error", isError);
}

async function requestState() {
  const response = await fetch("/api/state", { cache: "no-store" });
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "The display state is unavailable.");
  }

  return payload;
}

async function pollState() {
  if (view.polling) return;
  view.polling = true;

  try {
    const game = await requestState();
    if (game.version !== view.version) {
      setGameState(game);
    }
  } catch (error) {
    if (!view.game) {
      elements.loading.textContent = "Waiting for main_mcts.py to publish a state…";
    }
    showStatus(error.message || "Could not read the MCTS state.", true);
  } finally {
    view.polling = false;
    window.setTimeout(pollState, POLL_INTERVAL_MS);
  }
}

function setGameState(game) {
  view.game = game;
  view.version = game.version;
  elements.loading.hidden = true;
  updateInformationPanel();
  resizeAndDrawBoard();
  showStatus(game.message);
}

function updateInformationPanel() {
  const game = view.game;
  const nextPlayer = playerName(game.current_player);

  elements.boardTitle.textContent = `${game.rows} × ${game.cols} board`;
  if (game.game_over) {
    elements.turnPill.textContent = resultName(game.winner);
    elements.turnPill.className = "turn-pill";
  } else {
    elements.turnPill.textContent = `${nextPlayer} searching`;
    elements.turnPill.className =
      `turn-pill ${game.current_player === PLAYER_1 ? "player-one" : "player-two"}`;
  }

  elements.scorePlayer1.textContent = game.score.player_1;
  elements.scorePlayer2.textContent = game.score.player_2;
  elements.moveNumber.textContent = game.move_number;
  elements.lastMove.textContent = formatCoordinate(game.last_move);
  elements.legalMoves.textContent = game.legal_move_count;
  elements.capturedDots.textContent = formatCoordinates(game.last_captured_dots);
  elements.captureHappened.textContent = game.capture_happened ? "Yes" : "No";
  elements.captureHappened.classList.toggle("capture-yes", game.capture_happened);
  elements.gameResult.textContent = game.game_over
    ? resultName(game.winner)
    : "In progress";
  elements.rawBoard.textContent = formatMatrix(game.board);
  elements.rawTerritory.textContent = formatMatrix(game.territory);

  const stateDescription = game.game_over
    ? resultName(game.winner)
    : `${nextPlayer} is searching. ${game.legal_move_count} legal moves remain.`;
  canvas.setAttribute(
    "aria-label",
    `${game.rows} by ${game.cols} read-only Dots board. ${stateDescription}`,
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
}

const resizeObserver = new ResizeObserver(resizeAndDrawBoard);
resizeObserver.observe(canvasWrap);
window.addEventListener("resize", resizeAndDrawBoard);

pollState();
