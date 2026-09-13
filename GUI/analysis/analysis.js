"use strict";

import {
  DotsBoardRenderer,
  PLAYER_1,
  PLAYER_2,
} from "/shared/board_renderer.js?v=20260913-board-size";


const elements = {
  fileInput: document.querySelector("#npz-file"),
  fileButton: document.querySelector("#file-button"),
  emptyFileButton: document.querySelector("#open-empty-file"),
  dropTarget: document.querySelector("#drop-target"),
  dropOverlay: document.querySelector("#drop-overlay"),
  workspace: document.querySelector(".workspace"),
  gameTitle: document.querySelector("#game-title"),
  gameSubtitle: document.querySelector("#game-subtitle"),
  summaryBoard: document.querySelector("#summary-board"),
  summaryMoves: document.querySelector("#summary-moves"),
  summaryResult: document.querySelector("#summary-result"),
  summarySchema: document.querySelector("#summary-schema"),
  boardTitle: document.querySelector("#board-title"),
  board: document.querySelector("#analysis-board"),
  boardEmpty: document.querySelector("#board-empty"),
  playerPill: document.querySelector("#player-pill"),
  overlaySwitcher: document.querySelector("#overlay-switcher"),
  overlayDescription: document.querySelector("#overlay-description"),
  overlayScale: document.querySelector("#overlay-scale"),
  scaleLow: document.querySelector("#scale-low"),
  scaleHigh: document.querySelector("#scale-high"),
  frameAction: document.querySelector("#frame-action"),
  frameScore: document.querySelector("#frame-score"),
  frameLegal: document.querySelector("#frame-legal"),
  frameRollouts: document.querySelector("#frame-rollouts"),
  frameElapsed: document.querySelector("#frame-elapsed"),
  frameThroughput: document.querySelector("#frame-throughput"),
  cellTitle: document.querySelector("#cell-inspector-title"),
  cellContents: document.querySelector("#cell-contents"),
  cellLegal: document.querySelector("#cell-legal"),
  cellRawQ: document.querySelector("#cell-raw-q"),
  cellValue: document.querySelector("#cell-value"),
  cellVisits: document.querySelector("#cell-visits"),
  cellPolicy: document.querySelector("#cell-policy"),
  headValuePanel: document.querySelector("#head-value-panel"),
  headValueTitle: document.querySelector("#head-value-title"),
  headValueModel: document.querySelector("#head-value-model"),
  headValueMessage: document.querySelector("#head-value-message"),
  headValueLoss: document.querySelector("#head-value-loss"),
  headValueDraw: document.querySelector("#head-value-draw"),
  headValueWin: document.querySelector("#head-value-win"),
  headValueScore: document.querySelector("#head-value-score"),
  frameCounter: document.querySelector("#frame-counter"),
  previousFrame: document.querySelector("#previous-frame"),
  playTimeline: document.querySelector("#play-timeline"),
  nextFrame: document.querySelector("#next-frame"),
  timelineWheel: document.querySelector("#timeline-wheel"),
  timelineList: document.querySelector("#timeline-list"),
  timelineEmpty: document.querySelector("#timeline-empty"),
  status: document.querySelector("#status"),
  diagnosticsOutput: document.querySelector("#diagnostics-output"),
  diagnosticsConnection: document.querySelector("#diagnostics-connection"),
  copyDiagnostics: document.querySelector("#copy-diagnostics"),
  clearDiagnostics: document.querySelector("#clear-diagnostics"),
};


const view = {
  analysis: null,
  frame: null,
  frameIndex: 0,
  selectedCell: null,
  overlay: "value",
  importing: false,
  requestNumber: 0,
  frameCache: new Map(),
  pendingFrames: new Map(),
  headValueCache: new Map(),
  headValueRequestNumber: 0,
  headValueController: null,
  scrollTimer: null,
  centeringTimer: null,
  isCenteringTimeline: false,
  playTimer: null,
  dragDepth: 0,
  diagnosticsCursor: 0,
  diagnosticsPollTimer: null,
  lastDiagnosticFrameKey: null,
};


const MAX_DIAGNOSTIC_LINES = 160;


const boardRenderer = new DotsBoardRenderer(
  elements.board,
  selectBoardCell,
);


function updateTimelinePadding() {
  const centeredPadding = Math.max(
    0,
    elements.timelineWheel.clientHeight / 2 - 39,
  );
  elements.timelineList.style.setProperty(
    "--timeline-center-padding",
    `${centeredPadding}px`,
  );
}


const timelineResizeObserver = new ResizeObserver(() => {
  updateTimelinePadding();
  if (view.analysis) centerTimelineItem(view.frameIndex, false);
});
timelineResizeObserver.observe(elements.timelineWheel);


function updateHeadValueWorkspaceHeight() {
  if (elements.headValuePanel.hidden) return;
  const panelStyles = getComputedStyle(elements.headValuePanel);
  const panelSpace =
    elements.headValuePanel.getBoundingClientRect().height +
    Number.parseFloat(panelStyles.marginTop || "0");
  elements.workspace.style.setProperty(
    "--head-value-panel-space",
    `${panelSpace}px`,
  );
}


const headValueResizeObserver = new ResizeObserver(updateHeadValueWorkspaceHeight);
headValueResizeObserver.observe(elements.headValuePanel);


function playerName(player) {
  return player === PLAYER_1 ? "Player 1" : "Player 2";
}


function resultName(result) {
  if (result === PLAYER_1) return "Player 1 won";
  if (result === PLAYER_2) return "Player 2 won";
  return "Draw";
}


function formatCoordinate(coordinate) {
  return coordinate ? `(${coordinate[0]}, ${coordinate[1]})` : "—";
}


function formatInteger(value) {
  return new Intl.NumberFormat().format(value);
}


function formatDecimal(value, digits = 2, showSign = false) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const formatted = value.toFixed(digits);
  return showSign && value >= 0 ? `+${formatted}` : formatted;
}


function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(2)} s`;
}


function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}


function diagnosticTimestamp(timestamp) {
  const date = timestamp ? new Date(timestamp) : new Date();
  if (Number.isNaN(date.getTime())) return "--:--:--.---";
  const clock = date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  return `${clock}.${String(date.getMilliseconds()).padStart(3, "0")}`;
}


function logDiagnostic(message, { level = "info", source = "UI", timestamp } = {}) {
  const wasAtBottom =
    elements.diagnosticsOutput.scrollHeight -
      elements.diagnosticsOutput.scrollTop -
      elements.diagnosticsOutput.clientHeight < 24;
  const line = document.createElement("div");
  const time = document.createElement("time");
  const severity = document.createElement("span");
  const scope = document.createElement("span");
  const contents = document.createElement("span");

  line.className = `diagnostic-line is-${level}`;
  time.textContent = diagnosticTimestamp(timestamp);
  severity.textContent = level.toUpperCase();
  scope.textContent = String(source).toUpperCase();
  contents.textContent = String(message);
  line.dataset.plainText =
    `${time.textContent} ${severity.textContent.padEnd(7)} ` +
    `${scope.textContent.padEnd(8)} ${contents.textContent}`;
  line.append(time, severity, scope, contents);
  elements.diagnosticsOutput.append(line);

  while (elements.diagnosticsOutput.childElementCount > MAX_DIAGNOSTIC_LINES) {
    elements.diagnosticsOutput.firstElementChild.remove();
  }
  if (wasAtBottom) {
    elements.diagnosticsOutput.scrollTop = elements.diagnosticsOutput.scrollHeight;
  }
}


function logAnalysisMetadata(game) {
  logDiagnostic(
    `game ${game.game_id} · schema v${game.schema_version} · ` +
      `${game.board.rows}x${game.board.cols} · ${game.frame_count} frames`,
    { level: "success", source: "SESSION" },
  );
  logDiagnostic(
    `Q perspective ${game.q_perspective} · result perspective ` +
      `${game.final_result_perspective} · batch ${game.search.rollout_batch_size}`,
    { source: "DATA" },
  );
  logDiagnostic(
    `${formatInteger(game.search.total_rollouts)} rollouts · ` +
      `${formatDuration(game.search.total_elapsed_seconds)} total · ` +
      `${formatInteger(Math.round(game.search.average_rollouts_per_second))}/s average`,
    { source: "SEARCH" },
  );
}


async function pollDiagnostics() {
  try {
    const response = await fetch(
      `/api/diagnostics?after=${view.diagnosticsCursor}`,
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error(`Diagnostics returned HTTP ${response.status}`);
    const payload = await response.json();
    for (const event of payload.events) {
      logDiagnostic(event.message, event);
    }
    view.diagnosticsCursor = payload.cursor;
    elements.diagnosticsConnection.classList.remove("is-delayed");
    elements.diagnosticsConnection.lastChild.textContent = "Live";
  } catch (_error) {
    elements.diagnosticsConnection.classList.add("is-delayed");
    elements.diagnosticsConnection.lastChild.textContent = "Retrying";
  } finally {
    view.diagnosticsPollTimer = window.setTimeout(pollDiagnostics, 900);
  }
}


function showStatus(message, kind = "normal") {
  elements.status.querySelector("p").textContent = message;
  elements.status.classList.toggle("is-error", kind === "error");
  elements.status.classList.toggle("is-busy", kind === "busy");
}


async function responsePayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return null;
}


function cancelHeadValuePrediction() {
  view.headValueRequestNumber += 1;
  view.headValueController?.abort();
  view.headValueController = null;
}


function resetHeadValuePanel(message) {
  elements.headValuePanel.classList.remove(
    "is-loading",
    "is-error",
    "is-positive",
    "is-negative",
  );
  elements.headValueTitle.textContent = "Select a legal move";
  elements.headValueMessage.textContent = message ||
    "Click an empty legal position to evaluate the result after that move.";
  for (const result of [
    elements.headValueLoss,
    elements.headValueDraw,
    elements.headValueWin,
    elements.headValueScore,
  ]) {
    result.textContent = "—";
  }
}


function showHeadValueError(cell, message) {
  resetHeadValuePanel(message);
  elements.headValuePanel.classList.add("is-error");
  elements.headValueTitle.textContent = `Cannot evaluate ${formatCoordinate(cell)}`;
}


function displayHeadValuePrediction(prediction) {
  elements.headValuePanel.classList.remove("is-loading", "is-error");
  elements.headValuePanel.classList.toggle("is-positive", prediction.value > 0);
  elements.headValuePanel.classList.toggle("is-negative", prediction.value < 0);
  elements.headValueTitle.textContent =
    `${formatCoordinate(prediction.coordinate)} for ${playerName(prediction.player)}`;
  elements.headValueModel.textContent = prediction.model;
  elements.headValueModel.title = prediction.model;
  elements.headValueMessage.textContent = prediction.source === "terminal_result"
    ? "Exact result: this candidate move ends the game."
    : "Prediction after this move, from the moving player's perspective.";
  elements.headValueLoss.textContent = `${(prediction.loss * 100).toFixed(1)}%`;
  elements.headValueDraw.textContent = `${(prediction.draw * 100).toFixed(1)}%`;
  elements.headValueWin.textContent = `${(prediction.win * 100).toFixed(1)}%`;
  elements.headValueScore.textContent = formatDecimal(prediction.value, 3, true);
}


async function requestHeadValue(cell) {
  if (!view.analysis || !view.frame) return;

  const frameIndex = view.frameIndex;
  const [row, col] = cell;
  const cacheKey = `${view.analysis.analysis_id}:${frameIndex}:${row}:${col}`;
  const startedAt = performance.now();
  const isCached = view.headValueCache.has(cacheKey);
  cancelHeadValuePrediction();
  const requestNumber = view.headValueRequestNumber;

  logDiagnostic(
    `${isCached ? "cache lookup" : "predict"} · frame ${frameIndex + 1} · ` +
      `move ${formatCoordinate(cell)}`,
    { source: "MODEL" },
  );

  elements.headValuePanel.classList.remove("is-error", "is-positive", "is-negative");
  elements.headValuePanel.classList.add("is-loading");
  elements.headValueTitle.textContent = `Evaluating ${formatCoordinate(cell)}…`;
  elements.headValueMessage.textContent = "Running the local value-head model.";
  elements.headValueLoss.textContent = "—";
  elements.headValueDraw.textContent = "—";
  elements.headValueWin.textContent = "—";
  elements.headValueScore.textContent = "•••";

  try {
    let prediction = view.headValueCache.get(cacheKey);
    if (!prediction) {
      view.headValueController = new AbortController();
      const query = new URLSearchParams({ row: String(row), col: String(col) });
      const response = await fetch(
        `/api/analyses/${view.analysis.analysis_id}/frames/${frameIndex}/head-value?${query}`,
        { cache: "no-store", signal: view.headValueController.signal },
      );
      const payload = await responsePayload(response);
      if (!response.ok) {
        throw new Error(payload?.detail || "The value-head prediction failed.");
      }
      prediction = payload;
      view.headValueCache.set(cacheKey, prediction);
    }

    if (requestNumber !== view.headValueRequestNumber) return;
    view.headValueController = null;
    displayHeadValuePrediction(prediction);
    const roundTripMs = performance.now() - startedAt;
    const timing = isCached
      ? "cache hit"
      : Number.isFinite(prediction.elapsed_ms)
        ? `${prediction.elapsed_ms.toFixed(1)} ms server · ${roundTripMs.toFixed(1)} ms round trip`
        : `${roundTripMs.toFixed(1)} ms round trip`;
    logDiagnostic(
      `${prediction.model} · value ${formatDecimal(prediction.value, 3, true)} · ` +
        `L/D/W ${(prediction.loss * 100).toFixed(1)}/` +
        `${(prediction.draw * 100).toFixed(1)}/${(prediction.win * 100).toFixed(1)}% · ${timing}`,
      { level: "success", source: "MODEL" },
    );
  } catch (error) {
    if (error.name === "AbortError" || requestNumber !== view.headValueRequestNumber) {
      return;
    }
    view.headValueController = null;
    showHeadValueError(cell, error.message || "The value-head prediction failed.");
    logDiagnostic(
      error.message || "The value-head prediction failed.",
      { level: "error", source: "MODEL" },
    );
  }
}


async function importGame(file) {
  if (view.importing) return;

  view.importing = true;
  stopPlayback();
  elements.fileButton.classList.add("is-busy");
  elements.fileInput.disabled = true;
  elements.emptyFileButton.disabled = true;
  elements.fileButton.querySelector("span").textContent = "Reading game…";
  elements.gameTitle.textContent = file.name;
  elements.gameSubtitle.textContent = "Validating the saved trajectory…";
  showStatus(`Reading ${file.name}…`, "busy");
  logDiagnostic(
    `reading ${file.name} · ${formatFileSize(file.size)}`,
    { source: "NPZ" },
  );

  try {
    const response = await fetch("/api/analyses", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-File-Name": encodeURIComponent(file.name),
      },
      body: file,
    });
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(payload?.detail || "The selected NPZ game could not be read.");
    }

    const previousAnalysisId = view.analysis?.analysis_id;
    view.analysis = payload;
    view.frame = null;
    view.frameIndex = 0;
    view.selectedCell = null;
    view.requestNumber += 1;
    view.frameCache.clear();
    view.pendingFrames.clear();
    view.lastDiagnosticFrameKey = null;
    cancelHeadValuePrediction();
    view.headValueCache.clear();
    resetHeadValuePanel();

    updateGameOverview();
    buildTimeline();
    logAnalysisMetadata(payload);
    await selectFrame(0, { centerTimeline: true, smooth: false });
    showStatus(
      `Loaded ${payload.frame_count} decision frames from ${payload.file_name}.`,
    );

    // A previous file is no longer visible. Releasing its in-memory session
    // keeps repeated imports bounded without making the new import wait.
    if (previousAnalysisId) {
      fetch(`/api/analyses/${previousAnalysisId}`, { method: "DELETE" }).catch(
        () => {},
      );
    }
  } catch (error) {
    elements.gameSubtitle.textContent = "The selected file was not loaded.";
    showStatus(error.message || "Could not read the selected game.", "error");
    logDiagnostic(
      error.message || "Could not read the selected game.",
      { level: "error", source: "NPZ" },
    );
  } finally {
    view.importing = false;
    elements.fileInput.value = "";
    elements.fileButton.classList.remove("is-busy");
    elements.fileInput.disabled = false;
    elements.emptyFileButton.disabled = false;
    elements.fileButton.querySelector("span").textContent = view.analysis
      ? "Open another game"
      : "Open NPZ game";
  }
}


function updateGameOverview() {
  const game = view.analysis;
  elements.dropTarget.classList.add("has-analysis");
  const createdAt = new Date(game.created_at_utc);
  const createdLabel = Number.isNaN(createdAt.getTime())
    ? game.created_at_utc
    : createdAt.toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
      });
  const searchBudget = game.search.budget_type === "seconds"
    ? `${formatDuration(game.search.requested_simulation_seconds)} per move`
    : `${formatInteger(game.search.requested_simulations)} simulations per move`;

  elements.gameTitle.textContent = game.file_name;
  elements.gameTitle.title = game.file_name;
  elements.gameSubtitle.textContent =
    `${createdLabel} · ${searchBudget} · ` +
    `${formatInteger(game.search.total_rollouts)} total rollouts`;
  elements.summaryBoard.textContent = `${game.board.rows} × ${game.board.cols}`;
  elements.summaryMoves.textContent = formatInteger(game.frame_count);
  elements.summaryResult.textContent = resultName(game.final_result);
  elements.summarySchema.textContent = `v${game.schema_version}`;
}


function buildTimeline() {
  elements.timelineList.replaceChildren();
  elements.timelineEmpty.hidden = true;

  for (const descriptor of view.analysis.timeline) {
    const listItem = document.createElement("li");
    const button = document.createElement("button");
    const heading = document.createElement("span");
    const playerDot = document.createElement("i");
    const details = document.createElement("span");
    const value = document.createElement("span");

    listItem.className = "timeline-entry";
    button.className = "timeline-item";
    button.type = "button";
    button.dataset.frameIndex = String(descriptor.index);
    button.setAttribute(
      "aria-label",
      `Before move ${descriptor.move_number}, ${playerName(descriptor.player_to_move)} to move`,
    );

    heading.className = "timeline-item-heading";
    playerDot.className = descriptor.player_to_move === PLAYER_1
      ? "timeline-player player-one"
      : "timeline-player player-two";
    heading.append(
      playerDot,
      document.createTextNode(`Before move ${descriptor.move_number}`),
    );

    details.className = "timeline-item-details";
    details.textContent =
      `${playerName(descriptor.player_to_move)} · ` +
      `action ${formatCoordinate(descriptor.selected_action)}`;

    value.className = "timeline-item-value";
    value.textContent = descriptor.selected_mean_value === null
      ? "unvisited"
      : `Q/N ${formatDecimal(descriptor.selected_mean_value, 2, true)}`;

    button.append(heading, details, value);
    button.addEventListener("click", () => {
      selectFrame(descriptor.index, { centerTimeline: true, smooth: true });
    });
    listItem.append(button);
    elements.timelineList.append(listItem);
  }

  elements.previousFrame.disabled = false;
  elements.playTimeline.disabled = false;
  elements.nextFrame.disabled = false;
  updateTimelinePadding();
  elements.timelineWheel.scrollTop = 0;
}


async function fetchFrame(frameIndex) {
  if (view.frameCache.has(frameIndex)) {
    return view.frameCache.get(frameIndex);
  }
  if (view.pendingFrames.has(frameIndex)) {
    return view.pendingFrames.get(frameIndex);
  }

  const analysisId = view.analysis.analysis_id;
  const pendingRequest = fetch(
    `/api/analyses/${analysisId}/frames/${frameIndex}`,
    { cache: "no-store" },
  )
    .then(async (response) => {
      const payload = await responsePayload(response);
      if (!response.ok) {
        throw new Error(payload?.detail || `Could not read frame ${frameIndex + 1}.`);
      }
      view.frameCache.set(frameIndex, payload);
      return payload;
    })
    .finally(() => {
      view.pendingFrames.delete(frameIndex);
    });

  view.pendingFrames.set(frameIndex, pendingRequest);
  return pendingRequest;
}


async function selectFrame(
  requestedIndex,
  { centerTimeline = false, smooth = false } = {},
) {
  if (!view.analysis) return;

  const frameIndex = Math.max(
    0,
    Math.min(requestedIndex, view.analysis.frame_count - 1),
  );
  cancelHeadValuePrediction();
  if (view.overlay === "head-value") resetHeadValuePanel();
  const requestNumber = ++view.requestNumber;
  view.frameIndex = frameIndex;
  updateActiveTimelineItem();
  updatePlaybackControls();

  if (centerTimeline) centerTimelineItem(frameIndex, smooth);
  if (!view.frameCache.has(frameIndex)) {
    showStatus(`Loading decision frame ${frameIndex + 1}…`, "busy");
  }

  try {
    const frame = await fetchFrame(frameIndex);
    if (requestNumber !== view.requestNumber) return;

    view.frame = frame;
    view.selectedCell = [...frame.selected_action];
    updateFrameDisplay();
    preloadNeighboringFrames(frameIndex);
    showStatus(
      `Before move ${frame.move_number}: ${playerName(frame.player_to_move)} selected ` +
      `${formatCoordinate(frame.selected_action)} after ${formatInteger(frame.completed_rollouts)} rollouts.`,
    );
    const diagnosticFrameKey = `${view.analysis.analysis_id}:${frameIndex}`;
    if (view.lastDiagnosticFrameKey !== diagnosticFrameKey) {
      view.lastDiagnosticFrameKey = diagnosticFrameKey;
      logDiagnostic(
        `move ${frame.move_number}/${view.analysis.frame_count} · ` +
          `${playerName(frame.player_to_move)} · action ` +
          `${formatCoordinate(frame.selected_action)} · ` +
          `${formatInteger(frame.completed_rollouts)} rollouts · ` +
          `${formatDuration(frame.elapsed_seconds)}`,
        { source: "FRAME" },
      );
    }
  } catch (error) {
    if (requestNumber !== view.requestNumber) return;
    showStatus(error.message || "Could not read the selected frame.", "error");
    logDiagnostic(
      error.message || "Could not read the selected frame.",
      { level: "error", source: "FRAME" },
    );
  }
}


function preloadNeighboringFrames(frameIndex) {
  for (const neighbor of [frameIndex - 1, frameIndex + 1]) {
    if (neighbor >= 0 && neighbor < view.analysis.frame_count) {
      fetchFrame(neighbor).catch(() => {});
    }
  }
}


function updateFrameDisplay() {
  const frame = view.frame;
  const selected = frame.selected_action_statistics;

  elements.boardEmpty.hidden = true;
  elements.boardTitle.textContent = `Before move ${frame.move_number}`;
  elements.playerPill.textContent = `${playerName(frame.player_to_move)} to move`;
  elements.playerPill.className = frame.player_to_move === PLAYER_1
    ? "player-pill player-one"
    : "player-pill player-two";
  elements.frameCounter.textContent =
    `${frame.move_number} / ${view.analysis.frame_count}`;
  elements.frameAction.textContent =
    `${formatCoordinate(frame.selected_action)} · ` +
    `${formatDecimal(selected.mean_value, 2, true)} Q/N`;
  elements.frameScore.textContent =
    `${frame.scores.player_1} — ${frame.scores.player_2}`;
  elements.frameLegal.textContent = formatInteger(frame.legal_move_count);
  elements.frameRollouts.textContent = formatInteger(frame.completed_rollouts);
  elements.frameElapsed.textContent = formatDuration(frame.elapsed_seconds);
  elements.frameThroughput.textContent =
    `${formatInteger(Math.round(frame.rollouts_per_second))} / s`;
  for (const metric of document.querySelectorAll(".frame-summary strong")) {
    metric.title = metric.textContent;
  }

  elements.board.setAttribute(
    "aria-label",
    `${frame.rows} by ${frame.cols} Dots board before move ${frame.move_number}. ` +
    `${playerName(frame.player_to_move)} to move. Selected action ` +
    `${formatCoordinate(frame.selected_action)}.`,
  );

  boardRenderer.setFrame(frame);
  boardRenderer.setOverlay(view.overlay);
  selectBoardCell(view.selectedCell, false);
}


function selectBoardCell(cell, runHeadValuePrediction = true) {
  if (!view.frame || !cell) return;

  const [row, col] = cell;
  const contents = view.frame.board[row][col];
  const territoryOwner = view.frame.territory[row][col];
  const isLegal = view.frame.legal_mask[row][col] === 1;
  const rawQ = view.frame.q_values[row][col];
  const visits = view.frame.visit_counts[row][col];
  const totalVisits = view.frame.visit_counts
    .flat()
    .reduce((sum, value) => sum + value, 0);
  const meanValue = visits ? rawQ / visits : null;
  const policy = totalVisits ? visits / totalVisits : 0;
  const isSelectedAction =
    row === view.frame.selected_action[0] && col === view.frame.selected_action[1];

  let contentsLabel = "Empty";
  if (contents === PLAYER_1) contentsLabel = "Player 1 dot";
  if (contents === PLAYER_2) contentsLabel = "Player 2 dot";
  if (territoryOwner !== 0) {
    contentsLabel += ` · ${playerName(territoryOwner)} territory`;
  }

  view.selectedCell = [row, col];
  elements.cellTitle.textContent =
    `Position ${formatCoordinate(cell)}${isSelectedAction ? " · selected action" : ""}`;
  elements.cellContents.textContent = contentsLabel;
  elements.cellLegal.textContent = isLegal ? "Yes" : "No";
  elements.cellRawQ.textContent = visits
    ? formatDecimal(rawQ, 0, true)
    : isLegal ? "Unvisited" : "—";
  elements.cellValue.textContent = visits
    ? formatDecimal(meanValue, 3, true)
    : isLegal ? "Unvisited" : "—";
  elements.cellVisits.textContent = isLegal ? formatInteger(visits) : "—";
  elements.cellPolicy.textContent = isLegal
    ? `${(policy * 100).toFixed(2)}%`
    : "—";
  boardRenderer.setSelectedCell(cell);

  if (view.overlay === "head-value" && runHeadValuePrediction) {
    if (isLegal) requestHeadValue(cell);
    else {
      showHeadValueError(cell, "Choose an empty position marked as legal.");
      logDiagnostic(
        `rejected move ${formatCoordinate(cell)} · position is not legal`,
        { level: "warning", source: "MODEL" },
      );
    }
  }
}


function updateActiveTimelineItem() {
  const timelineItems = elements.timelineList.querySelectorAll(".timeline-item");
  for (const item of timelineItems) {
    const isActive = Number(item.dataset.frameIndex) === view.frameIndex;
    item.classList.toggle("is-active", isActive);
    item.setAttribute("aria-current", isActive ? "step" : "false");
  }
}


function centerTimelineItem(frameIndex, smooth) {
  const item = elements.timelineList.querySelector(
    `[data-frame-index="${frameIndex}"]`,
  );
  if (!item) return;

  view.isCenteringTimeline = true;
  window.clearTimeout(view.centeringTimer);
  const wheel = elements.timelineWheel;
  const itemBounds = item.getBoundingClientRect();
  const wheelBounds = wheel.getBoundingClientRect();
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Center inside the timeline without scrolling the surrounding workspace.
  wheel.scrollTo({
    top: wheel.scrollTop + itemBounds.top - wheelBounds.top -
      wheel.clientTop - (wheel.clientHeight - itemBounds.height) / 2,
    behavior: smooth && !reduceMotion ? "smooth" : "instant",
  });
  view.centeringTimer = window.setTimeout(() => {
    view.isCenteringTimeline = false;
  }, smooth ? 420 : 80);
}


function timelineIndexNearestCenter() {
  const wheelBounds = elements.timelineWheel.getBoundingClientRect();
  const wheelCenter = wheelBounds.top + wheelBounds.height / 2;
  let nearestIndex = view.frameIndex;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (const item of elements.timelineList.querySelectorAll(".timeline-item")) {
    const itemBounds = item.getBoundingClientRect();
    const itemCenter = itemBounds.top + itemBounds.height / 2;
    const distance = Math.abs(itemCenter - wheelCenter);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = Number(item.dataset.frameIndex);
    }
  }
  return nearestIndex;
}


function handleTimelineScroll() {
  if (!view.analysis || view.isCenteringTimeline) return;
  window.clearTimeout(view.scrollTimer);
  view.scrollTimer = window.setTimeout(() => {
    const nearestIndex = timelineIndexNearestCenter();
    if (nearestIndex !== view.frameIndex) {
      stopPlayback();
      selectFrame(nearestIndex, { centerTimeline: false });
    }
  }, 90);
}


function moveFrame(direction) {
  if (!view.analysis) return;
  stopPlayback();
  selectFrame(view.frameIndex + direction, {
    centerTimeline: true,
    smooth: true,
  });
}


function updatePlaybackControls() {
  if (!view.analysis) return;
  elements.previousFrame.disabled = view.frameIndex === 0;
  elements.nextFrame.disabled = view.frameIndex === view.analysis.frame_count - 1;
}


function startPlayback() {
  if (!view.analysis || view.playTimer) return;
  if (view.frameIndex === view.analysis.frame_count - 1) {
    selectFrame(0, { centerTimeline: true, smooth: false });
  }

  elements.playTimeline.textContent = "Pause";
  elements.playTimeline.classList.add("is-playing");
  view.playTimer = window.setInterval(() => {
    if (view.frameIndex >= view.analysis.frame_count - 1) {
      stopPlayback();
      return;
    }
    selectFrame(view.frameIndex + 1, {
      centerTimeline: true,
      smooth: true,
    });
  }, 900);
}


function stopPlayback() {
  if (view.playTimer) window.clearInterval(view.playTimer);
  view.playTimer = null;
  elements.playTimeline.textContent = "Play";
  elements.playTimeline.classList.remove("is-playing");
}


function togglePlayback() {
  if (view.playTimer) stopPlayback();
  else startPlayback();
}


elements.fileInput.addEventListener("change", () => {
  const selectedFile = elements.fileInput.files[0];
  if (selectedFile) importGame(selectedFile);
});

elements.emptyFileButton.addEventListener("click", () => elements.fileInput.click());


elements.overlaySwitcher.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-overlay]");
  if (!button) return;

  view.overlay = button.dataset.overlay;
  for (const overlayButton of elements.overlaySwitcher.querySelectorAll("button")) {
    overlayButton.classList.toggle("is-active", overlayButton === button);
    overlayButton.setAttribute("aria-pressed", String(overlayButton === button));
  }
  const descriptions = {
    "head-value": "Click a legal position · prediction appears below",
    value: "Mean result · player-to-move perspective",
    "raw-q": "Win/loss balance · player-to-move perspective",
    visits: "Completed visits · brighter means more visits",
    policy: "Share of visits · brighter means higher probability",
    none: "Board and placed dots · search overlays hidden",
  };
  const isHeadValue = view.overlay === "head-value";
  const isSequential = view.overlay === "visits" || view.overlay === "policy";
  elements.overlayDescription.textContent = descriptions[view.overlay];
  elements.overlayScale.hidden = view.overlay === "none" || isHeadValue;
  elements.overlayScale.classList.toggle("is-sequential", isSequential);
  elements.scaleLow.textContent = isSequential ? "Low" : "Negative";
  elements.scaleHigh.textContent = isSequential ? "High" : "Positive";
  elements.headValuePanel.hidden = !isHeadValue;
  if (isHeadValue) updateHeadValueWorkspaceHeight();
  cancelHeadValuePrediction();
  if (isHeadValue) {
    resetHeadValuePanel(
      view.frame
        ? undefined
        : "Open a saved 10 × 10 game, then click a legal position.",
    );
  }
  boardRenderer.setOverlay(view.overlay);
});


elements.previousFrame.addEventListener("click", () => moveFrame(-1));
elements.nextFrame.addEventListener("click", () => moveFrame(1));
elements.playTimeline.addEventListener("click", togglePlayback);
elements.timelineWheel.addEventListener("scroll", handleTimelineScroll, {
  passive: true,
});
elements.timelineWheel.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp") {
    event.preventDefault();
    moveFrame(-1);
  }
  if (event.key === "ArrowDown") {
    event.preventDefault();
    moveFrame(1);
  }
  if (event.key === "Home") {
    event.preventDefault();
    stopPlayback();
    selectFrame(0, { centerTimeline: true, smooth: true });
  }
  if (event.key === "End" && view.analysis) {
    event.preventDefault();
    stopPlayback();
    selectFrame(view.analysis.frame_count - 1, {
      centerTimeline: true,
      smooth: true,
    });
  }
  if (event.key === " ") {
    event.preventDefault();
    togglePlayback();
  }
});


for (const eventName of ["dragenter", "dragover"]) {
  elements.dropTarget.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "dragenter") view.dragDepth += 1;
    elements.dropOverlay.hidden = false;
  });
}


elements.dropTarget.addEventListener("dragleave", (event) => {
  event.preventDefault();
  view.dragDepth = Math.max(0, view.dragDepth - 1);
  if (view.dragDepth === 0) elements.dropOverlay.hidden = true;
});


elements.dropTarget.addEventListener("drop", (event) => {
  event.preventDefault();
  view.dragDepth = 0;
  elements.dropOverlay.hidden = true;
  const droppedFile = event.dataTransfer.files[0];
  if (droppedFile) importGame(droppedFile);
});


elements.clearDiagnostics.addEventListener("click", () => {
  elements.diagnosticsOutput.replaceChildren();
  logDiagnostic("diagnostic buffer cleared", { source: "SYSTEM" });
});


elements.copyDiagnostics.addEventListener("click", async () => {
  try {
    const plainText = [...elements.diagnosticsOutput.children]
      .map((line) => line.dataset.plainText)
      .join("\n");
    await navigator.clipboard.writeText(plainText);
    elements.copyDiagnostics.textContent = "Copied";
    window.setTimeout(() => {
      elements.copyDiagnostics.textContent = "Copy";
    }, 1200);
  } catch (error) {
    logDiagnostic(
      error.message || "Could not copy diagnostics.",
      { level: "error", source: "SYSTEM" },
    );
  }
});


logDiagnostic("analysis workspace ready · waiting for a saved game", {
  level: "success",
  source: "SYSTEM",
});
logDiagnostic("Python hook ready · from analysis import PRINT_T", {
  source: "SYSTEM",
});
pollDiagnostics();
