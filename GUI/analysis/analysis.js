"use strict";

import {
  DotsBoardRenderer,
  PLAYER_1,
  PLAYER_2,
} from "/shared/board_renderer.js?v=20260921-policy-split";


const elements = {
  fileInput: document.querySelector("#npz-file"),
  fileButton: document.querySelector("#file-button"),
  screenshotButton: document.querySelector("#screenshot-button"),
  panelScreenshotButton: document.querySelector("#panel-screenshot-button"),
  emptyFileButton: document.querySelector("#open-empty-file"),
  dropTarget: document.querySelector("#drop-target"),
  dropOverlay: document.querySelector("#drop-overlay"),
  workspace: document.querySelector(".workspace"),
  boardPanel: document.querySelector(".board-panel"),
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
  cellPrior: document.querySelector("#cell-prior"),
  cellRawQ: document.querySelector("#cell-raw-q"),
  cellValue: document.querySelector("#cell-value"),
  cellVisits: document.querySelector("#cell-visits"),
  cellVisitShare: document.querySelector("#cell-visit-share"),
  cellPolicyFlow: document.querySelector("#cell-policy-flow"),
  moveComparisonNote: document.querySelector("#move-comparison-note"),
  moveComparisonBody: document.querySelector("#move-comparison-body"),
  headValuePanel: document.querySelector("#head-value-panel"),
  headValueTitle: document.querySelector("#head-value-title"),
  headValueModel: document.querySelector("#head-value-model"),
  headValueMessage: document.querySelector("#head-value-message"),
  headValueLoss: document.querySelector("#head-value-loss"),
  headValueDraw: document.querySelector("#head-value-draw"),
  headValueWin: document.querySelector("#head-value-win"),
  headValueScore: document.querySelector("#head-value-score"),
  headPolicyPanel: document.querySelector("#head-policy-panel"),
  headPolicyTitle: document.querySelector("#head-policy-title"),
  headPolicyModel: document.querySelector("#head-policy-model"),
  headPolicyMessage: document.querySelector("#head-policy-message"),
  headPolicySelected: document.querySelector("#head-policy-selected"),
  headPolicySelectedPrior: document.querySelector("#head-policy-selected-prior"),
  headPolicyTopMove: document.querySelector("#head-policy-top-move"),
  headPolicyTopPrior: document.querySelector("#head-policy-top-prior"),
  experimentPanel: document.querySelector(".experiment-panel"),
  experimentTitle: document.querySelector("#experiment-title"),
  experimentStatus: document.querySelector("#experiment-status"),
  startExperiment: document.querySelector("#start-experiment"),
  stepExperiment: document.querySelector("#step-experiment"),
  continueExperiment: document.querySelector("#continue-experiment"),
  showExperiment: document.querySelector("#show-experiment"),
  timelinePanel: document.querySelector(".timeline-panel"),
  timelineTitle: document.querySelector("#timeline-title"),
  timelineSourceSwitcher: document.querySelector("#timeline-source-switcher"),
  originalTimelineRange: document.querySelector("#original-timeline-range"),
  replayTimelineRange: document.querySelector("#replay-timeline-range"),
  frameCounter: document.querySelector("#frame-counter"),
  previousFrame: document.querySelector("#previous-frame"),
  playTimeline: document.querySelector("#play-timeline"),
  nextFrame: document.querySelector("#next-frame"),
  timelineWheel: document.querySelector("#timeline-wheel"),
  timelineList: document.querySelector("#timeline-list"),
  timelineEmpty: document.querySelector("#timeline-empty"),
  timelineEmptyTitle: document.querySelector("#timeline-empty-title"),
  timelineEmptyMessage: document.querySelector("#timeline-empty-message"),
  status: document.querySelector("#status"),
  diagnosticsOutput: document.querySelector("#diagnostics-output"),
  diagnosticsConnection: document.querySelector("#diagnostics-connection"),
  copyDiagnostics: document.querySelector("#copy-diagnostics"),
  clearDiagnostics: document.querySelector("#clear-diagnostics"),
};


const view = {
  mode: "analysis",
  analysis: null,
  frame: null,
  frameIndex: 0,
  originalFrameIndex: 0,
  replayFrameIndex: 0,
  timelineSource: "original",
  selectedCell: null,
  overlay: "value",
  importing: false,
  requestNumber: 0,
  frameCache: new Map(),
  pendingFrames: new Map(),
  headValueCache: new Map(),
  headValueRequestNumber: 0,
  headValueController: null,
  headPolicyCache: new Map(),
  headPolicyPrediction: null,
  headPolicyRequestNumber: 0,
  headPolicyController: null,
  experiment: null,
  experimentRequestPending: false,
  experimentPollTimer: null,
  displayingExperiment: false,
  displayedExperimentMoves: 0,
  scrollTimer: null,
  centeringTimer: null,
  isCenteringTimeline: false,
  playTimer: null,
  dragDepth: 0,
  diagnosticsCursor: 0,
  diagnosticsPollTimer: null,
  liveRevision: null,
  livePollTimer: null,
  lastDiagnosticFrameKey: null,
  screenshotting: false,
};


const MAX_DIAGNOSTIC_LINES = 160;
const TOP_MOVE_ROW_COUNT = 6;
const PANEL_EXPORT_WIDTH = 1180;
const PANEL_EXPORT_BOARD_HEIGHT = 590;


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


function updateModelPanelWorkspaceHeight() {
  const panel = [elements.headValuePanel, elements.headPolicyPanel]
    .find((candidate) => !candidate.hidden);
  if (!panel) return;
  const panelStyles = getComputedStyle(panel);
  const panelSpace =
    panel.getBoundingClientRect().height +
    Number.parseFloat(panelStyles.marginTop || "0");
  elements.workspace.style.setProperty(
    "--head-value-panel-space",
    `${panelSpace}px`,
  );
}


const headValueResizeObserver = new ResizeObserver(updateModelPanelWorkspaceHeight);
headValueResizeObserver.observe(elements.headValuePanel);
headValueResizeObserver.observe(elements.headPolicyPanel);


function playerName(player) {
  return player === PLAYER_1 ? "Player 1" : "Player 2";
}


function resultName(result) {
  if (result === null || result === undefined) return "In progress";
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


function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}


function policyPriorsForCurrentFrame() {
  return view.frame?.policy_priors || view.headPolicyPrediction?.policy || null;
}


function policyPriorAt(cell) {
  if (!cell) return null;
  const priors = policyPriorsForCurrentFrame();
  const value = priors?.[cell[0]]?.[cell[1]];
  return Number.isFinite(value) ? value : null;
}


function topMovesFromPolicy(policy, legalMask, limit = 5) {
  if (!policy || !legalMask) return [];
  const moves = [];
  for (let row = 0; row < policy.length; row += 1) {
    for (let col = 0; col < policy[row].length; col += 1) {
      const probability = policy[row][col];
      if (legalMask[row]?.[col] && Number.isFinite(probability)) {
        moves.push({ coordinate: [row, col], probability });
      }
    }
  }
  moves.sort((left, right) => right.probability - left.probability);
  return moves.slice(0, limit);
}


function embeddedHeadPolicyPrediction(frame) {
  if (!frame?.policy_priors) return null;
  return {
    player: frame.player_to_move,
    move_number: frame.move_number,
    model: "search-time prior",
    policy: frame.policy_priors,
    top_moves: topMovesFromPolicy(
      frame.policy_priors,
      frame.legal_mask,
    ),
    source: "search",
  };
}


function renderMoveComparison() {
  elements.moveComparisonBody.replaceChildren();
  const frame = view.frame;
  if (!frame || !frame.selected_action) {
    elements.moveComparisonNote.textContent = "Available after a search completes";
    return;
  }

  const priors = policyPriorsForCurrentFrame();
  const totalVisits = frame.visit_counts
    .flat()
    .reduce((sum, value) => sum + value, 0);
  const actions = [];
  for (let row = 0; row < frame.rows; row += 1) {
    for (let col = 0; col < frame.cols; col += 1) {
      if (!frame.legal_mask[row][col]) continue;
      const visits = frame.visit_counts[row][col];
      const prior = priors?.[row]?.[col];
      if (!visits && !Number.isFinite(prior)) continue;
      const rawQ = frame.q_values[row][col];
      actions.push({
        coordinate: [row, col],
        prior: Number.isFinite(prior) ? prior : null,
        meanValue: visits ? rawQ / visits : null,
        visits,
        visitShare: totalVisits ? visits / totalVisits : 0,
      });
    }
  }

  for (const action of actions) {
    action.priorRank = action.prior === null
      ? null
      : 1 + actions.filter((candidate) =>
        candidate.prior !== null && candidate.prior > action.prior
      ).length;
    action.visitRank = 1 + actions.filter(
      (candidate) => candidate.visits > action.visits
    ).length;
  }
  const byVisits = [...actions].sort((left, right) =>
    right.visits - left.visits ||
    (right.prior ?? -1) - (left.prior ?? -1)
  );
  const byPrior = priors
    ? [...actions].sort((left, right) =>
      (right.prior ?? -1) - (left.prior ?? -1) ||
      right.visits - left.visits
    )
    : [];
  const selectedKey = frame.selected_action.join(":");
  const selectedAction = actions.find(
    (candidate) => candidate.coordinate.join(":") === selectedKey
  );
  const candidatesByKey = new Map();
  for (const action of [...byVisits, ...byPrior]) {
    candidatesByKey.set(action.coordinate.join(":"), action);
  }
  const candidates = [...candidatesByKey.values()].sort((left, right) => {
    const leftBestRank = Math.min(
      left.visitRank,
      left.priorRank ?? Number.POSITIVE_INFINITY,
    );
    const rightBestRank = Math.min(
      right.visitRank,
      right.priorRank ?? Number.POSITIVE_INFINITY,
    );
    const leftRankSum = left.visitRank + (left.priorRank ?? left.visitRank);
    const rightRankSum = right.visitRank + (right.priorRank ?? right.visitRank);
    return leftBestRank - rightBestRank ||
      leftRankSum - rightRankSum ||
      left.visitRank - right.visitRank;
  });
  const displayedByKey = new Map();
  if (selectedAction) displayedByKey.set(selectedKey, selectedAction);
  for (const action of candidates) {
    if (displayedByKey.size >= TOP_MOVE_ROW_COUNT) break;
    displayedByKey.set(action.coordinate.join(":"), action);
  }
  const displayedActions = [...displayedByKey.values()].sort((left, right) =>
    left.visitRank - right.visitRank ||
    (left.priorRank ?? Number.POSITIVE_INFINITY) -
      (right.priorRank ?? Number.POSITIVE_INFINITY)
  );

  elements.moveComparisonNote.textContent = frame.policy_priors
    ? "Exact prior used by this search"
    : priors
      ? "Fresh model inference; not stored with this search"
      : "Model prior unavailable in this recording";

  for (const action of displayedActions) {
    const row = document.createElement("tr");
    const isSelected =
      action.coordinate[0] === frame.selected_action[0] &&
      action.coordinate[1] === frame.selected_action[1];
    row.classList.toggle("is-selected", isSelected);

    const moveCell = document.createElement("td");
    const moveButton = document.createElement("button");
    moveButton.type = "button";
    moveButton.textContent = formatCoordinate(action.coordinate);
    moveButton.title = `Inspect ${formatCoordinate(action.coordinate)}`;
    moveButton.addEventListener("click", () => selectBoardCell(action.coordinate));
    moveCell.append(moveButton);

    const values = [
      formatPercent(action.prior),
      formatDecimal(action.meanValue, 3, true),
      formatInteger(action.visits),
      formatPercent(action.visitShare),
      action.priorRank === null
        ? `— → #${action.visitRank}`
        : `#${action.priorRank} → #${action.visitRank}`,
    ];
    row.append(moveCell);
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.moveComparisonBody.append(row);
  }
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


function safeFileStem(fileName) {
  return fileName
    .replace(/\.npz$/i, "")
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "-")
    .replace(/\s+/g, "-")
    .replace(/^-+|-+$/g, "") || "dots-analysis";
}


function screenshotFileName() {
  const game = safeFileStem(view.analysis?.file_name || "dots-analysis");
  const move = view.frame?.move_number || view.frameIndex + 1;
  const selected = view.selectedCell || view.frame?.selected_action;
  const cell = selected ? `-r${selected[0]}-c${selected[1]}` : "";
  return `${game}-move-${move}-${view.overlay}${cell}.png`;
}


function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("The browser could not create the PNG screenshot."));
    }, "image/png");
  });
}


function updateScreenshotAvailability() {
  elements.screenshotButton.disabled =
    !view.frame || view.importing || view.screenshotting;
}


async function downloadSquareScreenshot() {
  if (!view.frame || view.screenshotting) return;

  view.screenshotting = true;
  elements.screenshotButton.classList.add("is-busy");
  elements.screenshotButton.title = "Creating square screenshot…";
  updateScreenshotAvailability();

  try {
    const screenshot = boardRenderer.renderSquareCanvas(1600);
    const blob = await canvasToPngBlob(screenshot);
    const fileName = screenshotFileName();
    const objectUrl = URL.createObjectURL(blob);
    const download = document.createElement("a");
    download.href = objectUrl;
    download.download = fileName;
    document.body.append(download);
    download.click();
    download.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);

    showStatus(`Saved ${fileName} as a 1:1 PNG.`);
    logDiagnostic(
      `square screenshot · frame ${view.frameIndex + 1} · ${view.overlay} overlay · 1600 × 1600 PNG`,
      { level: "success", source: "EXPORT" },
    );
  } catch (error) {
    showStatus(error.message || "Could not save the screenshot.", "error");
    logDiagnostic(
      error.message || "Could not save the screenshot.",
      { level: "error", source: "EXPORT" },
    );
  } finally {
    view.screenshotting = false;
    elements.screenshotButton.classList.remove("is-busy");
    elements.screenshotButton.title = "Download square screenshot";
    updateScreenshotAvailability();
  }
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
  const schemaLabel = game.source_type === "live"
    ? "live source"
    : `schema v${game.schema_version}`;
  logDiagnostic(
    `game ${game.game_id} · ${schemaLabel} · ` +
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


function experimentBudgetLabel(search) {
  if (!search) return "saved search settings";
  const budget = search.budget_type === "seconds"
    ? `${formatDuration(search.simulation_seconds)} per move`
    : `${formatInteger(search.simulations_number)} simulations per move`;
  const execution = search.rollout_batch_size === 1
    ? "sequential"
    : `${formatInteger(search.rollout_batch_size)} rollout workers`;
  const exploration = Number.isFinite(search.uct_c_param)
    ? `UCT c ${formatDecimal(search.uct_c_param, 2)}`
    : "UCT c unknown";
  return `${budget} · ${execution} · ${exploration}`;
}


function stopExperimentPolling() {
  if (view.experimentPollTimer) window.clearTimeout(view.experimentPollTimer);
  view.experimentPollTimer = null;
}


function replayFrames() {
  return view.experiment?.frames || [];
}


function timelineLength() {
  return view.timelineSource === "replay"
    ? replayFrames().length
    : view.analysis?.frame_count || 0;
}


function updateTimelineSourceControls() {
  const originalButton = elements.timelineSourceSwitcher.querySelector(
    '[data-timeline-source="original"]',
  );
  const replayButton = elements.timelineSourceSwitcher.querySelector(
    '[data-timeline-source="replay"]',
  );
  const frames = replayFrames();
  const hasOriginal = Boolean(view.analysis);
  const hasReplay = frames.length > 0;

  originalButton.disabled = !hasOriginal;
  replayButton.disabled = !hasReplay;
  elements.originalTimelineRange.textContent = hasOriginal
    ? view.analysis.frame_count
      ? `Moves 1–${view.analysis.frame_count}`
      : view.mode === "live" ? "Waiting for move 1" : "No frames"
    : "Not loaded";
  elements.replayTimelineRange.textContent = hasReplay
    ? `Moves ${frames[0].move_number}–${frames.at(-1).move_number}`
    : view.experiment
      ? `Starts at move ${view.experiment.source_move_number}`
      : "Not started";

  for (const button of elements.timelineSourceSwitcher.querySelectorAll("button")) {
    const isActive = button.dataset.timelineSource === view.timelineSource;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
  elements.timelinePanel.classList.toggle(
    "is-replay",
    view.timelineSource === "replay",
  );
  elements.timelineTitle.textContent = view.timelineSource === "replay"
    ? "Forced replay"
    : view.mode === "live" ? "Live moves" : "Moves";
}


function updateExperimentControls() {
  if (view.mode === "live") {
    updateTimelineSourceControls();
    return;
  }
  const experiment = view.experiment;
  const hasSavedFrame = Boolean(
    view.analysis && view.frame && view.timelineSource === "original",
  );
  const isRunning = experiment?.status === "running";
  const isFinished = Boolean(experiment?.current_state?.game_over);
  const isUnavailable = view.importing || view.experimentRequestPending;

  elements.experimentPanel.classList.toggle("is-running", isRunning);
  elements.experimentPanel.classList.toggle(
    "is-failed",
    experiment?.status === "failed",
  );
  elements.startExperiment.disabled = !hasSavedFrame || isRunning || isUnavailable;
  elements.stepExperiment.disabled = !experiment || isRunning || isFinished || isUnavailable;
  elements.continueExperiment.disabled = !experiment || isRunning || isFinished || isUnavailable;
  elements.showExperiment.disabled = replayFrames().length === 0 || isUnavailable;
  elements.startExperiment.textContent = experiment
    ? `Reset from move ${view.originalFrameIndex + 1}`
    : `Start from move ${view.originalFrameIndex + 1}`;
  updateTimelineSourceControls();

  if (!view.analysis) {
    elements.experimentTitle.textContent = "Branch from the selected saved frame";
    elements.experimentStatus.textContent =
      "Open a game and choose the position you want to replay.";
    return;
  }
  if (!experiment) {
    elements.experimentTitle.textContent =
      `Start an alternative branch before move ${view.originalFrameIndex + 1}`;
    elements.experimentStatus.textContent =
      "The imported timeline will remain unchanged.";
    return;
  }

  elements.experimentTitle.textContent =
    `Branch from before move ${experiment.source_move_number}`;
  if (experiment.status === "running") {
    const action = experiment.mode === "continue"
      ? "Playing the continuation"
      : `Searching move ${experiment.next_move_number}`;
    elements.experimentStatus.textContent =
      `${action} · ${experimentBudgetLabel(experiment.search)}`;
    return;
  }
  if (experiment.status === "failed") {
    elements.experimentStatus.textContent =
      experiment.error || "The forced replay search failed.";
    return;
  }
  if (isFinished) {
    elements.experimentStatus.textContent =
      `Game over · ${resultName(experiment.current_state.winner)} · ` +
      `score ${experiment.current_state.scores.player_1}—` +
      `${experiment.current_state.scores.player_2}`;
    return;
  }
  elements.experimentStatus.textContent =
    `${formatInteger(experiment.moves_completed)} forced replay moves · ` +
    `next is move ${experiment.next_move_number} · ` +
    experimentBudgetLabel(experiment.search);
}


async function startExperimentFromSelectedFrame() {
  if (!view.analysis || !view.frame || view.timelineSource !== "original") return;
  stopPlayback();
  stopExperimentPolling();
  view.experimentRequestPending = true;
  updateExperimentControls();
  showStatus(
    `Preparing a forced replay before move ${view.originalFrameIndex + 1}…`,
    "busy",
  );

  try {
    const query = new URLSearchParams({
      frame_index: String(view.originalFrameIndex),
    });
    const response = await fetch(
      `/api/analyses/${view.analysis.analysis_id}/experiment?${query}`,
      { method: "POST", cache: "no-store" },
    );
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(payload?.detail || "Could not start the forced replay.");
    }
    view.experiment = payload;
    view.replayFrameIndex = 0;
    view.displayedExperimentMoves = 0;
    logDiagnostic(
      `branch created before move ${payload.source_move_number} · ` +
        experimentBudgetLabel(payload.search),
      { level: "success", source: "MCTS" },
    );
    showStatus(
      `Forced replay ready before move ${payload.source_move_number}.`,
    );
  } catch (error) {
    showStatus(error.message || "Could not start the forced replay.", "error");
    logDiagnostic(
      error.message || "Could not start the forced replay.",
      { level: "error", source: "MCTS" },
    );
  } finally {
    view.experimentRequestPending = false;
    updateExperimentControls();
  }
}


async function runExperimentCommand(command) {
  if (!view.analysis || !view.experiment || view.experimentRequestPending) return;
  view.experimentRequestPending = true;
  updateExperimentControls();

  try {
    const response = await fetch(
      `/api/analyses/${view.analysis.analysis_id}/experiment/${command}`,
      { method: "POST", cache: "no-store" },
    );
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(payload?.detail || "Could not start the MCTS search.");
    }
    view.experiment = payload;
    showStatus(
      command === "step"
        ? `Searching experimental move ${payload.next_move_number}…`
        : `Continuing the experiment from move ${payload.next_move_number}…`,
      "busy",
    );
    logDiagnostic(
      `${command === "step" ? "one step" : "continue game"} requested · ` +
        `next move ${payload.next_move_number}`,
      { source: "MCTS" },
    );
    pollExperiment();
  } catch (error) {
    showStatus(error.message || "Could not start the MCTS search.", "error");
    logDiagnostic(
      error.message || "Could not start the MCTS search.",
      { level: "error", source: "MCTS" },
    );
  } finally {
    view.experimentRequestPending = false;
    updateExperimentControls();
  }
}


async function pollExperiment() {
  stopExperimentPolling();
  if (!view.analysis || !view.experiment) return;

  try {
    const previousFrameCount = replayFrames().length;
    const wasFollowingLatest =
      view.timelineSource === "replay" &&
      view.replayFrameIndex >= previousFrameCount - 1;
    const response = await fetch(
      `/api/analyses/${view.analysis.analysis_id}/experiment`,
      { cache: "no-store" },
    );
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(payload?.detail || "Could not read the forced replay status.");
    }
    const hasNewMove = payload.frames.length > previousFrameCount;
    view.experiment = payload;
    if (hasNewMove) {
      view.displayedExperimentMoves = payload.moves_completed;
      if (previousFrameCount === 0) {
        switchTimelineSource("replay", { selectLatest: true });
      } else if (view.timelineSource === "replay") {
        view.replayFrameIndex = wasFollowingLatest
          ? payload.frames.length - 1
          : Math.min(view.replayFrameIndex, payload.frames.length - 1);
        view.frameIndex = view.replayFrameIndex;
        buildTimeline();
        selectFrame(view.frameIndex, {
          centerTimeline: true,
          smooth: wasFollowingLatest,
        });
      }
    }
    updateExperimentControls();

    if (payload.status === "running") {
      view.experimentPollTimer = window.setTimeout(pollExperiment, 750);
      return;
    }
    if (payload.status === "failed") {
      showStatus(payload.error || "The forced replay failed.", "error");
      return;
    }
    if (payload.current_state.game_over) {
      showStatus(
        `Forced replay complete: ${resultName(payload.current_state.winner)}, ` +
          `score ${payload.current_state.scores.player_1}—` +
          `${payload.current_state.scores.player_2}.`,
      );
    }
  } catch (error) {
    showStatus(error.message || "Could not read the forced replay status.", "error");
    logDiagnostic(
      error.message || "Could not read the forced replay status.",
      { level: "error", source: "MCTS" },
    );
    if (view.experiment?.status === "running") {
      view.experimentPollTimer = window.setTimeout(pollExperiment, 1500);
    }
  }
}


function cancelHeadValuePrediction() {
  view.headValueRequestNumber += 1;
  view.headValueController?.abort();
  view.headValueController = null;
}


function cancelHeadPolicyPrediction() {
  view.headPolicyRequestNumber += 1;
  view.headPolicyController?.abort();
  view.headPolicyController = null;
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


function resetHeadPolicyPanel(message) {
  view.headPolicyPrediction = null;
  boardRenderer.setHeadPolicy(null);
  elements.headPolicyPanel.classList.remove("is-loading", "is-error");
  elements.headPolicyTitle.textContent = "Current position";
  elements.headPolicyMessage.textContent = message ||
    "Open a saved frame to predict probabilities for all legal moves.";
  for (const result of [
    elements.headPolicySelected,
    elements.headPolicySelectedPrior,
    elements.headPolicyTopMove,
    elements.headPolicyTopPrior,
  ]) {
    result.textContent = "—";
  }
  renderMoveComparison();
}


function showHeadPolicyError(message) {
  resetHeadPolicyPanel(message);
  elements.headPolicyPanel.classList.add("is-error");
  elements.headPolicyTitle.textContent = "Policy unavailable";
}


function displayHeadPolicySelection(cell) {
  const prediction = view.headPolicyPrediction;
  if (!prediction || !cell) return;
  const [row, col] = cell;
  const probability = prediction.policy?.[row]?.[col];
  elements.headPolicySelected.textContent = formatCoordinate(cell);
  elements.headPolicySelectedPrior.textContent = Number.isFinite(probability)
    ? `${(probability * 100).toFixed(2)}%`
    : "—";
}


function displayHeadPolicyPrediction(prediction) {
  view.headPolicyPrediction = prediction;
  boardRenderer.setHeadPolicy(prediction.policy);
  elements.headPolicyPanel.classList.remove("is-loading", "is-error");
  elements.headPolicyTitle.textContent =
    `Before move ${prediction.move_number} for ${playerName(prediction.player)}`;
  elements.headPolicyModel.textContent = prediction.model;
  elements.headPolicyModel.title = prediction.model;
  elements.headPolicyMessage.textContent = prediction.source === "search"
    ? "Exact neural priors captured before MCTS changed the move distribution."
    : "Probabilities are normalized over legal moves in the current position.";

  const topMove = prediction.top_moves?.[0];
  elements.headPolicyTopMove.textContent = topMove
    ? formatCoordinate(topMove.coordinate)
    : "—";
  elements.headPolicyTopPrior.textContent = topMove
    ? `${(topMove.probability * 100).toFixed(2)}%`
    : "—";
  displayHeadPolicySelection(view.selectedCell);
  renderMoveComparison();
  if (view.selectedCell) selectBoardCell(view.selectedCell, false);
}


async function requestHeadPolicy() {
  if (!view.analysis || !view.frame || view.displayingExperiment) return;

  const embeddedPrediction = embeddedHeadPolicyPrediction(view.frame);
  if (embeddedPrediction) {
    cancelHeadPolicyPrediction();
    resetHeadPolicyPanel();
    displayHeadPolicyPrediction(embeddedPrediction);
    return;
  }
  if (view.mode === "live") {
    showHeadPolicyError(
      "This live frame does not contain a neural model prior.",
    );
    return;
  }

  const frameIndex = view.frameIndex;
  const cacheKey = `${view.analysis.analysis_id}:${frameIndex}`;
  const startedAt = performance.now();
  const isCached = view.headPolicyCache.has(cacheKey);
  cancelHeadPolicyPrediction();
  const requestNumber = view.headPolicyRequestNumber;

  resetHeadPolicyPanel();
  elements.headPolicyPanel.classList.add("is-loading");
  elements.headPolicyTitle.textContent = `Predicting move ${frameIndex + 1}…`;
  elements.headPolicyMessage.textContent = "Running the local policy-head model.";
  elements.headPolicyTopPrior.textContent = "•••";

  logDiagnostic(
    `${isCached ? "policy cache lookup" : "predict policy"} · frame ${frameIndex + 1}`,
    { source: "MODEL" },
  );

  try {
    let prediction = view.headPolicyCache.get(cacheKey);
    if (!prediction) {
      view.headPolicyController = new AbortController();
      const response = await fetch(
        `/api/analyses/${view.analysis.analysis_id}/frames/${frameIndex}/head-policy`,
        { cache: "no-store", signal: view.headPolicyController.signal },
      );
      const payload = await responsePayload(response);
      if (!response.ok) {
        throw new Error(payload?.detail || "The policy-head prediction failed.");
      }
      prediction = payload;
      view.headPolicyCache.set(cacheKey, prediction);
    }

    if (requestNumber !== view.headPolicyRequestNumber) return;
    view.headPolicyController = null;
    displayHeadPolicyPrediction(prediction);
    const roundTripMs = performance.now() - startedAt;
    const topMove = prediction.top_moves?.[0];
    const timing = isCached
      ? "cache hit"
      : Number.isFinite(prediction.elapsed_ms)
        ? `${prediction.elapsed_ms.toFixed(1)} ms server · ${roundTripMs.toFixed(1)} ms round trip`
        : `${roundTripMs.toFixed(1)} ms round trip`;
    logDiagnostic(
      `${prediction.model} · top ${formatCoordinate(topMove?.coordinate)} ` +
        `${topMove ? `${(topMove.probability * 100).toFixed(2)}%` : "—"} · ${timing}`,
      { level: "success", source: "MODEL" },
    );
  } catch (error) {
    if (error.name === "AbortError" || requestNumber !== view.headPolicyRequestNumber) {
      return;
    }
    view.headPolicyController = null;
    showHeadPolicyError(error.message || "The policy-head prediction failed.");
    logDiagnostic(
      error.message || "The policy-head prediction failed.",
      { level: "error", source: "MODEL" },
    );
  }
}


async function importGame(file) {
  if (view.mode !== "analysis" || view.importing) return;

  view.importing = true;
  stopPlayback();
  stopExperimentPolling();
  updateScreenshotAvailability();
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
    view.originalFrameIndex = 0;
    view.replayFrameIndex = 0;
    view.timelineSource = "original";
    view.selectedCell = null;
    view.requestNumber += 1;
    view.frameCache.clear();
    view.pendingFrames.clear();
    view.lastDiagnosticFrameKey = null;
    view.experiment = null;
    view.displayingExperiment = false;
    view.displayedExperimentMoves = 0;
    cancelHeadValuePrediction();
    view.headValueCache.clear();
    resetHeadValuePanel();
    cancelHeadPolicyPrediction();
    view.headPolicyCache.clear();
    resetHeadPolicyPanel();
    updateExperimentControls();

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
    updateScreenshotAvailability();
    updateExperimentControls();
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

  elements.gameTitle.textContent = view.mode === "live"
    ? "Live MCTS game"
    : game.file_name;
  elements.gameTitle.title = game.file_name;
  elements.gameSubtitle.textContent =
    `${createdLabel} · ${searchBudget} · ` +
    `${formatInteger(game.search.total_rollouts)} total rollouts`;
  elements.summaryBoard.textContent = `${game.board.rows} × ${game.board.cols}`;
  elements.summaryMoves.textContent = formatInteger(game.frame_count);
  elements.summaryResult.textContent = resultName(game.final_result);
  elements.summarySchema.textContent = view.mode === "live"
    ? "Live"
    : `v${game.schema_version}`;
}


function displayLivePosition(frame) {
  view.requestNumber += 1;
  cancelHeadPolicyPrediction();
  view.headPolicyPrediction = null;
  boardRenderer.setHeadPolicy(null);
  view.timelineSource = "original";
  view.displayingExperiment = false;
  view.frame = frame;
  view.frameIndex = Math.max(0, view.analysis.frame_count - 1);
  view.originalFrameIndex = view.frameIndex;
  view.selectedCell = null;
  updateActiveTimelineItem();
  updatePlaybackControls();
  if (view.analysis.frame_count) {
    centerTimelineItem(view.frameIndex, false);
  }
  updateFrameDisplay();
}


async function refreshLiveGame() {
  try {
    const response = await fetch("/api/live", { cache: "no-store" });
    const payload = await responsePayload(response);
    if (!response.ok) {
      throw new Error(payload?.detail || "The live MCTS game is not available.");
    }
    if (payload.revision === view.liveRevision) return;

    const previousFrameCount = view.analysis?.frame_count || 0;
    const wasFollowingLatest =
      !view.frame ||
      view.frame.live_position ||
      view.originalFrameIndex >= Math.max(0, previousFrameCount - 1);
    view.liveRevision = payload.revision;
    view.analysis = payload.analysis;
    updateGameOverview();

    if (payload.analysis.frame_count !== previousFrameCount) {
      buildTimeline();
    }

    if (wasFollowingLatest) {
      if (payload.status === "complete" || payload.analysis.frame_count === 0) {
        displayLivePosition(payload.current_frame);
      } else if (payload.analysis.frame_count > previousFrameCount) {
        await selectOriginalFrame(payload.analysis.frame_count - 1, {
          centerTimeline: true,
          smooth: true,
        });
      }
    }

    showStatus(
      payload.message,
      payload.status === "searching" ? "busy" : "normal",
    );
  } catch (error) {
    showStatus(error.message || "Waiting for the live MCTS game…", "busy");
  } finally {
    view.livePollTimer = window.setTimeout(refreshLiveGame, 500);
  }
}


function configureLiveMode() {
  document.body.classList.add("live-mode");
  elements.fileButton.hidden = true;
  elements.dropOverlay.hidden = true;
  elements.experimentPanel.hidden = true;
  elements.timelineSourceSwitcher.hidden = true;
  for (const name of ["head-value"]) {
    const button = elements.overlaySwitcher.querySelector(
      `[data-overlay="${name}"]`,
    );
    if (button) button.hidden = true;
  }
  document.querySelector(".brand-block p").textContent = "MCTS workspace";
  document.querySelector(".game-identity .eyebrow").lastChild.textContent =
    " Live self-play trajectory";
  elements.summarySchema.previousElementSibling.textContent = "Source";
  elements.timelineEmptyTitle.textContent = "Waiting for the first move";
  elements.timelineEmptyMessage.textContent =
    "The current position is visible while MCTS searches.";
  logDiagnostic("live MCTS workspace ready", {
    level: "success",
    source: "SYSTEM",
  });
  refreshLiveGame();
}


async function initializeApplication() {
  try {
    const response = await fetch("/api/runtime", { cache: "no-store" });
    const payload = await responsePayload(response);
    if (!response.ok || !["analysis", "live"].includes(payload?.mode)) {
      throw new Error("The GUI runtime mode is unavailable.");
    }
    view.mode = payload.mode;
  } catch (error) {
    showStatus(error.message || "Could not initialize the GUI.", "error");
    logDiagnostic(error.message || "Could not initialize the GUI.", {
      level: "error",
      source: "SYSTEM",
    });
    return;
  }

  if (view.mode === "live") {
    configureLiveMode();
  } else {
    logDiagnostic("analysis workspace ready · waiting for a saved game", {
      level: "success",
      source: "SYSTEM",
    });
    updateExperimentControls();
  }
  logDiagnostic("Python hook ready · from analysis import PRINT_T", {
    source: "SYSTEM",
  });
  pollDiagnostics();
}


function switchTimelineSource(source, { selectLatest = false } = {}) {
  if (!view.analysis || !["original", "replay"].includes(source)) return;
  if (source === "replay" && replayFrames().length === 0) return;

  stopPlayback();
  view.timelineSource = source;
  view.displayingExperiment = source === "replay";
  if (source === "replay") {
    view.replayFrameIndex = selectLatest
      ? replayFrames().length - 1
      : Math.min(view.replayFrameIndex, replayFrames().length - 1);
    view.frameIndex = view.replayFrameIndex;
  } else {
    view.frameIndex = view.originalFrameIndex;
  }

  buildTimeline();
  selectFrame(view.frameIndex, { centerTimeline: true, smooth: false });
}


function buildTimeline() {
  elements.timelineList.replaceChildren();
  const isReplay = view.timelineSource === "replay";
  const descriptors = isReplay
    ? replayFrames().map((frame, index) => ({
        index,
        move_number: frame.move_number,
        player_to_move: frame.player_to_move,
        selected_action: frame.selected_action,
        selected_mean_value: frame.selected_action_statistics?.mean_value ?? null,
        selected_prior: frame.selected_action_statistics?.prior ?? null,
        selected_visit_share:
          frame.selected_action_statistics?.visit_share ?? null,
      }))
    : view.analysis?.timeline || [];

  elements.timelineEmpty.hidden = descriptors.length > 0;
  elements.timelineEmptyTitle.textContent = isReplay
    ? "Replay waiting for its first move"
    : "Your decision trail";
  elements.timelineEmptyMessage.textContent = isReplay
    ? "Run one step or continue the game to populate this timeline."
    : "Move by move, the story of the search appears here.";

  for (const descriptor of descriptors) {
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
    const policyFlow = Number.isFinite(descriptor.selected_prior)
      ? ` · P ${formatPercent(descriptor.selected_prior, 1)} → ` +
        `π ${formatPercent(descriptor.selected_visit_share, 1)}`
      : Number.isFinite(descriptor.selected_visit_share)
        ? ` · MCTS π ${formatPercent(descriptor.selected_visit_share, 1)}`
        : "";
    details.textContent =
      `${playerName(descriptor.player_to_move)} · ` +
      `action ${formatCoordinate(descriptor.selected_action)}${policyFlow}`;

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

  elements.playTimeline.disabled = descriptors.length === 0;
  updateTimelinePadding();
  elements.timelineWheel.scrollTop = 0;
  updatePlaybackControls();
  updateTimelineSourceControls();
}


async function fetchFrame(frameIndex) {
  if (view.frameCache.has(frameIndex)) {
    return view.frameCache.get(frameIndex);
  }
  if (view.pendingFrames.has(frameIndex)) {
    return view.pendingFrames.get(frameIndex);
  }

  const analysisId = view.analysis.analysis_id;
  const endpoint = view.mode === "live"
    ? `/api/live/frames/${frameIndex}`
    : `/api/analyses/${analysisId}/frames/${frameIndex}`;
  const pendingRequest = fetch(
    endpoint,
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


async function selectOriginalFrame(
  requestedIndex,
  { centerTimeline = false, smooth = false } = {},
) {
  if (!view.analysis) return;

  const frameIndex = Math.max(
    0,
    Math.min(requestedIndex, view.analysis.frame_count - 1),
  );
  cancelHeadValuePrediction();
  cancelHeadPolicyPrediction();
  view.headPolicyPrediction = null;
  boardRenderer.setHeadPolicy(null);
  if (view.overlay === "head-value") resetHeadValuePanel();
  if (view.overlay === "head-policy") resetHeadPolicyPanel();
  const requestNumber = ++view.requestNumber;
  view.frameIndex = frameIndex;
  view.originalFrameIndex = frameIndex;
  view.displayingExperiment = false;
  updateActiveTimelineItem();
  updatePlaybackControls();
  updateExperimentControls();

  if (centerTimeline) centerTimelineItem(frameIndex, smooth);
  if (!view.frameCache.has(frameIndex)) {
    showStatus(`Loading decision frame ${frameIndex + 1}…`, "busy");
  }

  try {
    const frame = await fetchFrame(frameIndex);
    if (requestNumber !== view.requestNumber) return;

    view.frame = frame;
    view.selectedCell = frame.selected_action ? [...frame.selected_action] : null;
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
    updateExperimentControls();
  } catch (error) {
    if (requestNumber !== view.requestNumber) return;
    showStatus(error.message || "Could not read the selected frame.", "error");
    logDiagnostic(
      error.message || "Could not read the selected frame.",
      { level: "error", source: "FRAME" },
    );
  }
}


function selectReplayFrame(
  requestedIndex,
  { centerTimeline = false, smooth = false } = {},
) {
  const frames = replayFrames();
  if (!frames.length) return;

  const frameIndex = Math.max(0, Math.min(requestedIndex, frames.length - 1));
  const frame = frames[frameIndex];
  view.requestNumber += 1;
  cancelHeadValuePrediction();
  cancelHeadPolicyPrediction();
  view.headPolicyPrediction = null;
  boardRenderer.setHeadPolicy(null);
  if (view.overlay === "head-value") {
    resetHeadValuePanel(
      "Value-head requests are available on saved frames, not forced replay frames.",
    );
  }
  if (view.overlay === "head-policy") {
    showHeadPolicyError(
      "Policy-head requests are available on saved frames, not forced replay frames.",
    );
  }
  view.frameIndex = frameIndex;
  view.replayFrameIndex = frameIndex;
  view.displayingExperiment = true;
  view.frame = frame;
  view.selectedCell = [...frame.selected_action];
  updateActiveTimelineItem();
  updatePlaybackControls();
  if (centerTimeline) centerTimelineItem(frameIndex, smooth);
  updateFrameDisplay();
  showStatus(
    `Forced replay move ${frame.move_number}: ${playerName(frame.player_to_move)} selected ` +
      `${formatCoordinate(frame.selected_action)} after ` +
      `${formatInteger(frame.completed_rollouts)} rollouts.`,
  );

  const diagnosticFrameKey =
    `${view.analysis.analysis_id}:replay:${frame.move_number}`;
  if (view.lastDiagnosticFrameKey !== diagnosticFrameKey) {
    view.lastDiagnosticFrameKey = diagnosticFrameKey;
    logDiagnostic(
      `forced replay move ${frame.move_number} · ` +
        `${playerName(frame.player_to_move)} · action ` +
        `${formatCoordinate(frame.selected_action)} · ` +
        `${formatInteger(frame.completed_rollouts)} rollouts · ` +
        `${formatDuration(frame.elapsed_seconds)}`,
      { source: "FRAME" },
    );
  }
}


function selectFrame(requestedIndex, options = {}) {
  if (view.timelineSource === "replay") {
    selectReplayFrame(requestedIndex, options);
    return;
  }
  selectOriginalFrame(requestedIndex, options);
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
  const isExperiment = Boolean(frame.experiment);
  const isLivePosition = Boolean(frame.live_position);

  elements.boardEmpty.hidden = true;
  elements.boardTitle.textContent = isLivePosition
    ? frame.game_over ? "Final position" : `Move ${frame.move_number} · searching`
    : isExperiment
      ? `Forced replay · before move ${frame.move_number}`
      : `Before move ${frame.move_number}`;
  elements.playerPill.textContent = frame.game_over
    ? resultName(frame.winner)
    : `${playerName(frame.player_to_move)} to move`;
  elements.playerPill.className = frame.player_to_move === PLAYER_1
    ? "player-pill player-one"
    : "player-pill player-two";
  elements.frameCounter.textContent = isLivePosition
    ? `${view.analysis.frame_count} completed`
    : isExperiment
    ? `${frame.move_number} · ${view.replayFrameIndex + 1}/${replayFrames().length}`
    : `${frame.move_number} / ${view.analysis.frame_count}`;
  elements.frameAction.textContent = selected
    ? `${formatCoordinate(frame.selected_action)} · ` +
      `${formatDecimal(selected.mean_value, 2, true)} Q/N · ` +
      `${Number.isFinite(selected.prior) ? `P ${formatPercent(selected.prior, 1)} → ` : ""}` +
      `π ${formatPercent(selected.visit_share ?? selected.policy, 1)}`
    : frame.game_over ? "Game complete" : "Searching…";
  elements.frameScore.textContent =
    `${frame.scores.player_1} — ${frame.scores.player_2}`;
  elements.frameLegal.textContent = formatInteger(frame.legal_move_count);
  elements.frameRollouts.textContent = isLivePosition
    ? "—"
    : formatInteger(frame.completed_rollouts);
  elements.frameElapsed.textContent = isLivePosition
    ? "—"
    : formatDuration(frame.elapsed_seconds);
  elements.frameThroughput.textContent = isLivePosition
    ? "—"
    : `${formatInteger(Math.round(frame.rollouts_per_second))} / s`;
  for (const metric of document.querySelectorAll(".frame-summary strong")) {
    metric.title = metric.textContent;
  }

  elements.board.setAttribute(
    "aria-label",
    `${frame.rows} by ${frame.cols} Dots board ` +
    `${isExperiment ? "in the forced replay " : ""}` +
    `${isLivePosition && frame.game_over ? "at game end" : `before move ${frame.move_number}`}. ` +
    `${frame.game_over ? resultName(frame.winner) : `${playerName(frame.player_to_move)} to move`}.` +
    `${frame.selected_action ? ` Selected action ${formatCoordinate(frame.selected_action)}.` : ""}`,
  );

  boardRenderer.setFrame(frame);
  boardRenderer.setOverlay(view.overlay);
  selectBoardCell(view.selectedCell, false);
  renderMoveComparison();
  if (view.overlay === "head-policy") {
    if (isExperiment) {
      showHeadPolicyError(
        "Policy-head requests are available on saved frames, not forced replay frames.",
      );
    } else if (isLivePosition && !frame.policy_priors) {
      showHeadPolicyError(
        "The model prior appears after the current search completes.",
      );
    } else {
      requestHeadPolicy();
    }
  }
  updateScreenshotAvailability();
  updateExperimentControls();
}


function selectBoardCell(cell, runHeadValuePrediction = true) {
  if (!view.frame) return;
  if (!cell) {
    view.selectedCell = null;
    elements.cellTitle.textContent = "Select a cell";
    for (const value of [
      elements.cellContents,
      elements.cellLegal,
      elements.cellPrior,
      elements.cellRawQ,
      elements.cellValue,
      elements.cellVisits,
      elements.cellVisitShare,
      elements.cellPolicyFlow,
    ]) {
      value.textContent = "—";
    }
    elements.headPolicySelected.textContent = "—";
    elements.headPolicySelectedPrior.textContent = "—";
    boardRenderer.setSelectedCell(null);
    return;
  }

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
  const prior = policyPriorAt(cell);
  const isSelectedAction =
    Boolean(view.frame.selected_action) &&
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
  elements.cellPrior.textContent = isLegal ? formatPercent(prior) : "—";
  elements.cellRawQ.textContent = visits
    ? formatDecimal(rawQ, 0, true)
    : isLegal ? "Unvisited" : "—";
  elements.cellValue.textContent = visits
    ? formatDecimal(meanValue, 3, true)
    : isLegal ? "Unvisited" : "—";
  elements.cellVisits.textContent = isLegal ? formatInteger(visits) : "—";
  elements.cellVisitShare.textContent = isLegal ? formatPercent(policy) : "—";
  elements.cellPolicyFlow.textContent = isLegal && prior !== null
    ? `${formatPercent(prior)} → ${formatPercent(policy)}`
    : "—";
  boardRenderer.setSelectedCell(cell);

  if (view.overlay === "head-policy") {
    displayHeadPolicySelection(cell);
  }

  if (
    view.mode === "analysis" &&
    view.overlay === "head-value" &&
    runHeadValuePrediction
  ) {
    if (view.displayingExperiment) {
      showHeadValueError(
        cell,
        "Value-head requests are available on saved frames, not experiment results.",
      );
    } else if (isLegal) requestHeadValue(cell);
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
  const length = timelineLength();
  elements.previousFrame.disabled = length === 0 || view.frameIndex === 0;
  elements.nextFrame.disabled = length === 0 || view.frameIndex === length - 1;
  elements.playTimeline.disabled = length === 0;
}


function startPlayback() {
  const length = timelineLength();
  if (!view.analysis || view.playTimer || length === 0) return;
  if (view.frameIndex === length - 1) {
    selectFrame(0, { centerTimeline: true, smooth: false });
  }

  elements.playTimeline.textContent = "Pause";
  elements.playTimeline.classList.add("is-playing");
  view.playTimer = window.setInterval(() => {
    if (view.frameIndex >= timelineLength() - 1) {
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
elements.screenshotButton.addEventListener("click", downloadSquareScreenshot);
elements.startExperiment.addEventListener(
  "click",
  startExperimentFromSelectedFrame,
);
elements.stepExperiment.addEventListener("click", () => {
  runExperimentCommand("step");
});
elements.continueExperiment.addEventListener("click", () => {
  runExperimentCommand("continue");
});
elements.showExperiment.addEventListener("click", () => {
  if (replayFrames().length) {
    switchTimelineSource("replay", { selectLatest: true });
  }
});

elements.timelineSourceSwitcher.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-timeline-source]");
  if (!button || button.disabled) return;
  switchTimelineSource(button.dataset.timelineSource);
});


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
    "head-policy": "Model prior P · before MCTS search",
    value: "Mean result · player-to-move perspective",
    "raw-q": "Win/loss balance · player-to-move perspective",
    visits: "Completed visits · brighter means more visits",
    policy: "MCTS visit share π · after search",
    none: "Board and placed dots · search overlays hidden",
  };
  const isHeadValue = view.overlay === "head-value";
  const isHeadPolicy = view.overlay === "head-policy";
  const isSequential = ["visits", "policy", "head-policy"].includes(view.overlay);
  elements.overlayDescription.textContent = descriptions[view.overlay];
  elements.overlayScale.hidden = view.overlay === "none" || isHeadValue;
  elements.overlayScale.classList.toggle("is-sequential", isSequential);
  elements.scaleLow.textContent = isSequential ? "Low" : "Negative";
  elements.scaleHigh.textContent = isSequential ? "High" : "Positive";
  elements.headValuePanel.hidden = !isHeadValue;
  elements.headPolicyPanel.hidden = !isHeadPolicy;
  if (isHeadValue || isHeadPolicy) updateModelPanelWorkspaceHeight();
  cancelHeadValuePrediction();
  cancelHeadPolicyPrediction();
  if (isHeadValue) {
    resetHeadValuePanel(
      view.frame
        ? undefined
        : "Open a saved game, then click a legal position.",
    );
  }
  boardRenderer.setOverlay(view.overlay);
  if (isHeadPolicy) {
    if (!view.frame) {
      resetHeadPolicyPanel("Open a saved game to inspect its model prior.");
    } else if (view.displayingExperiment) {
      showHeadPolicyError(
        "Policy-head requests are available on saved frames, not forced replay frames.",
      );
    } else {
      requestHeadPolicy();
    }
  }
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
    selectFrame(timelineLength() - 1, {
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
    if (view.mode !== "analysis") return;
    event.preventDefault();
    if (eventName === "dragenter") view.dragDepth += 1;
    elements.dropOverlay.hidden = false;
  });
}


elements.dropTarget.addEventListener("dragleave", (event) => {
  if (view.mode !== "analysis") return;
  event.preventDefault();
  view.dragDepth = Math.max(0, view.dragDepth - 1);
  if (view.dragDepth === 0) elements.dropOverlay.hidden = true;
});


elements.dropTarget.addEventListener("drop", (event) => {
  if (view.mode !== "analysis") return;
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


initializeApplication();
