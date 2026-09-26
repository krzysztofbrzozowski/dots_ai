"use strict";

import { DotsBoardRenderer, PLAYER_1 } from "/shared/board_renderer.js?v=20260922-monochrome";

const elements = {
  status: document.querySelector("#status"),
  statusMessage: document.querySelector("#status-message"),
  player1: document.querySelector("#player-1"),
  player2: document.querySelector("#player-2"),
  humanName1: document.querySelector("#human-name-1"),
  humanName2: document.querySelector("#human-name-2"),
  start: document.querySelector("#start-match"),
  end: document.querySelector("#end-match"),
  matchTitle: document.querySelector("#match-title"),
  moveCount: document.querySelector("#move-count"),
  boardMoveCount: document.querySelector("#board-move-count"),
  summaryResult: document.querySelector("#summary-result"),
  boardResult: document.querySelector("#board-result"),
  matchStatus: document.querySelector("#match-status"),
  lastAction: document.querySelector("#last-action"),
  legalMoves: document.querySelector("#legal-moves"),
  scoreSummary: document.querySelector("#score-summary"),
  player1Name: document.querySelector("#player-1-name"),
  player2Name: document.querySelector("#player-2-name"),
  score1: document.querySelector("#score-1"),
  score2: document.querySelector("#score-2"),
  turn: document.querySelector("#turn-indicator"),
  positionCaption: document.querySelector("#position-caption"),
  boardShell: document.querySelector("#board-shell"),
  boardHint: document.querySelector("#board-hint"),
  board: document.querySelector("#arena-board"),
  bestPlayer: document.querySelector("#best-player"),
  bestRating: document.querySelector("#best-rating"),
  worstPlayer: document.querySelector("#worst-player"),
  worstRating: document.querySelector("#worst-rating"),
  matchCount: document.querySelector("#match-count"),
  rankingList: document.querySelector("#ranking-list"),
  movesList: document.querySelector("#moves-list"),
};

let latestState = null;
let latestRevision = null;
let movePending = false;

const renderer = new DotsBoardRenderer(elements.board, submitBoardMove);

async function responsePayload(response) {
  try {
    return await response.json();
  } catch (_error) {
    return null;
  }
}

function playerFromState(number) {
  return latestState?.players?.[String(number)] || null;
}

function updateHumanInput(select, input) {
  input.hidden = select.value !== "human";
  if (!input.hidden) input.focus();
}

function setStatus(message, tone = "idle") {
  elements.statusMessage.textContent = message;
  elements.status.classList.toggle("is-busy", tone === "busy");
  elements.status.classList.toggle("is-error", tone === "error");
}

function activeStatus(status) {
  return ["starting", "searching", "waiting_human"].includes(status);
}

function resultName(result) {
  if (result === PLAYER_1) return "Player 1 won";
  if (result === -PLAYER_1) return "Player 2 won";
  if (result === 0) return "Draw";
  return "—";
}

function statusName(status) {
  return {
    idle: "Idle",
    starting: "Starting",
    searching: "Searching",
    waiting_human: "Human turn",
    complete: "Complete",
    stopped: "Stopped",
    error: "Error",
  }[status] || status;
}

function renderRankings(ratings) {
  const players = ratings?.players || [];
  elements.matchCount.textContent = String(ratings?.match_count || 0);
  elements.rankingList.replaceChildren();

  if (!players.length) {
    const empty = document.createElement("p");
    empty.className = "arena-empty";
    empty.textContent = "The leaderboard will appear after the first rated game.";
    elements.rankingList.append(empty);
  } else {
    players.forEach((player, index) => {
      const row = document.createElement("div");
      row.className = "ranking-row";
      const position = document.createElement("span");
      position.className = "ranking-position";
      position.textContent = String(index + 1);
      const identity = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = String(player.display_name || "");
      const record = document.createElement("small");
      record.textContent = `${player.wins}W · ${player.draws}D · ${player.losses}L`;
      identity.append(name, record);
      const rating = document.createElement("b");
      rating.textContent = Number(player.rating).toFixed(0);
      row.append(position, identity, rating);
      elements.rankingList.append(row);
    });
  }

  const best = ratings?.best;
  const worst = ratings?.worst;
  elements.bestPlayer.textContent = best?.display_name || "—";
  elements.bestRating.textContent = best ? `${Number(best.rating).toFixed(0)} Elo` : "No games";
  elements.worstPlayer.textContent = worst?.display_name || "—";
  elements.worstRating.textContent = worst ? `${Number(worst.rating).toFixed(0)} Elo` : "No games";
}

function renderMoves(moves) {
  elements.movesList.replaceChildren();
  if (!moves?.length) {
    const empty = document.createElement("li");
    empty.className = "arena-empty";
    empty.textContent = "No moves yet.";
    elements.movesList.append(empty);
    return;
  }
  [...moves].reverse().forEach((move) => {
    const row = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${move.number}. ${move.player_name}`;
    const action = document.createElement("strong");
    action.textContent = `(${move.action[0]}, ${move.action[1]})`;
    row.append(label, action);
    elements.movesList.append(row);
  });
}

function renderState(state) {
  latestState = state;
  renderer.setFrame(state.board);
  const first = playerFromState(PLAYER_1);
  const second = playerFromState(-PLAYER_1);
  const firstName = first?.display_name || "Player 1";
  const secondName = second?.display_name || "Player 2";
  const result = resultName(state.game_result);
  const lastAction = state.board.selected_action;
  const legalMoves = state.board.legal_mask
    .flat()
    .reduce((total, value) => total + value, 0);

  elements.player1Name.textContent = firstName;
  elements.player2Name.textContent = secondName;
  elements.score1.textContent = state.scores.player_1;
  elements.score2.textContent = state.scores.player_2;
  elements.scoreSummary.textContent = `${state.scores.player_1} — ${state.scores.player_2}`;
  elements.moveCount.textContent = state.move_count;
  elements.boardMoveCount.textContent = state.move_count;
  elements.legalMoves.textContent = legalMoves;
  elements.lastAction.textContent = lastAction ? `(${lastAction[0]}, ${lastAction[1]})` : "—";
  elements.summaryResult.textContent = result;
  elements.boardResult.textContent = result;
  elements.matchStatus.textContent = statusName(state.status);
  elements.matchTitle.textContent = state.match_id
    ? `${firstName} vs ${secondName}`
    : "Board state";

  const canMove = Boolean(state.human_can_move) && !movePending;
  elements.boardShell.classList.toggle("human-turn", canMove);
  elements.boardHint.textContent = canMove
    ? "Your turn — click a legal board position."
    : movePending
      ? "Submitting the selected move…"
    : state.status === "searching"
      ? "The AI is searching this position."
      : state.status === "complete"
        ? "The match is complete."
        : "Start a match to activate the board.";

  elements.turn.classList.remove("player-one", "player-two");
  if (state.current_player) {
    elements.turn.textContent = `To move · ${state.current_player.display_name}`;
    elements.turn.classList.add(state.next_to_move === PLAYER_1 ? "player-one" : "player-two");
    elements.positionCaption.textContent = `${state.current_player.display_name} to move · live rated position`;
  } else {
    elements.turn.textContent = state.status === "complete" ? "Match complete" : "No active match";
    elements.positionCaption.textContent = "Live rated position · player-to-move perspective";
  }

  elements.start.disabled = activeStatus(state.status);
  elements.end.disabled = !activeStatus(state.status);
  setStatus(
    state.message,
    state.status === "error" ? "error" : activeStatus(state.status) ? "busy" : "idle",
  );
  renderRankings(state.ratings);
  renderMoves(state.recent_moves);
}

async function submitBoardMove([row, col]) {
  if (!latestState?.human_can_move || movePending) return;
  if (latestState.board.legal_mask[row]?.[col] !== 1) {
    setStatus("The selected position is not a legal move.", "error");
    return;
  }
  movePending = true;
  elements.boardShell.classList.remove("human-turn");
  try {
    const response = await fetch("/api/moves", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ row, col }),
    });
    const payload = await responsePayload(response);
    if (!response.ok) throw new Error(payload?.detail || "The move could not be submitted.");
    renderState(payload);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    movePending = false;
  }
}

async function startMatch() {
  elements.start.disabled = true;
  setStatus("Starting the match…", "busy");
  try {
    const response = await fetch("/api/matches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        player_1: elements.player1.value,
        player_2: elements.player2.value,
        human_name_1: elements.humanName1.value,
        human_name_2: elements.humanName2.value,
      }),
    });
    const payload = await responsePayload(response);
    if (!response.ok) throw new Error(payload?.detail || "The match could not be started.");
    renderState(payload);
  } catch (error) {
    elements.start.disabled = false;
    setStatus(error.message, "error");
  }
}

async function endMatch() {
  elements.end.disabled = true;
  setStatus("Ending the match…", "busy");
  try {
    const response = await fetch("/api/matches/end", { method: "POST" });
    const payload = await responsePayload(response);
    if (!response.ok) throw new Error(payload?.detail || "The match could not be ended.");
    renderState(payload);
  } catch (error) {
    setStatus(error.message, "error");
    if (latestState) elements.end.disabled = !activeStatus(latestState.status);
  }
}

async function pollState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    const payload = await responsePayload(response);
    if (!response.ok) throw new Error(payload?.detail || "The arena is unavailable.");
    if (payload.revision !== latestRevision) {
      latestRevision = payload.revision;
      renderState(payload);
    }
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    window.setTimeout(pollState, 500);
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    const config = await responsePayload(response);
    if (!response.ok) throw new Error(config?.detail || "The configuration could not be loaded.");
    for (const player of config.players) {
      for (const select of [elements.player1, elements.player2]) {
        const option = document.createElement("option");
        option.value = player.id;
        option.textContent = player.label;
        select.append(option);
      }
    }
    elements.player1.value = config.players[0].id;
    elements.player2.value = config.players[1]?.id || "human";
    updateHumanInput(elements.player1, elements.humanName1);
    updateHumanInput(elements.player2, elements.humanName2);
    await pollState();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

elements.player1.addEventListener("change", () => updateHumanInput(elements.player1, elements.humanName1));
elements.player2.addEventListener("change", () => updateHumanInput(elements.player2, elements.humanName2));
elements.start.addEventListener("click", startMatch);
elements.end.addEventListener("click", endMatch);

initialize();
