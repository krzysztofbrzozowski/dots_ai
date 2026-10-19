"""Thread-safe orchestration for one live 25x25 Elo match at a time."""

import hashlib
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from threading import Condition, RLock, Thread
from uuid import uuid4

from game.enclosure import DotsGame, PLAYER_1, PLAYER_2
from mcts.enclosure import SearchStats, TwoPlayerMCTSNode, rollout_state
from training import SelfPlayTrajectory

from .config import (
    COLS,
    HUMAN_PRESET_ID,
    PRESETS_BY_ID,
    ROWS,
    TRAINING_DATA_DIRECTORY,
)
from .players import MoveDecision, build_runtime_player


POOL_ID = f"{ROWS}x{COLS}"
ACTIVE_STATUSES = {"starting", "searching", "waiting_human"}


@dataclass(frozen=True, slots=True)
class Competitor:
    rating_id: str
    display_name: str
    kind: str
    preset_id: str


def _human_competitor(name):
    display_name = " ".join(str(name or "").split())
    if not display_name:
        raise ValueError("enter a name for the human player")
    if len(display_name) > 40:
        raise ValueError("a human player name may contain at most 40 characters")
    normalized = unicodedata.normalize("NFKC", display_name).casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:24] or "player"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return Competitor(
        rating_id=f"human:{slug}:{digest}",
        display_name=display_name,
        kind="human",
        preset_id=HUMAN_PRESET_ID,
    )


def resolve_competitor(selection, human_name=None):
    if selection == HUMAN_PRESET_ID:
        return _human_competitor(human_name)
    try:
        preset = PRESETS_BY_ID[selection]
    except KeyError as error:
        raise ValueError(f"unknown player: {selection}") from error
    return Competitor(
        rating_id=preset.id,
        display_name=preset.label,
        kind=preset.kind,
        preset_id=preset.id,
    )


class EloArenaSession:
    """Own the live board, human input hand-off, and rating update."""

    def __init__(self, rating_store, training_data_directory=TRAINING_DATA_DIRECTORY):
        self.rating_store = rating_store
        self.training_data_directory = training_data_directory
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._revision = 0
        self._status = "idle"
        self._message = "Choose two players and start a match."
        self._state = DotsGame(ROWS, COLS)
        self._players = {}
        self._moves = []
        self._pending_human_move = None
        self._stop_requested = False
        self._match_id = None
        self._rating_change = None
        self._adjudicated_result = None
        self._trajectory = None
        self._training_data_path = None

    def start_match(
        self,
        player_1_selection,
        player_2_selection,
        human_name_1=None,
        human_name_2=None,
    ):
        player_1 = resolve_competitor(player_1_selection, human_name_1)
        player_2 = resolve_competitor(player_2_selection, human_name_2)
        if player_1.rating_id == player_2.rating_id:
            raise ValueError("choose two distinct rated player identities")

        with self._condition:
            if self._status in ACTIVE_STATUSES:
                raise RuntimeError("the previous match is still running")
            self._match_id = uuid4().hex
            self._state = DotsGame(ROWS, COLS)
            self._players = {PLAYER_1: player_1, PLAYER_2: player_2}
            self._moves = []
            self._pending_human_move = None
            self._stop_requested = False
            self._rating_change = None
            self._adjudicated_result = None
            self._trajectory = self._new_trajectory(player_1, player_2)
            self._training_data_path = None
            self._status = "starting"
            self._message = "Preparing the players…"
            self._revision += 1
            match_id = self._match_id

        Thread(
            target=self._run_match,
            args=(match_id,),
            name=f"elo-arena-{match_id[:8]}",
            daemon=True,
        ).start()
        return self.snapshot()

    def end_match(self):
        """Stop and rate the active match using the current board score."""
        with self._condition:
            if self._status not in ACTIVE_STATUSES:
                raise RuntimeError("there is no active match to end")
            self._stop_requested = True
            self._pending_human_move = None
            first = self._players[PLAYER_1]
            second = self._players[PLAYER_2]
            if self._state.score[PLAYER_1] > self._state.score[PLAYER_2]:
                result = PLAYER_1
                result_message = f"{first.display_name} won on the current score."
            elif self._state.score[PLAYER_2] > self._state.score[PLAYER_1]:
                result = PLAYER_2
                result_message = f"{second.display_name} won on the current score."
            else:
                result = 0
                result_message = "The current score was tied."
            self._adjudicated_result = result
            saved_path = self._save_trajectory(result, "stopped")
            self._rating_change = self.rating_store.record_match(
                POOL_ID,
                first,
                second,
                result,
                match_id=self._match_id,
                termination="stopped",
            )
            self._status = "stopped"
            saved_message = (
                f" Saved game: {saved_path.name}."
                if saved_path is not None
                else " No NPZ was created because no move was played."
            )
            self._message = (
                f"Match ended by the user. {result_message} Ratings were updated."
                f"{saved_message}"
            )
            self._revision += 1
            self._condition.notify_all()
        return self.snapshot()

    def submit_human_move(self, row, col):
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError("row must be an integer")
        if isinstance(col, bool) or not isinstance(col, int):
            raise ValueError("column must be an integer")
        with self._condition:
            if self._status != "waiting_human":
                raise RuntimeError("it is not a human turn")
            competitor = self._players[self._state.next_to_move]
            if competitor.kind != "human":
                raise RuntimeError("the current player is not human")
            if not self._state.is_legal_move(row, col):
                raise ValueError("the selected position is not a legal move")
            if self._pending_human_move is not None:
                raise RuntimeError("a move has already been submitted")
            self._pending_human_move = (row, col)
            self._message = f"Submitting {competitor.display_name}'s move ({row}, {col})…"
            self._revision += 1
            self._condition.notify_all()
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            state = self._state
            rows, cols = state.board.shape
            legal_mask = [
                [int(state.is_legal_move(row, col)) for col in range(cols)]
                for row in range(rows)
            ]
            players = {
                str(number): {
                    "rating_id": competitor.rating_id,
                    "display_name": competitor.display_name,
                    "kind": competitor.kind,
                    "preset_id": competitor.preset_id,
                }
                for number, competitor in self._players.items()
            }
            current = self._players.get(state.next_to_move)
            match_is_active = self._status in ACTIVE_STATUSES
            last_action = list(state.last_move) if state.last_move is not None else None
            payload = {
                "revision": self._revision,
                "status": self._status,
                "message": self._message,
                "match_id": self._match_id,
                "board": {
                    "rows": rows,
                    "cols": cols,
                    "board": state.board.tolist(),
                    "territory": state.territory.tolist(),
                    "legal_mask": legal_mask,
                    "q_values": [[0.0 for _ in range(cols)] for _ in range(rows)],
                    "visit_counts": [[0 for _ in range(cols)] for _ in range(rows)],
                    "selected_action": last_action,
                },
                "scores": {
                    "player_1": int(state.score[PLAYER_1]),
                    "player_2": int(state.score[PLAYER_2]),
                },
                "players": players,
                "next_to_move": int(state.next_to_move),
                "current_player": (
                    {
                        "display_name": current.display_name,
                        "kind": current.kind,
                    }
                    if (
                        current is not None
                        and state.game_result is None
                        and match_is_active
                    )
                    else None
                ),
                "human_can_move": (
                    self._status == "waiting_human"
                    and current is not None
                    and current.kind == "human"
                ),
                "game_result": (
                    state.game_result
                    if state.game_result is not None
                    else self._adjudicated_result
                ),
                "move_count": len(self._moves),
                "recent_moves": list(self._moves[-12:]),
                "rating_change": self._rating_change,
                "training_data_path": (
                    str(self._training_data_path)
                    if self._training_data_path is not None
                    else None
                ),
            }
        payload["ratings"] = self.rating_store.snapshot(POOL_ID)
        return payload

    def _new_trajectory(self, player_1, player_2):
        if self.training_data_directory is None:
            return None
        presets = [
            PRESETS_BY_ID[player.preset_id]
            for player in (player_1, player_2)
            if player.kind != "human"
        ]
        simulation_seconds = max(
            (preset.simulation_seconds for preset in presets),
            default=1.0,
        )
        rollout_batch_size = max(
            (
                preset.workers
                for preset in presets
                if preset.kind == "classic_mcts"
            ),
            default=1,
        )
        return SelfPlayTrajectory(
            simulation_seconds=simulation_seconds,
            rollout_batch_size=rollout_batch_size,
        )

    @staticmethod
    def _human_root(state, action):
        root = TwoPlayerMCTSNode(state=state)
        root.children.append(
            TwoPlayerMCTSNode(
                state=state.move(action),
                parent=root,
                action=action,
            )
        )
        return root

    @staticmethod
    def _filename_slug(value):
        return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-") or "player"

    def _save_trajectory(self, result, termination):
        if self._trajectory is None or len(self._trajectory) == 0:
            return None
        first = self._players[PLAYER_1]
        second = self._players[PLAYER_2]
        suffix = "__".join(
            (
                f"elo-{termination}",
                f"match-{self._match_id[:8]}",
                f"p1-{self._filename_slug(first.rating_id)}",
                f"p2-{self._filename_slug(second.rating_id)}",
            )
        )
        saved_path = self._trajectory.save(
            self.training_data_directory,
            final_result=result,
            filename_suffix=suffix,
        )
        self._trajectory = None
        self._training_data_path = saved_path
        return saved_path

    def _finish_match(self, match_id):
        result = self._state.game_result
        first = self._players[PLAYER_1]
        second = self._players[PLAYER_2]
        saved_path = self._save_trajectory(result, "completed")
        self._rating_change = self.rating_store.record_match(
            POOL_ID,
            first,
            second,
            result,
            match_id=match_id,
        )
        self._status = "complete"
        if result == PLAYER_1:
            result_message = f"{first.display_name} won."
        elif result == PLAYER_2:
            result_message = f"{second.display_name} won."
        else:
            result_message = "Draw."
        saved_message = (
            f" Saved game: {saved_path.name}."
            if saved_path is not None
            else ""
        )
        self._message = (
            f"Game over. {result_message} Ratings were updated."
            f"{saved_message}"
        )
        self._revision += 1
        self._condition.notify_all()

    def _run_match(self, match_id):
        try:
            with self._lock:
                competitors = dict(self._players)
            presets = [
                PRESETS_BY_ID[competitor.preset_id]
                for competitor in competitors.values()
                if competitor.kind != "human"
            ]
            runtime_by_preset = {
                preset.id: build_runtime_player(preset)
                for preset in presets
            }
            worker_limit = max(
                (
                    preset.workers
                    for preset in presets
                    if preset.kind == "classic_mcts" and preset.workers > 1
                ),
                default=0,
            )
            if worker_limit:
                with ProcessPoolExecutor(
                    max_workers=worker_limit,
                    mp_context=get_context("spawn"),
                ) as executor:
                    self._warm_workers(executor, worker_limit)
                    self._play_loop(match_id, runtime_by_preset, executor)
            else:
                self._play_loop(match_id, runtime_by_preset, None)
        except Exception as error:
            with self._condition:
                if self._match_id == match_id and not self._stop_requested:
                    self._status = "error"
                    self._message = f"Match stopped: {error}"
                    self._revision += 1
                    self._condition.notify_all()

    @staticmethod
    def _warm_workers(executor, workers):
        terminal_state = DotsGame(1, 1).move((0, 0))
        futures = [
            executor.submit(rollout_state, terminal_state, seed)
            for seed in range(workers)
        ]
        for future in futures:
            future.result()

    def _play_loop(self, match_id, runtime_by_preset, executor):
        while True:
            with self._condition:
                if self._match_id != match_id:
                    return
                if self._stop_requested:
                    return
                if self._state.game_result is not None:
                    self._finish_match(match_id)
                    return
                moving_player = self._state.next_to_move
                competitor = self._players[moving_player]
                state_before = self._state

                if competitor.kind == "human":
                    self._status = "waiting_human"
                    self._message = f"Waiting for {competitor.display_name} to move."
                    self._pending_human_move = None
                    self._revision += 1
                    while (
                        self._pending_human_move is None
                        and self._match_id == match_id
                        and not self._stop_requested
                    ):
                        self._condition.wait()
                    if self._match_id != match_id or self._stop_requested:
                        return
                    action = self._pending_human_move
                    decision = MoveDecision(
                        action=action,
                        completed_rollouts=0,
                        elapsed_seconds=0.0,
                        root=self._human_root(state_before, action),
                    )
                else:
                    self._status = "searching"
                    self._message = f"{competitor.display_name} is searching…"
                    self._revision += 1
                    decision = None

            if decision is None:
                runtime_player = runtime_by_preset[competitor.preset_id]
                decision = runtime_player.choose_move(state_before, executor)

            next_state = state_before.move(decision.action)
            with self._condition:
                if self._match_id != match_id or self._stop_requested:
                    return
                if self._trajectory is not None:
                    self._trajectory.record_search(
                        state=state_before,
                        root=decision.root,
                        selected_action=decision.action,
                        search_stats=decision,
                    )
                self._state = next_state
                self._pending_human_move = None
                self._moves.append(
                    {
                        "number": len(self._moves) + 1,
                        "player": int(moving_player),
                        "player_name": competitor.display_name,
                        "kind": competitor.kind,
                        "action": list(decision.action),
                        "completed_rollouts": decision.completed_rollouts,
                        "elapsed_seconds": round(decision.elapsed_seconds, 4),
                    }
                )
                self._message = (
                    f"{competitor.display_name} played "
                    f"({decision.action[0]}, {decision.action[1]})."
                )
                self._revision += 1
                if self._state.game_result is not None:
                    self._finish_match(match_id)
                    return
