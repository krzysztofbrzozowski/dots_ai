"""Tests for the standalone NPZ timeline analysis application."""

from io import BytesIO
from threading import Event
from time import monotonic, sleep

import numpy as np
from fastapi.testclient import TestClient

from analysis import AnalysisFileError, PRINT_T, load_analysis_bytes
from analysis.mcts_forced_replay import ExperimentManager
from analysis.server import create_app
from analysis.service import AnalysisStore, analysis_frame, analysis_summary


def make_schema_v1_npz(**overrides):
    """Create a small but internally consistent two-frame game in memory."""
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "game_id": np.asarray("test-game"),
        "created_at_utc": np.asarray("2026-08-29T10:00:00+00:00"),
        "board_shape": np.asarray((2, 2), dtype=np.int16),
        "boards": np.asarray(
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
            ],
            dtype=np.int8,
        ),
        "territories": np.zeros((2, 2, 2), dtype=np.int8),
        "next_players": np.asarray((1, -1), dtype=np.int8),
        "scores": np.asarray(((0, 0), (0, 0)), dtype=np.int32),
        "q_values": np.asarray(
            [
                [[2, -1], [0, 0]],
                [[0, 1], [-1, 0]],
            ],
            dtype=np.float32,
        ),
        "q_perspective": np.asarray("player_to_move"),
        "visit_counts": np.asarray(
            [
                [[2, 1], [1, 0]],
                [[0, 2], [1, 1]],
            ],
            dtype=np.int64,
        ),
        "legal_masks": np.asarray(
            [
                [[1, 1], [1, 1]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.uint8,
        ),
        "selected_actions": np.asarray(((0, 0), (0, 1)), dtype=np.int16),
        "completed_rollouts": np.asarray((4, 4), dtype=np.int64),
        "search_elapsed_seconds": np.asarray((0.5, 0.25), dtype=np.float64),
        "final_result": np.asarray(0, dtype=np.int8),
        "final_result_perspective": np.asarray("player_1"),
        "search_budget_type": np.asarray("seconds"),
        "requested_simulation_seconds": np.asarray(1.0, dtype=np.float64),
        "requested_simulations": np.asarray(-1, dtype=np.int64),
        "rollout_batch_size": np.asarray(2, dtype=np.int32),
    }
    arrays.update(overrides)

    output = BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def test_schema_v1_loader_returns_read_only_canonical_game():
    game = load_analysis_bytes(make_schema_v1_npz(), "example.npz")

    assert game.file_name == "example.npz"
    assert game.schema_version == 1
    assert game.board_shape == (2, 2)
    assert game.frame_count == 2
    assert game.requested_simulation_seconds == 1.0
    assert game.requested_simulations is None
    assert game.boards.flags.writeable is False

    try:
        game.boards[0, 0, 0] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("loaded analysis arrays remained mutable")


def test_summary_and_frame_derive_search_statistics():
    game = load_analysis_bytes(make_schema_v1_npz(), "example.npz")

    summary = analysis_summary("session-1", game)
    frame = analysis_frame(game, 0)

    assert summary["analysis_id"] == "session-1"
    assert summary["frame_count"] == 2
    assert summary["search"]["total_rollouts"] == 8
    assert abs(summary["search"]["average_rollouts_per_second"] - (8 / 0.75)) < 1e-9
    assert summary["timeline"][0]["selected_mean_value"] == 1.0
    assert frame["selected_action"] == [0, 0]
    assert frame["selected_action_statistics"]["raw_q"] == 2.0
    assert frame["selected_action_statistics"]["visits"] == 2
    assert frame["selected_action_statistics"]["mean_value"] == 1.0
    assert frame["selected_action_statistics"]["policy"] == 0.5
    assert frame["selected_action_statistics"]["value_rank"] == 1
    assert frame["rollouts_per_second"] == 8.0


def test_loader_rejects_a_missing_required_field():
    file_bytes = make_schema_v1_npz()
    with np.load(BytesIO(file_bytes), allow_pickle=False) as stored:
        arrays = {
            name: stored[name].copy()
            for name in stored.files
            if name != "q_values"
        }

    output = BytesIO()
    np.savez_compressed(output, **arrays)

    try:
        load_analysis_bytes(output.getvalue(), "missing-q.npz")
    except AnalysisFileError as error:
        assert "q_values" in str(error)
    else:
        raise AssertionError("a file missing q_values was accepted")


def test_loader_rejects_an_illegal_selected_action():
    legal_masks = np.asarray(
        [
            [[0, 1], [1, 1]],
            [[0, 1], [1, 1]],
        ],
        dtype=np.uint8,
    )

    try:
        load_analysis_bytes(
            make_schema_v1_npz(legal_masks=legal_masks),
            "illegal-action.npz",
        )
    except AnalysisFileError as error:
        assert "selected action" in str(error).lower()
        assert "legal" in str(error).lower()
    else:
        raise AssertionError("an illegal selected action was accepted")


def test_loader_rejects_an_unknown_schema_version():
    try:
        load_analysis_bytes(
            make_schema_v1_npz(schema_version=np.asarray(9, dtype=np.int16)),
            "future.npz",
        )
    except AnalysisFileError as error:
        assert "Unsupported NPZ schema version 9" in str(error)
    else:
        raise AssertionError("an unknown schema version was accepted")


def test_loader_rejects_a_non_integer_schema_version_cleanly():
    try:
        load_analysis_bytes(
            make_schema_v1_npz(schema_version=np.asarray("one")),
            "bad-version.npz",
        )
    except AnalysisFileError as error:
        assert "schema_version" in str(error)
        assert "integer" in str(error)
    else:
        raise AssertionError("a text schema version was accepted")


def test_loader_rejects_non_npz_bytes_and_file_extensions():
    for file_bytes, file_name, expected_message in (
        (b"not a zip archive", "broken.npz", "valid NPZ archive"),
        (make_schema_v1_npz(), "game.zip", ".npz extension"),
    ):
        try:
            load_analysis_bytes(file_bytes, file_name)
        except AnalysisFileError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(f"invalid file {file_name!r} was accepted")


def test_analysis_api_imports_reads_and_releases_a_game():
    class StubValueHeadPredictor:
        def __init__(self):
            self.calls = []

        def predict_analysis_move(self, game, frame_index, move):
            self.calls.append((game.game_id, frame_index, move))
            return {
                "coordinate": list(move),
                "player": int(game.next_players[frame_index]),
                "model": "test-value-head.keras",
                "loss": 0.2,
                "draw": 0.3,
                "win": 0.5,
                "value": 0.3,
                "source": "model",
            }

    predictor = StubValueHeadPredictor()
    client = TestClient(
        create_app(
            AnalysisStore(maximum_sessions=2),
            predictor=predictor,
        )
    )

    imported = client.post(
        "/api/analyses",
        content=make_schema_v1_npz(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-File-Name": "browser-game.npz",
        },
    )

    assert imported.status_code == 201
    summary = imported.json()
    analysis_id = summary["analysis_id"]
    assert summary["file_name"] == "browser-game.npz"
    assert summary["timeline"][1]["selected_action"] == [0, 1]

    frame_response = client.get(f"/api/analyses/{analysis_id}/frames/1")
    assert frame_response.status_code == 200
    assert frame_response.json()["board"] == [[1, 0], [0, 0]]

    head_value_response = client.get(
        f"/api/analyses/{analysis_id}/frames/0/head-value?row=0&col=1"
    )
    assert head_value_response.status_code == 200
    head_value = head_value_response.json()
    assert head_value["coordinate"] == [0, 1]
    assert head_value["player"] == 1
    assert head_value["model"] == "test-value-head.keras"
    assert head_value["value"] == 0.3
    assert head_value["elapsed_ms"] >= 0
    assert predictor.calls == [("test-game", 0, (0, 1))]

    illegal_head_value = client.get(
        f"/api/analyses/{analysis_id}/frames/1/head-value?row=0&col=0"
    )
    assert illegal_head_value.status_code == 422
    assert "not a legal move" in illegal_head_value.json()["detail"]

    missing_frame = client.get(f"/api/analyses/{analysis_id}/frames/10")
    assert missing_frame.status_code == 404

    deleted = client.delete(f"/api/analyses/{analysis_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_analysis_api_runs_a_disposable_experiment_step():
    search_finished = Event()

    def fake_search(state, _config, _executor, move_number):
        action = state.get_legal_actions()[0]
        next_state = state.move(action)
        search_finished.set()
        return next_state, {
            "experiment": True,
            "move_number": move_number,
            "selected_action": list(action),
            "completed_rollouts": 4,
        }

    manager = ExperimentManager(search_runner=fake_search)
    client = TestClient(
        create_app(
            AnalysisStore(),
            experiment_manager=manager,
        )
    )
    game_bytes = make_schema_v1_npz(
        search_budget_type=np.asarray("simulations"),
        requested_simulation_seconds=np.asarray(np.nan, dtype=np.float64),
        requested_simulations=np.asarray(4, dtype=np.int64),
        rollout_batch_size=np.asarray(1, dtype=np.int32),
    )
    imported = client.post(
        "/api/analyses",
        content=game_bytes,
        headers={"X-File-Name": "experiment.npz"},
    ).json()
    analysis_id = imported["analysis_id"]

    missing = client.post(f"/api/analyses/{analysis_id}/experiment/step")
    assert missing.status_code == 404

    started = client.post(
        f"/api/analyses/{analysis_id}/experiment?frame_index=0"
    )
    assert started.status_code == 201
    assert started.json()["source_move_number"] == 1
    assert started.json()["status"] == "ready"

    step = client.post(f"/api/analyses/{analysis_id}/experiment/step")
    assert step.status_code == 202
    assert step.json()["status"] == "running"
    assert search_finished.wait(1)

    deadline = monotonic() + 1
    while manager.snapshot(analysis_id)["status"] == "running":
        assert monotonic() < deadline
        sleep(0.001)

    result = client.get(f"/api/analyses/{analysis_id}/experiment")
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "ready"
    assert payload["moves_completed"] == 1
    assert payload["next_move_number"] == 2
    assert payload["latest_frame"]["selected_action"] == [0, 0]
    assert payload["current_state"]["player_to_move"] == -1

    continued = client.post(
        f"/api/analyses/{analysis_id}/experiment/continue"
    )
    assert continued.status_code == 202
    deadline = monotonic() + 1
    while manager.snapshot(analysis_id)["status"] == "running":
        assert monotonic() < deadline
        sleep(0.001)

    completed = client.get(
        f"/api/analyses/{analysis_id}/experiment"
    ).json()
    assert completed["status"] == "complete"
    assert completed["moves_completed"] == 4
    assert completed["current_state"]["game_over"] is True
    assert completed["latest_frame"]["move_number"] == 4

    saved_frame = client.get(
        f"/api/analyses/{analysis_id}/frames/0"
    ).json()
    assert saved_frame["board"] == [[0, 0], [0, 0]]


def test_analysis_server_serves_the_gui_and_shared_renderer():
    client = TestClient(create_app(AnalysisStore()))

    page = client.get("/")
    script = client.get("/assets/analysis.js")
    renderer = client.get("/shared/board_renderer.js")
    health = client.get("/api/health")

    assert page.status_code == 200
    assert "Decision timeline" in page.text
    assert 'id="diagnostics-output"' in page.text
    assert 'id="screenshot-button"' in page.text
    assert 'data-overlay="head-value"' in page.text
    assert 'data-overlay="none"' in page.text
    assert 'id="start-experiment"' in page.text
    assert 'id="step-experiment"' in page.text
    assert 'id="continue-experiment"' in page.text
    assert page.headers["cache-control"] == "no-store"
    assert script.status_code == 200
    assert "importGame" in script.text
    assert "requestHeadValue" in script.text
    assert "logDiagnostic" in script.text
    assert "downloadSquareScreenshot" in script.text
    assert "startExperimentFromSelectedFrame" in script.text
    assert "runExperimentCommand" in script.text
    assert "pollExperiment" in script.text
    assert renderer.status_code == 200
    assert "DotsBoardRenderer" in renderer.text
    assert "renderSquareCanvas" in renderer.text
    assert 'this.overlay !== "none"' in renderer.text
    assert renderer.headers["cache-control"] == "no-store"
    assert health.json() == {"status": "ready"}


def test_print_t_publishes_messages_to_the_gui_diagnostic_stream():
    event = PRINT_T("candidate", (4, 7), level="warning", source="custom")
    client = TestClient(create_app(AnalysisStore()))

    response = client.get(f"/api/diagnostics?after={event['id'] - 1}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cursor"] >= event["id"]
    assert payload["events"][0] == event
    assert event["message"] == "candidate (4, 7)"
    assert event["level"] == "warning"
    assert event["source"] == "CUSTOM"


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} analysis tests passed")
