"""Tests for the standalone NPZ timeline analysis application."""

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from analysis import AnalysisFileError, load_analysis_bytes
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
    client = TestClient(create_app(AnalysisStore(maximum_sessions=2)))

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

    missing_frame = client.get(f"/api/analyses/{analysis_id}/frames/10")
    assert missing_frame.status_code == 404

    deleted = client.delete(f"/api/analyses/{analysis_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_analysis_server_serves_the_gui_and_shared_renderer():
    client = TestClient(create_app(AnalysisStore()))

    page = client.get("/")
    script = client.get("/assets/analysis.js")
    renderer = client.get("/shared/board_renderer.js")
    health = client.get("/api/health")

    assert page.status_code == 200
    assert "Decision timeline" in page.text
    assert script.status_code == 200
    assert "importGame" in script.text
    assert renderer.status_code == 200
    assert "DotsBoardRenderer" in renderer.text
    assert health.json() == {"status": "ready"}


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
