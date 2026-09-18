"""Tests for the resumable paired-seed c_param experiment runner."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from main_c_param_testing import (
    CParamExperimentConfig,
    _game_plan,
    run_experiment,
)
from training import load_self_play_game


def small_config():
    return CParamExperimentConfig(
        rows=2,
        cols=2,
        simulation_seconds=0.001,
        games_per_c_param=1,
        c_param_start=0.0,
        c_param_end=0.2,
        c_param_step=0.2,
        base_seed=17,
        workers=1,
    )


def test_plan_reuses_each_seed_across_the_complete_c_param_grid():
    config = CParamExperimentConfig(
        rows=2,
        cols=2,
        simulation_seconds=0.001,
        games_per_c_param=2,
        c_param_start=0.0,
        c_param_end=0.4,
        c_param_step=0.2,
        base_seed=17,
        workers=1,
    )

    assert list(_game_plan(config)) == [
        (17, 0.0),
        (17, 0.2),
        (17, 0.4),
        (18, 0.0),
        (18, 0.2),
        (18, 0.4),
    ]


def test_runner_saves_named_games_directly_and_resumes_without_duplicates():
    config = small_config()
    with TemporaryDirectory() as directory:
        output = Path(directory) / "10_10_c_param"
        first = run_experiment(config, output_directory=output, progress=None)

        paths = sorted(output.glob("*.npz"))
        assert first["games_added"] == 2
        assert len(paths) == 2
        assert not (output / "2x2").exists()
        assert any("_c_param-0p0_seed-0000000017.npz" in path.name for path in paths)
        assert any("_c_param-0p2_seed-0000000017.npz" in path.name for path in paths)
        for path in paths:
            game = load_self_play_game(path)
            assert game["board_shape"].tolist() == [2, 2]
            assert float(game["requested_simulation_seconds"]) == 0.001
            assert int(game["rollout_batch_size"]) == 1

        manifest = json.loads((output / "manifest.json").read_text())
        assert [(item["seed"], item["c_param"]) for item in manifest] == [
            (17, 0.0),
            (17, 0.2),
        ]

        resumed = run_experiment(config, output_directory=output, progress=None)
        assert resumed["games_added"] == 0
        assert len(list(output.glob("*.npz"))) == 2


def test_max_games_stops_cleanly_and_next_run_fills_the_remaining_plan():
    config = small_config()
    with TemporaryDirectory() as directory:
        first = run_experiment(
            config,
            output_directory=directory,
            max_games=1,
            progress=None,
        )
        second = run_experiment(
            config,
            output_directory=directory,
            progress=None,
        )

        assert first["games_added"] == 1
        assert second["games_before"] == 1
        assert second["games_added"] == 1
        assert second["total_games"] == 2


def test_dry_run_does_not_create_the_output_directory():
    with TemporaryDirectory() as directory:
        output = Path(directory) / "not-created"
        result = run_experiment(
            small_config(),
            output_directory=output,
            dry_run=True,
            progress=None,
        )

        assert result["planned_games"] == 2
        assert not output.exists()


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} c_param experiment tests passed")
