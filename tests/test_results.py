import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from main import (
    _axis_aligned_baseline,
    _combine_two_histories,
    _collect_deterministic_results,
    _collect_optimized_results,
    _dwell_markers,
    _evaluation_set_labels,
    _four_case_rates,
    _generate_user_sets,
    _missing_training_history,
    _model_score,
    _plot8_history_available,
    _plot9_title,
    _policy_trajectory_source,
    _power_vs_budget,
    _power_vs_distance,
    _set_model_path,
    _watts_to_dbw,
    main,
)
import main as main_module
from train_model import build_model
from uav_environment import UAVThroughputEnv
from wireless_model import Config, random_users


EXPECTED_FIGURES = [
    "01_received_power_watts_vs_distance.png",
    "02_received_power_dbw_vs_distance.png",
    "03_received_power_watts_vs_transmit_power.png",
    "04_received_power_dbw_vs_transmit_power.png",
    "05_deterministic_sum_throughput.png",
    "06_deterministic_trajectories.png",
    "07_deterministic_individual_throughput.png",
    "08_ppo_convergence.png",
    "09_optimized_trajectories.png",
    "10_four_case_individual_throughput.png",
    "11_four_case_sum_throughput.png",
]

EXPECTED_CASE_LABELS = [
    "Deterministic + random beamforming",
    "Deterministic + optimal beamforming",
    "Optimized + random beamforming",
    "Optimized + optimal beamforming",
]


def test_watts_to_dbw_and_signal_curves_are_finite():
    cfg = Config()
    distances, watts_by_eta = _power_vs_distance(cfg)
    budgets, watts_by_users = _power_vs_budget(cfg)

    assert np.isclose(_watts_to_dbw(np.array([1.0]))[0], 0.0)
    assert np.all(np.isfinite(_watts_to_dbw(np.array([0.0, 1e-9]))))
    assert np.all(np.diff(distances) > 0.0)
    assert set(watts_by_eta) == {2.0, 2.5, 3.0, 3.5, 4.0}
    assert set(watts_by_users) == {1, 2, 3, 4}
    assert all(values.shape == budgets.shape for values in watts_by_users.values())


def test_cli_defaults_use_one_million_training_budget():
    args = main_module._parser().parse_args([])

    assert args.train_steps == 1_000_000
    assert args.model_path == Path("models") / "ppo_balanced"
    assert args.output_dir == Path("results")


def test_formal_training_settings_use_50k_checkpoints_and_16_evaluations():
    block_size, evaluation_episodes = main_module._training_settings(quick=False)

    assert block_size == 50_000
    assert evaluation_episodes == 16


def test_two_user_sets_are_reproducible_uniform_random_samples():
    cfg = Config()

    sets = _generate_user_sets(cfg, seed=9123)

    assert np.array_equal(sets["set_a"], random_users(cfg, seed=9123))
    assert np.array_equal(sets["set_b"], random_users(cfg, seed=9124))
    assert not np.array_equal(sets["set_a"], sets["set_b"])
    assert np.all((sets["set_a"] >= 0.0) & (sets["set_a"] <= cfg.area_size))
    assert np.all((sets["set_b"] >= 0.0) & (sets["set_b"] <= cfg.area_size))


def test_target_users_are_fixed_balanced_and_copied():
    first = main_module._target_users()
    second = main_module._target_users()

    np.testing.assert_array_equal(first, [[30.0, 70.0], [70.0, 30.0]])
    assert first is not second
    first[0, 0] = -1.0
    np.testing.assert_array_equal(second, [[30.0, 70.0], [70.0, 30.0]])


def test_diagnostic_users_are_a_different_reproducible_uniform_set():
    cfg = Config()

    diagnostic = main_module._diagnostic_users(cfg, seed=9123)

    np.testing.assert_array_equal(diagnostic, random_users(cfg, seed=9124))
    assert not np.array_equal(diagnostic, main_module._target_users())


def test_balanced_model_path_is_not_split_by_user_set(tmp_path: Path):
    path = tmp_path / "models" / "ppo_balanced"

    assert main_module._balanced_model_path(path) == path


def test_two_histories_create_two_model_convergence_curves():
    history_a = {"training_steps": [64, 128], "returns_by_set": [[1.0, 2.0]]}
    history_b = {"training_steps": [64, 128], "returns_by_set": [[3.0, 4.0]]}

    combined = _combine_two_histories(history_a, history_b)

    assert combined == {
        "training_steps": [64, 128],
        "returns_by_set": [[1.0, 2.0], [3.0, 4.0]],
        "evaluation_set_roles": ["set_a", "set_b"],
    }
    assert _evaluation_set_labels(combined) == [
        "PPO-A / User set A",
        "PPO-B / User set B",
    ]


def test_moving_return_statistics_use_requested_window():
    mean, std = main_module._moving_return_statistics(
        [1.0, 2.0, 3.0, 4.0],
        window=2,
    )

    np.testing.assert_allclose(mean, [1.0, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(std, [0.0, 0.5, 0.5, 0.5])


def test_plot8_requires_two_aligned_user_set_evaluation_series():
    valid = {
        "training_steps": [50_000, 100_000],
        "returns_by_set": [[1.0, 2.0], [3.0, 4.0]],
        "returns_std_by_set": [[0.1, 0.2], [0.3, 0.4]],
        "evaluation_set_roles": ["target", "diagnostic"],
    }

    assert main_module._plot8_history_available(valid)
    assert not main_module._plot8_history_available({"episode_returns": [1.0]})
    assert not main_module._plot8_history_available(
        {
            **valid,
            "returns_std_by_set": [[0.1], [0.3, 0.4]],
        }
    )


def test_plot8_draws_two_user_set_means_and_variability_bands():
    fig, axis = plt.subplots()
    steps = np.arange(1, 21, dtype=float) * 50_000.0
    history = {
        "training_steps": steps.tolist(),
        "returns_by_set": [
            np.linspace(-100.0, 200.0, 20).tolist(),
            np.linspace(-200.0, 100.0, 20).tolist(),
        ],
        "returns_std_by_set": [
            np.full(20, 20.0).tolist(),
            np.full(20, 30.0).tolist(),
        ],
        "evaluation_set_roles": ["target", "diagnostic"],
    }

    main_module._draw_plot8(axis, Config(), history)

    labels = {line.get_label() for line in axis.lines}
    assert labels == {"Target fixed users", "Secondary diagnostic users"}
    assert len(axis.collections) == 2
    assert axis.get_xlabel() == "PPO training step"
    assert axis.get_ylabel() == "Mean evaluation episode return"
    assert "converged" not in axis.get_title().lower()
    assert len(axis.child_axes) == 0
    plt.close(fig)


def test_model_score_prioritizes_joint_completion(tmp_path: Path):
    joint_results = [
        {
            "users_served": [True, True],
            "reached_end": True,
            "sum_throughput": 10.0,
        }
    ]
    throughput_only_results = [
        {
            "users_served": [True, True],
            "reached_end": False,
            "sum_throughput": 9999.0,
        }
    ]

    assert _model_score(joint_results) > _model_score(throughput_only_results)
    assert _set_model_path(tmp_path / "ppo", "set_a") == tmp_path / "ppo_set_a"
    assert _set_model_path(tmp_path / "ppo", "set_b") == tmp_path / "ppo_set_b"


def test_deterministic_baseline_is_an_alternating_staircase_to_end():
    cfg = Config()
    trajectory = _axis_aligned_baseline(cfg)
    changes = np.diff(trajectory, axis=0)

    assert np.allclose(trajectory[0], cfg.start)
    assert np.allclose(trajectory[-1], cfg.end)
    assert changes.shape == (16, 2)
    assert np.allclose(changes[0::2], [cfg.move_distance, 0.0])
    assert np.allclose(changes[1::2], [0.0, cfg.move_distance])


def test_trajectory_sets_have_sixteen_axis_aligned_runs():
    cfg = Config()
    deterministic = _collect_deterministic_results(cfg, seed=9123, count=16)
    expected_staircase = _axis_aligned_baseline(cfg)

    assert len(deterministic) == 16
    for result in deterministic:
        trajectory = np.asarray(result["trajectory"], dtype=float)
        changes = np.diff(trajectory, axis=0)
        assert np.allclose(trajectory, expected_staircase)
        assert np.all((changes[:, 0] == 0.0) | (changes[:, 1] == 0.0))
        assert np.asarray(result["user_positions"]).shape == (cfg.num_users, 2)


def test_optimized_results_use_one_fixed_user_set_and_reward_ppo():
    cfg = Config()
    users = np.array([[25.0, 75.0], [70.0, 25.0]])
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=users)
    model = build_model(env, seed=9123)

    optimized = _collect_optimized_results(
        cfg,
        model=model,
        user_positions=users,
        seeds=main_module._plot9_seeds(9123, 16),
    )

    assert len(optimized) == 16
    assert all(np.allclose(result["user_positions"], users) for result in optimized)
    assert all(
        result["trajectory_source"] == "ppo_reward_trained"
        for result in optimized
    )
    assert all("teacher_accuracy" not in result for result in optimized)


def test_dwell_markers_scale_only_repeated_coordinates():
    trajectory = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [20.0, 0.0]]
    )

    points, sizes = _dwell_markers(trajectory)

    assert points.tolist() == [[10.0, 0.0]]
    assert sizes.tolist() == [42.0]


def test_plot9_panel_has_two_service_zones_labels_and_direction_arrows():
    fig, axis = plt.subplots()
    result = {
        "trajectory": np.array(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 10.0]]
        ),
        "user_positions": np.array([[30.0, 70.0], [70.0, 30.0]]),
        "sum_throughput": 123.0,
        "users_served": np.array([True, True]),
        "reached_end": True,
    }

    main_module._draw_trajectory_panel(axis, Config(), result, run_index=1)

    labels = {text.get_text() for text in axis.texts}
    assert {"U1", "U2"}.issubset(labels)
    service_zones = [
        patch for patch in axis.patches if patch.get_gid() == "service-zone"
    ]
    assert len(service_zones) == 2
    assert sum(
        collection.get_gid() == "direction-arrows"
        for collection in axis.collections
    ) == 1
    assert "served=yes, end=yes" in axis.get_title()
    plt.close(fig)


def test_joint_success_count_requires_service_and_endpoint():
    results = [
        {"users_served": [True, True], "reached_end": True},
        {"users_served": [True, True], "reached_end": False},
        {"users_served": [True, False], "reached_end": True},
    ]

    assert main_module._joint_success_count(results) == 1


def test_plot9_seed_range_is_disjoint_from_validation_range():
    validation = set(main_module._validation_seeds(9123, 100_000, 16, 0))
    plot9 = set(main_module._plot9_seeds(9123, 16))

    assert len(validation) == 16
    assert len(plot9) == 16
    assert validation.isdisjoint(plot9)


def test_plot9_acceptance_requires_all_three_16_counts():
    assert main_module._plot9_gate(16, 16, 16, total=16)
    assert not main_module._plot9_gate(16, 15, 15, total=16)
    assert not main_module._plot9_gate(15, 16, 15, total=16)


def test_current_training_metadata_has_truthful_plot_labels_and_source():
    history = {
        "training_steps": [64],
        "returns_by_set": [[1.0], [2.0]],
        "returns_std_by_set": [[0.1], [0.2]],
        "episode_returns": [1.0, 2.0],
        "evaluation_set_roles": ["target", "diagnostic"],
        "checkpoint_scope": "target_user_set_only",
        "training_user_positions": [[25.0, 75.0], [70.0, 25.0]],
    }

    assert _policy_trajectory_source(history, training_performed_this_run=False) == (
        "ppo_reward_trained"
    )
    assert _policy_trajectory_source({}, training_performed_this_run=True) == (
        "ppo_reward_trained"
    )
    assert _evaluation_set_labels(history) == [
        "Target fixed users",
        "Secondary diagnostic users",
    ]
    assert _plot8_history_available(history)


def test_legacy_training_metadata_uses_unknown_source_and_generic_labels():
    history = {
        "training_steps": [64],
        "returns_by_set": [[1.0], [2.0]],
    }

    assert _policy_trajectory_source(history, training_performed_this_run=False) == (
        "ppo_training_provenance_unknown"
    )
    assert _evaluation_set_labels(history) == [
        "Evaluation set 1 (legacy provenance unknown)",
        "Evaluation set 2 (legacy provenance unknown)",
    ]
    assert "Reward-Trained" not in _plot9_title(
        "ppo_training_provenance_unknown"
    )


def test_missing_training_history_has_no_invented_convergence_data():
    history = _missing_training_history()

    assert history["training_steps"] == []
    assert history["returns_by_set"] == [[], []]
    assert history["training_user_positions"] is None
    assert not _plot8_history_available(history)


def test_four_case_rates_use_expected_labels_and_user_shape():
    cfg = Config()
    users = random_users(cfg, seed=9123)
    deterministic = _collect_deterministic_results(cfg, 9123, 1)[0]
    cases = _four_case_rates(
        cfg,
        users,
        deterministic,
        deterministic,
        seed=9123,
    )

    assert list(cases) == EXPECTED_CASE_LABELS
    assert all(
        values.shape == (cfg.episode_steps, cfg.num_users)
        for values in cases.values()
    )
    assert all(np.all(np.isfinite(values)) for values in cases.values())
    assert all(np.all(values >= 0.0) for values in cases.values())


def test_quick_flow_creates_all_required_results(tmp_path: Path):
    output_dir = tmp_path / "results"
    model_path = tmp_path / "models" / "ppo_throughput"

    exit_code = main(
        [
            "--quick",
            "--train-steps",
            "64",
            "--output-dir",
            str(output_dir),
            "--model-path",
            str(model_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (output_dir / "results_summary.json").read_text(encoding="utf-8")
    )
    assert list(summary["throughput_cases"]) == EXPECTED_CASE_LABELS
    for values in summary["throughput_cases"].values():
        assert len(values["individual"]) == 2
        assert np.isclose(values["sum"], sum(values["individual"]))

    for filename in [*EXPECTED_FIGURES, "results_summary.json"]:
        path = output_dir / filename
        assert path.exists(), filename
        assert path.stat().st_size > 0

    assert summary["figure_files"] == EXPECTED_FIGURES
    assert summary["trajectory_run_counts"] == {
        "deterministic": 16,
        "optimized": 16,
    }
    assert len(summary["plot_9_runs"]) == 16
    assert summary["formal_objective"] == "maximize_200_step_sum_throughput"
    assert summary["fixed_horizon_steps"] == 200
    assert summary["hard_endpoint"] == [80.0, 80.0]
    assert summary["plot_9_mean_sum_throughput"] == pytest.approx(
        np.mean([run["sum_throughput"] for run in summary["plot_9_runs"]])
    )
    for run in summary["plot_9_runs"]:
        assert run["step_count"] == 200
        assert run["final_position"] == [80.0, 80.0]
        assert run["action_override_count"] >= 0
    assert summary["plot_9_joint_success_count"] == main_module._joint_success_count(
        summary["plot_9_runs"]
    )
    plot9_both = sum(all(run["users_served"]) for run in summary["plot_9_runs"])
    plot9_endpoints = sum(run["reached_end"] for run in summary["plot_9_runs"])
    assert summary["plot_9_both_users_served_count"] == plot9_both
    assert summary["plot_9_endpoint_success_count"] == plot9_endpoints
    assert summary["plot_9_acceptance_passed"] is (
        plot9_both == 16
        and plot9_endpoints == 16
        and summary["plot_9_joint_success_count"] == 16
    )
    assert summary["plot_9_evaluation_seeds"] == main_module._plot9_seeds(9123, 16)
    validation_seeds = {
        seed
        for user_set in summary["training_history"]["evaluation_seeds_by_set"]
        for checkpoint in user_set
        for seed in checkpoint
    }
    assert validation_seeds.isdisjoint(summary["plot_9_evaluation_seeds"])
    assert summary["plot_8_episode_return_samples"] == len(
        summary["training_history"]["episode_returns"]
    )
    assert summary["plot_9_trajectory_source"] == "ppo_reward_trained"
    assert summary["action_space_size"] == 5
    assert summary["movement_step_metres"] == 10.0
    expected_users = [[30.0, 70.0], [70.0, 30.0]]
    assert summary["user_position_source"] == "fixed_balanced_hidden_users"
    assert summary["plot_9_user_positions"] == expected_users
    assert summary["training_user_positions"] == summary["plot_9_user_positions"]
    assert summary["training_user_mode"] == "one_fixed_target_set"
    assert summary["checkpoint_scope"] == "formal_target_user_set_only"
    assert summary["training_history"]["training_user_positions"] == expected_users
    assert summary["training_history"]["evaluation_set_roles"] == [
        "target",
        "diagnostic",
    ]
    assert [
        stage["name"] for stage in summary["training_history"]["curriculum_stages"]
    ] == ["service_discovery", "service_and_return", "formal_task"]
    assert len(summary["training_history"]["returns_by_set"]) == 2
    assert len(summary["training_history"]["returns_std_by_set"]) == 2
    assert summary["plot_8_evaluation_user_sets"]["target"] == expected_users
    assert summary["plot_8_evaluation_user_sets"]["diagnostic"] != expected_users
    assert summary["selected_model_path"].endswith("ppo_throughput.zip")
    assert (tmp_path / "models" / "ppo_throughput.zip").exists()
    assert not (tmp_path / "models" / "ppo_throughput_set_a.zip").exists()
    assert not (tmp_path / "models" / "ppo_throughput_set_b.zip").exists()
    assert summary["requested_training_steps"] == 64
    assert summary["recorded_training_steps"] == 64
    assert summary["training_performed_this_run"] is True
    assert summary["training_steps"] == 64
    assert "teacher_accuracy" not in json.dumps(summary)
    assert "channel_search_service_seconds" not in summary
