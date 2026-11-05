"""Run UAV PPO training, evaluation, and result plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from train_model import evaluate_policy, load_model, train_curriculum
from uav_environment import UAVThroughputEnv
from wireless_model import (
    Config,
    channel_vector,
    optimise_power_split,
    random_users,
    sinr_and_throughput,
)


FIGURE_NAMES = [
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

TARGET_USER_POSITIONS = np.array(
    [[30.0, 70.0], [70.0, 30.0]],
    dtype=float,
)
VALIDATION_SEED_OFFSET = 100_000
DIAGNOSTIC_SEED_OFFSET = 10_000
PLOT9_SEED_OFFSET = 500_000


def _validation_seeds(
    seed: int,
    completed: int,
    count: int,
    set_index: int,
) -> list[int]:
    """Return reproducible checkpoint seeds outside the Plot 9 range."""

    start = seed + VALIDATION_SEED_OFFSET + completed
    if set_index:
        start += DIAGNOSTIC_SEED_OFFSET * set_index
    return list(range(start, start + count))


def _plot9_seeds(seed: int, count: int) -> list[int]:
    """Return the independent final-evaluation seeds."""

    start = seed + PLOT9_SEED_OFFSET
    return list(range(start, start + count))


def _target_users() -> np.ndarray:
    """Return an independent copy of the fixed hidden-user scenario."""

    return TARGET_USER_POSITIONS.copy()


def _diagnostic_users(cfg: Config, seed: int) -> np.ndarray:
    """Return a reproducible second user set used only for Plot 8 evaluation."""

    return random_users(cfg, seed=seed + 1)


def _generate_user_sets(cfg: Config, seed: int) -> dict[str, np.ndarray]:
    """Legacy helper retained for compatibility with older result tests."""

    return {
        "set_a": random_users(cfg, seed=seed),
        "set_b": random_users(cfg, seed=seed + 1),
    }


def _set_model_path(base_path: Path, set_name: str) -> Path:
    """Return the independent checkpoint path for one user set."""

    return base_path.parent / f"{base_path.name}_{set_name}"


def _balanced_model_path(base_path: Path) -> Path:
    """Use one model because training and evaluation share one fixed scenario."""

    return base_path


def _combine_two_histories(history_a: dict, history_b: dict) -> dict:
    """Build the two independent training curves required by Plot 8."""

    steps_a = history_a.get("training_steps", [])
    steps_b = history_b.get("training_steps", [])
    if steps_a != steps_b:
        raise ValueError("PPO-A and PPO-B training steps must match")
    returns_a = history_a.get("returns_by_set", [[]])
    returns_b = history_b.get("returns_by_set", [[]])
    return {
        "training_steps": list(steps_a),
        "returns_by_set": [list(returns_a[0]), list(returns_b[0])],
        "evaluation_set_roles": ["set_a", "set_b"],
    }


def _model_score(results: list[dict]) -> tuple[int, int, int, float]:
    """Rank complete-task success before throughput."""

    joint = sum(
        bool(np.all(result["users_served"]) and result["reached_end"])
        for result in results
    )
    both = sum(bool(np.all(result["users_served"])) for result in results)
    endpoints = sum(bool(result["reached_end"]) for result in results)
    mean_throughput = float(
        np.mean([result["sum_throughput"] for result in results])
    )
    return joint, both, endpoints, mean_throughput


def _policy_trajectory_source(
    training_history: dict,
    training_performed_this_run: bool,
) -> str:
    """Return a source label only when reward-training provenance is known."""

    known_loaded_history = (
        training_history.get("checkpoint_scope")
        in {"target_user_set_only", "formal_target_user_set_only"}
        and training_history.get("training_user_positions") is not None
    )
    if training_performed_this_run or known_loaded_history:
        return "ppo_reward_trained"
    return "ppo_training_provenance_unknown"


def _evaluation_set_labels(training_history: dict) -> list[str]:
    """Label evaluation curves without inventing roles for legacy histories."""

    returns_by_set = training_history.get("returns_by_set", [])
    roles = training_history.get("evaluation_set_roles")
    role_labels = {
        "target": "Target fixed users",
        "diagnostic": "Secondary diagnostic users",
        "set_a": "PPO-A / User set A",
        "set_b": "PPO-B / User set B",
    }
    if (
        isinstance(roles, list)
        and len(roles) == len(returns_by_set)
        and all(role in role_labels for role in roles)
    ):
        return [role_labels[role] for role in roles]
    return [
        f"Evaluation set {index} (legacy provenance unknown)"
        for index in range(1, len(returns_by_set) + 1)
    ]


def _plot8_history_available(training_history: dict) -> bool:
    """Return whether two aligned user-set evaluation series are available."""

    steps = training_history.get("training_steps", [])
    means = training_history.get("returns_by_set", [])
    deviations = training_history.get("returns_std_by_set", [])
    return bool(
        steps
        and len(means) == 2
        and len(deviations) == 2
        and all(len(series) == len(steps) for series in [*means, *deviations])
    )


def _moving_return_statistics(
    values: Sequence[float],
    window: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trailing mean and standard deviation for each episode."""

    if window < 1:
        raise ValueError("window must be positive")
    array = np.asarray(values, dtype=float)
    means = np.empty(array.size, dtype=float)
    deviations = np.empty(array.size, dtype=float)
    for index in range(array.size):
        sample = array[max(0, index - window + 1) : index + 1]
        means[index] = float(np.mean(sample))
        deviations[index] = float(np.std(sample))
    return means, deviations


def _missing_training_history() -> dict:
    """Return an explicit unknown-history record without fabricated samples."""

    return {
        "training_steps": [],
        "returns_by_set": [[], []],
        "returns_std_by_set": [[], []],
        "both_users_served_rates": [],
        "endpoint_success_rates": [],
        "joint_success_rates": [],
        "mean_sum_throughput": [],
        "step_rewards": [],
        "episode_returns": [],
        "evaluation_set_roles": [],
        "checkpoint_scope": "unknown_legacy_history",
        "training_user_positions": None,
    }


def _plot9_title(trajectory_source: str) -> str:
    """Return a Plot 9 title consistent with known policy provenance."""

    if trajectory_source == "ppo_reward_trained":
        return "Plot 9: Reward-Trained PPO Optimized Trajectories"
    return "Plot 9: PPO Optimized Trajectories (Training Provenance Unknown)"


def _axis_aligned_baseline(cfg: Config) -> np.ndarray:
    """Return the fixed horizontal/vertical staircase from start to end."""

    position = np.asarray(cfg.start, dtype=float).copy()
    destination = np.asarray(cfg.end, dtype=float)
    trajectory = [position.copy()]
    move_x_next = True
    while not np.allclose(position, destination):
        delta = destination - position
        if move_x_next and abs(delta[0]) > 1e-9:
            position[0] += np.sign(delta[0]) * min(abs(delta[0]), cfg.move_distance)
        elif abs(delta[1]) > 1e-9:
            position[1] += np.sign(delta[1]) * min(abs(delta[1]), cfg.move_distance)
        else:
            position[0] += np.sign(delta[0]) * min(abs(delta[0]), cfg.move_distance)
        trajectory.append(position.copy())
        move_x_next = not move_x_next
    return np.asarray(trajectory)


def _random_beamformers(cfg: Config, rng: np.random.Generator) -> list[np.ndarray]:
    beams = []
    power = cfg.transmit_power / cfg.num_users
    for _ in range(cfg.num_users):
        vector = rng.normal(size=cfg.num_antennas) + 1j * rng.normal(
            size=cfg.num_antennas
        )
        vector /= np.linalg.norm(vector)
        beams.append(np.sqrt(power) * vector)
    return beams


def _trajectory_rates(
    cfg: Config,
    trajectory: np.ndarray,
    users: np.ndarray,
    optimized_transmit: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    history = []
    for position in _episode_positions(cfg, trajectory):
        channels = [channel_vector(cfg, position, user) for user in users]
        if optimized_transmit:
            _, _, rates = optimise_power_split(cfg, channels, grid_points=21)
        else:
            beams = _random_beamformers(cfg, rng)
            _, rates = sinr_and_throughput(cfg, channels, beams)
        history.append(rates)
    return np.asarray(history)


def _episode_positions(cfg: Config, trajectory: np.ndarray) -> np.ndarray:
    """Return exactly one UAV position for every episode step."""

    points = np.asarray(trajectory, dtype=float)
    positions = points[1 : cfg.episode_steps + 1]
    if positions.shape[0] == 0:
        positions = np.asarray([cfg.start], dtype=float)
    if positions.shape[0] < cfg.episode_steps:
        padding = np.repeat(
            positions[-1][None, :],
            cfg.episode_steps - positions.shape[0],
            axis=0,
        )
        positions = np.vstack([positions, padding])
    return positions


def _four_case_rates(
    cfg: Config,
    users: np.ndarray,
    deterministic_result: dict,
    optimized_result: dict,
    seed: int,
) -> dict[str, np.ndarray]:
    """Evaluate the four trajectory/beamforming cases from the reference."""

    deterministic_path = np.asarray(deterministic_result["trajectory"])
    optimized_path = np.asarray(optimized_result["trajectory"])
    return {
        "Deterministic + random beamforming": _trajectory_rates(
            cfg, deterministic_path, users, False, seed
        ),
        "Deterministic + optimal beamforming": _trajectory_rates(
            cfg, deterministic_path, users, True, seed
        ),
        "Optimized + random beamforming": _trajectory_rates(
            cfg, optimized_path, users, False, seed
        ),
        "Optimized + optimal beamforming": _trajectory_rates(
            cfg, optimized_path, users, True, seed
        ),
    }


def _smooth(values: Sequence[float], width: int = 25) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return array
    width = min(width, array.size)
    kernel = np.ones(width) / width
    return np.convolve(array, kernel, mode="valid")


def _watts_to_dbw(power_watts: np.ndarray) -> np.ndarray:
    """Convert watts to dBW without allowing log10(0)."""

    return 10.0 * np.log10(
        np.maximum(np.asarray(power_watts, dtype=float), 1e-300)
    )


def _power_vs_distance(cfg: Config) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Return received-power curves for the path-loss exponents in Plots 1-2."""

    distances = np.geomspace(cfg.height, 160.0, 150)
    curves = {
        eta: cfg.transmit_power * cfg.reference_gain / distances**eta
        for eta in (2.0, 2.5, 3.0, 3.5, 4.0)
    }
    return distances, curves


def _power_vs_budget(cfg: Config) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return per-user power curves for the user counts in Plots 3-4."""

    budgets = np.linspace(0.01, 1.0, 120)
    representative_distance = 70.0
    curves = {
        user_count: budgets
        / user_count
        * cfg.reference_gain
        / representative_distance**cfg.path_loss_exponent
        for user_count in (1, 2, 3, 4)
    }
    return budgets, curves


def _collect_deterministic_results(
    cfg: Config,
    seed: int,
    count: int,
) -> list[dict]:
    """Pair the fixed staircase baseline with reproducible hidden-user sets."""

    results = []
    trajectory = _axis_aligned_baseline(cfg)
    for run in range(count):
        users = random_users(cfg, seed=seed + run)
        results.append(
            {
                "trajectory": trajectory.copy(),
                "user_positions": users,
            }
        )
    return results


def _collect_optimized_results(
    cfg: Config,
    model,
    user_positions: np.ndarray,
    seeds: Sequence[int],
    trajectory_source: str = "ppo_reward_trained",
) -> list[dict]:
    """Collect stochastic trajectories from one PPO policy."""

    users = np.asarray(user_positions, dtype=float)
    results = []
    for evaluation_seed in seeds:
        env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=users)
        result = evaluate_policy(
            model,
            env,
            seed=evaluation_seed,
            deterministic=False,
        )
        result["trajectory_source"] = trajectory_source
        result["user_positions"] = users.copy()
        results.append(result)
    return results


def _dwell_markers(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return repeated coordinates and marker sizes proportional to dwell time."""

    points, counts = np.unique(
        np.asarray(trajectory, dtype=float),
        axis=0,
        return_counts=True,
    )
    repeated = counts > 1
    return points[repeated], 12.0 + 10.0 * counts[repeated]


def _joint_success_count(results: Sequence[dict]) -> int:
    """Count runs that both serve every user and finish at the endpoint."""

    return sum(
        bool(np.all(result["users_served"]) and result["reached_end"])
        for result in results
    )


def _plot9_gate(
    both: int,
    endpoints: int,
    joint: int,
    total: int = 16,
) -> bool:
    """Accept only when every run serves both users and reaches the endpoint."""

    return both == total and endpoints == total and joint == total


def _draw_trajectory_panel(
    axis,
    cfg: Config,
    result: dict,
    run_index: int,
) -> None:
    """Draw one independently styled PPO trajectory panel."""

    trajectory = np.asarray(result["trajectory"], dtype=float)
    users = np.asarray(result["user_positions"], dtype=float)
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        linewidth=1.25,
        marker="o",
        markersize=1.4,
        color="tab:blue",
        zorder=3,
    )

    changes = np.diff(trajectory, axis=0)
    moving_indices = np.flatnonzero(np.any(np.abs(changes) > 1e-12, axis=1))
    if moving_indices.size:
        sample_positions = np.linspace(
            0,
            moving_indices.size - 1,
            min(6, moving_indices.size),
            dtype=int,
        )
        selected = moving_indices[np.unique(sample_positions)]
        arrows = axis.quiver(
            trajectory[selected, 0],
            trajectory[selected, 1],
            changes[selected, 0],
            changes[selected, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="tab:blue",
            width=0.006,
            zorder=4,
        )
        arrows.set_gid("direction-arrows")

    user_colours = ("tab:purple", "tab:orange")
    for index, (user, colour) in enumerate(zip(users, user_colours), start=1):
        zone = Circle(
            tuple(user),
            cfg.locality_radius,
            facecolor=colour,
            edgecolor=colour,
            alpha=0.12,
            linewidth=1.0,
            zorder=1,
        )
        zone.set_gid("service-zone")
        axis.add_patch(zone)
        axis.scatter(*user, s=28, color=colour, zorder=5)
        axis.annotate(
            f"U{index}",
            tuple(user),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color=colour,
            zorder=6,
        )

    axis.scatter(*cfg.start, marker="s", s=26, color="tab:green", zorder=6)
    axis.scatter(*cfg.end, marker="*", s=48, color="tab:red", zorder=6)
    served = "yes" if bool(np.all(result["users_served"])) else "no"
    reached = "yes" if bool(result["reached_end"]) else "no"
    axis.set_title(
        f"Run {run_index}: T={result['sum_throughput']:.0f}\n"
        f"served={served}, end={reached}, "
        f"overrides={result.get('action_override_count', 0)}",
        fontsize=8,
    )


def _plot_trajectory_grid(
    output_path: Path,
    cfg: Config,
    results: list[dict],
    title: str,
    show_dwell: bool = False,
) -> None:
    """Save a trajectory grid for 10 or 16 episode results."""

    layouts = {16: (4, 4), 10: (2, 5)}
    if len(results) not in layouts:
        raise ValueError("trajectory grid requires exactly 10 or 16 results")
    rows, columns = layouts[len(results)]
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.75 * columns, 2.5 * rows),
        constrained_layout=True,
    )
    for run_index, (axis, result) in enumerate(
        zip(np.asarray(axes).flat, results),
        start=1,
    ):
        if show_dwell:
            _draw_trajectory_panel(axis, cfg, result, run_index)
        else:
            trajectory = np.asarray(result["trajectory"], dtype=float)
            users = np.asarray(result["user_positions"], dtype=float)
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=1.1,
                color="tab:blue",
            )
            axis.scatter(users[:, 0], users[:, 1], marker="x", s=24, color="black")
            axis.scatter(*cfg.start, marker="s", s=24, color="tab:green")
            axis.scatter(*cfg.end, marker="*", s=42, color="tab:red")
            axis.set_title(f"Run {run_index}", fontsize=8)
        axis.set_xlim(-5.0, cfg.area_size + 5.0)
        axis.set_ylim(-5.0, cfg.area_size + 5.0)
        axis.set_xticks(np.arange(0.0, cfg.area_size + 1.0, 20.0))
        axis.set_yticks(np.arange(0.0, cfg.area_size + 1.0, 20.0))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
        axis.tick_params(labelsize=7)
        if run_index > (rows - 1) * columns:
            axis.set_xlabel("x (m)", fontsize=8)
        if (run_index - 1) % columns == 0:
            axis.set_ylabel("y (m)", fontsize=8)
    if show_dwell:
        joint = _joint_success_count(results)
        fig.suptitle(
            f"{title}\njoint success = {joint}/{len(results)}",
            fontsize=14,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color="tab:blue", label="PPO trajectory"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:purple", label="U1"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", label="U2"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:green", label="Start"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:red", markersize=10, label="Endpoint"),
                Patch(facecolor="grey", alpha=0.12, label="Service radius"),
            ],
            loc="lower center",
            ncol=6,
            fontsize=8,
        )
    else:
        fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_plot8(axis, cfg: Config, training_history: dict) -> None:
    """Draw target and diagnostic user-set evaluation return evidence."""

    if _plot8_history_available(training_history):
        steps = np.asarray(training_history["training_steps"], dtype=float)
        labels = _evaluation_set_labels(training_history)
        colours = ("tab:blue", "tab:orange")
        for means, deviations, label, colour in zip(
            training_history["returns_by_set"],
            training_history["returns_std_by_set"],
            labels,
            colours,
        ):
            mean_array = np.asarray(means, dtype=float)
            std_array = np.asarray(deviations, dtype=float)
            axis.plot(
                steps,
                mean_array,
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=colour,
                label=label,
            )
            axis.fill_between(
                steps,
                mean_array - std_array,
                mean_array + std_array,
                color=colour,
                alpha=0.14,
            )
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Two-user-set evaluation history unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("PPO training step")
    axis.set_ylabel("Mean evaluation episode return")
    axis.set_title("Plot 8: PPO Evaluation Return Across User Sets")
    axis.grid(True, alpha=0.3)


def _save_figures(
    output_dir: Path,
    cfg: Config,
    training_history: dict,
    deterministic_results: list[dict],
    optimized_results: list[dict],
    four_cases: dict[str, np.ndarray],
    trajectory_source: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    distances, distance_curves = _power_vs_distance(cfg)
    plt.figure(figsize=(7.5, 5.0))
    for eta, received in distance_curves.items():
        plt.plot(distances, received, label=f"Path-loss exponent = {eta:g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance (m, log scale)")
    plt.ylabel("Received power (W)")
    plt.title("Plot 1: Received Power vs Distance")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / FIGURE_NAMES[0], dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    for eta, received in distance_curves.items():
        plt.plot(distances, _watts_to_dbw(received), label=f"Path-loss exponent = {eta:g}")
    plt.xscale("log")
    plt.xlabel("Distance (m, log scale)")
    plt.ylabel("Received power (dBW)")
    plt.title("Plot 2: Received Power vs Distance in dBW")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / FIGURE_NAMES[1], dpi=180)
    plt.close()

    budgets, budget_curves = _power_vs_budget(cfg)
    plt.figure(figsize=(7.5, 5.0))
    for user_count, received in budget_curves.items():
        plt.plot(budgets, received, label=f"K = {user_count} users")
    plt.xlabel("Transmit power budget (W)")
    plt.ylabel("Per-user received power (W)")
    plt.title("Plot 3: Received Power vs Transmit Power")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / FIGURE_NAMES[2], dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    for user_count, received in budget_curves.items():
        plt.plot(budgets, _watts_to_dbw(received), label=f"K = {user_count} users")
    plt.xlabel("Transmit power budget (W)")
    plt.ylabel("Per-user received power (dBW)")
    plt.title("Plot 4: Received Power vs Transmit Power in dBW")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / FIGURE_NAMES[3], dpi=180)
    plt.close()

    case_labels = list(four_cases)
    deterministic_labels = case_labels[:2]
    deterministic_sums = [float(four_cases[label].sum()) for label in deterministic_labels]
    two_short_labels = ["Random\nbeamformers", "Optimal\nbeamformers"]
    plt.figure(figsize=(7.5, 5.0))
    plt.bar(two_short_labels, deterministic_sums, color="#d90079")
    plt.ylabel("Sum throughput (bps/Hz)")
    plt.title("Plot 5: Deterministic Trajectory Sum Throughput")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / FIGURE_NAMES[4], dpi=180)
    plt.close()

    _plot_trajectory_grid(
        output_dir / FIGURE_NAMES[5],
        cfg,
        deterministic_results,
        "Plot 6: Deterministic Trajectory Runs",
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.2), constrained_layout=True)
    for user_index, axis in enumerate(axes):
        values = [
            float(four_cases[label][:, user_index].sum())
            for label in deterministic_labels
        ]
        axis.bar(two_short_labels, values, color="#d90079")
        axis.set_ylabel("Individual throughput (bps/Hz)")
        axis.set_title(f"User {user_index + 1}")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Plot 7: Deterministic Individual Throughput")
    fig.savefig(output_dir / FIGURE_NAMES[6], dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    _draw_plot8(axis, cfg, training_history)
    fig.tight_layout()
    fig.savefig(output_dir / FIGURE_NAMES[7], dpi=180)
    plt.close(fig)

    _plot_trajectory_grid(
        output_dir / FIGURE_NAMES[8],
        cfg,
        optimized_results,
        _plot9_title(trajectory_source),
        show_dwell=True,
    )

    four_short_labels = [
        "Deterministic\nrandom BF",
        "Deterministic\noptimal BF",
        "Optimized\nrandom BF",
        "Optimized\noptimal BF",
    ]
    bar_colors = ["#d90079", "#8e44ad", "#1f77b4", "#2ca02c"]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), constrained_layout=True)
    for user_index, axis in enumerate(axes):
        values = [float(four_cases[label][:, user_index].sum()) for label in case_labels]
        axis.bar(four_short_labels, values, color=bar_colors)
        axis.set_ylabel("Individual throughput (bps/Hz)")
        axis.set_title(f"User {user_index + 1}")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Plot 10: Four-Case Individual Throughput")
    fig.savefig(output_dir / FIGURE_NAMES[9], dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9.0, 5.5))
    totals = [float(four_cases[label].sum()) for label in case_labels]
    plt.bar(four_short_labels, totals, color=bar_colors)
    plt.ylabel("Sum throughput (bps/Hz)")
    plt.title("Plot 11: Four-Case Sum Throughput")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / FIGURE_NAMES[10], dpi=180)
    plt.close()


def _training_settings(quick: bool) -> tuple[int, int]:
    """Return checkpoint size and target evaluation episode count."""

    return (64, 1) if quick else (50_000, 16)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short validation run")
    parser.add_argument("--train-steps", type=int, default=1_000_000)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--seed", type=int, default=9123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "ppo_balanced",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = Config()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    target_users = _target_users()
    diagnostic_users = _diagnostic_users(cfg, args.seed)
    balanced_model_path = _balanced_model_path(model_path)
    history_file = balanced_model_path.with_suffix(".history.json")
    environment = UAVThroughputEnv(
        cfg=cfg,
        fixed_user_positions=target_users,
        seed=args.seed,
    )
    train_steps = args.train_steps
    block_size, evaluation_episodes = _training_settings(args.quick)
    training_performed_this_run = not args.skip_training

    if args.skip_training:
        model = load_model(balanced_model_path, environment)
        training_history = (
            json.loads(history_file.read_text(encoding="utf-8"))
            if history_file.exists()
            else _missing_training_history()
        )
    else:
        model, training_history = train_curriculum(
            cfg=cfg,
            fixed_user_positions=target_users,
            model_path=balanced_model_path,
            total_timesteps=train_steps,
            evaluation_user_sets=[target_users, diagnostic_users],
            evaluation_episodes=evaluation_episodes,
            checkpoint_interval=min(block_size, train_steps),
            seed=args.seed,
            validation_seed_fn=_validation_seeds,
        )
        history_file.write_text(
            json.dumps(training_history, indent=2),
            encoding="utf-8",
        )

    trajectory_source = _policy_trajectory_source(
        training_history,
        training_performed_this_run=training_performed_this_run,
    )
    recorded_steps = training_history.get("training_steps", [])
    recorded_training_steps = int(recorded_steps[-1]) if recorded_steps else None

    deterministic_results = _collect_deterministic_results(
        cfg,
        seed=args.seed,
        count=16,
    )
    plot9_evaluation_seeds = _plot9_seeds(args.seed, 16)
    optimized_results = _collect_optimized_results(
        cfg,
        model=model,
        user_positions=target_users,
        seeds=plot9_evaluation_seeds,
        trajectory_source=trajectory_source,
    )
    plot9_both = sum(
        bool(np.all(result["users_served"])) for result in optimized_results
    )
    plot9_endpoints = sum(
        bool(result["reached_end"]) for result in optimized_results
    )
    plot9_joint = _joint_success_count(optimized_results)
    plot9_passed = _plot9_gate(
        plot9_both,
        plot9_endpoints,
        plot9_joint,
        total=len(optimized_results),
    )
    representative = max(
        optimized_results,
        key=lambda result: float(result["sum_throughput"]),
    )
    four_cases = _four_case_rates(
        cfg,
        target_users,
        deterministic_results[0],
        representative,
        seed=args.seed,
    )
    _save_figures(
        output_dir,
        cfg,
        training_history,
        deterministic_results,
        optimized_results,
        four_cases,
        trajectory_source,
    )

    summary = {
        "seed": args.seed,
        "formal_objective": "maximize_200_step_sum_throughput",
        "fixed_horizon_steps": cfg.episode_steps,
        "hard_endpoint": list(cfg.end),
        "requested_training_steps": train_steps,
        "recorded_training_steps": recorded_training_steps,
        "training_performed_this_run": training_performed_this_run,
        "training_steps": recorded_training_steps,
        "training_history": training_history,
        "user_position_source": "fixed_balanced_hidden_users",
        "selected_model_path": balanced_model_path.with_suffix(".zip").name,
        "plot_9_user_positions": target_users.tolist(),
        "training_user_positions": target_users.tolist(),
        "training_user_mode": "one_fixed_target_set",
        "plot_8_evaluation_user_sets": {
            "target": target_users.tolist(),
            "diagnostic": diagnostic_users.tolist(),
        },
        "checkpoint_scope": training_history.get(
            "checkpoint_scope", "unknown_legacy_history"
        ),
        "policy_observation_contains_user_positions": False,
        "allowed_actions": ["right", "left", "up", "down", "hover"],
        "action_space_size": int(environment.action_space.n),
        "movement_step_metres": cfg.move_distance,
        "service_requirement_seconds": cfg.service_steps_required * cfg.step_seconds,
        "ppo_service_seconds": np.asarray(representative["service_seconds"]).tolist(),
        "ppo_users_served": np.asarray(
            representative["users_served"], dtype=bool
        ).tolist(),
        "ppo_reached_end": bool(representative["reached_end"]),
        "ppo_sum_throughput": float(representative["sum_throughput"]),
        "ppo_individual_throughput": np.asarray(
            representative["individual_throughput"]
        ).tolist(),
        "ppo_episode_reward": float(representative["episode_reward"]),
        "ppo_action_override_count": int(representative["action_override_count"]),
        "ppo_final_position": np.asarray(
            representative["final_position"], dtype=float
        ).tolist(),
        "ppo_trajectory": np.asarray(representative["trajectory"]).tolist(),
        "plot_9_runs": [
            {
                "sum_throughput": float(result["sum_throughput"]),
                "users_served": np.asarray(
                    result["users_served"], dtype=bool
                ).tolist(),
                "reached_end": bool(result["reached_end"]),
                "episode_reward": float(result["episode_reward"]),
                "step_count": int(result["step_count"]),
                "final_position": np.asarray(
                    result["final_position"], dtype=float
                ).tolist(),
                "action_override_count": int(result["action_override_count"]),
            }
            for result in optimized_results
        ],
        "plot_9_mean_sum_throughput": float(
            np.mean([result["sum_throughput"] for result in optimized_results])
        ),
        "plot_9_evaluation_seeds": plot9_evaluation_seeds,
        "plot_9_both_users_served_count": plot9_both,
        "plot_9_endpoint_success_count": plot9_endpoints,
        "plot_8_episode_return_samples": len(
            training_history.get("episode_returns", [])
        ),
        "plot_9_joint_success_count": plot9_joint,
        "plot_9_acceptance_passed": plot9_passed,
        "throughput_cases": {
            label: {
                "individual": rates.sum(axis=0).tolist(),
                "sum": float(rates.sum()),
            }
            for label, rates in four_cases.items()
        },
        "trajectory_run_counts": {
            "deterministic": len(deterministic_results),
            "optimized": len(optimized_results),
        },
        "all_trajectory_runs_axis_aligned": {
            "deterministic": all(
                np.all(
                    (np.diff(np.asarray(result["trajectory"]), axis=0)[:, 0] == 0.0)
                    | (np.diff(np.asarray(result["trajectory"]), axis=0)[:, 1] == 0.0)
                )
                for result in deterministic_results
            ),
            "optimized": all(
                np.all(
                    (np.diff(np.asarray(result["trajectory"]), axis=0)[:, 0] == 0.0)
                    | (np.diff(np.asarray(result["trajectory"]), axis=0)[:, 1] == 0.0)
                )
                for result in optimized_results
            ),
        },
        "plot_9_trajectory_source": trajectory_source,
        "all_optimized_runs_served": all(
            bool(np.all(result["users_served"])) for result in optimized_results
        ),
        "all_optimized_runs_reached_end": all(
            bool(result["reached_end"]) for result in optimized_results
        ),
        "figure_files": FIGURE_NAMES,
    }
    (output_dir / "results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Results saved to: {output_dir}")
    print(f"PPO service seconds: {summary['ppo_service_seconds']}")
    print(f"PPO reached end: {summary['ppo_reached_end']}")
    print(f"Plot 9 served: {plot9_both}/16")
    print(f"Plot 9 endpoint: {plot9_endpoints}/16")
    print(f"Plot 9 joint: {plot9_joint}/16")
    print(f"Plot 9 acceptance passed: {plot9_passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
