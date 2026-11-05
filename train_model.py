"""Reward-only PPO training, loading, and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from uav_environment import UAVThroughputEnv
from wireless_model import Config


@dataclass(frozen=True)
class CurriculumStage:
    """One declared phase of the single-policy curriculum."""

    name: str
    timesteps: int
    service_steps_required: int
    objective: str
    entropy_start: float
    entropy_end: float


def curriculum_stages(total_timesteps: int) -> list[CurriculumStage]:
    """Use short warmups and devote the remaining budget to the formal task."""

    if total_timesteps < 3:
        raise ValueError("curriculum requires at least three timesteps")
    service_steps = min(20_000, max(1, int(round(total_timesteps * 0.20))))
    joint_steps = min(30_000, max(1, int(round(total_timesteps * 0.30))))
    formal_steps = total_timesteps - service_steps - joint_steps
    return [
        CurriculumStage(
            "service_discovery",
            service_steps,
            1,
            "service",
            0.01,
            0.002,
        ),
        CurriculumStage(
            "service_and_return",
            joint_steps,
            1,
            "joint",
            0.002,
            0.0,
        ),
        CurriculumStage(
            "formal_task",
            formal_steps,
            3,
            "joint",
            0.0,
            0.0,
        ),
    ]


class RewardHistoryCallback(BaseCallback):
    """Record step rewards and genuine completed-episode returns."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.rewards: list[float] = []
        self.episode_returns: list[float] = []
        self._running_returns: np.ndarray | None = None

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            reward_values = np.asarray(rewards, dtype=float).reshape(-1)
            self.rewards.append(float(np.mean(reward_values)))
            if (
                self._running_returns is None
                or self._running_returns.shape != reward_values.shape
            ):
                self._running_returns = np.zeros_like(reward_values)
            self._running_returns += reward_values
            dones = np.asarray(
                self.locals.get("dones", np.zeros_like(reward_values, dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            for index in np.flatnonzero(dones):
                self.episode_returns.append(float(self._running_returns[index]))
                self._running_returns[index] = 0.0
        return True


def build_model(
    env: UAVThroughputEnv,
    seed: int = 9123,
    verbose: int = 0,
) -> PPO:
    """Create the single PPO policy used for autonomous throughput learning."""

    torch.set_num_threads(1)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gamma=0.995,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        verbose=verbose,
        device="cpu",
    )


def entropy_schedule(block_count: int, initial: float = 0.01) -> list[float]:
    """Return one entropy coefficient per block, ending at zero."""

    if block_count < 1:
        raise ValueError("block_count must be positive")
    if initial < 0.0:
        raise ValueError("initial entropy coefficient must be non-negative")
    if block_count == 1:
        return [0.0]
    return np.linspace(initial, 0.0, block_count).tolist()


def default_validation_seeds(
    seed: int,
    completed: int,
    count: int,
    set_index: int,
) -> list[int]:
    """Return reproducible checkpoint seeds outside the Plot 9 range."""

    start = seed + 100_000 + completed + set_index * 10_000
    return list(range(start, start + count))


def train_and_save(
    env: UAVThroughputEnv,
    model_path: str | Path,
    total_timesteps: int,
    seed: int = 9123,
    model: PPO | None = None,
) -> tuple[PPO, dict[str, list[float]]]:
    """Train only through ``model.learn()``, save, and return reward history."""

    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    trained_model = model or build_model(env, seed=seed)
    callback = RewardHistoryCallback()
    trained_model.learn(total_timesteps=total_timesteps, callback=callback)
    trained_model.save(str(target))
    return trained_model, {
        "step_rewards": callback.rewards,
        "episode_returns": callback.episode_returns,
    }


def train_in_blocks(
    env: UAVThroughputEnv,
    model_path: str | Path,
    total_timesteps: int,
    block_size: int,
    evaluation_user_sets: list[np.ndarray],
    evaluation_episodes: int = 4,
    seed: int = 9123,
    verbose: int = 0,
    validation_seed_fn: Callable[[int, int, int, int], list[int]] | None = None,
) -> tuple[PPO, dict[str, Any]]:
    """Train in blocks; set 0 targets checkpoints and later sets are diagnostic."""

    if total_timesteps < 1 or block_size < 1:
        raise ValueError("total_timesteps and block_size must be positive")
    if not evaluation_user_sets or evaluation_episodes < 1:
        raise ValueError("evaluation sets and episodes must be non-empty")
    if env.fixed_user_positions is None:
        raise ValueError("train_in_blocks requires fixed training users")
    validation_seed_fn = validation_seed_fn or default_validation_seeds

    training_users = np.asarray(env.fixed_user_positions, dtype=float)
    checkpoint_users = np.asarray(evaluation_user_sets[0], dtype=float)
    if (
        checkpoint_users.shape != training_users.shape
        or not np.array_equal(checkpoint_users, training_users)
    ):
        raise ValueError(
            "first evaluation set must match fixed training users exactly"
        )

    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(env, seed=seed, verbose=verbose)
    block_count = int(np.ceil(total_timesteps / block_size))
    entropy_values = entropy_schedule(
        block_count,
        initial=float(model.ent_coef),
    )
    history: dict[str, Any] = {
        "training_steps": [],
        "returns_by_set": [[] for _ in evaluation_user_sets],
        "returns_std_by_set": [[] for _ in evaluation_user_sets],
        "both_users_served_rates": [],
        "endpoint_success_rates": [],
        "joint_success_rates": [],
        "mean_sum_throughput": [],
        "step_rewards": [],
        "episode_returns": [],
        "entropy_coefficients": [],
        "evaluation_seeds_by_set": [[] for _ in evaluation_user_sets],
        "evaluation_mode": "stochastic",
        "evaluation_set_roles": [
            "target",
            *["diagnostic" for _ in evaluation_user_sets[1:]],
        ],
        "checkpoint_scope": "target_user_set_only",
        "training_user_positions": training_users.tolist(),
        "checkpoint_user_positions": checkpoint_users.tolist(),
    }
    best_score: tuple[int, int, int, float] | None = None
    completed = 0
    block_index = 0

    while completed < total_timesteps:
        current_block = min(block_size, total_timesteps - completed)
        entropy_coefficient = float(entropy_values[block_index])
        model.ent_coef = entropy_coefficient
        history["entropy_coefficients"].append(entropy_coefficient)
        callback = RewardHistoryCallback()
        model.learn(
            total_timesteps=current_block,
            callback=callback,
            reset_num_timesteps=False,
        )
        completed += current_block
        history["training_steps"].append(completed)
        history["step_rewards"].extend(callback.rewards)
        history["episode_returns"].extend(callback.episode_returns)

        results_by_set: list[list[dict[str, Any]]] = []
        for set_index, users in enumerate(evaluation_user_sets):
            evaluation_seeds = validation_seed_fn(
                seed,
                completed,
                evaluation_episodes,
                set_index,
            )
            history["evaluation_seeds_by_set"][set_index].append(evaluation_seeds)
            set_results = [
                evaluate_policy(
                    model,
                    UAVThroughputEnv(
                        cfg=env.cfg,
                        fixed_user_positions=np.asarray(users, dtype=float),
                    ),
                    seed=evaluation_seed,
                    deterministic=False,
                )
                for evaluation_seed in evaluation_seeds
            ]
            results_by_set.append(set_results)
            history["returns_by_set"][set_index].append(
                float(np.mean([result["episode_reward"] for result in set_results]))
            )
            history["returns_std_by_set"][set_index].append(
                float(np.std([result["episode_reward"] for result in set_results]))
            )

        target_results = results_by_set[0]
        both_served = sum(
            bool(np.all(result["users_served"])) for result in target_results
        )
        endpoints = sum(bool(result["reached_end"]) for result in target_results)
        joint = sum(
            bool(np.all(result["users_served"]) and result["reached_end"])
            for result in target_results
        )
        mean_throughput = float(
            np.mean([result["sum_throughput"] for result in target_results])
        )
        count = len(target_results)
        history["both_users_served_rates"].append(both_served / count)
        history["endpoint_success_rates"].append(endpoints / count)
        history["joint_success_rates"].append(joint / count)
        history["mean_sum_throughput"].append(mean_throughput)

        score = (joint, both_served, endpoints, mean_throughput)
        if best_score is None or score > best_score:
            best_score = score
            model.save(str(target))
        block_index += 1

    return load_model(target, env), history


def train_curriculum(
    cfg: Config,
    fixed_user_positions: np.ndarray,
    model_path: str | Path,
    total_timesteps: int,
    evaluation_user_sets: list[np.ndarray],
    evaluation_episodes: int = 16,
    checkpoint_interval: int = 50_000,
    seed: int = 9123,
    verbose: int = 0,
    validation_seed_fn: Callable[[int, int, int, int], list[int]] | None = None,
) -> tuple[PPO, dict[str, Any]]:
    """Continue one PPO policy through three stages and validate the formal task."""

    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if not evaluation_user_sets or evaluation_episodes < 1:
        raise ValueError("evaluation sets and episodes must be non-empty")
    users = np.asarray(fixed_user_positions, dtype=float)
    target_users = np.asarray(evaluation_user_sets[0], dtype=float)
    if users.shape != target_users.shape or not np.array_equal(users, target_users):
        raise ValueError("first evaluation set must match fixed training users exactly")

    stages = curriculum_stages(total_timesteps)
    validation_seed_fn = validation_seed_fn or default_validation_seeds
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first_stage = stages[0]
    first_env = UAVThroughputEnv(
        cfg=replace(
            cfg,
            service_steps_required=first_stage.service_steps_required,
        ),
        fixed_user_positions=users,
        objective=first_stage.objective,
    )
    model = build_model(first_env, seed=seed, verbose=verbose)
    history: dict[str, Any] = {
        "training_steps": [],
        "returns_by_set": [[] for _ in evaluation_user_sets],
        "returns_std_by_set": [[] for _ in evaluation_user_sets],
        "both_users_served_rates": [],
        "endpoint_success_rates": [],
        "joint_success_rates": [],
        "mean_sum_throughput": [],
        "mean_policy_entropies": [],
        "step_rewards": [],
        "episode_returns": [],
        "entropy_coefficients": [],
        "evaluation_seeds_by_set": [[] for _ in evaluation_user_sets],
        "evaluation_mode": "stochastic",
        "evaluation_set_roles": [
            "target",
            *["diagnostic" for _ in evaluation_user_sets[1:]],
        ],
        "checkpoint_scope": "formal_target_user_set_only",
        "training_user_positions": users.tolist(),
        "checkpoint_user_positions": target_users.tolist(),
        "curriculum_stages": [],
        "selected_checkpoint_step": None,
    }
    best_score: tuple[int, int, int, float] | None = None
    completed = 0

    for stage in stages:
        stage_start = completed
        history["curriculum_stages"].append(
            {
                "name": stage.name,
                "start_step": stage_start,
                "end_step": stage_start + stage.timesteps,
                "service_steps_required": stage.service_steps_required,
                "objective": stage.objective,
            }
        )
        stage_env = UAVThroughputEnv(
            cfg=replace(
                cfg,
                service_steps_required=stage.service_steps_required,
            ),
            fixed_user_positions=users,
            objective=stage.objective,
        )
        model.set_env(stage_env)
        block_count = int(np.ceil(stage.timesteps / checkpoint_interval))
        entropy_values = np.linspace(
            stage.entropy_start,
            stage.entropy_end,
            block_count,
        )
        stage_completed = 0

        for entropy_coefficient in entropy_values:
            current_block = min(
                checkpoint_interval,
                stage.timesteps - stage_completed,
            )
            model.ent_coef = float(entropy_coefficient)
            history["entropy_coefficients"].append(float(entropy_coefficient))
            callback = RewardHistoryCallback()
            model.learn(
                total_timesteps=current_block,
                callback=callback,
                reset_num_timesteps=False,
            )
            stage_completed += current_block
            completed += current_block
            history["training_steps"].append(completed)
            history["step_rewards"].extend(callback.rewards)
            history["episode_returns"].extend(callback.episode_returns)

            results_by_set: list[list[dict[str, Any]]] = []
            for set_index, evaluation_users in enumerate(evaluation_user_sets):
                evaluation_seeds = validation_seed_fn(
                    seed,
                    completed,
                    evaluation_episodes,
                    set_index,
                )
                history["evaluation_seeds_by_set"][set_index].append(
                    evaluation_seeds
                )
                set_results = [
                    evaluate_policy(
                        model,
                        UAVThroughputEnv(
                            cfg=cfg,
                            fixed_user_positions=np.asarray(
                                evaluation_users,
                                dtype=float,
                            ),
                            objective="joint",
                        ),
                        seed=evaluation_seed,
                        deterministic=False,
                    )
                    for evaluation_seed in evaluation_seeds
                ]
                results_by_set.append(set_results)
                history["returns_by_set"][set_index].append(
                    float(
                        np.mean(
                            [result["episode_reward"] for result in set_results]
                        )
                    )
                )
                history["returns_std_by_set"][set_index].append(
                    float(
                        np.std(
                            [result["episode_reward"] for result in set_results]
                        )
                    )
                )

            target_results = results_by_set[0]
            both_served = sum(
                bool(np.all(result["users_served"]))
                for result in target_results
            )
            endpoints = sum(
                bool(result["reached_end"]) for result in target_results
            )
            joint = sum(
                bool(np.all(result["users_served"]) and result["reached_end"])
                for result in target_results
            )
            mean_throughput = float(
                np.mean([result["sum_throughput"] for result in target_results])
            )
            entropy_samples = [
                float(result["mean_policy_entropy"])
                for result in target_results
                if result.get("mean_policy_entropy") is not None
            ]
            mean_entropy = (
                float(np.mean(entropy_samples)) if entropy_samples else None
            )
            count = len(target_results)
            history["both_users_served_rates"].append(both_served / count)
            history["endpoint_success_rates"].append(endpoints / count)
            history["joint_success_rates"].append(joint / count)
            history["mean_sum_throughput"].append(mean_throughput)
            history["mean_policy_entropies"].append(mean_entropy)

            score = (joint, both_served, endpoints, mean_throughput)
            should_save = best_score is None or score > best_score
            if should_save:
                best_score = score
                history["selected_checkpoint_step"] = completed
                model.save(str(target))

    formal_env = UAVThroughputEnv(
        cfg=cfg,
        fixed_user_positions=users,
        objective="joint",
    )
    return load_model(target, formal_env), history


def load_model(model_path: str | Path, env: UAVThroughputEnv) -> PPO:
    """Load a saved PPO archive and attach the supplied environment."""

    target = Path(model_path)
    archive = target if target.suffix == ".zip" else target.with_suffix(".zip")
    if not archive.exists():
        raise FileNotFoundError(f"Saved PPO model not found: {archive}")
    return PPO.load(str(target), env=env, device="cpu")


def _model_action(model: Any, observation: np.ndarray, deterministic: bool) -> int:
    prediction = model.predict(observation, deterministic=deterministic)
    action = prediction[0] if isinstance(prediction, tuple) else prediction
    return int(np.asarray(action).item())


def _action_entropy(model: Any, observation: np.ndarray) -> float | None:
    """Return categorical action entropy when an SB3 policy exposes it."""

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "get_distribution"):
        return None
    with torch.no_grad():
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        distribution = policy.get_distribution(tensor).distribution
        return float(distribution.entropy().mean().cpu().item())


def evaluate_policy(
    model: Any,
    env: UAVThroughputEnv,
    seed: int = 9123,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run one complete episode and return measured trajectory and throughput."""

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation, _ = env.reset(seed=seed)
    actions: list[int] = []
    effective_actions: list[int] = []
    step_distances: list[float] = []
    gain_ratio_history: list[np.ndarray] = []
    service_increment_history: list[np.ndarray] = []
    reward_component_history: list[dict[str, float | bool]] = []
    action_entropies: list[float] = []
    last_info: dict[str, Any] = {}

    try:
        while True:
            action_entropy = _action_entropy(model, observation)
            if action_entropy is not None:
                action_entropies.append(action_entropy)
            action = _model_action(model, observation, deterministic)
            observation, _, terminated, truncated, last_info = env.step(action)
            actions.append(action)
            effective_actions.append(int(last_info["effective_action"]))
            step_distances.append(float(last_info["step_distance"]))
            gain_ratio_history.append(
                np.asarray(last_info["gain_ratios"], dtype=float)
            )
            service_increment_history.append(
                np.asarray(last_info["service_incremented"], dtype=bool)
            )
            reward_components: dict[str, float | bool] = {
                key: float(last_info[key])
                for key in (
                    "reward_throughput",
                    "reward_locality",
                    "reward_service_progress",
                    "reward_service_completion",
                    "reward_channel_change",
                    "reward_step",
                    "reward_boundary",
                    "reward_endpoint_progress",
                    "reward_endpoint_distance",
                    "reward_endpoint_dwell",
                    "reward_terminal",
                )
            }
            reward_components["raw_sum_throughput"] = float(
                last_info["raw_sum_throughput"]
            )
            reward_components["return_phase"] = bool(last_info["return_phase"])
            reward_components["endpoint_constraint_active"] = bool(
                last_info["endpoint_constraint_active"]
            )
            reward_component_history.append(reward_components)
            if terminated or truncated:
                break
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    throughput = np.asarray(env.throughput_history, dtype=float)
    individual = (
        throughput.sum(axis=0)
        if throughput.size
        else np.zeros(env.cfg.num_users, dtype=float)
    )
    action_array = np.asarray(actions, dtype=int)
    effective_action_array = np.asarray(effective_actions, dtype=int)
    return {
        "trajectory_source": "ppo_reward_trained",
        "trajectory": np.asarray(env.trajectory, dtype=float),
        "actions": action_array,
        "effective_actions": effective_action_array,
        "action_override_count": int(
            np.count_nonzero(action_array != effective_action_array)
        ),
        "step_distances": np.asarray(step_distances, dtype=float),
        "gain_ratio_history": np.asarray(gain_ratio_history, dtype=float),
        "service_increment_history": np.asarray(
            service_increment_history,
            dtype=bool,
        ),
        "reward_component_history": reward_component_history,
        "service_seconds": env.service_steps.astype(float) * env.cfg.step_seconds,
        "users_served": env.users_served.copy(),
        "individual_throughput": individual,
        "sum_throughput": float(individual.sum()),
        "throughput_history": throughput,
        "episode_reward": float(np.sum(env.reward_history)),
        "mean_policy_entropy": (
            float(np.mean(action_entropies)) if action_entropies else None
        ),
        "reached_end": bool(last_info.get("reached_end", False)),
        "step_count": int(env.step_count),
        "final_position": env.uav_position.copy(),
        "user_positions": env.user_positions_for_evaluation(),
    }
