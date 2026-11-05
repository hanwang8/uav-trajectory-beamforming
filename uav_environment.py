"""Gymnasium UAV environment with hidden ground-user coordinates."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wireless_model import (
    Config,
    channel_vector,
    maximum_normalised_channel_power,
    normalised_channel_power,
    optimise_power_split,
)

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3
HOVER = 4

ACTION_CHANGES = {
    RIGHT: np.array([1.0, 0.0]),
    LEFT: np.array([-1.0, 0.0]),
    UP: np.array([0.0, 1.0]),
    DOWN: np.array([0.0, -1.0]),
    HOVER: np.array([0.0, 0.0]),
}


class UAVThroughputEnv(gym.Env[np.ndarray, int]):
    """Five-action UAV environment with throughput and service rewards."""

    metadata = {"render_modes": ["ansi"]}
    observation_names = (
        "uav_x",
        "uav_y",
        "end_dx",
        "end_dy",
        "gain_1",
        "gain_2",
        "gain_change_1",
        "gain_change_2",
        "service_progress_1",
        "service_progress_2",
        "remaining_time",
    )

    def __init__(
        self,
        cfg: Config | None = None,
        fixed_user_positions: np.ndarray | None = None,
        seed: int | None = None,
        objective: str = "joint",
    ) -> None:
        super().__init__()
        self.cfg = cfg or Config()
        if objective not in {"service", "joint"}:
            raise ValueError("objective must be 'service' or 'joint'")
        self.objective = objective
        self.fixed_user_positions = (
            None
            if fixed_user_positions is None
            else np.asarray(fixed_user_positions, dtype=float).copy()
        )
        expected_shape = (self.cfg.num_users, 2)
        if (
            self.fixed_user_positions is not None
            and self.fixed_user_positions.shape != expected_shape
        ):
            raise ValueError(f"fixed_user_positions must have shape {expected_shape}")

        self.action_space = spaces.Discrete(5, seed=seed)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        self._user_positions = np.zeros(expected_shape, dtype=float)
        self.uav_position = np.asarray(self.cfg.start, dtype=float)
        self.previous_gains = np.zeros(self.cfg.num_users, dtype=float)
        self.current_gains = np.zeros(self.cfg.num_users, dtype=float)
        self.gain_changes = np.zeros(self.cfg.num_users, dtype=float)
        self.service_steps = np.zeros(self.cfg.num_users, dtype=int)
        self.users_served = np.zeros(self.cfg.num_users, dtype=bool)
        self.step_count = 0
        self.trajectory: list[np.ndarray] = []
        self.throughput_history: list[np.ndarray] = []
        self.reward_history: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.objective == "joint":
            self._validate_endpoint_constraint()
        if self.fixed_user_positions is None:
            self._user_positions = self.np_random.uniform(
                0.0,
                self.cfg.area_size,
                size=(self.cfg.num_users, 2),
            )
        else:
            self._user_positions = self.fixed_user_positions.copy()

        self.uav_position = np.asarray(self.cfg.start, dtype=float)
        self.current_gains = self._channel_gain_ratios()
        self.previous_gains = self.current_gains.copy()
        self.gain_changes = np.zeros(self.cfg.num_users, dtype=float)
        self.service_steps = np.zeros(self.cfg.num_users, dtype=int)
        self.users_served = np.zeros(self.cfg.num_users, dtype=bool)
        self.step_count = 0
        self.trajectory = [self.uav_position.copy()]
        self.throughput_history = []
        self.reward_history = []
        return self._observation(), self._info_template()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        requested = int(action)
        if self.objective == "joint":
            effective, action_overridden = self._project_action_for_endpoint(requested)
            endpoint_constraint_active = self._endpoint_constraint_active()
        else:
            effective = requested
            action_overridden = False
            endpoint_constraint_active = False
        return_phase = endpoint_constraint_active
        old_position = self.uav_position.copy()
        old_end_distance = self._manhattan_distance_to_end(old_position)
        proposed = old_position + ACTION_CHANGES[effective] * self.cfg.move_distance
        clipped = np.clip(proposed, 0.0, self.cfg.area_size)
        boundary_hit = not np.allclose(proposed, clipped)
        self.uav_position = clipped

        self.previous_gains = self.current_gains.copy()
        self.current_gains = self._channel_gain_ratios()
        self.gain_changes = np.clip(
            self.current_gains - self.previous_gains,
            -1.0,
            1.0,
        )

        distances = np.linalg.norm(self._user_positions - self.uav_position, axis=1)
        near_users = distances <= self.cfg.locality_radius + 1e-9
        first_entry = near_users & (self.service_steps == 0)
        previously_served = self.users_served.copy()
        service_incremented = near_users & (
            self.service_steps < self.cfg.service_steps_required
        )
        self.service_steps = np.minimum(
            self.service_steps + near_users.astype(int),
            self.cfg.service_steps_required,
        )
        self.users_served = self.service_steps >= self.cfg.service_steps_required
        newly_completed = self.users_served & ~previously_served

        channels = [
            channel_vector(self.cfg, self.uav_position, user)
            for user in self._user_positions
        ]
        powers, _, rates = optimise_power_split(self.cfg, channels, grid_points=11)
        self.throughput_history.append(rates.copy())

        raw_sum_throughput = float(rates.sum())
        throughput_multiplier = (
            self.cfg.throughput_reward_multiplier
            if self.objective == "joint"
            else 1.0
        )
        reward_throughput = raw_sum_throughput * throughput_multiplier
        reward_locality = self.cfg.reward_first_entry * float(first_entry.sum())
        reward_service_progress = self.cfg.reward_service_progress * float(
            service_incremented.sum()
        )
        reward_service_completion = self.cfg.reward_service_completion * float(
            newly_completed.sum()
        )
        reward_channel_change = self.cfg.reward_channel_change * float(
            self.current_gains.sum() - self.previous_gains.sum()
        )
        reward_boundary = self.cfg.reward_boundary if boundary_hit else 0.0

        new_end_distance = self._manhattan_distance_to_end(self.uav_position)
        required_to_end = int(
            np.ceil(new_end_distance / self.cfg.move_distance - 1e-12)
        )
        endpoint_phase = endpoint_constraint_active
        reached_end = bool(np.allclose(self.uav_position, self.cfg.end))
        reward_endpoint_progress = (
            self.cfg.reward_endpoint_progress
            * (old_end_distance - new_end_distance)
            / self.cfg.move_distance
            if endpoint_phase
            else 0.0
        )
        reward_endpoint_distance = (
            self.cfg.reward_endpoint_distance_per_step
            * new_end_distance
            / self.cfg.move_distance
            if endpoint_phase
            else 0.0
        )
        reward_endpoint_dwell = 0.0

        self.step_count += 1
        service_success = bool(np.all(self.users_served))
        joint_success = bool(service_success and reached_end)
        horizon_reached = self.step_count >= self.cfg.episode_steps
        if self.objective == "service":
            terminated = service_success
        else:
            terminated = bool(horizon_reached and joint_success)
        truncated = bool(horizon_reached and not terminated)
        reward_step = self.cfg.reward_step
        reward_terminal = 0.0
        if horizon_reached:
            reward_terminal += self.cfg.reward_missing_user * float(
                (~self.users_served).sum()
            )
            if self.objective == "joint" and not reached_end:
                reward_terminal += self.cfg.reward_missing_endpoint

        reward_unscaled = float(
            reward_throughput
            + reward_locality
            + reward_service_progress
            + reward_service_completion
            + reward_channel_change
            + reward_step
            + reward_boundary
            + reward_endpoint_progress
            + reward_endpoint_distance
            + reward_endpoint_dwell
            + reward_terminal
        )
        reward = reward_unscaled * self.cfg.reward_scale
        self.trajectory.append(self.uav_position.copy())
        self.reward_history.append(reward)

        info = {
            "boundary_hit": bool(boundary_hit),
            "proposed_action": requested,
            "effective_action": effective,
            "action_overridden": bool(action_overridden),
            "endpoint_constraint_active": bool(endpoint_constraint_active),
            "step_distance": float(np.linalg.norm(self.uav_position - old_position)),
            "near_users": near_users.copy(),
            "service_incremented": service_incremented.copy(),
            "service_steps": self.service_steps.copy(),
            "users_served": self.users_served.copy(),
            "previous_gain_ratios": self.previous_gains.copy(),
            "gain_ratios": self.current_gains.copy(),
            "power_allocation": powers.copy(),
            "throughputs": rates.copy(),
            "return_phase": return_phase,
            "raw_sum_throughput": raw_sum_throughput,
            "reached_end": reached_end,
            "required_steps_to_end": required_to_end,
            "endpoint_phase": bool(endpoint_phase),
            "reward_throughput": reward_throughput,
            "reward_locality": reward_locality,
            "reward_service_progress": reward_service_progress,
            "reward_service_completion": reward_service_completion,
            "reward_channel_change": reward_channel_change,
            "reward_step": reward_step,
            "reward_boundary": reward_boundary,
            "reward_endpoint_progress": reward_endpoint_progress,
            "reward_endpoint_distance": reward_endpoint_distance,
            "reward_endpoint_dwell": reward_endpoint_dwell,
            "reward_terminal": reward_terminal,
            "reward_unscaled": reward_unscaled,
        }
        return self._observation(), reward, terminated, truncated, info

    def user_positions_for_evaluation(self) -> np.ndarray:
        """Expose hidden coordinates only to evaluation and plotting code."""

        return self._user_positions.copy()

    def render(self) -> str:
        return (
            f"step={self.step_count}, position={self.uav_position.tolist()}, "
            f"service_steps={self.service_steps.tolist()}"
        )

    def _channel_gain_ratios(self) -> np.ndarray:
        maximum = maximum_normalised_channel_power(self.cfg)
        ratios = [
            normalised_channel_power(channel_vector(self.cfg, self.uav_position, user))
            / maximum
            for user in self._user_positions
        ]
        return np.clip(np.asarray(ratios, dtype=float), 0.0, 1.0)

    def _observation(self) -> np.ndarray:
        values = np.concatenate(
            [
                self.uav_position / self.cfg.area_size,
                (np.asarray(self.cfg.end) - self.uav_position) / self.cfg.area_size,
                self.current_gains,
                self.gain_changes,
                self.service_steps / self.cfg.service_steps_required,
                np.array([1.0 - self.step_count / self.cfg.episode_steps]),
            ]
        )
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    def _manhattan_distance_to_end(self, position: np.ndarray) -> float:
        return float(np.abs(np.asarray(self.cfg.end) - position).sum())

    def _required_steps_to_end(self, position: np.ndarray) -> int:
        distance = self._manhattan_distance_to_end(position)
        return int(round(distance / self.cfg.move_distance))

    def _clipped_position(self, action: int) -> np.ndarray:
        proposed = self.uav_position + ACTION_CHANGES[action] * self.cfg.move_distance
        return np.clip(proposed, 0.0, self.cfg.area_size)

    def _endpoint_constraint_active(self) -> bool:
        remaining_steps = self.cfg.episode_steps - self.step_count
        return self._required_steps_to_end(self.uav_position) >= remaining_steps

    def _project_action_for_endpoint(self, requested: int) -> tuple[int, bool]:
        remaining_after = self.cfg.episode_steps - self.step_count - 1
        requested_position = self._clipped_position(requested)
        if self._required_steps_to_end(requested_position) <= remaining_after:
            return requested, False

        feasible_actions = [
            action
            for action in ACTION_CHANGES
            if self._required_steps_to_end(self._clipped_position(action))
            <= remaining_after
        ]
        if not feasible_actions:
            raise RuntimeError("endpoint is no longer reachable within the fixed horizon")
        effective = min(
            feasible_actions,
            key=lambda action: (
                self._required_steps_to_end(self._clipped_position(action)),
                action,
            ),
        )
        return effective, True

    def _validate_endpoint_constraint(self) -> None:
        delta_steps = (
            np.abs(np.asarray(self.cfg.end) - np.asarray(self.cfg.start))
            / self.cfg.move_distance
        )
        if not np.allclose(delta_steps, np.round(delta_steps), atol=1e-9):
            raise ValueError("start cannot reach endpoint exactly on the movement grid")
        required_steps = int(np.round(delta_steps).sum())
        if required_steps > self.cfg.episode_steps:
            raise ValueError("cannot reach endpoint within the fixed horizon")

    def _info_template(self) -> dict[str, Any]:
        return {
            "service_steps": self.service_steps.copy(),
            "users_served": self.users_served.copy(),
            "gain_ratios": self.current_gains.copy(),
            "reached_end": bool(np.allclose(self.uav_position, self.cfg.end)),
        }
