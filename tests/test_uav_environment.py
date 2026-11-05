from dataclasses import replace

import numpy as np
import pytest

from uav_environment import DOWN, HOVER, LEFT, RIGHT, UP, UAVThroughputEnv
from wireless_model import Config


FIXED_USERS = np.array([[25.0, 75.0], [70.0, 25.0]])


def test_observation_hides_user_coordinates_and_has_eleven_values():
    env = UAVThroughputEnv(fixed_user_positions=FIXED_USERS, seed=7)
    first, _ = env.reset(seed=9123)
    second, _ = env.reset(seed=9123)

    assert env.action_space.n == 5
    assert env.observation_space.shape == (11,)
    assert first.shape == (11,)
    assert np.allclose(first, second)
    assert not any(
        "user_x" in name or "user_y" in name or "distance" in name
        for name in env.observation_names
    )
    assert np.all(np.isfinite(first))


def test_fixed_users_do_not_change_across_different_reset_seeds():
    env = UAVThroughputEnv(fixed_user_positions=FIXED_USERS)

    first_observation, _ = env.reset(seed=1)
    first_users = env.user_positions_for_evaluation()
    second_observation, _ = env.reset(seed=999)
    second_users = env.user_positions_for_evaluation()

    assert np.array_equal(first_users, FIXED_USERS)
    assert np.array_equal(second_users, FIXED_USERS)
    assert first_observation.shape == (11,)
    assert second_observation.shape == (11,)


def test_actions_move_ten_metres_on_one_axis_or_hover():
    expected = {
        RIGHT: np.array([10.0, 0.0]),
        LEFT: np.array([-10.0, 0.0]),
        UP: np.array([0.0, 10.0]),
        DOWN: np.array([0.0, -10.0]),
        HOVER: np.array([0.0, 0.0]),
    }
    for action, change in expected.items():
        cfg = replace(Config(), start=(50.0, 50.0))
        env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
        env.reset()
        old = env.uav_position.copy()
        _, _, _, _, info = env.step(action)
        assert np.allclose(env.uav_position - old, change)
        assert info["effective_action"] == action
        assert info["action_overridden"] is False


def test_three_locality_steps_serve_user_without_scripted_loop():
    cfg = replace(Config(), start=(20.0, 70.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    for expected_progress in (1, 2, 3):
        _, _, _, _, info = env.step(HOVER)
        assert env.service_steps[0] == expected_progress
        assert info["effective_action"] == HOVER

    assert env.users_served.tolist() == [True, False]


def test_reward_uses_sum_throughput_as_primary_component():
    cfg = replace(Config(), start=(50.0, 50.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    _, reward, _, _, info = env.step(HOVER)

    assert np.isclose(
        info["reward_throughput"],
        info["throughputs"].sum() * cfg.throughput_reward_multiplier,
    )
    components = (
        info["reward_throughput"]
        + info["reward_locality"]
        + info["reward_service_progress"]
        + info["reward_service_completion"]
        + info["reward_channel_change"]
        + info["reward_step"]
        + info["reward_boundary"]
        + info["reward_endpoint_progress"]
        + info["reward_endpoint_distance"]
        + info["reward_endpoint_dwell"]
        + info["reward_terminal"]
    )
    assert np.isclose(info["reward_unscaled"], components)
    assert np.isclose(reward, components * cfg.reward_scale)
    assert "reward_channel" not in info
    assert "loiter_loop_confirmed" not in info


def test_boundary_is_clipped_and_penalized_without_action_override():
    env = UAVThroughputEnv(fixed_user_positions=FIXED_USERS)
    env.reset()
    _, _, _, _, info = env.step(LEFT)

    assert np.allclose(env.uav_position, [0.0, 0.0])
    assert info["effective_action"] == LEFT
    assert info["action_overridden"] is False
    assert info["boundary_hit"] is True
    assert info["reward_boundary"] == env.cfg.reward_boundary


def test_each_locality_step_rewards_progress_and_third_step_rewards_completion():
    cfg = replace(Config(), start=(20.0, 70.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, _, _, first = env.step(HOVER)
    _, _, _, _, second = env.step(HOVER)
    _, _, _, _, third = env.step(HOVER)

    assert first["reward_service_progress"] == cfg.reward_service_progress
    assert first["reward_service_completion"] == 0.0
    assert second["reward_service_progress"] == cfg.reward_service_progress
    assert second["reward_service_completion"] == 0.0
    assert third["reward_service_progress"] == cfg.reward_service_progress
    assert third["reward_service_completion"] == cfg.reward_service_completion
    assert env.users_served.tolist() == [True, False]


def test_served_user_does_not_keep_receiving_service_reward():
    cfg = replace(Config(), start=(20.0, 70.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    for _ in range(3):
        env.step(HOVER)

    _, _, _, _, info = env.step(HOVER)

    assert info["reward_service_progress"] == 0.0
    assert info["reward_service_completion"] == 0.0


def _mark_both_users_served(env):
    env.service_steps[:] = env.cfg.service_steps_required
    env.users_served[:] = True


def test_joint_completion_at_endpoint_does_not_end_before_horizon():
    cfg = replace(Config(), start=(70.0, 80.0), episode_steps=2)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    _mark_both_users_served(env)

    _, _, terminated, truncated, info = env.step(RIGHT)

    assert terminated is False
    assert truncated is False
    assert info["reached_end"] is True
    assert info["reward_terminal"] == 0.0


def test_endpoint_without_service_does_not_terminate():
    cfg = replace(Config(), start=(70.0, 80.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, terminated, truncated, info = env.step(RIGHT)

    assert terminated is False
    assert truncated is False
    assert info["reached_end"] is True


def test_service_curriculum_terminates_after_both_users_are_served():
    cfg = replace(Config(), start=(70.0, 30.0), service_steps_required=1)
    env = UAVThroughputEnv(
        cfg=cfg,
        fixed_user_positions=FIXED_USERS,
        objective="service",
    )
    env.reset()
    env.service_steps[:] = [1, 0]
    env.users_served[:] = [True, False]

    _, _, terminated, truncated, _ = env.step(HOVER)

    assert terminated is True
    assert truncated is False


def test_service_completion_does_not_start_return_before_reachability_deadline():
    cfg = replace(Config(), start=(60.0, 80.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    _mark_both_users_served(env)

    _, _, _, _, info = env.step(RIGHT)

    assert np.allclose(env.uav_position, [70.0, 80.0])
    assert info["endpoint_phase"] is False
    assert info["reward_endpoint_progress"] == 0.0
    assert info["reward_endpoint_distance"] == 0.0
    assert info["reward_endpoint_dwell"] == 0.0


def test_served_users_do_not_activate_competing_endpoint_reward():
    cfg = replace(Config(), start=(60.0, 80.0), episode_steps=22)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    _mark_both_users_served(env)

    _, _, _, _, info = env.step(RIGHT)

    assert np.allclose(env.uav_position, [70.0, 80.0])
    assert info["endpoint_phase"] is False
    assert info["reward_endpoint_progress"] == 0.0
    assert info["reward_endpoint_distance"] == 0.0


def test_endpoint_distance_penalty_is_zero_before_endpoint_phase():
    cfg = replace(Config(), start=(70.0, 80.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, _, _, info = env.step(HOVER)

    assert info["endpoint_phase"] is False
    assert info["reward_endpoint_distance"] == 0.0


def test_policy_hover_at_endpoint_waits_until_fixed_horizon():
    cfg = replace(Config(), start=(80.0, 80.0), episode_steps=2)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    _mark_both_users_served(env)

    _, _, terminated, truncated, info = env.step(HOVER)

    assert info["effective_action"] == HOVER
    assert info["action_overridden"] is False
    assert info["reward_endpoint_progress"] == 0.0
    assert info["reward_endpoint_dwell"] == 0.0
    assert info["reward_terminal"] == 0.0
    assert terminated is False
    assert truncated is False


def test_endpoint_dwell_reward_requires_both_users_served():
    cfg = replace(Config(), start=(80.0, 80.0), episode_steps=1)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, _, truncated, info = env.step(HOVER)

    assert truncated is True
    assert info["endpoint_phase"] is False
    assert info["reward_endpoint_dwell"] == 0.0


def test_final_failure_uses_explicit_missing_constraint_penalties():
    cfg = replace(Config(), start=(70.0, 80.0), episode_steps=1)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, _, truncated, info = env.step(HOVER)

    assert truncated is True
    assert info["reward_terminal"] == (
        2 * cfg.reward_missing_user
    )


def test_service_curriculum_timeout_penalizes_each_missing_user():
    cfg = replace(Config(), episode_steps=1, service_steps_required=1)
    env = UAVThroughputEnv(
        cfg=cfg,
        fixed_user_positions=FIXED_USERS,
        objective="service",
    )
    env.reset()

    _, _, terminated, truncated, info = env.step(HOVER)

    assert not terminated
    assert truncated
    assert info["reward_terminal"] == 2 * cfg.reward_missing_user


def test_final_joint_success_receives_explicit_bonus():
    cfg = replace(Config(), start=(80.0, 80.0), episode_steps=1)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    _mark_both_users_served(env)

    _, _, terminated, truncated, info = env.step(HOVER)

    assert terminated is True
    assert truncated is False
    assert info["reward_terminal"] == 0.0


def test_service_phase_keeps_measured_throughput_in_learning_reward():
    env = UAVThroughputEnv(
        cfg=replace(Config(), start=(50.0, 50.0)),
        fixed_user_positions=FIXED_USERS,
    )
    env.reset()

    _, _, _, _, info = env.step(HOVER)

    assert info["return_phase"] is False
    assert info["raw_sum_throughput"] == info["throughputs"].sum()
    assert info["reward_throughput"] == pytest.approx(
        info["raw_sum_throughput"] * env.cfg.throughput_reward_multiplier
    )


def test_service_curriculum_uses_unmultiplied_throughput_for_constraint_learning():
    env = UAVThroughputEnv(
        cfg=replace(Config(), start=(50.0, 50.0)),
        fixed_user_positions=FIXED_USERS,
        objective="service",
    )
    env.reset()

    _, _, _, _, info = env.step(HOVER)

    assert info["reward_throughput"] == info["raw_sum_throughput"]


def test_throughput_remains_in_learning_reward_after_service():
    env = UAVThroughputEnv(
        cfg=replace(Config(), start=(70.0, 30.0)),
        fixed_user_positions=FIXED_USERS,
    )
    env.reset()
    _mark_both_users_served(env)

    _, _, _, _, info = env.step(UP)

    assert info["return_phase"] is False
    assert info["raw_sum_throughput"] > 0.0
    assert info["reward_throughput"] == pytest.approx(
        info["raw_sum_throughput"] * env.cfg.throughput_reward_multiplier
    )
    assert info["reward_locality"] == 0.0
    assert info["reward_service_progress"] == 0.0
    assert info["reward_service_completion"] == 0.0
    assert info["reward_endpoint_progress"] == 0.0


def test_step_completing_second_user_keeps_completion_reward_without_early_return():
    cfg = replace(Config(), start=(70.0, 30.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()
    env.service_steps[:] = [
        cfg.service_steps_required,
        cfg.service_steps_required - 1,
    ]
    env.users_served[:] = [True, False]

    _, _, _, _, completion = env.step(HOVER)
    _, _, _, _, returning = env.step(UP)

    assert completion["return_phase"] is False
    assert completion["reward_service_completion"] == cfg.reward_service_completion
    assert returning["return_phase"] is False
    assert returning["reward_service_completion"] == 0.0


def test_feasible_action_is_not_overridden_near_endpoint_deadline():
    cfg = replace(Config(), start=(70.0, 70.0), episode_steps=2)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, terminated, truncated, info = env.step(RIGHT)

    assert not terminated
    assert not truncated
    assert info["effective_action"] == RIGHT
    assert info["action_overridden"] is False
    assert np.allclose(env.uav_position, [80.0, 70.0])


def test_last_infeasible_action_is_projected_to_exact_endpoint():
    cfg = replace(Config(), start=(70.0, 80.0), episode_steps=1)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, _, truncated, info = env.step(HOVER)

    assert truncated
    assert info["proposed_action"] == HOVER
    assert info["effective_action"] == RIGHT
    assert info["action_overridden"] is True
    assert info["endpoint_constraint_active"] is True
    assert np.allclose(env.uav_position, cfg.end)


def test_joint_episode_runs_full_horizon_and_finishes_exactly_at_endpoint():
    env = UAVThroughputEnv(fixed_user_positions=FIXED_USERS)
    env.reset()

    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(HOVER)

    assert env.step_count == env.cfg.episode_steps
    assert len(env.trajectory) == env.cfg.episode_steps + 1
    assert len(env.throughput_history) == env.cfg.episode_steps
    assert np.allclose(env.uav_position, env.cfg.end)


def test_joint_reset_rejects_horizon_too_short_for_endpoint():
    cfg = replace(Config(), episode_steps=1)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)

    with pytest.raises(ValueError, match="cannot reach endpoint"):
        env.reset()


def test_service_phase_includes_time_and_signed_channel_shaping():
    cfg = replace(Config(), start=(0.0, 0.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, reward, _, _, info = env.step(UP)

    expected_channel = cfg.reward_channel_change * float(
        info["gain_ratios"].sum() - info["previous_gain_ratios"].sum()
    )
    assert info["reward_step"] == cfg.reward_step
    assert info["reward_channel_change"] == pytest.approx(expected_channel)
    components = (
        info["reward_throughput"]
        + info["reward_locality"]
        + info["reward_service_progress"]
        + info["reward_service_completion"]
        + info["reward_channel_change"]
        + info["reward_step"]
        + info["reward_boundary"]
        + info["reward_endpoint_progress"]
        + info["reward_endpoint_distance"]
        + info["reward_terminal"]
    )
    assert info["reward_unscaled"] == pytest.approx(components)
    assert reward == pytest.approx(components * cfg.reward_scale)


def test_timeout_remains_truncated_and_not_terminated():
    cfg = replace(Config(), start=(70.0, 80.0), episode_steps=1)
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, _, terminated, truncated, _ = env.step(HOVER)

    assert terminated is False
    assert truncated is True


def test_learning_reward_scales_the_reported_unscaled_components():
    cfg = replace(Config(), start=(50.0, 50.0))
    env = UAVThroughputEnv(cfg=cfg, fixed_user_positions=FIXED_USERS)
    env.reset()

    _, reward, _, _, info = env.step(HOVER)

    assert cfg.reward_scale == 0.001
    assert reward == pytest.approx(info["reward_unscaled"] * cfg.reward_scale)
