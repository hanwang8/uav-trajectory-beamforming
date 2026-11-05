from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import train_model
from train_model import (
    build_model,
    evaluate_policy,
    load_model,
    train_and_save,
    train_in_blocks,
)
from uav_environment import UAVThroughputEnv


FIXED_USERS = np.array([[25.0, 75.0], [70.0, 25.0]])


def test_training_module_exposes_no_teacher_pipeline():
    assert not hasattr(train_model, "ChannelGradientController")
    assert not hasattr(train_model, "ChannelSearchController")
    assert not hasattr(train_model, "distill_controller_policy")
    assert not hasattr(train_model, "pretrain_from_controller")


def test_build_model_limits_torch_to_one_cpu_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_model.torch,
        "set_num_threads",
        lambda count: calls.append(count),
    )

    build_model(UAVThroughputEnv(fixed_user_positions=FIXED_USERS), seed=1)

    assert calls == [1]


def test_build_model_uses_small_exploration_entropy():
    model = build_model(
        UAVThroughputEnv(fixed_user_positions=FIXED_USERS),
        seed=1,
    )

    assert model.ent_coef == 0.01


def test_default_curriculum_caps_warmup_and_prioritizes_formal_training():
    stages = train_model.curriculum_stages(1_000_000)

    assert [(stage.name, stage.timesteps) for stage in stages] == [
        ("service_discovery", 20_000),
        ("service_and_return", 30_000),
        ("formal_task", 950_000),
    ]
    assert [stage.service_steps_required for stage in stages] == [1, 1, 3]
    assert [stage.objective for stage in stages] == ["service", "joint", "joint"]
    assert stages[-1].entropy_start == 0.0
    assert stages[-1].entropy_end == 0.0


def test_short_curriculum_preserves_all_three_nonempty_stages():
    stages = train_model.curriculum_stages(64)

    assert sum(stage.timesteps for stage in stages) == 64
    assert all(stage.timesteps > 0 for stage in stages)


def test_build_model_uses_long_rollout_and_larger_network():
    model = build_model(
        UAVThroughputEnv(fixed_user_positions=FIXED_USERS),
        seed=1,
    )

    assert model.n_steps == 512
    assert model.batch_size == 128
    assert model.gamma == pytest.approx(0.995)
    assert model.policy.mlp_extractor.policy_net[0].out_features == 128


def test_entropy_schedule_decays_monotonically_to_zero():
    values = train_model.entropy_schedule(block_count=10, initial=0.01)

    assert len(values) == 10
    assert values[0] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(0.0)
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_entropy_schedule_for_one_block_is_zero():
    assert train_model.entropy_schedule(1, initial=0.01) == [0.0]


def test_reward_callback_records_completed_episode_returns():
    callback = train_model.RewardHistoryCallback()
    for reward, done in [(1.0, False), (2.0, False), (3.0, True)]:
        callback.locals = {
            "rewards": np.array([reward], dtype=float),
            "dones": np.array([done], dtype=bool),
        }
        assert callback._on_step()

    assert callback.episode_returns == [6.0]


def test_short_reward_training_creates_loadable_model(tmp_path: Path):
    env = UAVThroughputEnv(fixed_user_positions=FIXED_USERS)
    model_path = tmp_path / "ppo_throughput"
    model, history = train_and_save(
        env,
        model_path=model_path,
        total_timesteps=64,
        seed=4,
    )
    loaded = load_model(model_path, env)

    assert model is not None
    assert loaded is not None
    assert (tmp_path / "ppo_throughput.zip").exists()
    assert history["step_rewards"]


def test_policy_evaluation_reports_real_throughput_and_source():
    env = UAVThroughputEnv(fixed_user_positions=FIXED_USERS)
    model = build_model(env, seed=5)
    result = evaluate_policy(model, env, seed=5, deterministic=False)

    assert result["trajectory_source"] == "ppo_reward_trained"
    assert result["trajectory"].shape == (env.cfg.episode_steps + 1, 2)
    assert result["throughput_history"].shape == (env.cfg.episode_steps, 2)
    assert result["actions"].shape == (env.cfg.episode_steps,)
    assert result["effective_actions"].shape == (env.cfg.episode_steps,)
    assert result["step_count"] == env.cfg.episode_steps
    assert np.allclose(result["final_position"], env.cfg.end)
    assert result["action_override_count"] == int(
        np.count_nonzero(result["actions"] != result["effective_actions"])
    )
    assert np.isclose(
        result["sum_throughput"],
        result["individual_throughput"].sum(),
    )
    assert set(result["reward_component_history"][0]) == {
        "raw_sum_throughput",
        "return_phase",
        "endpoint_constraint_active",
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
    }
    assert result["mean_policy_entropy"] is not None
    assert result["mean_policy_entropy"] >= 0.0
    assert np.allclose(result["trajectory"][-1], env.cfg.end)


def test_stochastic_evaluation_seed_reproduces_policy_actions():
    class RandomPolicy:
        def predict(self, observation, deterministic):
            assert deterministic is False
            return np.array(np.random.randint(0, 5)), None

    first = evaluate_policy(
        RandomPolicy(),
        UAVThroughputEnv(fixed_user_positions=FIXED_USERS),
        seed=123,
        deterministic=False,
    )
    np.random.seed(999)
    np.random.random(20)
    second = evaluate_policy(
        RandomPolicy(),
        UAVThroughputEnv(fixed_user_positions=FIXED_USERS),
        seed=123,
        deterministic=False,
    )

    np.testing.assert_array_equal(first["actions"], second["actions"])


def test_train_in_blocks_records_two_hidden_user_sets(tmp_path: Path):
    fixed_sets = [
        np.array([[25.0, 75.0], [70.0, 25.0]]),
        np.array([[20.0, 30.0], [75.0, 70.0]]),
    ]
    env = UAVThroughputEnv(seed=12, fixed_user_positions=fixed_sets[0])
    _, history = train_in_blocks(
        env,
        model_path=tmp_path / "best_ppo",
        total_timesteps=128,
        block_size=64,
        evaluation_user_sets=fixed_sets,
        evaluation_episodes=1,
        seed=12,
    )

    assert history["training_steps"] == [64, 128]
    assert len(history["returns_by_set"]) == 2
    assert all(len(values) == 2 for values in history["returns_by_set"])
    assert len(history["returns_std_by_set"]) == 2
    assert all(len(values) == 2 for values in history["returns_std_by_set"])
    assert history["evaluation_mode"] == "stochastic"
    assert (tmp_path / "best_ppo.zip").exists()


class _FakeBlockModel:
    def __init__(self):
        self.version = 0
        self.save_count = 0
        self.saved_versions = []
        self.ent_coef = 0.01
        self.learned_entropy_coefficients = []

    def learn(self, total_timesteps, callback, reset_num_timesteps):
        self.learned_entropy_coefficients.append(self.ent_coef)
        self.version += 1
        return self

    def save(self, path):
        self.save_count += 1
        self.saved_versions.append(self.version)


def test_train_in_blocks_selects_checkpoint_from_target_set_only(
    tmp_path: Path,
    monkeypatch,
):
    target_users = FIXED_USERS
    diagnostic_users = np.array([[20.0, 30.0], [75.0, 70.0]])
    fake_model = _FakeBlockModel()
    calls = {"count": 0}
    expected_positions = [
        target_users,
        diagnostic_users,
        target_users,
        diagnostic_users,
    ]
    expected_seeds = [100076, 110076, 100140, 110140]
    scripted_results = [
        {
            "episode_reward": 10.0,
            "users_served": np.array([True, True]),
            "reached_end": True,
            "sum_throughput": 10.0,
        },
        {
            "episode_reward": -10.0,
            "users_served": np.array([False, False]),
            "reached_end": False,
            "sum_throughput": 1.0,
        },
        {
            "episode_reward": 20.0,
            "users_served": np.array([True, True]),
            "reached_end": False,
            "sum_throughput": 1000.0,
        },
        {
            "episode_reward": 999.0,
            "users_served": np.array([True, True]),
            "reached_end": True,
            "sum_throughput": 9999.0,
        },
    ]

    def fake_evaluate_policy(model, env, seed, deterministic):
        call_index = calls["count"]
        result = scripted_results[call_index]
        assert np.array_equal(
            env.fixed_user_positions,
            expected_positions[call_index],
        )
        assert seed == expected_seeds[call_index]
        calls["count"] += 1
        assert deterministic is False
        return result

    monkeypatch.setattr(train_model, "build_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(train_model, "evaluate_policy", fake_evaluate_policy)
    monkeypatch.setattr(
        train_model,
        "load_model",
        lambda *args, **kwargs: SimpleNamespace(version=fake_model.saved_versions[-1]),
    )

    selected_model, history = train_in_blocks(
        UAVThroughputEnv(fixed_user_positions=target_users),
        model_path=tmp_path / "target_best",
        total_timesteps=128,
        block_size=64,
        evaluation_user_sets=[target_users, diagnostic_users],
        evaluation_episodes=1,
        seed=12,
    )

    assert fake_model.save_count == 1
    assert fake_model.saved_versions == [1]
    assert fake_model.version == 2
    assert fake_model.learned_entropy_coefficients == [0.01, 0.0]
    assert selected_model.version == 1
    assert history["returns_by_set"] == [[10.0, 20.0], [-10.0, 999.0]]
    assert history["joint_success_rates"] == [1.0, 0.0]
    assert history["both_users_served_rates"] == [1.0, 1.0]
    assert history["endpoint_success_rates"] == [1.0, 0.0]
    assert history["mean_sum_throughput"] == [10.0, 1000.0]
    assert history["entropy_coefficients"] == [0.01, 0.0]
    assert history["checkpoint_scope"] == "target_user_set_only"
    assert history["training_user_positions"] == FIXED_USERS.tolist()
    assert history["checkpoint_user_positions"] == FIXED_USERS.tolist()
    assert history["evaluation_seeds_by_set"] == [
        [[100076], [100140]],
        [[110076], [110140]],
    ]


def test_train_in_blocks_rejects_random_training_users_before_build(
    tmp_path: Path,
    monkeypatch,
):
    def fail_build(*args, **kwargs):
        raise AssertionError("build_model must not be called")

    monkeypatch.setattr(train_model, "build_model", fail_build)

    with pytest.raises(ValueError, match="fixed training users"):
        train_in_blocks(
            UAVThroughputEnv(seed=12),
            model_path=tmp_path / "best_ppo",
            total_timesteps=1,
            block_size=1,
            evaluation_user_sets=[FIXED_USERS],
            evaluation_episodes=1,
        )


def test_train_in_blocks_rejects_mismatched_first_evaluation_set_before_build(
    tmp_path: Path,
    monkeypatch,
):
    def fail_build(*args, **kwargs):
        raise AssertionError("build_model must not be called")

    monkeypatch.setattr(train_model, "build_model", fail_build)

    with pytest.raises(ValueError, match="first evaluation set must match"):
        train_in_blocks(
            UAVThroughputEnv(fixed_user_positions=FIXED_USERS),
            model_path=tmp_path / "best_ppo",
            total_timesteps=1,
            block_size=1,
            evaluation_user_sets=[np.array([[1.0, 2.0], [3.0, 4.0]])],
            evaluation_episodes=1,
        )


class _FakeCurriculumModel:
    def __init__(self):
        self.build_count = 1
        self.ent_coef = 0.01
        self.learn_calls = []
        self.environment_calls = []
        self.version = 0

    def set_env(self, env):
        self.environment_calls.append(
            {
                "objective": env.objective,
                "service_steps_required": env.cfg.service_steps_required,
            }
        )

    def learn(self, total_timesteps, callback, reset_num_timesteps):
        self.learn_calls.append(total_timesteps)
        self.version += 1
        return self

    def save(self, path):
        return None


def test_curriculum_continues_one_model_and_validates_formal_task(
    tmp_path,
    monkeypatch,
):
    fake_model = _FakeCurriculumModel()
    validation_calls = []

    def fake_evaluate(model, env, seed, deterministic):
        validation_calls.append(
            {
                "objective": env.objective,
                "service_steps_required": env.cfg.service_steps_required,
            }
        )
        return {
            "episode_reward": 1.0,
            "users_served": np.array([True, True]),
            "reached_end": True,
            "sum_throughput": float(model.version),
            "mean_policy_entropy": 0.1,
        }

    monkeypatch.setattr(train_model, "build_model", lambda *a, **k: fake_model)
    monkeypatch.setattr(train_model, "evaluate_policy", fake_evaluate)
    monkeypatch.setattr(train_model, "load_model", lambda *a, **k: fake_model)

    _, history = train_model.train_curriculum(
        cfg=train_model.Config(),
        fixed_user_positions=FIXED_USERS,
        model_path=tmp_path / "curriculum",
        total_timesteps=9,
        evaluation_user_sets=[FIXED_USERS],
        evaluation_episodes=1,
        checkpoint_interval=50_000,
        seed=12,
    )

    assert fake_model.build_count == 1
    assert fake_model.learn_calls == [2, 3, 4]
    assert [call["objective"] for call in fake_model.environment_calls] == [
        "service",
        "joint",
        "joint",
    ]
    assert history["curriculum_stages"] == [
        {
            "name": "service_discovery",
            "start_step": 0,
            "end_step": 2,
            "service_steps_required": 1,
            "objective": "service",
        },
        {
            "name": "service_and_return",
            "start_step": 2,
            "end_step": 5,
            "service_steps_required": 1,
            "objective": "joint",
        },
        {
            "name": "formal_task",
            "start_step": 5,
            "end_step": 9,
            "service_steps_required": 3,
            "objective": "joint",
        },
    ]
    assert all(call["service_steps_required"] == 3 for call in validation_calls)
    assert all(call["objective"] == "joint" for call in validation_calls)
    assert history["selected_checkpoint_step"] == 9
