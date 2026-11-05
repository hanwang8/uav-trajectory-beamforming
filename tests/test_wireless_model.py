import numpy as np

from wireless_model import (
    Config,
    channel_vector,
    maximum_normalised_channel_power,
    mrt_beamformer,
    normalised_channel_power,
    optimise_power_split,
    random_users,
    sinr_and_throughput,
)


def test_throughput_first_reward_weights_are_explicit():
    cfg = Config()

    assert cfg.reward_scale == 0.001
    assert cfg.throughput_reward_multiplier == 10.0
    assert cfg.reward_first_entry == 100.0
    assert cfg.reward_service_progress == 300.0
    assert cfg.reward_service_completion == 3000.0
    assert cfg.reward_channel_change == 250.0
    assert cfg.reward_step == 0.0
    assert cfg.reward_boundary == -50.0
    assert cfg.reward_endpoint_progress == 0.0
    assert cfg.reward_endpoint_distance_per_step == 0.0
    assert cfg.reward_endpoint_dwell == 0.0
    assert cfg.reward_missing_user == -10000.0
    assert cfg.reward_missing_endpoint == -10000.0
    assert cfg.reward_joint_success == 0.0
    maximum_service_shaping = (
        cfg.num_users * cfg.reward_first_entry
        + cfg.num_users
        * cfg.service_steps_required
        * cfg.reward_service_progress
        + cfg.num_users * cfg.reward_service_completion
    )
    conservative_throughput_reward = (
        cfg.episode_steps * 8.0 * cfg.throughput_reward_multiplier
    )
    assert maximum_service_shaping < conservative_throughput_reward


def test_random_users_are_reproducible_and_inside_area():
    cfg = Config()
    first = random_users(cfg, seed=9123)
    second = random_users(cfg, seed=9123)

    assert first.shape == (cfg.num_users, 2)
    assert np.allclose(first, second)
    assert np.all((first >= 0.0) & (first <= cfg.area_size))


def test_channel_power_grows_when_uav_moves_closer():
    cfg = Config()
    user = np.array([50.0, 50.0])
    far = normalised_channel_power(channel_vector(cfg, (0.0, 0.0), user))
    near = normalised_channel_power(channel_vector(cfg, (50.0, 50.0), user))

    assert near > far
    assert near <= maximum_normalised_channel_power(cfg)
    assert np.isclose(near, maximum_normalised_channel_power(cfg), rtol=2e-5)


def test_mrt_and_power_split_respect_total_power_budget():
    cfg = Config()
    users = np.array([[30.0, 60.0], [70.0, 40.0]])
    channels = [channel_vector(cfg, (40.0, 50.0), user) for user in users]

    beam = mrt_beamformer(channels[0], 0.2)
    powers, beams, rates = optimise_power_split(cfg, channels, grid_points=21)

    assert np.isclose(np.vdot(beam, beam).real, 0.2)
    assert np.isclose(powers.sum(), cfg.transmit_power)
    assert sum(np.vdot(item, item).real for item in beams) <= cfg.transmit_power + 1e-12
    assert np.all(np.isfinite(rates))
    assert np.all(rates >= 0.0)


def test_sinr_and_throughput_are_finite_and_nonnegative():
    cfg = Config()
    users = np.array([[25.0, 65.0], [65.0, 35.0]])
    channels = [channel_vector(cfg, (30.0, 60.0), user) for user in users]
    beams = [mrt_beamformer(channel, cfg.transmit_power / 2) for channel in channels]

    sinr, rates = sinr_and_throughput(cfg, channels, beams)

    assert sinr.shape == (cfg.num_users,)
    assert rates.shape == (cfg.num_users,)
    assert np.all(np.isfinite(sinr))
    assert np.all(sinr >= 0.0)
    assert np.allclose(rates, np.log2(1.0 + sinr))
