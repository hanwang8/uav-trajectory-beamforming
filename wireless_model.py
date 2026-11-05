"""Wireless channel and transmission helpers for the UAV downlink."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class Config:
    """All fixed simulation parameters."""

    area_size: float = 100.0
    height: float = 50.0
    start: tuple[float, float] = (0.0, 0.0)
    end: tuple[float, float] = (80.0, 80.0)
    num_users: int = 2
    num_antennas: int = 8
    transmit_power: float = 0.5
    noise_power_db: float = -100.0
    frequency: float = 2.4e9
    path_loss_exponent: float = 2.5
    episode_steps: int = 200
    step_seconds: float = 1.0
    move_distance: float = 10.0
    locality_radius: float = 10.0
    service_steps_required: int = 3
    endpoint_safety_steps: int = 20
    reward_scale: float = 0.001
    throughput_reward_multiplier: float = 10.0
    reward_first_entry: float = 100.0
    reward_service_progress: float = 300.0
    reward_service_completion: float = 3000.0
    reward_channel_change: float = 250.0
    reward_step: float = 0.0
    reward_boundary: float = -50.0
    reward_endpoint_progress: float = 0.0
    reward_endpoint_distance_per_step: float = 0.0
    reward_endpoint_dwell: float = 0.0
    reward_missing_user: float = -10000.0
    reward_missing_endpoint: float = -10000.0
    reward_joint_success: float = 0.0

    @property
    def wavelength(self) -> float:
        return 299_792_458.0 / self.frequency

    @property
    def reference_gain(self) -> float:
        return (self.wavelength / (4.0 * pi)) ** 2

    @property
    def noise_power_watts(self) -> float:
        return 10.0 ** (self.noise_power_db / 10.0)


def random_users(cfg: Config, seed: int | None = None) -> np.ndarray:
    """Generate ground-user XY coordinates; the environment keeps them hidden."""

    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, cfg.area_size, size=(cfg.num_users, 2))


def _antenna_positions(cfg: Config, uav_xy: tuple[float, float] | np.ndarray) -> np.ndarray:
    """Return ULA element positions, centred on the UAV and aligned with Y."""

    spacing = cfg.wavelength / 2.0
    offsets = (np.arange(cfg.num_antennas) - (cfg.num_antennas - 1) / 2.0) * spacing
    positions = np.zeros((cfg.num_antennas, 3), dtype=float)
    positions[:, 0] = float(uav_xy[0])
    positions[:, 1] = float(uav_xy[1]) + offsets
    positions[:, 2] = cfg.height
    return positions


def channel_vector(
    cfg: Config,
    uav_xy: tuple[float, float] | np.ndarray,
    user_xy: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """Compute the complex LoS channel from all UAV antennas to one user."""

    antennas = _antenna_positions(cfg, uav_xy)
    user = np.array([float(user_xy[0]), float(user_xy[1]), 0.0])
    distances = np.linalg.norm(antennas - user, axis=1)
    amplitude = np.sqrt(cfg.reference_gain / distances**cfg.path_loss_exponent)
    phase = np.exp(-1j * 2.0 * pi * distances / cfg.wavelength)
    return amplitude * phase


def normalised_channel_power(channel: np.ndarray) -> float:
    """Average channel power per antenna, independent of antenna count."""

    values = np.asarray(channel)
    return float(np.mean(np.abs(values) ** 2))


def maximum_normalised_channel_power(cfg: Config) -> float:
    """Ideal upper bound when the UAV array centre is directly above a user."""

    return cfg.reference_gain / cfg.height**cfg.path_loss_exponent


def mrt_beamformer(channel: np.ndarray, power: float) -> np.ndarray:
    """Maximum-ratio transmission beam with exactly the requested power."""

    h = np.asarray(channel, dtype=np.complex128)
    if power <= 0.0:
        return np.zeros_like(h)
    norm = np.linalg.norm(h)
    if norm == 0.0:
        return np.zeros_like(h)
    return np.sqrt(power) * h / norm


def sinr_and_throughput(
    cfg: Config,
    channels: list[np.ndarray],
    beamformers: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-user downlink SINR and normalized throughput."""

    if len(channels) != len(beamformers):
        raise ValueError("channels and beamformers must have the same length")

    sinr_values = np.zeros(len(channels), dtype=float)
    for user_index, channel in enumerate(channels):
        received = np.array(
            [abs(np.vdot(channel, beam)) ** 2 for beam in beamformers],
            dtype=float,
        )
        desired = received[user_index]
        interference = float(received.sum() - desired)
        sinr_values[user_index] = desired / (interference + cfg.noise_power_watts)

    rates = np.log2(1.0 + sinr_values)
    return sinr_values, rates


def optimise_power_split(
    cfg: Config,
    channels: list[np.ndarray],
    grid_points: int = 31,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Choose a two-user MRT power split that maximizes instantaneous sum-rate."""

    if len(channels) != 2:
        raise ValueError("This implementation expects exactly two users")
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")

    best_sum_rate = -np.inf
    best_powers = np.array([cfg.transmit_power / 2.0] * 2)
    best_beams: list[np.ndarray] = []
    best_rates = np.zeros(2, dtype=float)

    for first_power in np.linspace(0.0, cfg.transmit_power, grid_points):
        powers = np.array([first_power, cfg.transmit_power - first_power])
        beams = [mrt_beamformer(channel, power) for channel, power in zip(channels, powers)]
        _, rates = sinr_and_throughput(cfg, channels, beams)
        sum_rate = float(rates.sum())
        if sum_rate > best_sum_rate:
            best_sum_rate = sum_rate
            best_powers = powers
            best_beams = beams
            best_rates = rates

    return best_powers, best_beams, best_rates
