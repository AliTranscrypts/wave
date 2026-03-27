"""Tests for eagle motion module."""

import numpy as np

from thermal_tracker.eagle import (
    EagleConfig, EagleState, LissajousMotion, RandomWalkMotion,
    SplineMotion, create_motion_generator, MotionType,
)
from thermal_tracker.world import WorldConfig


def test_lissajous_deterministic():
    config = EagleConfig(motion_type=MotionType.LISSAJOUS)
    world = WorldConfig()
    gen = LissajousMotion(config, world)
    state = gen.reset(42)

    # Generate two trajectories with same seed — should be identical
    gen2 = LissajousMotion(config, world)
    state2 = gen2.reset(42)

    for _ in range(50):
        state = gen.step(state, 0.1)
        state2 = gen2.step(state2, 0.1)
        assert np.allclose(state.position, state2.position)


def test_lissajous_matches_analytical():
    config = EagleConfig(
        motion_type=MotionType.LISSAJOUS,
        lissajous_amplitudes=[100, 200, 50],
        lissajous_frequencies=[0.1, 0.2, 0.05],
        lissajous_phases=[0, 0, 0],
        lissajous_z_center=100.0,
    )
    world = WorldConfig()
    gen = LissajousMotion(config, world)
    state = gen.reset(0)

    t = 1.0
    state = gen.step(state, t)
    expected = np.array([
        100 * np.sin(0.1 * t),
        200 * np.sin(0.2 * t),
        100 + 50 * np.sin(0.05 * t),
    ])
    assert np.allclose(state.position, expected, atol=1e-10)


def test_random_walk_bounds():
    config = EagleConfig(
        motion_type=MotionType.RANDOM_WALK,
        max_speed=20.0,
        altitude_range=[50, 200],
        initial_position=[0, 0, 100],
    )
    world = WorldConfig(x_min=-500, x_max=500, y_min=-500, y_max=500, z_min=0, z_max=300)
    gen = RandomWalkMotion(config, world)
    state = gen.reset(42)

    for _ in range(1000):
        state = gen.step(state, 0.1)
        assert state.position[0] >= world.x_min
        assert state.position[0] <= world.x_max
        assert state.position[2] >= config.altitude_range[0]
        assert state.position[2] <= config.altitude_range[1]
        assert np.linalg.norm(state.velocity) <= config.max_speed + 1e-6


def test_random_walk_reproducible():
    config = EagleConfig(motion_type=MotionType.RANDOM_WALK)
    world = WorldConfig()
    gen1 = RandomWalkMotion(config, world)
    gen2 = RandomWalkMotion(config, world)

    s1 = gen1.reset(123)
    s2 = gen2.reset(123)
    for _ in range(100):
        s1 = gen1.step(s1, 0.1)
        s2 = gen2.step(s2, 0.1)
    assert np.allclose(s1.position, s2.position)


def test_trajectory_generation():
    config = EagleConfig(motion_type=MotionType.LISSAJOUS)
    world = WorldConfig()
    gen = create_motion_generator(config, world)
    traj = gen.generate_trajectory(50, 0.1, seed=42)
    assert traj.shape == (50, 7)
    assert traj[0, 0] == 0.0  # first timestamp
