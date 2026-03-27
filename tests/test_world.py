"""Tests for world module."""

import numpy as np
import yaml

from thermal_tracker.world import WorldConfig, is_within_bounds, clamp_to_bounds


def test_world_config_defaults():
    config = WorldConfig()
    assert config.x_min == -2500.0
    assert config.z_max == 500.0
    assert np.array_equal(config.bounds_min, [-2500, -2500, 0])


def test_world_config_yaml_roundtrip():
    config = WorldConfig(x_min=-100, x_max=100)
    data = config.model_dump()
    yaml_str = yaml.dump(data)
    loaded = yaml.safe_load(yaml_str)
    config2 = WorldConfig.model_validate(loaded)
    assert config2.x_min == config.x_min
    assert config2.x_max == config.x_max


def test_is_within_bounds():
    config = WorldConfig(x_min=-10, x_max=10, y_min=-10, y_max=10, z_min=0, z_max=20)
    assert is_within_bounds(np.array([0, 0, 10]), config)
    assert is_within_bounds(np.array([-10, -10, 0]), config)  # boundary
    assert not is_within_bounds(np.array([11, 0, 10]), config)
    assert not is_within_bounds(np.array([0, 0, -1]), config)


def test_clamp_to_bounds():
    config = WorldConfig(x_min=-10, x_max=10, y_min=-10, y_max=10, z_min=0, z_max=20)
    assert np.array_equal(clamp_to_bounds(np.array([0, 0, 10]), config), [0, 0, 10])
    result = clamp_to_bounds(np.array([100, -100, -5]), config)
    assert np.array_equal(result, [10, -10, 0])
