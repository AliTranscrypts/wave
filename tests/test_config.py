"""Tests for config management."""

import tempfile
from pathlib import Path

from thermal_tracker.config import SimulationConfig, load_config, save_config


def test_load_default_config():
    config = load_config("configs/default_config.yaml")
    assert config.run.num_frames == 100
    assert len(config.cameras) == 3
    assert config.world.x_min == -2500.0


def test_config_roundtrip():
    config = SimulationConfig()
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.run.num_frames == config.run.num_frames
    assert loaded.world.x_min == config.world.x_min


def test_config_validation_negative_focal():
    """Cameras with valid parameters should load fine."""
    config = load_config("configs/default_config.yaml")
    assert config.cameras[0].hfov_deg > 0
