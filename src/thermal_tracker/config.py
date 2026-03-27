"""Configuration management (Task 2.1.2)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from thermal_tracker.camera import CameraConfig
from thermal_tracker.eagle import EagleConfig
from thermal_tracker.rendering import RenderingConfig
from thermal_tracker.world import WorldConfig


class SimulationRunConfig(BaseModel):
    """Runtime parameters for a simulation run."""
    num_frames: int = 100
    dt: float = 0.1  # seconds per frame
    random_seed: int = 42


class SimulationConfig(BaseModel):
    """Top-level configuration composing all sub-configs."""
    world: WorldConfig = Field(default_factory=WorldConfig)
    eagle: EagleConfig = Field(default_factory=EagleConfig)
    eagles: list[EagleConfig] = Field(default_factory=list)  # Multi-eagle support
    cameras: list[CameraConfig] = Field(default_factory=list)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    run: SimulationRunConfig = Field(default_factory=SimulationRunConfig)


def load_config(path: str | Path) -> SimulationConfig:
    """Load simulation config from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SimulationConfig.model_validate(data)


def save_config(config: SimulationConfig, path: str | Path) -> None:
    """Save simulation config to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
