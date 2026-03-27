"""World representation and bounds utilities (Task 2.1.1)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class WorldConfig(BaseModel):
    """Canonical world coordinate frame and bounding volume.

    Right-handed coordinate system with Z-up.
    All distances in meters, temperatures in Celsius at config layer.
    """

    origin: list[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    x_min: float = -2500.0
    x_max: float = 2500.0
    y_min: float = -2500.0
    y_max: float = 2500.0
    z_min: float = 0.0
    z_max: float = 500.0

    @property
    def bounds_min(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.z_min])

    @property
    def bounds_max(self) -> np.ndarray:
        return np.array([self.x_max, self.y_max, self.z_max])


def is_within_bounds(point: np.ndarray, config: WorldConfig) -> bool:
    """Check if a 3D point is within the world bounds."""
    p = np.asarray(point)
    return bool(
        np.all(p >= config.bounds_min) and np.all(p <= config.bounds_max)
    )


def clamp_to_bounds(point: np.ndarray, config: WorldConfig) -> np.ndarray:
    """Project a point onto the nearest boundary face if outside."""
    return np.clip(np.asarray(point, dtype=np.float64), config.bounds_min, config.bounds_max)
