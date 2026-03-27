"""Sparse voxel grid for MVS tracking (Tasks 5.1.1-5.1.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class VoxelData:
    """Per-voxel payload."""
    log_odds: float = 0.0
    last_updated_frame: int = 0

    @property
    def probability(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.log_odds))


class VoxelGridConfig(BaseModel):
    """Configuration for the sparse voxel grid."""
    voxel_size: float = 1.0
    roi_min: list[float] = Field(default=[-250.0, -250.0, 50.0], min_length=3, max_length=3)
    roi_max: list[float] = Field(default=[250.0, 250.0, 200.0], min_length=3, max_length=3)
    occupancy_threshold: float = 0.8
    log_odds_clamp: float = 10.0
    temporal_decay_rate: float = 0.1
    pruning_threshold: float = 0.05
    max_active_voxels: int | None = None


class SparseVoxelGrid:
    """Hash-map-based sparse voxel grid."""

    def __init__(self, config: VoxelGridConfig):
        self.config = config
        self._voxels: dict[tuple[int, int, int], VoxelData] = {}
        self._roi_min = np.array(config.roi_min, dtype=np.float64)
        self._roi_max = np.array(config.roi_max, dtype=np.float64)

    def world_to_grid(self, point: np.ndarray) -> tuple[int, int, int]:
        """Convert world coordinates to integer grid indices."""
        p = np.asarray(point, dtype=np.float64)
        idx = np.floor((p - self._roi_min) / self.config.voxel_size).astype(int)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def grid_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """Return the world-space center of the voxel."""
        return self._roi_min + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * self.config.voxel_size

    def get(self, ix: int, iy: int, iz: int) -> VoxelData | None:
        return self._voxels.get((ix, iy, iz))

    def get_or_create(self, ix: int, iy: int, iz: int) -> VoxelData:
        key = (ix, iy, iz)
        if key not in self._voxels:
            self._voxels[key] = VoxelData()
        return self._voxels[key]

    def set(self, ix: int, iy: int, iz: int, data: VoxelData) -> None:
        self._voxels[(ix, iy, iz)] = data

    def get_occupied(self, threshold: float | None = None) -> list[tuple[tuple[int, int, int], VoxelData]]:
        """Return voxels with probability above threshold."""
        thresh = threshold if threshold is not None else self.config.occupancy_threshold
        return [
            (k, v) for k, v in self._voxels.items()
            if v.probability > thresh
        ]

    def active_voxels(self) -> Iterator[tuple[tuple[int, int, int], VoxelData]]:
        yield from self._voxels.items()

    def num_active(self) -> int:
        return len(self._voxels)

    def clear(self) -> None:
        self._voxels.clear()

    def get_neighborhood(self, ix: int, iy: int, iz: int, connectivity: int = 26) -> list[tuple[int, int, int]]:
        """Return neighbor grid indices.

        connectivity=6: face neighbors only.
        connectivity=26: full cube neighbors.
        """
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if connectivity == 6 and abs(dx) + abs(dy) + abs(dz) != 1:
                        continue
                    neighbors.append((ix + dx, iy + dy, iz + dz))
        return neighbors

    def enumerate_region(self, min_grid: tuple[int, int, int],
                         max_grid: tuple[int, int, int]) -> Iterator[tuple[int, int, int]]:
        """Iterate over all grid cells in a bounding box."""
        for ix in range(min_grid[0], max_grid[0] + 1):
            for iy in range(min_grid[1], max_grid[1] + 1):
                for iz in range(min_grid[2], max_grid[2] + 1):
                    yield (ix, iy, iz)

    # -----------------------------------------------------------------------
    # Temporal decay and pruning (Task 5.1.2)
    # -----------------------------------------------------------------------

    def apply_temporal_decay(self, current_frame: int) -> None:
        """Decay log-odds of voxels not recently updated."""
        decay_rate = self.config.temporal_decay_rate
        for voxel in self._voxels.values():
            frames_since = current_frame - voxel.last_updated_frame
            if frames_since > 0:
                voxel.log_odds *= (1.0 - decay_rate) ** frames_since
                voxel.last_updated_frame = current_frame

    def prune(self) -> int:
        """Remove voxels with |log_odds| below pruning threshold. Returns count pruned."""
        thresh = self.config.pruning_threshold
        to_remove = [k for k, v in self._voxels.items() if abs(v.log_odds) < thresh]
        for k in to_remove:
            del self._voxels[k]
        return len(to_remove)
