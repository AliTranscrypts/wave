"""Tests for sparse voxel grid."""

import numpy as np

from thermal_tracker.voxel_grid import SparseVoxelGrid, VoxelData, VoxelGridConfig


def _make_grid() -> SparseVoxelGrid:
    config = VoxelGridConfig(
        voxel_size=1.0,
        roi_min=[-10.0, -10.0, -10.0],
        roi_max=[10.0, 10.0, 10.0],
    )
    return SparseVoxelGrid(config)


def test_world_to_grid_roundtrip():
    grid = _make_grid()
    p = np.array([5.5, -3.2, 7.8])
    ix, iy, iz = grid.world_to_grid(p)
    center = grid.grid_to_world(ix, iy, iz)
    # Center should be within voxel_size/2 of original point in each axis
    assert np.all(np.abs(center - p) < grid.config.voxel_size)


def test_insert_and_retrieve():
    grid = _make_grid()
    for i in range(100):
        grid.get_or_create(i, 0, 0).log_odds = 5.0
    assert grid.num_active() == 100


def test_get_occupied():
    grid = _make_grid()
    grid.get_or_create(0, 0, 0).log_odds = 5.0  # high probability
    grid.get_or_create(1, 0, 0).log_odds = -5.0  # low probability
    grid.get_or_create(2, 0, 0).log_odds = 0.0  # 0.5 probability

    occupied = grid.get_occupied(threshold=0.8)
    assert len(occupied) == 1
    assert occupied[0][0] == (0, 0, 0)


def test_neighborhood_6():
    grid = _make_grid()
    neighbors = grid.get_neighborhood(5, 5, 5, connectivity=6)
    assert len(neighbors) == 6


def test_neighborhood_26():
    grid = _make_grid()
    neighbors = grid.get_neighborhood(5, 5, 5, connectivity=26)
    assert len(neighbors) == 26


def test_temporal_decay():
    grid = _make_grid()
    voxel = grid.get_or_create(0, 0, 0)
    voxel.log_odds = 5.0
    voxel.last_updated_frame = 0

    grid.apply_temporal_decay(current_frame=10)
    assert abs(voxel.log_odds) < 5.0  # Should have decayed


def test_prune():
    grid = _make_grid()
    grid.get_or_create(0, 0, 0).log_odds = 0.01  # Below threshold
    grid.get_or_create(1, 0, 0).log_odds = 5.0  # Above threshold
    pruned = grid.prune()
    assert pruned == 1
    assert grid.num_active() == 1
