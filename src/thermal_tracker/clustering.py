"""Clustering occupied voxels into candidate detections (Tasks 5.3.1-5.3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field
from scipy import ndimage
from sklearn.cluster import DBSCAN

from thermal_tracker.voxel_grid import SparseVoxelGrid


class ClusteringConfig(BaseModel):
    """Clustering parameters."""
    method: str = "connected_components"  # or "dbscan"
    connectivity: int = 26
    min_cluster_size: int = 1
    max_cluster_size: int = 50
    dbscan_eps_factor: float = 1.5  # eps = factor * voxel_size
    dbscan_min_samples: int = 1
    weighted_centroid: bool = True


@dataclass
class VoxelCluster:
    """A cluster of occupied voxels representing a candidate detection."""
    cluster_id: int
    voxel_indices: list[tuple[int, int, int]]
    voxel_positions: np.ndarray  # (K, 3)
    centroid: np.ndarray  # (3,)
    size: int
    max_probability: float
    bounding_box: tuple[np.ndarray, np.ndarray]  # (min_corner, max_corner)


def cluster_occupied_voxels(
    voxel_grid: SparseVoxelGrid,
    config: ClusteringConfig,
) -> list[VoxelCluster]:
    """Group occupied voxels into spatially connected clusters."""
    if config.method == "dbscan":
        return _cluster_dbscan(voxel_grid, config)
    return _cluster_connected_components(voxel_grid, config)


def _cluster_connected_components(
    voxel_grid: SparseVoxelGrid,
    config: ClusteringConfig,
) -> list[VoxelCluster]:
    """Connected-component labeling via scipy on a dense subgrid."""
    occupied = voxel_grid.get_occupied()
    if not occupied:
        return []

    indices = [k for k, _ in occupied]
    data = {k: v for k, v in occupied}

    # Compute bounding box of occupied voxels
    arr_idx = np.array(indices)
    grid_min = arr_idx.min(axis=0)
    grid_max = arr_idx.max(axis=0)
    shape = tuple(grid_max - grid_min + 1)

    # Build dense boolean array
    dense = np.zeros(shape, dtype=bool)
    for idx in indices:
        local = tuple(np.array(idx) - grid_min)
        dense[local] = True

    # Connectivity structure
    if config.connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif config.connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    else:  # 26
        struct = ndimage.generate_binary_structure(3, 3)

    labels, num_features = ndimage.label(dense, structure=struct)

    clusters = []
    for label_id in range(1, num_features + 1):
        locs = np.argwhere(labels == label_id)
        voxel_idx = [tuple(loc + grid_min) for loc in locs]

        positions = np.array([voxel_grid.grid_to_world(*vi) for vi in voxel_idx])
        probs = np.array([data[vi].probability for vi in voxel_idx if vi in data])

        if len(voxel_idx) < config.min_cluster_size or len(voxel_idx) > config.max_cluster_size:
            continue

        # Centroid
        if config.weighted_centroid and len(probs) == len(positions):
            weights = probs / probs.sum()
            centroid = (positions * weights[:, None]).sum(axis=0)
        else:
            centroid = positions.mean(axis=0)

        clusters.append(VoxelCluster(
            cluster_id=label_id,
            voxel_indices=voxel_idx,
            voxel_positions=positions,
            centroid=centroid,
            size=len(voxel_idx),
            max_probability=float(probs.max()) if len(probs) > 0 else 0.0,
            bounding_box=(positions.min(axis=0), positions.max(axis=0)),
        ))

    return clusters


def _cluster_dbscan(
    voxel_grid: SparseVoxelGrid,
    config: ClusteringConfig,
) -> list[VoxelCluster]:
    """DBSCAN-based clustering of occupied voxels."""
    occupied = voxel_grid.get_occupied()
    if not occupied:
        return []

    indices = [k for k, _ in occupied]
    data_map = {k: v for k, v in occupied}
    positions = np.array([voxel_grid.grid_to_world(*idx) for idx in indices])
    probs = np.array([v.probability for _, v in occupied])

    eps = config.dbscan_eps_factor * voxel_grid.config.voxel_size
    db = DBSCAN(eps=eps, min_samples=config.dbscan_min_samples).fit(positions)

    clusters = []
    for label_id in set(db.labels_):
        if label_id == -1:
            continue
        mask = db.labels_ == label_id
        cluster_positions = positions[mask]
        cluster_probs = probs[mask]
        cluster_indices = [indices[i] for i in np.where(mask)[0]]

        if len(cluster_indices) < config.min_cluster_size or len(cluster_indices) > config.max_cluster_size:
            continue

        if config.weighted_centroid:
            weights = cluster_probs / cluster_probs.sum()
            centroid = (cluster_positions * weights[:, None]).sum(axis=0)
        else:
            centroid = cluster_positions.mean(axis=0)

        clusters.append(VoxelCluster(
            cluster_id=int(label_id),
            voxel_indices=cluster_indices,
            voxel_positions=cluster_positions,
            centroid=centroid,
            size=len(cluster_indices),
            max_probability=float(cluster_probs.max()),
            bounding_box=(cluster_positions.min(axis=0), cluster_positions.max(axis=0)),
        ))

    return clusters
