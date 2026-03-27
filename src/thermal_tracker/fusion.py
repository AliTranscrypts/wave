"""Occupancy fusion: Bayesian and space carving (Tasks 5.2.1-5.2.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from thermal_tracker.voxel_grid import SparseVoxelGrid

if TYPE_CHECKING:
    from thermal_tracker.camera import Camera


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FusionConfig(BaseModel):
    """Bayesian fusion parameters."""
    detection_threshold_sigma: float = 2.0
    p_hot_given_occupied: float = 0.9
    p_hot_given_empty: float = 0.05
    min_cameras_for_update: int = 2

    @property
    def log_lr_hot(self) -> float:
        """Log-likelihood ratio for a hot observation."""
        return float(np.log(self.p_hot_given_occupied / self.p_hot_given_empty))

    @property
    def log_lr_cold(self) -> float:
        """Log-likelihood ratio for a cold observation."""
        return float(np.log((1 - self.p_hot_given_occupied) / (1 - self.p_hot_given_empty)))


class SpaceCarvingConfig(BaseModel):
    """Space carving parameters."""
    min_cameras_vote: int = 2
    allow_vetoing: bool = True
    occupied_log_odds: float = 5.0
    empty_log_odds: float = -5.0


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_thermal_image(image: np.ndarray, config: FusionConfig) -> np.ndarray:
    """Threshold thermal image to produce a binary hot mask.

    Uses robust estimators (median + MAD) for sky statistics.
    Returns boolean mask where True = hot pixel.
    """
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    # Scale MAD to approximate std: std ≈ 1.4826 * MAD
    robust_std = 1.4826 * mad if mad > 0 else 1.0
    threshold = median + config.detection_threshold_sigma * robust_std
    return image > threshold


# ---------------------------------------------------------------------------
# Bayesian fusion (Task 5.2.1)
# ---------------------------------------------------------------------------

def fuse_frame(
    voxel_grid: SparseVoxelGrid,
    cameras: list[Camera],
    images: list[np.ndarray],
    fusion_config: FusionConfig,
    current_frame: int,
    search_region: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> None:
    """Per-frame Bayesian occupancy fusion.

    Vectorized: batch-projects all active voxel centers through each camera.
    """
    # Determine active voxel set
    if search_region is not None:
        min_g, max_g = search_region
        voxel_indices = list(voxel_grid.enumerate_region(min_g, max_g))
    else:
        # Use full ROI — compute grid extent
        cfg = voxel_grid.config
        roi_min = np.array(cfg.roi_min)
        roi_max = np.array(cfg.roi_max)
        grid_min = np.floor((roi_min - roi_min) / cfg.voxel_size).astype(int)
        grid_max = np.floor((roi_max - roi_min) / cfg.voxel_size).astype(int)
        voxel_indices = list(voxel_grid.enumerate_region(
            tuple(grid_min), tuple(grid_max)
        ))

    if not voxel_indices:
        return

    N = len(voxel_indices)
    # Compute world-space centers for all voxels (N, 3)
    centers = np.array([
        voxel_grid.grid_to_world(ix, iy, iz) for ix, iy, iz in voxel_indices
    ])

    # Preprocess images to hot masks
    hot_masks = [preprocess_thermal_image(img, fusion_config) for img in images]

    # Accumulate log-likelihood per voxel across cameras
    total_log_lr = np.zeros(N, dtype=np.float64)
    num_visible = np.zeros(N, dtype=np.int32)

    log_lr_hot = fusion_config.log_lr_hot
    log_lr_cold = fusion_config.log_lr_cold

    for cam, mask in zip(cameras, hot_masks):
        # Batch project
        pixels, visible = cam.project(centers)
        H, W = mask.shape

        # Valid projections: visible and within bounds
        u = np.round(pixels[:, 0]).astype(int)
        v = np.round(pixels[:, 1]).astype(int)
        in_bounds = visible & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Read hot mask at projected locations
        is_hot = np.zeros(N, dtype=bool)
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) > 0:
            is_hot[valid_idx] = mask[v[valid_idx], u[valid_idx]]

        # Compute log-likelihood contribution
        ll = np.where(is_hot, log_lr_hot, log_lr_cold)
        # Only count cameras where voxel is visible
        ll[~in_bounds] = 0.0
        total_log_lr += ll
        num_visible += in_bounds.astype(np.int32)

    # Update voxels that have sufficient camera coverage
    clamp = voxel_grid.config.log_odds_clamp
    for i in range(N):
        if num_visible[i] >= fusion_config.min_cameras_for_update:
            ix, iy, iz = voxel_indices[i]
            voxel = voxel_grid.get_or_create(ix, iy, iz)
            voxel.log_odds += total_log_lr[i]
            voxel.log_odds = np.clip(voxel.log_odds, -clamp, clamp)
            voxel.last_updated_frame = current_frame


# ---------------------------------------------------------------------------
# Space carving (Task 5.2.2)
# ---------------------------------------------------------------------------

def space_carve_frame(
    voxel_grid: SparseVoxelGrid,
    cameras: list[Camera],
    hot_masks: list[np.ndarray],
    config: SpaceCarvingConfig,
    current_frame: int,
    search_region: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> None:
    """Space carving via vote accumulation."""
    cfg = voxel_grid.config
    if search_region is not None:
        min_g, max_g = search_region
        voxel_indices = list(voxel_grid.enumerate_region(min_g, max_g))
    else:
        roi_min = np.array(cfg.roi_min)
        roi_max = np.array(cfg.roi_max)
        grid_min = np.floor((roi_min - roi_min) / cfg.voxel_size).astype(int)
        grid_max = np.floor((roi_max - roi_min) / cfg.voxel_size).astype(int)
        voxel_indices = list(voxel_grid.enumerate_region(tuple(grid_min), tuple(grid_max)))

    if not voxel_indices:
        return

    N = len(voxel_indices)
    centers = np.array([voxel_grid.grid_to_world(*idx) for idx in voxel_indices])

    vote_count = np.zeros(N, dtype=np.int32)
    veto_count = np.zeros(N, dtype=np.int32)

    for cam, mask in zip(cameras, hot_masks):
        pixels, visible = cam.project(centers)
        H, W = mask.shape
        u = np.round(pixels[:, 0]).astype(int)
        v = np.round(pixels[:, 1]).astype(int)
        in_bounds = visible & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) > 0:
            is_hot = mask[v[valid_idx], u[valid_idx]]
            vote_count[valid_idx] += is_hot.astype(np.int32)
            veto_count[valid_idx] += (~is_hot).astype(np.int32)

    # Decision rule
    for i in range(N):
        ix, iy, iz = voxel_indices[i]
        if vote_count[i] >= config.min_cameras_vote:
            if not config.allow_vetoing or veto_count[i] == 0:
                voxel = voxel_grid.get_or_create(ix, iy, iz)
                voxel.log_odds = config.occupied_log_odds
                voxel.last_updated_frame = current_frame
            else:
                voxel = voxel_grid.get(ix, iy, iz)
                if voxel is not None:
                    voxel.log_odds = config.empty_log_odds
                    voxel.last_updated_frame = current_frame
        elif vote_count[i] == 0 and veto_count[i] > 0:
            voxel = voxel_grid.get(ix, iy, iz)
            if voxel is not None:
                voxel.log_odds = config.empty_log_odds
                voxel.last_updated_frame = current_frame
