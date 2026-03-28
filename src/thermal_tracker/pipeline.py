"""End-to-end tracking pipeline and evaluation (Tasks 5.5-5.6)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from pydantic import BaseModel, Field

from thermal_tracker.camera import Camera
from thermal_tracker.clustering import ClusteringConfig, VoxelCluster, cluster_occupied_voxels
from thermal_tracker.engine import SimulationEngine
from thermal_tracker.fusion import FusionConfig, SpaceCarvingConfig, fuse_frame, space_carve_frame, preprocess_thermal_image
from thermal_tracker.rendering import FrameBundle
from thermal_tracker.tracking import Track, TrackState, Tracker, TrackingConfig
from thermal_tracker.voxel_grid import SparseVoxelGrid, VoxelGridConfig


# ---------------------------------------------------------------------------
# Tracking pipeline (Task 5.5.1)
# ---------------------------------------------------------------------------

@dataclass
class TrackingResult:
    """Results from processing one frame."""
    frame_index: int
    timestamp: float
    clusters: list[VoxelCluster]
    active_tracks: list[Track]
    primary_track: Track | None = None
    estimated_position: np.ndarray | None = None
    num_active_voxels: int = 0
    processing_time_ms: float = 0.0


class TrackingPipeline:
    """Composes fusion, clustering, and tracking into a single pipeline."""

    def __init__(
        self,
        cameras: list[Camera],
        voxel_config: VoxelGridConfig,
        fusion_config: FusionConfig,
        clustering_config: ClusteringConfig,
        tracking_config: TrackingConfig,
        fusion_method: str = "bayesian",
        space_carving_config: SpaceCarvingConfig | None = None,
    ):
        self.cameras = cameras
        self.voxel_grid = SparseVoxelGrid(voxel_config)
        self.fusion_config = fusion_config
        self.clustering_config = clustering_config
        self.tracker = Tracker(tracking_config)
        self.fusion_method = fusion_method
        self.space_carving_config = space_carving_config or SpaceCarvingConfig()
        self._prune_interval = 10

    def process_frame(self, frame_bundle: FrameBundle) -> TrackingResult:
        """Process one frame through the full pipeline."""
        t0 = time.time()

        images = list(frame_bundle.camera_images.values())
        frame_idx = frame_bundle.frame_index

        # Search region: use full ROI for multi-eagle, or narrow for single-eagle
        # Full ROI at 140K voxels runs at ~56ms/frame — no need to narrow
        search_region = None  # Always scan full ROI for multi-target support

        # Fusion
        if self.fusion_method == "space_carving":
            hot_masks = [preprocess_thermal_image(img, self.fusion_config) for img in images]
            space_carve_frame(
                self.voxel_grid, self.cameras, hot_masks,
                self.space_carving_config, frame_idx, search_region
            )
        else:
            fuse_frame(
                self.voxel_grid, self.cameras, images,
                self.fusion_config, frame_idx, search_region
            )

        # Decay and prune
        self.voxel_grid.apply_temporal_decay(frame_idx)
        if frame_idx % self._prune_interval == 0:
            self.voxel_grid.prune()

        # Cluster
        clusters = cluster_occupied_voxels(self.voxel_grid, self.clustering_config)

        # Track
        active_tracks = self.tracker.update(clusters, frame_idx)

        # Find primary track (highest confidence confirmed track)
        primary = None
        est_pos = None
        confirmed = [t for t in active_tracks if t.state == TrackState.CONFIRMED]
        if confirmed:
            primary = max(confirmed, key=lambda t: len(t.history))
            est_pos = primary.predicted_position

        elapsed = (time.time() - t0) * 1000.0

        return TrackingResult(
            frame_index=frame_idx,
            timestamp=frame_bundle.timestamp,
            clusters=clusters,
            active_tracks=active_tracks,
            primary_track=primary,
            estimated_position=est_pos,
            num_active_voxels=self.voxel_grid.num_active(),
            processing_time_ms=elapsed,
        )

    def _get_search_region(self) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        """Compute search region covering ALL confirmed tracks.

        For multi-eagle: expands the region to the bounding box of all tracks + margin.
        Returns None (full ROI scan) if no confirmed tracks exist.
        """
        active = self.tracker.get_active_tracks()
        if not active:
            return None

        confirmed = [t for t in active if t.state == TrackState.CONFIRMED]
        if not confirmed:
            return None

        # Collect all confirmed track positions
        positions = []
        for track in confirmed:
            pos = track.predicted_position
            if pos is not None:
                positions.append(pos)

        if not positions:
            return None

        # Bounding box of all tracks + margin
        positions = np.array(positions)
        margin = 50.0  # meters — covers eagle movement + voxel size
        min_world = positions.min(axis=0) - margin
        max_world = positions.max(axis=0) + margin

        # Clamp to ROI bounds so we don't search outside
        roi_min = np.array(self.voxel_grid.config.roi_min)
        roi_max = np.array(self.voxel_grid.config.roi_max)
        min_world = np.maximum(min_world, roi_min)
        max_world = np.minimum(max_world, roi_max)

        min_grid = self.voxel_grid.world_to_grid(min_world)
        max_grid = self.voxel_grid.world_to_grid(max_world)
        return min_grid, max_grid

    def reset(self) -> None:
        self.voxel_grid.clear()
        self.tracker = Tracker(self.tracker.config)


# ---------------------------------------------------------------------------
# Online simulation integration
# ---------------------------------------------------------------------------

def run_online_simulation(
    engine: SimulationEngine,
    pipeline: TrackingPipeline,
) -> list[TrackingResult]:
    """Run simulation in streaming mode, feeding frames to the tracking pipeline."""
    results = []
    for frame_bundle in engine.run_streaming():
        result = pipeline.process_frame(frame_bundle)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Evaluation metrics (Task 5.6.1)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Aggregated evaluation metrics."""
    # Position error
    position_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_error: float = 0.0
    median_error: float = 0.0
    max_error: float = 0.0
    rmse: float = 0.0
    percentile_95: float = 0.0

    # Detection
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    frames_to_first_detection: int = 0
    longest_detection_gap: int = 0

    # Timing
    mean_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0

    # Config
    num_frames: int = 0


def _get_gt_positions(ground_truth: np.ndarray, all_gt_positions: list[list[np.ndarray]] | None,
                      frame_idx: int) -> list[np.ndarray]:
    """Get all eagle ground truth positions for a frame.

    Supports both single-eagle (ground_truth array) and multi-eagle (all_gt_positions list).
    """
    if all_gt_positions and frame_idx < len(all_gt_positions):
        return all_gt_positions[frame_idx]
    # Fallback: single eagle from the trajectory array
    if frame_idx < len(ground_truth):
        gt = ground_truth[frame_idx, 1:4] if ground_truth.shape[1] >= 4 else ground_truth[frame_idx]
        return [gt]
    return []


def compute_position_error(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    all_gt_positions: list[list[np.ndarray]] | None = None,
) -> np.ndarray:
    """Compute per-frame min-distance error across all tracks and all GT eagles.

    For each track, find the nearest GT eagle. Report the mean of these min-distances.
    NaN for frames without any detection.
    """
    n = len(tracking_results)
    errors = np.full(n, np.nan)
    for i, result in enumerate(tracking_results):
        gt_list = _get_gt_positions(ground_truth, all_gt_positions, i)
        if not gt_list:
            continue

        # Collect all tracked positions (all confirmed + primary)
        tracked_positions = []
        confirmed = [t for t in result.active_tracks if t.state == TrackState.CONFIRMED]
        for t in confirmed:
            pos = t.predicted_position
            if pos is not None:
                tracked_positions.append(pos)

        if not tracked_positions:
            continue

        # For each GT eagle, find distance to nearest track (zone awareness)
        min_dists = []
        for gt_pos in gt_list:
            dists = [np.linalg.norm(tp - gt_pos) for tp in tracked_positions]
            min_dists.append(min(dists))

        # Report mean across all eagles (how well are we covering all of them?)
        errors[i] = np.mean(min_dists)
    return errors


def compute_detection_rate(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    distance_threshold: float = 5.0,
    all_gt_positions: list[list[np.ndarray]] | None = None,
) -> float:
    """Fraction of (frame, eagle) pairs where the eagle has a track within threshold."""
    total_eagle_frames = 0
    detected_eagle_frames = 0

    for i, result in enumerate(tracking_results):
        gt_list = _get_gt_positions(ground_truth, all_gt_positions, i)
        if not gt_list:
            continue

        confirmed = [t for t in result.active_tracks if t.state == TrackState.CONFIRMED]
        tracked = [t.predicted_position for t in confirmed if t.predicted_position is not None]

        for gt_pos in gt_list:
            total_eagle_frames += 1
            if tracked:
                min_dist = min(np.linalg.norm(tp - gt_pos) for tp in tracked)
                if min_dist < distance_threshold:
                    detected_eagle_frames += 1

    return detected_eagle_frames / max(total_eagle_frames, 1)


def compute_false_positive_rate(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    distance_threshold: float = 10.0,
    all_gt_positions: list[list[np.ndarray]] | None = None,
) -> float:
    """Fraction of frames with a track not near any GT eagle."""
    n_fp_frames = 0
    n_frames_with_tracks = 0

    for i, result in enumerate(tracking_results):
        gt_list = _get_gt_positions(ground_truth, all_gt_positions, i)
        confirmed = [t for t in result.active_tracks if t.state == TrackState.CONFIRMED]
        if not confirmed:
            continue
        n_frames_with_tracks += 1

        for track in confirmed:
            pos = track.predicted_position
            if pos is None:
                continue
            # Is this track near ANY gt eagle?
            near_any = any(np.linalg.norm(pos - gt) < distance_threshold for gt in gt_list) if gt_list else False
            if not near_any:
                n_fp_frames += 1
                break

    return n_fp_frames / max(n_frames_with_tracks, 1)


def generate_report(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    distance_threshold: float = 5.0,
    all_gt_positions: list[list[np.ndarray]] | None = None,
) -> EvaluationReport:
    """Generate a full evaluation report. Supports multi-eagle via all_gt_positions."""
    errors = compute_position_error(tracking_results, ground_truth, all_gt_positions)
    valid_errors = errors[~np.isnan(errors)]

    # Detection classification (multi-eagle aware)
    n = len(tracking_results)
    tp = 0
    fp = 0
    fn = 0
    first_detection = n
    current_gap = 0
    max_gap = 0

    for i, result in enumerate(tracking_results):
        gt_list = _get_gt_positions(ground_truth, all_gt_positions, i)
        confirmed = [t for t in result.active_tracks if t.state == TrackState.CONFIRMED]
        tracked = [t.predicted_position for t in confirmed if t.predicted_position is not None]

        # For each GT eagle: is it covered by a track?
        for gt_pos in gt_list:
            if tracked:
                min_dist = min(np.linalg.norm(tp - gt_pos) for tp in tracked)
                if min_dist < distance_threshold:
                    tp += 1
                    first_detection = min(first_detection, i)
                else:
                    fn += 1
            else:
                fn += 1

        # Any track not near any eagle?
        for pos in tracked:
            near_any = any(np.linalg.norm(pos - gt) < distance_threshold for gt in gt_list) if gt_list else False
            if not near_any:
                fp += 1

        # Gap tracking (any eagle detected this frame?)
        any_detected = False
        for gt_pos in gt_list:
            if tracked:
                min_dist = min(np.linalg.norm(tp_pos - gt_pos) for tp_pos in tracked)
                if min_dist < distance_threshold:
                    any_detected = True
                    break
        if any_detected:
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:
            current_gap += 1

    max_gap = max(max_gap, current_gap)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    times = np.array([r.processing_time_ms for r in tracking_results])

    return EvaluationReport(
        position_errors=errors,
        mean_error=float(np.mean(valid_errors)) if len(valid_errors) > 0 else float("inf"),
        median_error=float(np.median(valid_errors)) if len(valid_errors) > 0 else float("inf"),
        max_error=float(np.max(valid_errors)) if len(valid_errors) > 0 else float("inf"),
        rmse=float(np.sqrt(np.mean(valid_errors ** 2))) if len(valid_errors) > 0 else float("inf"),
        percentile_95=float(np.percentile(valid_errors, 95)) if len(valid_errors) > 0 else float("inf"),
        detection_rate=compute_detection_rate(tracking_results, ground_truth, distance_threshold, all_gt_positions),
        false_positive_rate=compute_false_positive_rate(tracking_results, ground_truth, distance_threshold * 2, all_gt_positions),
        precision=precision,
        recall=recall,
        f1_score=f1,
        frames_to_first_detection=first_detection if first_detection < n else -1,
        longest_detection_gap=max_gap,
        mean_processing_time_ms=float(np.mean(times)),
        max_processing_time_ms=float(np.max(times)),
        num_frames=n,
    )
