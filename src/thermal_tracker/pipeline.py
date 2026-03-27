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

        # Determine search region from previous detection
        search_region = self._get_search_region()

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
        """Compute search region from last known track position."""
        active = self.tracker.get_active_tracks()
        if not active:
            return None

        # Use the primary track's predicted position
        confirmed = [t for t in active if t.state == TrackState.CONFIRMED]
        if not confirmed:
            return None

        track = max(confirmed, key=lambda t: len(t.history))
        pos = track.predicted_position
        if pos is None:
            return None

        # Search margin: ~3x max speed * dt around predicted position
        margin = 30.0  # meters
        min_world = pos - margin
        max_world = pos + margin

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


def compute_position_error(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Compute per-frame position error. NaN for frames without detection."""
    n = len(tracking_results)
    errors = np.full(n, np.nan)
    for i, result in enumerate(tracking_results):
        if result.estimated_position is not None and i < len(ground_truth):
            gt = ground_truth[i, 1:4] if ground_truth.shape[1] >= 4 else ground_truth[i]
            errors[i] = np.linalg.norm(result.estimated_position - gt)
    return errors


def compute_detection_rate(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    distance_threshold: float = 5.0,
) -> float:
    """Fraction of frames where eagle is correctly detected within threshold."""
    errors = compute_position_error(tracking_results, ground_truth)
    valid = ~np.isnan(errors)
    if not np.any(valid):
        return 0.0
    correct = np.sum(errors[valid] < distance_threshold)
    return float(correct / len(errors))


def compute_false_positive_rate(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    distance_threshold: float = 10.0,
) -> float:
    """Fraction of frames with spurious tracks far from ground truth."""
    n_fp = 0
    for i, result in enumerate(tracking_results):
        confirmed = [t for t in result.active_tracks if t.state == TrackState.CONFIRMED]
        if not confirmed:
            continue
        # Check if any confirmed track is far from GT
        gt = ground_truth[i, 1:4] if ground_truth.shape[1] >= 4 else ground_truth[i]
        for track in confirmed:
            pos = track.predicted_position
            if pos is not None and np.linalg.norm(pos - gt) > distance_threshold:
                n_fp += 1
                break
    return n_fp / max(len(tracking_results), 1)


def generate_report(
    tracking_results: list[TrackingResult],
    ground_truth: np.ndarray,
    distance_threshold: float = 5.0,
) -> EvaluationReport:
    """Generate a full evaluation report."""
    errors = compute_position_error(tracking_results, ground_truth)
    valid_errors = errors[~np.isnan(errors)]

    # Detection classification
    n = len(tracking_results)
    tp = 0
    fp = 0
    fn = 0
    first_detection = n
    current_gap = 0
    max_gap = 0

    for i, result in enumerate(tracking_results):
        gt = ground_truth[i, 1:4] if ground_truth.shape[1] >= 4 else ground_truth[i]
        has_detection = result.estimated_position is not None
        if has_detection:
            dist = np.linalg.norm(result.estimated_position - gt)
            if dist < distance_threshold:
                tp += 1
                first_detection = min(first_detection, i)
                max_gap = max(max_gap, current_gap)
                current_gap = 0
            else:
                fp += 1
                current_gap += 1
        else:
            fn += 1
            current_gap += 1

    max_gap = max(max_gap, current_gap)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Timing
    times = np.array([r.processing_time_ms for r in tracking_results])

    return EvaluationReport(
        position_errors=errors,
        mean_error=float(np.mean(valid_errors)) if len(valid_errors) > 0 else float("inf"),
        median_error=float(np.median(valid_errors)) if len(valid_errors) > 0 else float("inf"),
        max_error=float(np.max(valid_errors)) if len(valid_errors) > 0 else float("inf"),
        rmse=float(np.sqrt(np.mean(valid_errors ** 2))) if len(valid_errors) > 0 else float("inf"),
        percentile_95=float(np.percentile(valid_errors, 95)) if len(valid_errors) > 0 else float("inf"),
        detection_rate=compute_detection_rate(tracking_results, ground_truth, distance_threshold),
        false_positive_rate=compute_false_positive_rate(tracking_results, ground_truth, distance_threshold * 2),
        precision=precision,
        recall=recall,
        f1_score=f1,
        frames_to_first_detection=first_detection if first_detection < n else -1,
        longest_detection_gap=max_gap,
        mean_processing_time_ms=float(np.mean(times)),
        max_processing_time_ms=float(np.max(times)),
        num_frames=n,
    )
