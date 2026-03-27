"""Integration tests for the full pipeline."""

import numpy as np

from thermal_tracker.config import load_config
from thermal_tracker.clustering import ClusteringConfig
from thermal_tracker.engine import SimulationEngine, SimulationOutputMode
from thermal_tracker.fusion import FusionConfig
from thermal_tracker.pipeline import (
    TrackingPipeline, run_online_simulation, generate_report,
)
from thermal_tracker.tracking import TrackingConfig
from thermal_tracker.voxel_grid import VoxelGridConfig


def test_simulation_engine_10_frames():
    """Run 10 frames and verify basic properties."""
    config = load_config("configs/default_config.yaml")
    config.run.num_frames = 10
    engine = SimulationEngine(config)
    result = engine.run(mode=SimulationOutputMode.BATCH)

    assert result.num_frames == 10
    assert len(result.frame_bundles) == 10
    assert result.trajectory.shape == (10, 7)

    # Verify images
    for fb in result.frame_bundles:
        assert len(fb.camera_images) == 3
        for cam_id, img in fb.camera_images.items():
            assert img.shape == (512, 640)
            assert img.dtype == np.float32
            assert not np.any(np.isnan(img))
            assert not np.any(np.isinf(img))

    # Verify timestamps are monotonically increasing
    for i in range(1, len(result.frame_bundles)):
        assert result.frame_bundles[i].timestamp > result.frame_bundles[i - 1].timestamp


def test_full_pipeline_lissajous():
    """End-to-end: Lissajous trajectory, 3 cameras, no noise, small ROI."""
    config = load_config("configs/default_config.yaml")
    config.run.num_frames = 30
    config.rendering.noise.enabled = False

    engine = SimulationEngine(config)

    # Set up tracking pipeline with a focused ROI around the expected trajectory
    voxel_config = VoxelGridConfig(
        voxel_size=5.0,
        roi_min=[-400.0, -400.0, 50.0],
        roi_max=[400.0, 400.0, 200.0],
        occupancy_threshold=0.7,
        temporal_decay_rate=0.05,
    )
    fusion_config = FusionConfig(
        detection_threshold_sigma=2.0,
        p_hot_given_occupied=0.9,
        p_hot_given_empty=0.05,
        min_cameras_for_update=2,
    )
    clustering_config = ClusteringConfig(
        method="connected_components",
        min_cluster_size=1,
        max_cluster_size=100,
        weighted_centroid=True,
    )
    tracking_config = TrackingConfig(
        max_association_distance=30.0,
        min_hits_to_confirm=2,
        max_frames_to_coast=5,
        dt=config.run.dt,
    )

    pipeline = TrackingPipeline(
        cameras=engine.cameras,
        voxel_config=voxel_config,
        fusion_config=fusion_config,
        clustering_config=clustering_config,
        tracking_config=tracking_config,
    )

    results = run_online_simulation(engine, pipeline)
    assert len(results) == 30

    # Re-run to get trajectory ground truth
    engine2 = SimulationEngine(config)
    sim_result = engine2.run(mode=SimulationOutputMode.BATCH)
    gt = sim_result.trajectory

    report = generate_report(results, gt, distance_threshold=20.0)
    assert report.num_frames == 30

    # With 5m voxels and no noise, we should get some detections
    # (Detection rate depends on ROI coverage and camera placement)
    print(f"Detection rate: {report.detection_rate:.2f}")
    print(f"Mean error: {report.mean_error:.2f} m")
    print(f"Mean processing time: {report.mean_processing_time_ms:.1f} ms")
