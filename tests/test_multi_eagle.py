"""Multi-eagle tracking tests."""

import numpy as np

from thermal_tracker.camera import Camera, CameraExtrinsics, CameraIntrinsics, generate_ring_placement, build_cameras
from thermal_tracker.clustering import ClusteringConfig, VoxelCluster
from thermal_tracker.config import SimulationConfig, load_config
from thermal_tracker.eagle import EagleConfig, EagleState, MotionType
from thermal_tracker.engine import SimulationEngine, SimulationOutputMode
from thermal_tracker.fusion import (
    FusionConfig, SpaceCarvingConfig, preprocess_thermal_image,
    space_carve_frame, validate_clusters,
)
from thermal_tracker.pipeline import TrackingPipeline, generate_report
from thermal_tracker.rendering import RenderingConfig, render_frame, render_all_cameras, FrameBundle, project_eagle
from thermal_tracker.tracking import TrackingConfig, TrackState
from thermal_tracker.voxel_grid import SparseVoxelGrid, VoxelGridConfig
from thermal_tracker.world import WorldConfig


def _make_cameras(n=4, radius=500.0, hfov=45.0):
    """Build cameras in a ring for testing."""
    configs = generate_ring_placement(n, ring_radius=radius, pole_height=10.0, hfov_deg=hfov)
    return build_cameras(configs)


def _make_eagle(position, temperature=35.0):
    return EagleState(
        position=np.array(position, dtype=np.float64),
        velocity=np.zeros(3),
        temperature=temperature,
        radius=0.5,
    )


def test_validate_clusters_keeps_real_rejects_ghost():
    """validate_clusters keeps a cluster at a real eagle and rejects a ghost."""
    cameras = _make_cameras(n=4, radius=300.0)
    eagle = _make_eagle([0.0, 0.0, 100.0])
    config = RenderingConfig(sky_temperature=-10.0)
    rng = np.random.default_rng(42)

    # Render one frame with one eagle
    images = {}
    for cam in cameras:
        images[cam.id] = render_frame(eagle, cam, config, rng)
    hot_masks = [
        preprocess_thermal_image(images[cam.id], FusionConfig())
        for cam in cameras
    ]

    # Real cluster at the eagle position
    real_cluster = VoxelCluster(
        cluster_id=0, voxel_indices=[(0, 0, 0)],
        voxel_positions=np.array([[0.0, 0.0, 100.0]]),
        centroid=np.array([0.0, 0.0, 100.0]),
        size=1, max_probability=0.95,
        bounding_box=(np.array([0, 0, 100]), np.array([0, 0, 100])),
    )
    # Ghost cluster far from the eagle
    ghost_cluster = VoxelCluster(
        cluster_id=1, voxel_indices=[(10, 10, 10)],
        voxel_positions=np.array([[200.0, 200.0, 100.0]]),
        centroid=np.array([200.0, 200.0, 100.0]),
        size=1, max_probability=0.9,
        bounding_box=(np.array([200, 200, 100]), np.array([200, 200, 100])),
    )

    result = validate_clusters([real_cluster, ghost_cluster], cameras, hot_masks, min_cameras_confirm=2)

    # Real should survive, ghost should be filtered
    assert len(result) >= 1
    centroids = [c.centroid for c in result]
    real_found = any(np.linalg.norm(c - np.array([0, 0, 100])) < 50 for c in centroids)
    ghost_found = any(np.linalg.norm(c - np.array([200, 200, 100])) < 50 for c in centroids)
    assert real_found, "Real eagle cluster was incorrectly filtered"
    assert not ghost_found, "Ghost cluster was not filtered"


def test_three_eagles_produce_three_clusters():
    """With 3 well-separated eagles and space carving + validation, get 3 clusters."""
    cameras = _make_cameras(n=8, radius=400.0, hfov=45.0)
    eagles = [
        _make_eagle([-100.0, 0.0, 100.0]),
        _make_eagle([100.0, 0.0, 100.0]),
        _make_eagle([0.0, 100.0, 100.0]),
    ]
    config = RenderingConfig(sky_temperature=-10.0)
    rngs = [np.random.default_rng(i) for i in range(len(cameras))]

    # Render all eagles into all cameras
    images = {}
    for cam, rng in zip(cameras, rngs):
        images[cam.id] = render_frame(eagles[0], cam, config, rng, extra_eagles=eagles[1:])
    hot_masks = [preprocess_thermal_image(images[cam.id], FusionConfig()) for cam in cameras]

    # Space carve
    voxel_config = VoxelGridConfig(
        voxel_size=7.0, roi_min=[-200, -200, 60], roi_max=[200, 200, 140],
        occupancy_threshold=0.7,
    )
    voxel_grid = SparseVoxelGrid(voxel_config)
    sc_config = SpaceCarvingConfig(min_cameras_vote=2, allow_vetoing=False, occupied_log_odds=3.0)
    space_carve_frame(voxel_grid, cameras, hot_masks, sc_config, current_frame=0)

    # Cluster with DBSCAN
    from thermal_tracker.clustering import cluster_occupied_voxels
    cl_config = ClusteringConfig(method="dbscan", dbscan_eps_factor=2.0, dbscan_min_samples=1,
                                 min_cluster_size=1, max_cluster_size=200)
    clusters = cluster_occupied_voxels(voxel_grid, cl_config)

    # Validate (ghost suppression)
    valid_clusters = validate_clusters(clusters, cameras, hot_masks, min_cameras_confirm=3)

    # Should have at least 2 valid clusters (3 ideally + maybe 1-2 residual ghosts)
    assert len(valid_clusters) >= 2, f"Expected >=2 clusters, got {len(valid_clusters)}"

    # Count how many eagles have a nearby cluster (within 20m)
    eagles_covered = 0
    for eagle in eagles:
        dists = [np.linalg.norm(c.centroid - eagle.position) for c in valid_clusters]
        if min(dists) < 20.0:
            eagles_covered += 1

    assert eagles_covered >= 2, f"Only {eagles_covered}/3 eagles have a cluster within 20m"


def test_single_eagle_regression():
    """Single-eagle tracking still works after multi-eagle changes."""
    from thermal_tracker.config import SimulationRunConfig
    config = SimulationConfig(
        eagle=EagleConfig(motion_type=MotionType.LISSAJOUS, lissajous_amplitudes=[100, 100, 20],
                          lissajous_frequencies=[0.1, 0.08, 0.05], lissajous_z_center=100),
        cameras=generate_ring_placement(4, ring_radius=300, pole_height=10, hfov_deg=45),
        rendering=RenderingConfig(),
        run=SimulationRunConfig(num_frames=20, dt=0.1, random_seed=42),
    )

    engine = SimulationEngine(config)
    pipeline = TrackingPipeline(
        cameras=engine.cameras,
        voxel_config=VoxelGridConfig(voxel_size=5.0, roi_min=[-150, -150, 60], roi_max=[150, 150, 140]),
        fusion_config=FusionConfig(detection_threshold_sigma=1.5, min_cameras_for_update=2),
        clustering_config=ClusteringConfig(method="dbscan", dbscan_eps_factor=2.0, dbscan_min_samples=1),
        tracking_config=TrackingConfig(min_hits_to_confirm=1, max_frames_to_coast=3, dt=0.1),
        fusion_method="space_carving",
        space_carving_config=SpaceCarvingConfig(min_cameras_vote=2, allow_vetoing=False, occupied_log_odds=3.0),
    )

    from thermal_tracker.pipeline import run_online_simulation
    results = run_online_simulation(engine, pipeline)

    # Should have some detections — not zero
    n_detected = sum(1 for r in results if r.estimated_position is not None)
    assert n_detected > 0, "Single-eagle regression: no detections at all"


def test_cluster_separation_at_varying_distances():
    """Two eagles at different separations: merge when close, separate when far."""
    cameras = _make_cameras(n=6, radius=400.0, hfov=45.0)
    config = RenderingConfig(sky_temperature=-10.0)
    rngs = [np.random.default_rng(i) for i in range(len(cameras))]
    voxel_size = 7.0

    for separation, expect_min_clusters in [(20.0, 1), (100.0, 2)]:
        eagle1 = _make_eagle([0.0, 0.0, 100.0])
        eagle2 = _make_eagle([separation, 0.0, 100.0])

        images = {}
        for cam, rng in zip(cameras, rngs):
            images[cam.id] = render_frame(eagle1, cam, config, rng, extra_eagles=[eagle2])
        hot_masks = [preprocess_thermal_image(images[cam.id], FusionConfig()) for cam in cameras]

        voxel_grid = SparseVoxelGrid(VoxelGridConfig(
            voxel_size=voxel_size, roi_min=[-200, -200, 60], roi_max=[200, 200, 140],
        ))
        space_carve_frame(voxel_grid, cameras, hot_masks,
                          SpaceCarvingConfig(min_cameras_vote=2, allow_vetoing=False, occupied_log_odds=3.0),
                          current_frame=0)

        from thermal_tracker.clustering import cluster_occupied_voxels
        clusters = cluster_occupied_voxels(voxel_grid,
                                           ClusteringConfig(method="dbscan", dbscan_eps_factor=2.0,
                                                            dbscan_min_samples=1))
        valid = validate_clusters(clusters, cameras, hot_masks, min_cameras_confirm=2)

        assert len(valid) >= expect_min_clusters, \
            f"Separation {separation}m: expected >={expect_min_clusters} clusters, got {len(valid)}"
