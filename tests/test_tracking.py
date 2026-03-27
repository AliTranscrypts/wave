"""Tests for tracking module."""

import numpy as np

from thermal_tracker.clustering import VoxelCluster
from thermal_tracker.tracking import KalmanFilter3D, Tracker, TrackingConfig, TrackState


def _make_cluster(centroid: np.ndarray, cluster_id: int = 0) -> VoxelCluster:
    return VoxelCluster(
        cluster_id=cluster_id,
        voxel_indices=[(0, 0, 0)],
        voxel_positions=centroid.reshape(1, 3),
        centroid=centroid,
        size=1,
        max_probability=0.95,
        bounding_box=(centroid, centroid),
    )


def test_kalman_constant_velocity():
    kf = KalmanFilter3D(dt=0.1, q=1.0, r=0.5)
    kf.initialize(np.array([0.0, 0.0, 0.0]))

    # Feed constant velocity trajectory
    for i in range(20):
        t = (i + 1) * 0.1
        measurement = np.array([t * 10.0, 0.0, 0.0])  # 10 m/s in x
        kf.predict()
        kf.update(measurement)

    vel = kf.get_velocity()
    assert abs(vel[0] - 10.0) < 1.0  # Should converge to ~10 m/s


def test_kalman_noise_filtering():
    kf = KalmanFilter3D(dt=0.1, q=1.0, r=0.5)
    rng = np.random.default_rng(42)
    # Initialize at the first true position
    kf.initialize(np.array([1.0, 0.0, 100.0]))

    errors_raw = []
    errors_filtered = []
    for i in range(50):
        t = (i + 1) * 0.1
        true_pos = np.array([t * 10.0, 0.0, 100.0])
        noisy_pos = true_pos + rng.normal(0, 0.5, 3)
        kf.predict()
        filtered = kf.update(noisy_pos)
        errors_raw.append(np.linalg.norm(noisy_pos - true_pos))
        errors_filtered.append(np.linalg.norm(filtered - true_pos))

    # After convergence, filtered error should be less than raw noise
    assert np.mean(errors_filtered[20:]) < np.mean(errors_raw[20:])


def test_tracker_single_target():
    config = TrackingConfig(
        max_association_distance=20.0,
        min_hits_to_confirm=3,
        max_frames_to_coast=5,
        dt=0.1,
    )
    tracker = Tracker(config)

    for i in range(10):
        centroid = np.array([float(i), 0.0, 100.0])
        det = _make_cluster(centroid)
        active = tracker.update([det], i)

    # Should have one confirmed track
    confirmed = [t for t in active if t.state == TrackState.CONFIRMED]
    assert len(confirmed) == 1
    assert len(confirmed[0].history) == 10


def test_tracker_coast_and_reacquire():
    config = TrackingConfig(
        max_association_distance=20.0,
        min_hits_to_confirm=2,
        max_frames_to_coast=5,
        dt=0.1,
    )
    tracker = Tracker(config)

    # 5 frames of detection
    for i in range(5):
        det = _make_cluster(np.array([float(i), 0.0, 100.0]))
        tracker.update([det], i)

    # 3 frames of no detection (within coast limit)
    for i in range(5, 8):
        tracker.update([], i)

    active = tracker.get_active_tracks()
    assert len(active) == 1  # Still coasting

    # Reacquire
    det = _make_cluster(np.array([8.0, 0.0, 100.0]))
    active = tracker.update([det], 8)
    assert len(active) >= 1


def test_tracker_spurious_detection():
    config = TrackingConfig(min_hits_to_confirm=3, dt=0.1)
    tracker = Tracker(config)

    # Single spurious detection
    det = _make_cluster(np.array([100.0, 100.0, 100.0]))
    active = tracker.update([det], 0)

    # No detection next frame — tentative track should die
    active = tracker.update([], 1)
    confirmed = [t for t in active if t.state == TrackState.CONFIRMED]
    assert len(confirmed) == 0
