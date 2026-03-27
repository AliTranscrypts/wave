"""Tests for camera module."""

import math

import numpy as np

from thermal_tracker.camera import (
    Camera, CameraConfig, CameraExtrinsics, CameraIntrinsics,
    OrientationMode, build_camera, generate_ring_placement, undistort_points,
)


def test_intrinsics_k_matrix():
    intr = CameraIntrinsics(fx=800, fy=800, cx=320, cy=256)
    K = intr.K()
    assert K[0, 0] == 800
    assert K[1, 1] == 800
    assert K[0, 2] == 320
    assert K[1, 2] == 256
    assert K[2, 2] == 1.0


def test_intrinsics_fov_roundtrip():
    hfov_deg = 30.0
    intr = CameraIntrinsics.flir_640x512(hfov_deg)
    hfov, vfov = intr.compute_fov()
    assert abs(math.degrees(hfov) - hfov_deg) < 0.01


def test_extrinsics_look_at():
    pos = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])
    ext = CameraExtrinsics.from_look_at(pos, target)
    view_dir = ext.view_direction()
    expected = np.array([0, 0, 1])
    assert np.allclose(view_dir, expected, atol=1e-6)


def test_extrinsics_world_to_camera_origin():
    pos = np.array([10.0, 20.0, 30.0])
    ext = CameraExtrinsics.from_look_at(pos, np.array([10.0, 20.0, 130.0]))
    cam_coords = ext.world_to_camera(pos)
    assert np.allclose(cam_coords, [0, 0, 0], atol=1e-10)


def test_camera_project_unproject_roundtrip():
    intr = CameraIntrinsics.flir_640x512(45.0)
    ext = CameraExtrinsics.from_look_at(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 100.0]),
    )
    cam = Camera("test", intr, ext)

    world_point = np.array([5.0, 3.0, 50.0])
    pixels, visible = cam.project(world_point.reshape(1, 3))
    assert visible[0]

    # Compute depth for unproject
    cam_pt = ext.world_to_camera(world_point)
    depth = np.linalg.norm(cam_pt)

    recovered = cam.unproject(pixels[0], depth)
    assert np.allclose(recovered, world_point, atol=0.5)  # sub-meter


def test_ring_placement():
    configs = generate_ring_placement(4, ring_radius=500, pole_height=10)
    assert len(configs) == 4
    for c in configs:
        assert c.orientation_mode == OrientationMode.LOOK_AT
        pos = np.array(c.position)
        assert abs(np.sqrt(pos[0] ** 2 + pos[1] ** 2) - 500) < 0.1
        assert pos[2] == 10.0
