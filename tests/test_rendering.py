"""Tests for rendering module."""

import numpy as np

from thermal_tracker.camera import Camera, CameraExtrinsics, CameraIntrinsics
from thermal_tracker.eagle import EagleState
from thermal_tracker.rendering import (
    RenderingConfig, generate_background, project_eagle,
    render_thermal_blob, render_frame, generate_vignetting_map,
    apply_vignetting, compute_snr, NoiseConfig,
)


def _make_test_camera() -> Camera:
    intr = CameraIntrinsics.flir_640x512(45.0)
    ext = CameraExtrinsics.from_look_at(
        np.array([500.0, 0.0, 10.0]),
        np.array([0.0, 0.0, 100.0]),
    )
    return Camera("test_cam", intr, ext)


def test_background():
    config = RenderingConfig(sky_temperature=-10.0)
    bg = generate_background(640, 512, config)
    assert bg.shape == (512, 640)
    assert np.allclose(bg, -10.0)


def test_project_eagle_visible():
    cam = _make_test_camera()
    eagle = EagleState(position=np.array([0.0, 0.0, 100.0]))
    proj = project_eagle(eagle, cam)
    assert proj["is_visible"]
    assert proj["distance"] > 0
    assert proj["projected_radius"] > 0


def test_render_blob_peak():
    cam = _make_test_camera()
    config = RenderingConfig(sky_temperature=-10.0, atmospheric_attenuation_coeff=0.0)
    eagle = EagleState(position=np.array([0.0, 0.0, 100.0]), temperature=35.0)

    bg = generate_background(cam.intrinsics.width, cam.intrinsics.height, config)
    proj = project_eagle(eagle, cam)
    image = render_thermal_blob(bg, proj, eagle, config)

    # Peak should be at the projected center, value = sky + deltaT
    peak = image.max()
    expected_peak = -10.0 + (35.0 - (-10.0))  # No attenuation
    assert peak > -10.0
    assert peak <= expected_peak + 1.0  # Some tolerance for sub-pixel


def test_vignetting_center_unaffected():
    shape = (512, 640)
    vmap = generate_vignetting_map(shape, strength=0.5)
    cy, cx = shape[0] // 2, shape[1] // 2
    assert abs(vmap[cy, cx] - 1.0) < 0.01


def test_vignetting_corners_attenuated():
    shape = (512, 640)
    vmap = generate_vignetting_map(shape, strength=0.5)
    assert vmap[0, 0] < vmap[shape[0] // 2, shape[1] // 2]


def test_snr():
    snr = compute_snr(35.0, -10.0, 0.0005, 1000.0, 0.05)
    expected = (35 - (-10)) * np.exp(-0.0005 * 1000) / 0.05
    assert abs(snr - expected) < 0.01


def test_render_frame_no_nan():
    cam = _make_test_camera()
    config = RenderingConfig()
    eagle = EagleState(position=np.array([0.0, 0.0, 100.0]))
    rng = np.random.default_rng(42)
    image = render_frame(eagle, cam, config, rng)
    assert not np.any(np.isnan(image))
    assert not np.any(np.isinf(image))
    assert image.shape == (cam.intrinsics.height, cam.intrinsics.width)
