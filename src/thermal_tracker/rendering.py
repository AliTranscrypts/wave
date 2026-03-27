"""Thermal rendering pipeline (Tasks 2.3.1-2.3.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from thermal_tracker.camera import Camera
    from thermal_tracker.eagle import EagleState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class NoiseConfig(BaseModel):
    """Sensor noise parameters."""
    enabled: bool = False
    gaussian_std: float = 0.05  # °C (NETD)
    shot_noise_enabled: bool = False
    fixed_pattern_noise_std: float = 0.0
    random_seed: int = 42


class RenderingMethod(str, Enum):
    PROJECTION = "projection"
    RAYCASTING = "raycasting"
    CUSTOM_RAYTRACER = "custom_raytracer"


class RenderingConfig(BaseModel):
    """Rendering pipeline configuration."""
    sky_temperature: float = -10.0  # °C
    ground_temperature: float = 15.0
    atmospheric_attenuation_coeff: float = 0.0005  # per meter
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    vignetting_strength: float = 0.0
    blur_sigma_pixels: float = 2.0
    rendering_method: RenderingMethod = RenderingMethod.PROJECTION


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def generate_background(width: int, height: int, config: RenderingConfig) -> np.ndarray:
    """Generate a uniform sky-temperature background image."""
    return np.full((height, width), config.sky_temperature, dtype=np.float32)


# ---------------------------------------------------------------------------
# Eagle projection and thermal blob
# ---------------------------------------------------------------------------

def project_eagle(eagle_state: EagleState, camera: Camera) -> dict:
    """Project eagle center to pixel coordinates and compute projected radius.

    Returns dict with keys: center_uv, projected_radius, distance, is_visible.
    """
    pos = eagle_state.position.reshape(1, 3)
    pixels, visible = camera.project(pos)
    center_uv = pixels[0]

    # Distance from camera to eagle
    diff = eagle_state.position - camera.extrinsics.position
    distance = float(np.linalg.norm(diff))

    # Projected radius in pixels
    if distance > 0:
        f = (camera.intrinsics.fx + camera.intrinsics.fy) / 2.0
        projected_radius = f * eagle_state.radius / distance
    else:
        projected_radius = 0.0

    # Visibility: in front of camera, projects within image (with margin for blob)
    pts_cam = camera.extrinsics.world_to_camera(pos)
    in_front = pts_cam[0, 2] > 0

    margin = 3 * max(projected_radius, 3.0)
    in_bounds = (
        center_uv[0] >= -margin and center_uv[0] < camera.intrinsics.width + margin and
        center_uv[1] >= -margin and center_uv[1] < camera.intrinsics.height + margin
    )

    return {
        "center_uv": center_uv,
        "projected_radius": projected_radius,
        "distance": distance,
        "is_visible": bool(in_front and in_bounds),
    }


def render_thermal_blob(
    image: np.ndarray,
    projection: dict,
    eagle_state: EagleState,
    config: RenderingConfig,
) -> np.ndarray:
    """Render eagle's thermal Gaussian blob onto the image."""
    if not projection["is_visible"]:
        return image

    u, v = projection["center_uv"]
    d = projection["distance"]
    r_proj = projection["projected_radius"]
    alpha = config.atmospheric_attenuation_coeff

    # Excess temperature with atmospheric attenuation
    delta_T = (eagle_state.temperature - config.sky_temperature) * np.exp(-alpha * d)

    # Sigma for Gaussian blob
    sigma = max(config.blur_sigma_pixels, r_proj, 1.0)

    # Bounding box (3-sigma)
    H, W = image.shape
    half_size = int(np.ceil(3 * sigma))
    i_min = max(0, int(v) - half_size)
    i_max = min(H, int(v) + half_size + 1)
    j_min = max(0, int(u) - half_size)
    j_max = min(W, int(u) + half_size + 1)

    if i_min >= i_max or j_min >= j_max:
        return image

    # Vectorized Gaussian
    rows = np.arange(i_min, i_max, dtype=np.float64)
    cols = np.arange(j_min, j_max, dtype=np.float64)
    jj, ii = np.meshgrid(cols, rows)
    r_sq = (ii - v) ** 2 + (jj - u) ** 2
    blob = delta_T * np.exp(-r_sq / (2.0 * sigma * sigma))

    image[i_min:i_max, j_min:j_max] += blob.astype(np.float32)
    return image


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def generate_fpn_map(shape: tuple[int, int], fpn_std: float, seed: int) -> np.ndarray:
    """Generate fixed-pattern noise map (static per camera)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, fpn_std, shape).astype(np.float32)


def apply_noise(
    image: np.ndarray,
    noise_config: NoiseConfig,
    fpn_map: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply sensor noise to a thermal image."""
    if not noise_config.enabled:
        return image

    result = image.copy()

    # Additive Gaussian noise
    if noise_config.gaussian_std > 0:
        result += rng.normal(0, noise_config.gaussian_std, result.shape).astype(np.float32)

    # Shot noise (simplified Poisson)
    if noise_config.shot_noise_enabled:
        # Convert to pseudo-photon counts (linear scaling, offset to positive)
        offset = abs(result.min()) + 1.0
        counts = np.clip((result + offset) * 100.0, 0, None)
        noisy_counts = rng.poisson(counts.astype(np.float64))
        result = (noisy_counts / 100.0 - offset).astype(np.float32)

    # Fixed-pattern noise
    if fpn_map is not None:
        result += fpn_map

    return result


# ---------------------------------------------------------------------------
# Vignetting
# ---------------------------------------------------------------------------

def generate_vignetting_map(shape: tuple[int, int], strength: float) -> np.ndarray:
    """Generate a radial vignetting map. 1.0 at center, reduced at edges."""
    H, W = shape
    cy, cx = H / 2.0, W / 2.0
    rows = np.arange(H, dtype=np.float64) - cy
    cols = np.arange(W, dtype=np.float64) - cx
    jj, ii = np.meshgrid(cols, rows)
    r_max = np.sqrt(cy * cy + cx * cx)
    r_norm = np.sqrt(ii * ii + jj * jj) / r_max
    vmap = 1.0 - strength * r_norm * r_norm
    return vmap.astype(np.float32)


def apply_vignetting(image: np.ndarray, vignetting_map: np.ndarray,
                     sky_temperature: float) -> np.ndarray:
    """Apply vignetting as multiplicative on temperature excess from sky."""
    excess = image - sky_temperature
    return sky_temperature + excess * vignetting_map


# ---------------------------------------------------------------------------
# SNR utility
# ---------------------------------------------------------------------------

def compute_snr(T_eagle: float, T_sky: float, alpha: float,
                distance: float, noise_std: float) -> float:
    """Compute peak signal-to-noise ratio of the eagle blob."""
    delta_T = (T_eagle - T_sky) * np.exp(-alpha * distance)
    if noise_std <= 0:
        return float("inf")
    return delta_T / noise_std


# ---------------------------------------------------------------------------
# Full frame rendering pipeline
# ---------------------------------------------------------------------------

def render_frame(
    eagle_state: EagleState,
    camera: Camera,
    config: RenderingConfig,
    rng: np.random.Generator,
    extra_eagles: list[EagleState] | None = None,
) -> np.ndarray:
    """Render one complete thermal frame for a single camera.

    Supports multiple eagles: pass additional eagles via extra_eagles.
    """
    W, H = camera.intrinsics.width, camera.intrinsics.height

    # 1. Background
    image = generate_background(W, H, config)

    # 2. Project and render all eagle blobs (additive)
    all_eagles = [eagle_state]
    if extra_eagles:
        all_eagles.extend(extra_eagles)

    for eagle in all_eagles:
        proj = project_eagle(eagle, camera)
        if proj["is_visible"]:
            image = render_thermal_blob(image, proj, eagle, config)

    # 3. Vignetting
    if camera.vignetting_map is not None:
        image = apply_vignetting(image, camera.vignetting_map, config.sky_temperature)

    # 4. Noise
    image = apply_noise(image, config.noise, camera.fpn_map, rng)

    return image


def render_all_cameras(
    eagle_state: EagleState,
    cameras: list[Camera],
    config: RenderingConfig,
    camera_rngs: list[np.random.Generator],
    extra_eagles: list[EagleState] | None = None,
) -> dict[str, np.ndarray]:
    """Render thermal images for all cameras at the same timestamp."""
    images = {}
    for cam, rng in zip(cameras, camera_rngs):
        images[cam.id] = render_frame(eagle_state, cam, config, rng, extra_eagles)
    return images


# ---------------------------------------------------------------------------
# FrameBundle
# ---------------------------------------------------------------------------

@dataclass
class FrameBundle:
    """All data for a single simulation time step."""
    timestamp: float
    frame_index: int
    camera_images: dict[str, np.ndarray] = field(default_factory=dict)
    ground_truth_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ground_truth_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ground_truth_2d: dict[str, np.ndarray] = field(default_factory=dict)
    visibility: dict[str, bool] = field(default_factory=dict)
    # Multi-eagle ground truth: list of (position, velocity) per eagle
    all_eagle_positions: list[np.ndarray] = field(default_factory=list)
    all_eagle_velocities: list[np.ndarray] = field(default_factory=list)
