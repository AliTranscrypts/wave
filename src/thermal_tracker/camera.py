"""Camera intrinsics, extrinsics, and full projection pipeline (Tasks 3.1-3.3)."""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# 3.1 Intrinsic Parameters
# ---------------------------------------------------------------------------

class CameraIntrinsics(BaseModel):
    """Pinhole camera model with Brown-Conrady distortion."""

    width: int = 640
    height: int = 512
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 256.0
    dist_coeffs: list[float] = Field(default=[0.0, 0.0, 0.0, 0.0, 0.0], min_length=5, max_length=5)

    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def compute_fov(self) -> tuple[float, float]:
        """Return (hfov, vfov) in radians."""
        hfov = 2.0 * math.atan(self.width / (2.0 * self.fx))
        vfov = 2.0 * math.atan(self.height / (2.0 * self.fy))
        return hfov, vfov

    @classmethod
    def flir_640x512(cls, hfov_deg: float = 24.0) -> CameraIntrinsics:
        w, h = 640, 512
        fx = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return cls(width=w, height=h, fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0)

    @classmethod
    def low_cost_320x256(cls, hfov_deg: float = 24.0) -> CameraIntrinsics:
        w, h = 320, 256
        fx = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return cls(width=w, height=h, fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0)


def distort_points(points_2d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Apply Brown-Conrady distortion to ideal normalized image coordinates.

    points_2d: (N, 2) undistorted pixel coordinates.
    Returns: (N, 2) distorted pixel coordinates.
    """
    pts = points_2d.reshape(-1, 1, 2).astype(np.float64)
    K = intrinsics.K()
    dist = np.array(intrinsics.dist_coeffs, dtype=np.float64)
    # Undistort to get normalized coords, then re-distort is non-trivial.
    # For simplicity, use cv2.projectPoints with identity rotation.
    # Convert pixels to normalised camera coords first
    pts_norm = cv2.undistortPoints(pts, K, np.zeros(5))  # remove K only
    # Now apply distortion manually via the Brown-Conrady model
    x = pts_norm[:, 0, 0]
    y = pts_norm[:, 0, 1]
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    # Back to pixel coordinates
    u = intrinsics.fx * xd + intrinsics.cx
    v = intrinsics.fy * yd + intrinsics.cy
    return np.column_stack([u, v])


def undistort_points(points_2d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Remove distortion from pixel coordinates. Returns undistorted pixels."""
    pts = points_2d.reshape(-1, 1, 2).astype(np.float64)
    K = intrinsics.K()
    dist = np.array(intrinsics.dist_coeffs, dtype=np.float64)
    result = cv2.undistortPoints(pts, K, dist, P=K)
    return result.reshape(-1, 2)


# ---------------------------------------------------------------------------
# 3.2 Extrinsic Parameters
# ---------------------------------------------------------------------------

class CameraExtrinsics:
    """Camera pose in world coordinates."""

    def __init__(self, position: np.ndarray, quaternion: np.ndarray | None = None,
                 rotation_matrix: np.ndarray | None = None):
        """
        Args:
            position: (3,) camera center in world coordinates.
            quaternion: (4,) as (x, y, z, w) for scipy convention.
            rotation_matrix: (3,3) world-to-camera rotation.
        """
        self.position = np.asarray(position, dtype=np.float64)
        if rotation_matrix is not None:
            self._rotation = Rotation.from_matrix(rotation_matrix)
        elif quaternion is not None:
            self._rotation = Rotation.from_quat(quaternion)  # scipy: (x,y,z,w)
        else:
            self._rotation = Rotation.identity()

    @classmethod
    def from_euler(cls, position: np.ndarray, roll: float, pitch: float, yaw: float) -> CameraExtrinsics:
        """Construct from Euler angles (radians, ZYX extrinsic)."""
        rot = Rotation.from_euler("ZYX", [yaw, pitch, roll])
        return cls(position=position, rotation_matrix=rot.as_matrix())

    @classmethod
    def from_look_at(cls, position: np.ndarray, target: np.ndarray,
                     up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> CameraExtrinsics:
        """Construct rotation that points camera optical axis (-Z in camera frame) toward target."""
        pos = np.asarray(position, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = tgt - pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        norm_right = np.linalg.norm(right)
        if norm_right < 1e-8:
            # forward is parallel to up — pick an arbitrary perpendicular
            up_alt = np.array([1.0, 0.0, 0.0]) if abs(forward[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up_alt)
            norm_right = np.linalg.norm(right)
        right = right / norm_right

        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)

        # Camera convention: X-right, Y-down, Z-forward (OpenCV)
        # R maps world vectors to camera vectors
        R = np.stack([right, down, forward], axis=0)  # (3,3)
        return cls(position=pos, rotation_matrix=R)

    def R(self) -> np.ndarray:
        """3x3 world-to-camera rotation matrix."""
        return self._rotation.as_matrix()

    def t(self) -> np.ndarray:
        """Translation vector t = -R @ C."""
        return -self.R() @ self.position

    def Rt(self) -> np.ndarray:
        """3x4 [R | t] matrix."""
        return np.hstack([self.R(), self.t().reshape(3, 1)])

    def view_direction(self) -> np.ndarray:
        """Unit vector along optical axis in world coordinates (3rd row of R transposed)."""
        return self.R()[2, :]  # Z-axis of camera in world frame

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform world points to camera coordinates.

        Args:
            points: (3,) or (N, 3) world coordinates.
        Returns:
            Same shape in camera coordinates.
        """
        R = self.R()
        t = self.t()
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            return R @ pts + t
        return (R @ pts.T).T + t

    def rvec(self) -> np.ndarray:
        """Rodrigues rotation vector for OpenCV."""
        rvec, _ = cv2.Rodrigues(self.R())
        return rvec.flatten()


# ---------------------------------------------------------------------------
# 3.3 Camera Class
# ---------------------------------------------------------------------------

class OrientationMode(str, Enum):
    EULER = "euler"
    LOOK_AT = "look_at"


class CameraConfig(BaseModel):
    """Serializable camera configuration."""

    id: str = "cam_01"
    position: list[float] = Field(default=[0.0, 0.0, 10.0], min_length=3, max_length=3)
    orientation_mode: OrientationMode = OrientationMode.LOOK_AT
    euler_angles_deg: list[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    look_at_target: list[float] = Field(default=[0.0, 0.0, 100.0], min_length=3, max_length=3)
    intrinsics_preset: str = "flir_640x512"
    hfov_deg: float = 24.0


class Camera:
    """Full camera with intrinsics, extrinsics, and projection pipeline."""

    def __init__(self, cam_id: str, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics,
                 vignetting_map: np.ndarray | None = None,
                 fpn_map: np.ndarray | None = None):
        self.id = cam_id
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.vignetting_map = vignetting_map
        self.fpn_map = fpn_map
        # Pre-compute undistortion maps
        K = intrinsics.K()
        dist = np.array(intrinsics.dist_coeffs, dtype=np.float64)
        self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (intrinsics.width, intrinsics.height), cv2.CV_32FC1
        )

    def project(self, points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D world points to pixel coordinates.

        Args:
            points_3d: (N, 3) or (3,) world coordinates.
        Returns:
            pixels: (N, 2) pixel coordinates.
            visible: (N,) bool mask — within image bounds and in front of camera.
        """
        pts = np.atleast_2d(np.asarray(points_3d, dtype=np.float64))
        N = pts.shape[0]

        rvec = self.extrinsics.rvec()
        tvec = self.extrinsics.t()
        K = self.intrinsics.K()
        dist = np.array(self.intrinsics.dist_coeffs, dtype=np.float64)

        projected, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
        pixels = projected.reshape(N, 2)

        # Check visibility: in front of camera and within image bounds
        pts_cam = self.extrinsics.world_to_camera(pts)
        in_front = pts_cam[:, 2] > 0
        in_bounds = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < self.intrinsics.width) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < self.intrinsics.height)
        )
        visible = in_front & in_bounds

        return pixels, visible

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """Unproject a pixel to a 3D world point at given depth.

        Args:
            pixel: (2,) pixel coordinates (u, v).
            depth: distance along ray from camera center.
        Returns:
            (3,) world point.
        """
        K = self.intrinsics.K()
        dist = np.array(self.intrinsics.dist_coeffs, dtype=np.float64)

        # Undistort pixel
        pt = np.array([[pixel]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pt, K, dist)  # returns normalized coords
        x_norm, y_norm = undistorted[0, 0]

        # Ray in camera frame
        d_cam = np.array([x_norm, y_norm, 1.0])
        d_cam = d_cam / np.linalg.norm(d_cam)

        # Transform to world
        R = self.extrinsics.R()
        d_world = R.T @ d_cam
        return self.extrinsics.position + depth * d_world

    def P(self) -> np.ndarray:
        """3x4 full projection matrix P = K @ [R | t]."""
        return self.intrinsics.K() @ self.extrinsics.Rt()

    def frustum_corners(self, near: float = 1.0, far: float = 1000.0) -> np.ndarray:
        """8 frustum corners in world coordinates. Shape (8, 3).

        Order: 4 near-plane corners then 4 far-plane corners.
        Corner order per plane: TL, TR, BR, BL.
        """
        w, h = self.intrinsics.width, self.intrinsics.height
        corners_px = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float64)
        result = []
        for depth in [near, far]:
            for px in corners_px:
                result.append(self.unproject(px, depth))
        return np.array(result)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an image."""
        return cv2.remap(image, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Camera placement utilities
# ---------------------------------------------------------------------------

def generate_ring_placement(
    num_cameras: int,
    ring_radius: float = 500.0,
    pole_height: float = 10.0,
    look_at_center: np.ndarray | None = None,
    hfov_deg: float = 45.0,
) -> list[CameraConfig]:
    """Place cameras equally spaced around a circle, all pointing toward center."""
    if look_at_center is None:
        look_at_center = np.array([0.0, 0.0, 100.0])
    configs = []
    for i in range(num_cameras):
        angle = 2.0 * math.pi * i / num_cameras
        x = ring_radius * math.cos(angle)
        y = ring_radius * math.sin(angle)
        configs.append(CameraConfig(
            id=f"cam_{i:02d}",
            position=[x, y, pole_height],
            orientation_mode=OrientationMode.LOOK_AT,
            look_at_target=look_at_center.tolist(),
            hfov_deg=hfov_deg,
        ))
    return configs


def build_camera(config: CameraConfig, vignetting_map: np.ndarray | None = None,
                 fpn_map: np.ndarray | None = None) -> Camera:
    """Construct a Camera from a CameraConfig."""
    # Build intrinsics
    if config.intrinsics_preset == "flir_640x512":
        intrinsics = CameraIntrinsics.flir_640x512(config.hfov_deg)
    elif config.intrinsics_preset == "low_cost_320x256":
        intrinsics = CameraIntrinsics.low_cost_320x256(config.hfov_deg)
    else:
        intrinsics = CameraIntrinsics.flir_640x512(config.hfov_deg)

    # Build extrinsics
    pos = np.array(config.position)
    if config.orientation_mode == OrientationMode.LOOK_AT:
        extrinsics = CameraExtrinsics.from_look_at(pos, np.array(config.look_at_target))
    else:
        r, p, y = [math.radians(a) for a in config.euler_angles_deg]
        extrinsics = CameraExtrinsics.from_euler(pos, r, p, y)

    return Camera(config.id, intrinsics, extrinsics, vignetting_map, fpn_map)


def build_cameras(configs: list[CameraConfig],
                  vignetting_strength: float = 0.0,
                  noise_fpn_std: float = 0.0,
                  seed: int = 42) -> list[Camera]:
    """Build Camera objects from configs with optional vignetting and FPN maps."""
    from thermal_tracker.rendering import generate_vignetting_map, generate_fpn_map

    cameras = []
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(configs))

    for i, cfg in enumerate(configs):
        # Determine intrinsics to get image dimensions
        if cfg.intrinsics_preset == "flir_640x512":
            intr = CameraIntrinsics.flir_640x512(cfg.hfov_deg)
        elif cfg.intrinsics_preset == "low_cost_320x256":
            intr = CameraIntrinsics.low_cost_320x256(cfg.hfov_deg)
        else:
            intr = CameraIntrinsics.flir_640x512(cfg.hfov_deg)

        shape = (intr.height, intr.width)
        vig_map = generate_vignetting_map(shape, vignetting_strength) if vignetting_strength > 0 else None
        cs_entropy = child_seeds[i].entropy
        cs_int = cs_entropy if isinstance(cs_entropy, int) else int(cs_entropy[0])
        fpn_map = generate_fpn_map(shape, noise_fpn_std, cs_int) if noise_fpn_std > 0 else None

        cameras.append(build_camera(cfg, vig_map, fpn_map))

    return cameras
