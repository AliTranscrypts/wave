"""Simulation engine and data pipeline (Tasks 4.1.1, 4.2.1)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from thermal_tracker.camera import Camera, build_cameras
from thermal_tracker.config import SimulationConfig
from thermal_tracker.eagle import EagleState, create_motion_generator, MotionGenerator
from thermal_tracker.rendering import (
    FrameBundle, RenderingConfig, render_all_cameras, project_eagle,
)


class SimulationOutputMode(str, Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    DISK = "disk"


@dataclass
class SimulationResult:
    """Metadata from a completed simulation run."""
    config: SimulationConfig
    num_frames: int
    wall_clock_seconds: float
    trajectory: np.ndarray  # (N, 7)
    frame_bundles: list[FrameBundle] = field(default_factory=list)


class SimulationEngine:
    """Main simulation loop orchestrator."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._cameras: list[Camera] = []
        self._motion_gen: MotionGenerator | None = None
        self._eagle_state: EagleState | None = None
        self._frame_index: int = 0
        self._camera_rngs: list[np.random.Generator] = []

        self._initialize()

    def _initialize(self) -> None:
        ss = np.random.SeedSequence(self.config.run.random_seed)
        motion_seed, cam_seed = ss.spawn(2)

        def _seed_int(seed_seq: np.random.SeedSequence) -> int:
            entropy = seed_seq.entropy
            if isinstance(entropy, int):
                return entropy
            return int(entropy[0])

        # Build cameras
        self._cameras = build_cameras(
            self.config.cameras,
            vignetting_strength=self.config.rendering.vignetting_strength,
            noise_fpn_std=self.config.rendering.noise.fixed_pattern_noise_std,
            seed=_seed_int(cam_seed),
        )

        # Motion generator
        self._motion_gen = create_motion_generator(self.config.eagle, self.config.world)
        self._eagle_state = self._motion_gen.reset(_seed_int(motion_seed))

        # Per-camera RNGs for noise
        cam_ss = np.random.SeedSequence(_seed_int(cam_seed))
        cam_seeds = cam_ss.spawn(len(self._cameras))
        self._camera_rngs = [np.random.default_rng(s) for s in cam_seeds]

        self._frame_index = 0

    @property
    def cameras(self) -> list[Camera]:
        return self._cameras

    def step_single(self) -> FrameBundle:
        """Advance simulation by one time step, return FrameBundle."""
        dt = self.config.run.dt
        timestamp = self._frame_index * dt

        # Render all cameras
        images = render_all_cameras(
            self._eagle_state, self._cameras, self.config.rendering, self._camera_rngs
        )

        # Ground truth 2D projections
        gt_2d = {}
        vis = {}
        for cam in self._cameras:
            proj = project_eagle(self._eagle_state, cam)
            gt_2d[cam.id] = proj["center_uv"]
            vis[cam.id] = proj["is_visible"]

        bundle = FrameBundle(
            timestamp=timestamp,
            frame_index=self._frame_index,
            camera_images=images,
            ground_truth_3d=self._eagle_state.position.copy(),
            ground_truth_velocity=self._eagle_state.velocity.copy(),
            ground_truth_2d=gt_2d,
            visibility=vis,
        )

        # Advance eagle state
        self._eagle_state = self._motion_gen.step(self._eagle_state, dt)
        self._frame_index += 1

        return bundle

    def run(self, mode: SimulationOutputMode = SimulationOutputMode.BATCH) -> SimulationResult:
        """Run full simulation."""
        start = time.time()
        n = self.config.run.num_frames
        dt = self.config.run.dt

        bundles: list[FrameBundle] = []
        traj = np.zeros((n, 7))

        for i in range(n):
            bundle = self.step_single()
            traj[i] = [
                bundle.timestamp,
                *bundle.ground_truth_3d,
                *bundle.ground_truth_velocity,
            ]
            if mode == SimulationOutputMode.BATCH:
                bundles.append(bundle)

        elapsed = time.time() - start
        return SimulationResult(
            config=self.config,
            num_frames=n,
            wall_clock_seconds=elapsed,
            trajectory=traj,
            frame_bundles=bundles,
        )

    def run_streaming(self) -> Iterator[FrameBundle]:
        """Yield frame bundles one at a time (streaming mode)."""
        for _ in range(self.config.run.num_frames):
            yield self.step_single()


# ---------------------------------------------------------------------------
# Data I/O (Task 4.2.1) — NPZ backend
# ---------------------------------------------------------------------------

class SimulationWriter(ABC):
    @abstractmethod
    def open(self, output_dir: str, config: SimulationConfig) -> None: ...
    @abstractmethod
    def write_frame(self, frame_bundle: FrameBundle) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


class SimulationReader(ABC):
    @abstractmethod
    def open(self, path: str) -> SimulationConfig: ...
    @abstractmethod
    def num_frames(self) -> int: ...
    @abstractmethod
    def read_frame(self, index: int) -> FrameBundle: ...
    @abstractmethod
    def read_trajectory(self) -> np.ndarray: ...


class NpzWriter(SimulationWriter):
    def __init__(self):
        self._output_dir: Path | None = None
        self._trajectory: list[np.ndarray] = []

    def open(self, output_dir: str, config: SimulationConfig) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "frames").mkdir(exist_ok=True)
        # Save config
        with open(self._output_dir / "config.yaml", "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
        self._trajectory = []

    def write_frame(self, fb: FrameBundle) -> None:
        assert self._output_dir is not None
        data = {}
        for cam_id, img in fb.camera_images.items():
            data[cam_id] = img
        data["gt_3d"] = fb.ground_truth_3d
        data["gt_velocity"] = fb.ground_truth_velocity
        for cam_id, uv in fb.ground_truth_2d.items():
            data[f"gt_2d_{cam_id}"] = uv

        path = self._output_dir / "frames" / f"frame_{fb.frame_index:06d}.npz"
        np.savez(path, **data)

        self._trajectory.append(np.array([
            fb.timestamp, *fb.ground_truth_3d, *fb.ground_truth_velocity
        ]))

    def close(self) -> None:
        if self._output_dir and self._trajectory:
            traj = np.stack(self._trajectory)
            np.savez(self._output_dir / "trajectory.npz", trajectory=traj)


class NpzReader(SimulationReader):
    def __init__(self):
        self._path: Path | None = None
        self._config: SimulationConfig | None = None
        self._n_frames: int = 0

    def open(self, path: str) -> SimulationConfig:
        self._path = Path(path)
        with open(self._path / "config.yaml") as f:
            data = yaml.safe_load(f)
        self._config = SimulationConfig.model_validate(data)
        frames_dir = self._path / "frames"
        self._n_frames = len(list(frames_dir.glob("frame_*.npz")))
        return self._config

    def num_frames(self) -> int:
        return self._n_frames

    def read_frame(self, index: int) -> FrameBundle:
        assert self._path is not None
        path = self._path / "frames" / f"frame_{index:06d}.npz"
        data = dict(np.load(path))

        gt_3d = data.pop("gt_3d")
        gt_vel = data.pop("gt_velocity")

        gt_2d = {}
        cam_images = {}
        for key, val in data.items():
            if key.startswith("gt_2d_"):
                cam_id = key[6:]
                gt_2d[cam_id] = val
            else:
                cam_images[key] = val

        return FrameBundle(
            timestamp=index * self._config.run.dt if self._config else 0.0,
            frame_index=index,
            camera_images=cam_images,
            ground_truth_3d=gt_3d,
            ground_truth_velocity=gt_vel,
            ground_truth_2d=gt_2d,
        )

    def read_trajectory(self) -> np.ndarray:
        assert self._path is not None
        return np.load(self._path / "trajectory.npz")["trajectory"]
