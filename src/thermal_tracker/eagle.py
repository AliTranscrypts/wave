"""Eagle state representation and motion generators (Tasks 2.2.1-2.2.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from scipy.interpolate import CubicSpline

from thermal_tracker.world import WorldConfig, clamp_to_bounds


@dataclass
class EagleState:
    """Eagle state at a single time step."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 100.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temperature: float = 35.0  # Celsius
    radius: float = 0.5  # meters


class MotionType(str, Enum):
    RANDOM_WALK = "random_walk"
    SPLINE = "spline"
    LISSAJOUS = "lissajous"


class EagleConfig(BaseModel):
    """Configuration for eagle motion and thermal properties."""

    temperature: float = 35.0
    radius: float = 0.5
    max_speed: float = 20.0
    max_acceleration: float = 5.0
    motion_type: MotionType = MotionType.LISSAJOUS
    altitude_range: list[float] = Field(default=[50.0, 200.0], min_length=2, max_length=2)
    # Lissajous params
    lissajous_amplitudes: list[float] = Field(default=[500.0, 500.0, 50.0], min_length=3, max_length=3)
    lissajous_frequencies: list[float] = Field(default=[0.05, 0.07, 0.03], min_length=3, max_length=3)
    lissajous_phases: list[float] = Field(default=[0.0, 1.57, 0.0], min_length=3, max_length=3)
    lissajous_z_center: float = 100.0
    # Spline params
    num_control_points: int = 10
    # Initial position
    initial_position: list[float] = Field(default=[0.0, 0.0, 100.0], min_length=3, max_length=3)


# ---------------------------------------------------------------------------
# Motion Generators
# ---------------------------------------------------------------------------

class MotionGenerator(ABC):
    @abstractmethod
    def reset(self, seed: int) -> EagleState:
        ...

    @abstractmethod
    def step(self, state: EagleState, dt: float) -> EagleState:
        ...

    def generate_trajectory(self, num_steps: int, dt: float, seed: int = 42) -> np.ndarray:
        """Generate full trajectory. Returns (N, 7): [t, px, py, pz, vx, vy, vz]."""
        state = self.reset(seed)
        traj = np.zeros((num_steps, 7))
        for i in range(num_steps):
            traj[i] = [i * dt, *state.position, *state.velocity]
            state = self.step(state, dt)
        return traj


class RandomWalkMotion(MotionGenerator):
    """Bounded random walk with Gaussian acceleration."""

    def __init__(self, config: EagleConfig, world_config: WorldConfig):
        self.config = config
        self.world = world_config
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> EagleState:
        self._rng = np.random.default_rng(seed)
        return EagleState(
            position=np.array(self.config.initial_position, dtype=np.float64),
            velocity=np.zeros(3),
            temperature=self.config.temperature,
            radius=self.config.radius,
        )

    def step(self, state: EagleState, dt: float) -> EagleState:
        assert self._rng is not None
        acc = self._rng.normal(0, self.config.max_acceleration / 3.0, size=3)
        acc_norm = np.linalg.norm(acc)
        if acc_norm > self.config.max_acceleration:
            acc = acc * self.config.max_acceleration / acc_norm

        new_vel = state.velocity + acc * dt
        speed = np.linalg.norm(new_vel)
        if speed > self.config.max_speed:
            new_vel = new_vel * self.config.max_speed / speed

        new_pos = state.position + new_vel * dt

        # Enforce altitude range
        z_min, z_max = self.config.altitude_range
        if new_pos[2] < z_min:
            new_pos[2] = z_min
            new_vel[2] = abs(new_vel[2])
        elif new_pos[2] > z_max:
            new_pos[2] = z_max
            new_vel[2] = -abs(new_vel[2])

        # Elastic reflection on world bounds
        bounds_min = self.world.bounds_min.copy()
        bounds_max = self.world.bounds_max.copy()
        for axis in range(3):
            if new_pos[axis] < bounds_min[axis]:
                new_pos[axis] = bounds_min[axis]
                new_vel[axis] = abs(new_vel[axis])
            elif new_pos[axis] > bounds_max[axis]:
                new_pos[axis] = bounds_max[axis]
                new_vel[axis] = -abs(new_vel[axis])

        return EagleState(
            position=new_pos,
            velocity=new_vel,
            temperature=state.temperature,
            radius=state.radius,
        )


class SplineMotion(MotionGenerator):
    """Catmull-Rom spline through random control points."""

    def __init__(self, config: EagleConfig, world_config: WorldConfig):
        self.config = config
        self.world = world_config
        self._spline_x: CubicSpline | None = None
        self._spline_y: CubicSpline | None = None
        self._spline_z: CubicSpline | None = None
        self._t_param: float = 0.0
        self._t_max: float = 1.0

    def reset(self, seed: int) -> EagleState:
        rng = np.random.default_rng(seed)
        n_pts = self.config.num_control_points
        z_lo, z_hi = self.config.altitude_range

        xs = rng.uniform(self.world.x_min * 0.3, self.world.x_max * 0.3, n_pts)
        ys = rng.uniform(self.world.y_min * 0.3, self.world.y_max * 0.3, n_pts)
        zs = rng.uniform(z_lo, z_hi, n_pts)

        # Prepend initial position
        xs = np.insert(xs, 0, self.config.initial_position[0])
        ys = np.insert(ys, 0, self.config.initial_position[1])
        zs = np.insert(zs, 0, self.config.initial_position[2])
        n_pts += 1

        t_knots = np.linspace(0, 1, n_pts)
        self._spline_x = CubicSpline(t_knots, xs, bc_type="clamped")
        self._spline_y = CubicSpline(t_knots, ys, bc_type="clamped")
        self._spline_z = CubicSpline(t_knots, zs, bc_type="clamped")
        self._t_param = 0.0
        self._t_max = 1.0

        pos = np.array([xs[0], ys[0], zs[0]])
        return EagleState(
            position=pos, velocity=np.zeros(3),
            temperature=self.config.temperature, radius=self.config.radius,
        )

    def step(self, state: EagleState, dt: float) -> EagleState:
        # Advance parameter — scale dt to spline parameter space
        # Approximate: advance by dt * max_speed / total_path_length
        speed_scale = self.config.max_speed * dt / 5000.0  # heuristic
        self._t_param = min(self._t_param + speed_scale, self._t_max)

        t = self._t_param
        pos = np.array([
            float(self._spline_x(t)),
            float(self._spline_y(t)),
            float(self._spline_z(t)),
        ])
        vel_raw = np.array([
            float(self._spline_x(t, 1)),
            float(self._spline_y(t, 1)),
            float(self._spline_z(t, 1)),
        ]) * speed_scale / dt if dt > 0 else np.zeros(3)

        speed = np.linalg.norm(vel_raw)
        if speed > self.config.max_speed:
            vel_raw = vel_raw * self.config.max_speed / speed

        return EagleState(
            position=pos, velocity=vel_raw,
            temperature=state.temperature, radius=state.radius,
        )


class LissajousMotion(MotionGenerator):
    """Deterministic Lissajous curve motion."""

    def __init__(self, config: EagleConfig, world_config: WorldConfig):
        self.config = config
        self.world = world_config
        self._time: float = 0.0

    def reset(self, seed: int) -> EagleState:
        self._time = 0.0
        pos = self._position_at(0.0)
        vel = self._velocity_at(0.0)
        return EagleState(
            position=pos, velocity=vel,
            temperature=self.config.temperature, radius=self.config.radius,
        )

    def _position_at(self, t: float) -> np.ndarray:
        A = self.config.lissajous_amplitudes
        w = self.config.lissajous_frequencies
        phi = self.config.lissajous_phases
        return np.array([
            A[0] * np.sin(w[0] * t + phi[0]),
            A[1] * np.sin(w[1] * t + phi[1]),
            self.config.lissajous_z_center + A[2] * np.sin(w[2] * t + phi[2]),
        ])

    def _velocity_at(self, t: float) -> np.ndarray:
        A = self.config.lissajous_amplitudes
        w = self.config.lissajous_frequencies
        phi = self.config.lissajous_phases
        return np.array([
            A[0] * w[0] * np.cos(w[0] * t + phi[0]),
            A[1] * w[1] * np.cos(w[1] * t + phi[1]),
            A[2] * w[2] * np.cos(w[2] * t + phi[2]),
        ])

    def step(self, state: EagleState, dt: float) -> EagleState:
        self._time += dt
        pos = self._position_at(self._time)
        vel = self._velocity_at(self._time)
        return EagleState(
            position=pos, velocity=vel,
            temperature=state.temperature, radius=state.radius,
        )


def create_motion_generator(eagle_config: EagleConfig, world_config: WorldConfig) -> MotionGenerator:
    """Factory for motion generators."""
    if eagle_config.motion_type == MotionType.RANDOM_WALK:
        return RandomWalkMotion(eagle_config, world_config)
    elif eagle_config.motion_type == MotionType.SPLINE:
        return SplineMotion(eagle_config, world_config)
    elif eagle_config.motion_type == MotionType.LISSAJOUS:
        return LissajousMotion(eagle_config, world_config)
    else:
        raise ValueError(f"Unknown motion type: {eagle_config.motion_type}")


def save_trajectory(trajectory: np.ndarray, path: str) -> None:
    np.savez(path, trajectory=trajectory)


def load_trajectory(path: str) -> np.ndarray:
    data = np.load(path)
    return data["trajectory"]
