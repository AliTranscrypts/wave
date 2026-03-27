"""Multi-frame tracking with Hungarian association and Kalman filter (Tasks 5.4.1-5.4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from thermal_tracker.clustering import VoxelCluster


# ---------------------------------------------------------------------------
# Kalman Filter (Task 5.4.2)
# ---------------------------------------------------------------------------

class KalmanFilter3D:
    """6-state (pos + vel) constant-velocity Kalman filter."""

    def __init__(self, dt: float, q: float = 5.0, r: float = 0.5):
        self.dt = dt
        self.state = np.zeros(6)
        self.P = np.eye(6) * 100.0

        # State transition
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Process noise
        G = np.array([dt * dt / 2, dt * dt / 2, dt * dt / 2, dt, dt, dt])
        self.Q = q * np.outer(G, G)

        # Measurement model (observe position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Measurement noise
        self.R = np.eye(3) * r * r

    def initialize(self, measurement: np.ndarray) -> None:
        self.state[:3] = measurement
        self.state[3:] = 0.0
        self.P = np.diag([self.R[0, 0], self.R[1, 1], self.R[2, 2], 100.0, 100.0, 100.0])

    def predict(self) -> np.ndarray:
        """Predict step. Returns predicted position."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.H @ self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update step with a position measurement. Returns filtered position."""
        y = measurement - self.H @ self.state  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.H @ self.state

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_position(self) -> np.ndarray:
        return self.state[:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.state[3:].copy()


# ---------------------------------------------------------------------------
# Track data structures
# ---------------------------------------------------------------------------

class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class TrackPoint:
    frame_index: int
    centroid: np.ndarray
    cluster: VoxelCluster | None = None


@dataclass
class Track:
    track_id: int
    history: list[TrackPoint] = field(default_factory=list)
    state: TrackState = TrackState.TENTATIVE
    frames_since_last_detection: int = 0
    kalman: KalmanFilter3D | None = None
    consecutive_hits: int = 0

    @property
    def last_position(self) -> np.ndarray | None:
        if self.history:
            return self.history[-1].centroid
        return None

    @property
    def predicted_position(self) -> np.ndarray | None:
        if self.kalman:
            return self.kalman.get_position()
        return self.last_position


class TrackingConfig(BaseModel):
    """Multi-frame tracking parameters."""
    max_association_distance: float = 20.0  # meters
    min_hits_to_confirm: int = 3
    max_frames_to_coast: int = 5
    max_tracks: int = 10
    kalman_q: float = 5.0
    kalman_r: float = 0.5
    dt: float = 0.1


# ---------------------------------------------------------------------------
# Tracker (Task 5.4.1)
# ---------------------------------------------------------------------------

class Tracker:
    """Multi-target tracker with Hungarian association and Kalman filtering."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._tracks: list[Track] = []
        self._next_id: int = 1

    def update(self, detections: list[VoxelCluster], frame_index: int) -> list[Track]:
        """Process detections for one frame. Returns active (non-lost) tracks."""
        # Predict all existing tracks
        for track in self._tracks:
            if track.state != TrackState.LOST and track.kalman:
                track.kalman.predict()

        # Build cost matrix
        active_tracks = [t for t in self._tracks if t.state != TrackState.LOST]
        n_tracks = len(active_tracks)
        n_dets = len(detections)

        if n_tracks == 0 and n_dets == 0:
            return self.get_active_tracks()

        if n_tracks == 0:
            # All detections start new tracks
            for det in detections:
                self._create_track(det, frame_index)
            return self.get_active_tracks()

        if n_dets == 0:
            # All tracks are missed
            for track in active_tracks:
                self._miss_track(track)
            return self.get_active_tracks()

        # Cost matrix: Euclidean distance between predicted position and detection centroid
        max_dist = self.config.max_association_distance
        cost = np.full((n_tracks + n_dets, n_dets + n_tracks), max_dist, dtype=np.float64)

        for i, track in enumerate(active_tracks):
            pred = track.predicted_position
            if pred is None:
                continue
            for j, det in enumerate(detections):
                dist = np.linalg.norm(pred - det.centroid)
                cost[i, j] = min(dist, max_dist)

        # Dummy rows/cols already filled with max_dist

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if r < n_tracks and c < n_dets and cost[r, c] < max_dist:
                self._update_track(active_tracks[r], detections[c], frame_index)
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched tracks: missed detection
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                self._miss_track(track)

        # Unmatched detections: new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det, frame_index)

        # Enforce max tracks
        active = [t for t in self._tracks if t.state != TrackState.LOST]
        if len(active) > self.config.max_tracks:
            # Keep the ones with longest history
            active.sort(key=lambda t: len(t.history), reverse=True)
            for t in active[self.config.max_tracks:]:
                t.state = TrackState.LOST

        return self.get_active_tracks()

    def _create_track(self, detection: VoxelCluster, frame_index: int) -> Track:
        track = Track(
            track_id=self._next_id,
            state=TrackState.TENTATIVE,
            kalman=KalmanFilter3D(
                dt=self.config.dt, q=self.config.kalman_q, r=self.config.kalman_r
            ),
            consecutive_hits=1,
        )
        track.kalman.initialize(detection.centroid)
        track.history.append(TrackPoint(frame_index, detection.centroid.copy(), detection))
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _update_track(self, track: Track, detection: VoxelCluster, frame_index: int) -> None:
        if track.kalman:
            track.kalman.update(detection.centroid)
        track.history.append(TrackPoint(frame_index, detection.centroid.copy(), detection))
        track.frames_since_last_detection = 0
        track.consecutive_hits += 1

        if track.state == TrackState.TENTATIVE and track.consecutive_hits >= self.config.min_hits_to_confirm:
            track.state = TrackState.CONFIRMED

    def _miss_track(self, track: Track) -> None:
        track.frames_since_last_detection += 1
        track.consecutive_hits = 0

        if track.state == TrackState.TENTATIVE:
            track.state = TrackState.LOST
        elif track.frames_since_last_detection > self.config.max_frames_to_coast:
            track.state = TrackState.LOST

    def get_active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state != TrackState.LOST]

    def get_all_tracks(self) -> list[Track]:
        return list(self._tracks)
