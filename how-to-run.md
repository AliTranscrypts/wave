# How to Run — Virtual Multi-View Thermal Tracking Simulation

## Prerequisites

- **Python 3.10+** (tested on 3.11)
- **pip** (comes with Python)
- **Git** (to clone the repo)

## 1. Clone and Install

```bash
git clone https://github.com/AliTranscrypts/wave.git
cd wave
git checkout claude/start-app-implementation-oOMaO

# Install the package and all dependencies (editable mode)
pip install -e ".[dev]"
```

This installs:
- `numpy`, `opencv-python-headless`, `scipy`, `scikit-learn` — core computation
- `pydantic`, `pyyaml` — configuration management
- `matplotlib` — visualization
- `pytest`, `pytest-cov` — testing (dev only)

## 2. Verify Installation

```bash
python -m pytest tests/ -v
```

You should see **39 tests passing**.

## 3. Run a Simulation

### Quick Start (Python script)

Create a file called `run_simulation.py`:

```python
from thermal_tracker.config import load_config
from thermal_tracker.engine import SimulationEngine, SimulationOutputMode
from thermal_tracker.clustering import ClusteringConfig
from thermal_tracker.fusion import FusionConfig
from thermal_tracker.tracking import TrackingConfig
from thermal_tracker.voxel_grid import VoxelGridConfig
from thermal_tracker.pipeline import TrackingPipeline, run_online_simulation, generate_report

# 1. Load config
config = load_config("configs/default_config.yaml")
config.run.num_frames = 100  # adjust as needed

# 2. Create simulation engine
engine = SimulationEngine(config)
print(f"Cameras: {[c.id for c in engine.cameras]}")

# 3. Set up tracking pipeline
voxel_config = VoxelGridConfig(
    voxel_size=5.0,
    roi_min=[-400.0, -400.0, 50.0],
    roi_max=[400.0, 400.0, 200.0],
    occupancy_threshold=0.7,
)
fusion_config = FusionConfig(
    detection_threshold_sigma=2.0,
    min_cameras_for_update=2,
)
clustering_config = ClusteringConfig(
    method="connected_components",
    min_cluster_size=1,
    max_cluster_size=100,
)
tracking_config = TrackingConfig(
    max_association_distance=30.0,
    min_hits_to_confirm=2,
    dt=config.run.dt,
)

pipeline = TrackingPipeline(
    cameras=engine.cameras,
    voxel_config=voxel_config,
    fusion_config=fusion_config,
    clustering_config=clustering_config,
    tracking_config=tracking_config,
)

# 4. Run end-to-end
print("Running simulation + tracking...")
results = run_online_simulation(engine, pipeline)

# 5. Get ground truth and evaluate
engine2 = SimulationEngine(config)
sim_result = engine2.run(mode=SimulationOutputMode.BATCH)
gt = sim_result.trajectory

report = generate_report(results, gt, distance_threshold=20.0)

print(f"\n=== Evaluation Report ===")
print(f"Frames:              {report.num_frames}")
print(f"Detection rate:      {report.detection_rate:.1%}")
print(f"Mean position error: {report.mean_error:.2f} m")
print(f"RMSE:                {report.rmse:.2f} m")
print(f"Precision:           {report.precision:.1%}")
print(f"Recall:              {report.recall:.1%}")
print(f"F1 score:            {report.f1_score:.1%}")
print(f"First detection:     frame {report.frames_to_first_detection}")
print(f"Avg processing time: {report.mean_processing_time_ms:.1f} ms/frame")
```

Run it:

```bash
python run_simulation.py
```

### Simulation Only (no tracking)

```python
from thermal_tracker.config import load_config
from thermal_tracker.engine import SimulationEngine, SimulationOutputMode

config = load_config("configs/default_config.yaml")
engine = SimulationEngine(config)
result = engine.run(mode=SimulationOutputMode.BATCH)

print(f"Generated {result.num_frames} frames in {result.wall_clock_seconds:.2f}s")
print(f"Trajectory shape: {result.trajectory.shape}")

# Access a frame
frame = result.frame_bundles[0]
print(f"Frame 0 cameras: {list(frame.camera_images.keys())}")
print(f"Eagle position:  {frame.ground_truth_3d}")
```

### Save Simulation to Disk

```python
from thermal_tracker.engine import SimulationEngine, NpzWriter
from thermal_tracker.config import load_config

config = load_config("configs/default_config.yaml")
engine = SimulationEngine(config)

writer = NpzWriter()
writer.open("output/my_run", config)

for frame in engine.run_streaming():
    writer.write_frame(frame)

writer.close()
print("Saved to output/my_run/")
```

### Load and Replay Saved Data

```python
from thermal_tracker.engine import NpzReader

reader = NpzReader()
config = reader.open("output/my_run")
print(f"Frames: {reader.num_frames()}")

trajectory = reader.read_trajectory()
frame_0 = reader.read_frame(0)
```

## 4. Configuration

All simulation parameters are in `configs/default_config.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `world` | 3D bounding volume (5km x 5km x 500m default) |
| `eagle` | Temperature, radius, motion type, trajectory params |
| `cameras` | Number, positions, orientations, FOV |
| `rendering` | Sky temp, noise, vignetting, attenuation |
| `run` | Number of frames, time step, random seed |

### Change Eagle Motion Type

In `configs/default_config.yaml`, set `eagle.motion_type` to:
- `lissajous` — deterministic figure-8 pattern (default, best for testing)
- `random_walk` — bounded random walk with velocity limits
- `spline` — smooth path through random control points

### Add More Cameras

Add entries to the `cameras` list, or generate programmatically:

```python
from thermal_tracker.camera import generate_ring_placement

configs = generate_ring_placement(
    num_cameras=6,
    ring_radius=500.0,
    pole_height=10.0,
    hfov_deg=45.0,
)
```

### Enable Noise

```yaml
rendering:
  noise:
    enabled: true
    gaussian_std: 0.05    # NETD in °C
    shot_noise_enabled: false
    fixed_pattern_noise_std: 0.02
```

## 5. Project Structure

```
src/thermal_tracker/
├── world.py          # World bounds & coordinate system
├── camera.py         # Camera intrinsics, extrinsics, projection
├── eagle.py          # Eagle state & 3 motion generators
├── rendering.py      # Thermal image rendering pipeline
├── config.py         # YAML configuration management
├── engine.py         # Simulation engine & NPZ I/O
├── voxel_grid.py     # Sparse voxel grid with decay/pruning
├── fusion.py         # Bayesian occupancy fusion & space carving
├── clustering.py     # Connected components & DBSCAN clustering
├── tracking.py       # Hungarian tracker + Kalman filter
└── pipeline.py       # End-to-end tracking pipeline & evaluation

configs/
└── default_config.yaml   # Default simulation parameters

tests/                    # 39 unit + integration tests
```

## 6. Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_camera.py -v

# With coverage
python -m pytest tests/ --cov=thermal_tracker --cov-report=term-missing
```

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: thermal_tracker` | Run `pip install -e ".[dev]"` from the repo root |
| `cv2` import error | Run `pip install opencv-python-headless` |
| Slow fusion (large ROI) | Reduce ROI size or increase `voxel_size` |
| No detections | Check that eagle trajectory is within camera FOV and ROI bounds |
| Out of memory | Use `engine.run_streaming()` instead of batch mode |
