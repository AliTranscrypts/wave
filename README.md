# Virtual Multi-View Thermal Tracking Simulation: High‑Level Technical Plan

## 1. System Overview

We will build a Python‑native simulation environment that models a national park airspace, multiple thermal cameras, a moving eagle, and a voxel‑based Multi‑View Stereo (MVS) tracking system. The simulation serves to validate camera placement, fusion algorithms, and tracking accuracy before hardware deployment. Key components:

* **World** – 3D Euclidean space with a bounding volume (e.g., 5 km × 5 km × 500 m).
* **Eagle** – A moving object emitting thermal radiation, represented as a 3D point or a small sphere with known temperature.
* **Cameras** – Virtual thermal sensors with full intrinsic/extrinsic parameters, capable of rendering synthetic thermal images.
* **MVS Tracker** – A sparse voxel grid that fuses observations from all cameras to estimate the eagle's 3D position and trajectory.

## 2. Simulation Environment

### 2.1 World Representation

* Use a right‑handed coordinate system with Z up (or Y up, consistent with rendering libraries).
* Define world bounds: `(x_min, x_max)`, `(y_min, y_max)`, `(z_min, z_max)`.
* Optionally include a simple terrain mesh (e.g., a plane) to simulate ground clutter, but not necessary for initial validation.

### 2.2 Eagle Motion Model

* Represent the eagle as a point with a fixed temperature $T_e$ (e.g., 35 °C) and a radius $r_e$ (e.g., 0.5 m) to simulate its thermal footprint.
* Motion: generate a trajectory that stays within a horizontal plane (or a slightly varying altitude) to mimic typical flight. Options:
  * Random walk with bounded velocity.
  * Spline‑based path (Catmull‑Rom) for smooth, realistic motion.
  * Parameterized curve (e.g., Lissajous) for repeatable tests.
* For each time step $t$, store ground truth position $\mathbf{p}_t$.

### 2.3 Thermal Rendering Model

For each camera, generate a synthetic thermal image at each frame. Use a simplified radiometric model:

* Background sky temperature $T_{sky}$ (e.g., –10 °C) and optional cloud patterns (cold).
* Eagle appears as a hot blob; we render its projection onto the image plane using a point‑spread function (Gaussian) to simulate lens blur and sensor integration.
* Additional effects:
  * **Atmospheric attenuation** – intensity decays with distance $d$: $I = I_0 \cdot e^{-\alpha d}$.
  * **Vignetting** – radial intensity falloff.
  * **Noise** – additive Gaussian, photon shot noise, fixed‑pattern noise.

Rendering method options (ordered by complexity):

1. **Direct geometric projection** – For each camera, compute which pixels are intersected by the eagle's projected circle; fill those pixels with a temperature value (fast, but no shadows or occlusions).
2. **Ray casting / rasterization** – Use a lightweight renderer like pyrender (OpenGL‑based) to render a mesh representing the eagle; assign material properties with temperature. This handles occlusion naturally if multiple eagles or obstacles are added.
3. **Custom ray tracer** – For full control over thermal physics, implement a ray‑casting loop that shoots rays from camera, tests intersection with eagle (sphere), and computes radiance.

Given Python performance constraints, option 1 (projection + Gaussian blur) is often sufficient for algorithm validation and is easy to implement with numpy and opencv.

## 3. Camera Simulation

### 3.1 Intrinsic Parameters

* Model the pinhole plus lens distortion.
* Define:
  * Focal length $f_x, f_y$ (pixels).
  * Principal point $c_x, c_y$.
  * Distortion coefficients $k_1, k_2, p_1, p_2, k_3$ (Brown‑Conrady model).
* Use opencv's `cv2.projectPoints` to convert world points to pixel coordinates.

### 3.2 Extrinsic Parameters

* Position $\mathbf{C}$ in world coordinates.
* Orientation: Euler angles (roll, pitch, yaw) or quaternion. Convert to rotation matrix $\mathbf{R}$.
* Build projection matrix $\mathbf{P} = \mathbf{K} [\mathbf{R} | \mathbf{t}]$ where $\mathbf{t} = -\mathbf{R}\mathbf{C}$.

### 3.3 Camera Configuration

* Define a set of cameras with realistic placements (e.g., on poles around the park, pointing upward).
* Provide each camera with a unique ID, intrinsic matrix, distortion, and extrinsic transform.

## 4. Synthetic Data Generation Pipeline

For each time step $t$:

1. Update eagle position $\mathbf{p}_t$ based on motion model.
2. For each camera:
   * Project the eagle's center and its sphere onto the image plane.
   * Create an empty thermal image (2D array) filled with sky temperature.
   * Draw the eagle's projected footprint as a Gaussian blob:
     * Compute the projected ellipse (circle becomes ellipse due to perspective).
     * For each pixel within bounding box, compute intensity:

       $$I(u,v) = T_{sky} + (T_e - T_{sky}) \cdot \exp\left(-\frac{r^2}{2\sigma^2}\right)$$

       where $r$ is the distance from pixel to the projected center in image coordinates, and $\sigma$ models blur (e.g., 2–3 pixels).
     * Apply distance‑based attenuation $e^{-\alpha |\mathbf{p}_t - \mathbf{C}|}$ to the excess temperature.
   * Add noise (Gaussian, etc.).
   * Apply vignetting (multiply by a radial falloff).
   * Optionally apply lens distortion (if not handled by projection).
3. Store:
   * The synthetic thermal image (e.g., as a numpy array, float32).
   * Ground truth 2D projections (optional for debugging).
   * Ground truth 3D position $\mathbf{p}_t$.

All synthetic images and ground truth are saved or streamed to the MVS tracker.

## 5. Voxel‑Based MVS Tracking

### 5.1 Voxel Grid Definition

* Choose a sparse representation to handle the large airspace:
  * **Octree** – only allocate voxels near detections.
  * **Hash map** – keyed by integer grid coordinates.
* Voxel size: trade‑off between accuracy and memory. Start with $1\,m^3$ in a region of interest (e.g., 500 m × 500 m × 200 m) and expand dynamically.
* Each voxel stores:
  * Occupancy probability $p$ (initialized to 0.5).
  * Last update timestamp.
  * Optionally, accumulated evidence (e.g., log‑odds ratio).

### 5.2 Occupancy Fusion per Frame

For each frame, for each voxel (in the active set), perform:

1. Project the voxel center to each camera (using `cv2.projectPoints` with the camera's intrinsics and extrinsics).
2. If the projection falls within the image bounds:
   * Read the corresponding pixel value (temperature) from the synthetic image.
   * Compute the likelihood $P(\text{hot} \mid \text{voxel occupied})$ based on whether the pixel exceeds a threshold (adaptive, e.g., 2σ above sky mean).
3. Combine evidence across cameras using a Bayesian update:

   $$\text{logit}(p_{\text{new}}) = \text{logit}(p_{\text{old}}) + \sum_{c} \log\frac{P_c(\text{hot} \mid \text{occupied})}{P_c(\text{hot} \mid \text{empty})}$$

   where $P_c(\text{hot} \mid \text{empty})$ is the probability of a hot pixel given no eagle (e.g., noise floor).
4. Threshold: voxels with $p > 0.8$ are considered "occupied" (potential eagle parts).

### 5.3 Space Carving Alternative

For real‑time scenarios, a simpler approach:

* For each camera, threshold its image to get a binary "eagle mask".
* For each mask pixel, cast a ray into the world; all voxels along that ray get a "vote".
* A voxel is considered occupied only if it receives votes from a minimum number of cameras (e.g., 2) and not vetoed by any camera seeing empty space along that ray.

### 5.4 Clustering and Tracking

* Group occupied voxels into connected components (3D connected‑component analysis, e.g., using `scipy.ndimage.label` on a dense subgrid or custom flood‑fill on the octree).
* Each cluster corresponds to a candidate eagle.
* Compute centroid of each cluster.
* Tracking: associate centroids across frames using Hungarian algorithm (nearest neighbor) and optionally apply a Kalman filter to smooth trajectory.

## 6. Evaluation Metrics

* **Position error**: Euclidean distance between tracked centroid and ground truth eagle position.
* **Detection rate**: ratio of frames where eagle is correctly detected (cluster exists near ground truth).
* **False positive rate**: number of frames with spurious clusters.
* **Processing time**: per‑frame fusion and tracking latency.

## 7. Implementation Roadmap (Phased)

### Phase 1: Core Simulation

* Define `Camera` class with intrinsics/extrinsics, projection methods.
* Implement eagle motion (simple random walk).
* Render thermal images using geometric projection + Gaussian blob (no noise).
* Verify correct projection by visualizing images and checking pixel‑space trajectory.

### Phase 2: Voxel Reconstruction (Offline)

* Implement dense voxel grid (small region) with Bayesian fusion.
* Process stored synthetic images frame‑by‑frame.
* Visualize occupied voxels (e.g., using matplotlib 3D scatter or open3d).

### Phase 3: Sparse & Real‑Time

* Replace dense grid with hash‑based sparse voxel map.
* Implement incremental updates.
* Add clustering and tracking.

### Phase 4: Realism Enhancements

* Add noise, vignetting, atmospheric attenuation to rendering.
* Add multiple eagles (for testing false associations).
* Add simple terrain obstacles.

### Phase 5: Validation and Export

* Run parameter sweeps (voxel size, number of cameras, noise level).
* Export optimal camera configurations and algorithm parameters for real‑world deployment.

## 8. Recommended Python Libraries

| Library | Purpose |
|---|---|
| `numpy` | Core array operations, geometry math |
| `opencv-python` | Camera projection (`cv2.projectPoints`), distortion, image processing |
| `scipy` | Sparse matrices, connected components, spline interpolation |
| `pyrender` / `trimesh` | (Optional) for more realistic rendering with mesh objects |
| `scikit-learn` | Clustering (DBSCAN), nearest neighbors |
| `matplotlib` / `plotly` / `open3d` | 3D visualization of voxels, cameras, trajectory |
| `numba` | (Optional) for accelerating projection loops over many voxels |
| `pyvista` | Alternative for 3D visualization and mesh handling |

## 9. Data Flow Diagram

```
Simulation Loop:
  For t = 0..T:
    Eagle.update_position(t)
    For each camera:
      img = render_thermal_image(eagle, camera, t)
      store img (or stream to tracker)
    (Optionally store ground truth 3D position)

Tracking Loop (could be offline or online):
  For each frame:
    For each active voxel (or all if dense):
      For each camera:
        proj = camera.project(voxel.center)
        if proj in image:
          update occupancy probability (Bayesian)
    Cluster occupied voxels
    Track clusters (Kalman / Hungarian)
    Compute error vs ground truth
```

## 10. Technical Notes

* **Synchronization**: In simulation, all cameras produce images at the same time stamp. The MVS algorithm should process them as a batch.
* **Performance**: Projecting every voxel against every camera per frame is costly. Use spatial hashing to limit voxels to regions near predicted eagle location (e.g., using previous frame's cluster centroid). Also, use numba to accelerate loops.
* **Extensibility**: Design `Camera`, `Eagle`, `VoxelGrid` as classes with clear interfaces to allow swapping of rendering methods or fusion logic.

## 11. Additional Considerations

### 11.1 Coordinate System and Units Convention

* Establish a clear convention document: all distances in meters, temperatures in Celsius (or Kelvin internally for radiometric calculations), angles in radians internally (degrees for configuration files).
* Define the origin point (e.g., center of the park or a known GPS reference point) for future real‑world alignment.

### 11.2 Configuration Management

* Use a YAML or JSON configuration file to define all simulation parameters (world bounds, camera placements, eagle motion parameters, noise levels, voxel sizes) so experiments are reproducible without code changes.
* Version configuration files alongside code for experiment tracking.

### 11.3 Sensor Resolution and Frame Rate

* Define default thermal camera resolution (e.g., 640×512 for FLIR‑style sensors or 320×256 for lower‑cost options).
* Define simulation frame rate (e.g., 10–30 Hz) and ensure the motion model's time step is consistent with this.
* Consider the Nyquist criterion: the frame rate must be high enough to capture the eagle's motion without aliasing (at max eagle speed of ~20 m/s and voxel size of 1 m, a minimum of 20 fps is needed).

### 11.4 Field of View and Coverage Analysis

* Before running full simulations, compute the overlap map of camera frustums to ensure sufficient multi‑view coverage (at least 2 cameras covering every point in the region of interest).
* Provide a utility to visualize camera frustums in 3D and compute coverage statistics.

### 11.5 Calibration Simulation

* Include a simulated calibration step where known 3D points (e.g., thermal targets at fixed positions) are projected into each camera to verify that the projection pipeline is correct end‑to‑end.
* This validates the intrinsic/extrinsic pipeline before running tracking experiments.

### 11.6 Temporal Decay and Voxel Pruning

* Implement a temporal decay mechanism for voxel occupancy: voxels that haven't received positive evidence for $N$ frames should have their probability decay toward 0.5 (prior) to avoid stale detections.
* Periodically prune voxels with near‑prior probability from the sparse map to bound memory usage.

### 11.7 Multi‑Target Data Association

* Even in Phase 1 with a single eagle, design the tracking interface to handle multiple detections per frame (for robustness to false positives).
* The Hungarian algorithm should operate on a cost matrix that includes a "missed detection" and "new track" column/row to handle appearance/disappearance gracefully.

### 11.8 Logging and Reproducibility

* Log all random seeds used for motion generation and noise so that any simulation run can be exactly reproduced.
* Store simulation results (trajectories, error metrics, configuration snapshots) in a structured format (e.g., HDF5 or a directory of CSV/NPZ files) for post‑hoc analysis.

### 11.9 Testing Strategy

* **Unit tests**: Verify camera projection (known 3D point → expected pixel), motion model bounds, voxel indexing.
* **Integration tests**: Run a short simulation (e.g., 10 frames) and verify that tracking error is below a known bound for a deterministic trajectory.
* **Regression tests**: Store baseline metrics and alert if tracking accuracy degrades after code changes.

### 11.10 Real‑World Transition Path

* Document which components are simulation‑only (rendering) vs. shared with production (camera model, voxel tracker, clustering).
* Design a `CameraSource` abstraction that can be backed by the synthetic renderer or a real camera feed (e.g., via GStreamer or a frame‑grabber SDK).
* Plan for clock synchronization (PTP/NTP) in real deployment, even though simulation assumes perfect sync.

## 12. Conclusion

This plan provides a complete technical framework for a Python‑native virtual simulation of a multi‑view thermal tracking system. By implementing this, a computer scientist or software engineer can validate the voxel‑based MVS approach, optimize camera configurations, and develop robust tracking algorithms before any real‑world hardware is deployed. The modular design ensures that the final tracking code can be directly transferred to a live system with only the rendering backend replaced by actual camera feeds.
