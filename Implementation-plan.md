# Task Document: Section 2 (Simulation Environment) & Section 3 (Camera Simulation)

**Scope:** All implementation work required to stand up the simulation world, eagle motion model, thermal rendering pipeline, and virtual camera system described in Sections 2 and 3 of the high-level technical plan.

**Audience:** Computer scientists, software engineers, system design engineers, physicists.

---

## SECTION 2 — SIMULATION ENVIRONMENT

---

### 2.1 World Representation

#### Task 2.1.1 — Define the Coordinate System and World Volume

**Objective:** Establish the canonical coordinate frame, units convention, and bounding volume that all downstream components (cameras, eagle, voxel grid) will reference.

**Sub-tasks:**

- **2.1.1a** Choose and document the handedness and axis assignment. The plan specifies right-handed with Z-up. If any rendering library (e.g., pyrender uses Y-up by default in OpenGL convention) is adopted later, define the transform between the simulation frame and the renderer frame explicitly as a 4×4 homogeneous matrix and apply it at the rendering boundary only — all internal math stays in the canonical frame.

- **2.1.1b** Define the `WorldConfig` data structure (or dataclass) containing:
  - `origin`: 3-vector, world-frame origin in meters. For future real-world alignment, this should map to a known geodetic coordinate (lat/lon/alt), but for simulation phase it can be `(0, 0, 0)`.
  - `bounds`: 6-tuple `(x_min, x_max, y_min, y_max, z_min, z_max)` in meters. Default: `(-2500, 2500, -2500, 2500, 0, 500)` giving the 5 km × 5 km × 500 m volume.
  - `units`: string enum, locked to `"meters"` for distance, `"celsius"` for temperature at the config layer (Kelvin internally for radiometric calculations), `"radians"` for angles internally.

- **2.1.1c** Implement bounds-checking utility: a function `is_within_bounds(point: np.ndarray) -> bool` and a clamping variant `clamp_to_bounds(point) -> np.ndarray` that projects a point onto the nearest boundary face. This will be consumed by the eagle motion model (Section 2.2) and the voxel grid (Section 5).

- **2.1.1d** (Optional, low priority) Define a ground-plane mesh: a single triangulated quad at `z = 0` with an assigned ground temperature `T_ground` (e.g., 15 °C). This supports future ground-clutter rendering but is not required for initial validation. If implemented, represent it as a `trimesh.Trimesh` object for compatibility with pyrender if that path is later adopted.

**Deliverables:**
- `WorldConfig` dataclass with serialization to/from YAML.
- Bounds utility functions with unit tests (point inside, point outside, clamping edge cases).
- Convention document (can be a docstring or a short markdown file) specifying axis orientation, units, and the simulation-to-renderer transform if applicable.

**Dependencies:** None. This is a root task.

**Acceptance criteria:**
- A round-trip test: instantiate `WorldConfig` from YAML, serialize back, compare equality.
- `is_within_bounds` and `clamp_to_bounds` pass tests for interior points, boundary points, and exterior points in each axis.

---

#### Task 2.1.2 — Configuration Management Infrastructure

**Objective:** Establish the YAML/JSON-based configuration system so that all simulation parameters (world, cameras, eagle, noise, voxel sizes) are declared externally and loaded at runtime.

**Sub-tasks:**

- **2.1.2a** Choose a schema-validation approach. Options: `pydantic` models (preferred — gives type checking, defaults, and YAML/JSON serde via `pydantic-settings` or manual `yaml.safe_load` + model construction), or `dataclasses` + `jsonschema`. Pydantic is recommended for nested config structures.

- **2.1.2b** Define top-level `SimulationConfig` model that composes:
  - `WorldConfig` (from 2.1.1)
  - `EagleConfig` (from 2.2)
  - `List[CameraConfig]` (from 3.3)
  - `RenderingConfig` (from 2.3)
  - `SimulationRunConfig`: `num_frames`, `dt` (seconds per frame), `random_seed`.

- **2.1.2c** Implement `load_config(path: str) -> SimulationConfig` and `save_config(config, path)`.

- **2.1.2d** Write a default configuration file (`default_config.yaml`) with sensible initial values for a minimal working simulation (e.g., 3 cameras, 100 frames, no noise).

**Deliverables:**
- `SimulationConfig` pydantic model hierarchy.
- `default_config.yaml`.
- Loader/saver utilities.

**Dependencies:** Task 2.1.1 (WorldConfig).

**Acceptance criteria:**
- Loading the default config succeeds with no validation errors.
- Changing a single parameter in YAML and reloading produces the expected change in the in-memory object.
- Invalid values (e.g., negative focal length) raise a validation error.

---

### 2.2 Eagle Motion Model

#### Task 2.2.1 — Eagle State Representation

**Objective:** Define the eagle's state vector and thermal properties.

**Sub-tasks:**

- **2.2.1a** Define `EagleState` dataclass:
  - `position`: `np.ndarray` shape `(3,)` — world coordinates in meters.
  - `velocity`: `np.ndarray` shape `(3,)` — m/s.
  - `temperature`: `float` — surface temperature in Celsius (converted to Kelvin for radiometric use). Default 35 °C.
  - `radius`: `float` — thermal footprint radius in meters. Default 0.5 m (half-wingspan approximation).

- **2.2.1b** Define `EagleConfig` dataclass:
  - `temperature`: float.
  - `radius`: float.
  - `max_speed`: float (m/s). Typical eagle cruising speed ~15–20 m/s.
  - `max_acceleration`: float (m/s²). Bounds for physically plausible maneuvers.
  - `motion_type`: enum `{RANDOM_WALK, SPLINE, LISSAJOUS}`.
  - `altitude_range`: tuple `(z_min, z_max)` to constrain vertical extent.
  - Motion-type-specific parameters (see sub-tasks below).

**Deliverables:**
- `EagleState` and `EagleConfig` dataclasses.
- Serialization into the `SimulationConfig` hierarchy.

**Dependencies:** Task 2.1.1 (coordinate conventions).

---

#### Task 2.2.2 — Motion Generators

**Objective:** Implement three trajectory generation strategies, each conforming to a common interface so they are interchangeable.

**Sub-tasks:**

- **2.2.2a** Define abstract interface `MotionGenerator`:
  ```python
  class MotionGenerator(ABC):
      def reset(self, seed: int) -> EagleState: ...
      def step(self, state: EagleState, dt: float) -> EagleState: ...
      def generate_trajectory(self, num_steps: int, dt: float) -> np.ndarray: ...  # (N, 3)
  ```

- **2.2.2b** Implement `RandomWalkMotion`:
  - At each step, sample an acceleration vector from a zero-mean Gaussian, scale to `max_acceleration`.
  - Integrate velocity: `v_{t+1} = v_t + a * dt`, clamp `||v||` to `max_speed`.
  - Integrate position: `p_{t+1} = p_t + v_{t+1} * dt`.
  - Enforce world bounds via elastic reflection (reverse velocity component on boundary contact) or clamping.
  - Enforce altitude constraints from `EagleConfig.altitude_range`.
  - Seed the RNG from `SimulationRunConfig.random_seed` for reproducibility.

- **2.2.2c** Implement `SplineMotion`:
  - Accept a set of control points (either user-specified or randomly sampled within bounds).
  - Fit a Catmull-Rom spline through them using `scipy.interpolate.CubicSpline` (or manual Catmull-Rom basis matrix).
  - Parameterize by arc length (or approximate via uniform time steps with velocity clamping) to ensure the eagle doesn't exceed `max_speed`.
  - `step()` advances a parameter `t` and evaluates the spline.
  - Expose control point generation as a seeded random process for reproducibility.

- **2.2.2d** Implement `LissajousMotion`:
  - Parameterized by amplitudes `(A_x, A_y, A_z)`, frequencies `(ω_x, ω_y, ω_z)`, and phase offsets `(φ_x, φ_y, φ_z)`.
  - `p(t) = (A_x sin(ω_x t + φ_x), A_y sin(ω_y t + φ_y), z_0 + A_z sin(ω_z t + φ_z))`.
  - Verify that the resulting velocity `||dp/dt||` doesn't exceed `max_speed`; if it does, scale frequencies or amplitudes and warn.
  - Deterministic — no RNG needed, but parameters should be part of `EagleConfig`.

- **2.2.2e** Implement ground-truth storage:
  - For each time step, record `(t, p_x, p_y, p_z, v_x, v_y, v_z)` into an `np.ndarray` of shape `(N, 7)`.
  - Provide `save_trajectory(path)` and `load_trajectory(path)` using `.npz` format.

**Deliverables:**
- Three `MotionGenerator` implementations.
- Trajectory I/O utilities.
- Unit tests: bounds enforcement (no position ever leaves world bounds), velocity cap enforcement, deterministic replay given same seed.

**Dependencies:** Task 2.2.1 (EagleState), Task 2.1.1 (bounds).

**Acceptance criteria:**
- `RandomWalkMotion`: 10,000-step trajectory stays within bounds; max velocity never exceeds cap; two runs with same seed are identical.
- `SplineMotion`: trajectory passes through or near control points; velocity stays within cap.
- `LissajousMotion`: trajectory matches analytical formula to float precision.

---

### 2.3 Thermal Rendering Model

#### Task 2.3.1 — Background Model

**Objective:** Generate the baseline thermal image that represents "no eagle present."

**Sub-tasks:**

- **2.3.1a** Define `RenderingConfig` dataclass:
  - `sky_temperature`: float (°C). Default −10 °C.
  - `ground_temperature`: float (°C). Default 15 °C (only relevant if ground plane is included).
  - `atmospheric_attenuation_coeff`: float `α` (per meter). Typical values for LWIR in clear air: ~0.0001–0.001 m⁻¹. Default 0.0005.
  - `noise_config`: sub-structure (see 2.3.3).
  - `vignetting_strength`: float. 0 = no vignetting, 1 = strong.
  - `blur_sigma_pixels`: float. Default 2.0.
  - `rendering_method`: enum `{PROJECTION, RAYCASTING, CUSTOM_RAYTRACER}`. Default `PROJECTION`.

- **2.3.1b** Implement `generate_background(camera: Camera, config: RenderingConfig) -> np.ndarray`:
  - Returns a `(H, W)` float32 array filled with `sky_temperature` (in °C or Kelvin depending on internal convention).
  - If ground plane is enabled and the camera's frustum intersects it, render ground pixels at `ground_temperature`. This requires ray-plane intersection for each pixel or a rasterization pass.
  - For Phase 1, a uniform fill at `sky_temperature` suffices.

**Deliverables:**
- `RenderingConfig` dataclass integrated into `SimulationConfig`.
- `generate_background()` function.

**Dependencies:** Task 2.1.1 (world config), Task 3.1 (camera intrinsics for image dimensions).

---

#### Task 2.3.2 — Eagle Projection and Thermal Blob Rendering

**Objective:** Render the eagle's thermal signature into a camera's image.

This is the core rendering step. The plan recommends Option 1 (direct geometric projection + Gaussian blob) for initial implementation.

**Sub-tasks:**

- **2.3.2a** Implement `project_eagle(eagle_state: EagleState, camera: Camera) -> dict`:
  - Project the eagle's center `p` to pixel coordinates `(u, v)` using the camera's full projection pipeline (intrinsics + extrinsics + distortion). Use `cv2.projectPoints`.
  - Compute the projected radius: the eagle is a sphere of radius `r_e` at distance `d = ||p - C||` from the camera. The projected radius in pixels is approximately `r_proj = f * r_e / d` where `f` is the focal length in pixels (average of `f_x`, `f_y`, or use the geometric mean).
  - Due to perspective, the circle may become a slight ellipse. For Phase 1, treat as a circle. For later phases, compute the full projected ellipse by projecting multiple points on the sphere's silhouette and fitting an ellipse.
  - Return a dict: `{center_uv: (u, v), projected_radius: r_proj, distance: d, is_visible: bool}`.
  - `is_visible` is `True` iff `(u, v)` falls within the image bounds (with some margin for the blob extent) and `d > 0` (eagle is in front of camera).

- **2.3.2b** Implement `render_thermal_blob(image: np.ndarray, projection: dict, eagle_state: EagleState, config: RenderingConfig) -> np.ndarray`:
  - Compute the excess temperature at the camera: `ΔT = (T_e − T_sky) * exp(−α * d)` where `α` is the atmospheric attenuation coefficient and `d` is the distance.
  - Define a bounding box around `(u, v)` of size `±3σ` where `σ = max(blur_sigma_pixels, r_proj)` (the blur kernel should be at least as wide as the projected footprint).
  - For each pixel `(i, j)` within the bounding box:
    - Compute `r² = (i − u)² + (j − v)²`.
    - Compute intensity contribution: `I(i, j) += ΔT * exp(−r² / (2σ²))`.
  - Clamp the bounding box to image dimensions.
  - Use vectorized numpy operations: create a meshgrid over the bounding box, compute the Gaussian in one vectorized call, add to the image slice.

- **2.3.2c** Handle edge cases:
  - Eagle behind camera (`d < 0` or dot product of view direction and eagle direction is negative): skip rendering.
  - Eagle far outside field of view but blob tails might bleed in: the 3σ bounding box handles this naturally if the center is near the edge.
  - Eagle very close to camera (fills most of the frame): clamp `σ` to prevent absurdly large kernels; warn if the eagle subtends more than, say, 50% of the image.
  - Multiple eagles (future): this function should be called per-eagle, accumulating onto the same image.

**Deliverables:**
- `project_eagle()` and `render_thermal_blob()` functions.
- Unit tests:
  - A known 3D point projects to the expected pixel (cross-referenced with manual calculation or `cv2.projectPoints` directly).
  - The peak pixel value equals `T_sky + ΔT` (within float tolerance) when the eagle is at the projected center.
  - Blob symmetry: the rendered blob is symmetric about `(u, v)`.

**Dependencies:** Task 2.2.1 (EagleState), Task 2.3.1 (background), Task 3.1–3.2 (Camera with projection pipeline).

---

#### Task 2.3.3 — Noise Model

**Objective:** Add realistic sensor noise to the synthetic thermal image.

**Sub-tasks:**

- **2.3.3a** Define `NoiseConfig` dataclass:
  - `enabled`: bool.
  - `gaussian_std`: float — standard deviation of additive Gaussian noise in °C (or equivalent digital counts). Typical NETD (Noise Equivalent Temperature Difference) for uncooled microbolometers: ~0.05 °C.
  - `shot_noise_enabled`: bool — photon shot noise, Poisson-distributed. Scale depends on signal level.
  - `fixed_pattern_noise_std`: float — spatially fixed but temporally constant noise (FPN), simulating per-pixel gain/offset variations. Generated once per camera and added to every frame.
  - `random_seed`: int — for reproducibility of noise realization.

- **2.3.3b** Implement `apply_noise(image: np.ndarray, noise_config: NoiseConfig, fpn_map: np.ndarray | None, rng: np.random.Generator) -> np.ndarray`:
  - Additive Gaussian: `image += rng.normal(0, gaussian_std, image.shape)`.
  - Shot noise: convert temperature to a pseudo-photon count (linear scaling), apply `rng.poisson(count)`, convert back. This is a simplification — in reality the relationship between scene radiance and detector output is nonlinear and depends on the sensor's spectral response, but a linear approximation is adequate for algorithm validation.
  - Fixed-pattern noise: `image += fpn_map`. The `fpn_map` is generated once per camera at initialization as `rng.normal(0, fpn_std, (H, W))`.

- **2.3.3c** Implement `generate_fpn_map(shape: tuple, fpn_std: float, seed: int) -> np.ndarray`.

**Deliverables:**
- `NoiseConfig` dataclass.
- `apply_noise()` and `generate_fpn_map()` functions.
- Unit tests: statistical tests — mean of noise over many samples ≈ 0, std ≈ configured value.

**Dependencies:** Task 2.3.1 (background image to add noise to).

---

#### Task 2.3.4 — Vignetting Model

**Objective:** Simulate radial intensity falloff from center to edges of the image, as seen in real lens systems.

**Sub-tasks:**

- **2.3.4a** Implement `generate_vignetting_map(shape: tuple, strength: float) -> np.ndarray`:
  - Compute normalized radial distance from image center for each pixel: `r_norm = sqrt((i - cy)² + (j - cx)²) / r_max` where `r_max = sqrt(cy² + cx²)`.
  - Apply a cos⁴ falloff model (physically motivated by the cos⁴ law of illumination falloff): `V(i,j) = 1 - strength * (1 - cos⁴(arctan(r_norm * tan(half_fov))))`.
  - Simplified alternative: `V(i,j) = 1 - strength * r_norm²` (parabolic approximation). Sufficient for validation.
  - The map is static per camera — generate once and cache.

- **2.3.4b** Implement `apply_vignetting(image: np.ndarray, vignetting_map: np.ndarray) -> np.ndarray`:
  - In radiometric terms, vignetting is multiplicative on radiance, but since we are working in temperature space (which is not linearly proportional to radiance), the correct approach is:
    1. Convert temperature to radiance: `L = σ * T⁴` (Stefan-Boltzmann, simplified — full Planck integration over the sensor's spectral band is more accurate but overkill for validation).
    2. Multiply: `L_vignetted = L * V`.
    3. Convert back to temperature: `T_vignetted = (L_vignetted / σ)^(1/4)`.
  - Alternatively, for small perturbations and algorithm validation purposes, a direct multiplicative approximation on the temperature difference from sky is acceptable: `T_out = T_sky + (T_in - T_sky) * V`. This preserves the sky background at edges while reducing the eagle signal, which is the physically relevant effect.

**Deliverables:**
- `generate_vignetting_map()` and `apply_vignetting()` functions.
- Unit test: center pixel is unattenuated, corner pixels are attenuated by the expected factor.

**Dependencies:** Task 2.3.1 (image dimensions), Task 3.1 (camera intrinsics for principal point).

---

#### Task 2.3.5 — Atmospheric Attenuation

**Objective:** Model the decay of thermal signal with distance between eagle and camera.

**Sub-tasks:**

- **2.3.5a** This is already folded into Task 2.3.2b (the `exp(−α * d)` factor applied to `ΔT`). This task is about parameterization and validation:
  - Verify that the attenuation coefficient `α` produces physically reasonable behavior. At `α = 0.0005 m⁻¹`, the signal at 1 km is attenuated to `exp(−0.5) ≈ 60%` of the source excess temperature. At 3 km, `exp(−1.5) ≈ 22%`. Confirm these are acceptable detection margins given the noise floor.
  - Implement a utility function `compute_snr(T_eagle, T_sky, alpha, distance, noise_std) -> float` that returns the signal-to-noise ratio of the eagle blob peak pixel. This is useful for coverage analysis (Section 11.4 of the plan): for a given camera placement, at what range does SNR drop below a detection threshold (e.g., SNR < 5)?

- **2.3.5b** (Optional, Phase 4) Implement wavelength-dependent attenuation if spectral realism is desired. LWIR (8–14 µm) atmospheric transmission can be approximated with MODTRAN lookup tables or a simplified Beer-Lambert model with wavelength-dependent `α(λ)`. This is not needed for initial validation.

**Deliverables:**
- `compute_snr()` utility.
- Validation report: SNR vs. distance plot for default parameters.

**Dependencies:** Task 2.3.2 (rendering pipeline), Task 2.3.3 (noise model).

---

#### Task 2.3.6 — Full Frame Rendering Pipeline

**Objective:** Compose all rendering sub-tasks into a single callable that produces a complete synthetic thermal frame.

**Sub-tasks:**

- **2.3.6a** Implement `render_frame(eagle_state: EagleState, camera: Camera, config: RenderingConfig, rng: np.random.Generator) -> np.ndarray`:
  1. `image = generate_background(camera, config)`.
  2. `proj = project_eagle(eagle_state, camera)`.
  3. If `proj['is_visible']`: `image = render_thermal_blob(image, proj, eagle_state, config)`.
  4. `image = apply_vignetting(image, camera.vignetting_map)`.
  5. `image = apply_noise(image, config.noise_config, camera.fpn_map, rng)`.
  6. Return `image`.

- **2.3.6b** Implement batch rendering: `render_all_cameras(eagle_state, cameras: List[Camera], config, rng) -> List[np.ndarray]`.
  - All cameras render at the same logical timestamp (perfect sync).

- **2.3.6c** Implement frame storage/streaming interface:
  - `FrameBundle` dataclass: `{timestamp: float, camera_images: Dict[str, np.ndarray], ground_truth_3d: np.ndarray, ground_truth_2d: Dict[str, tuple]}`.
  - Save to disk as `.npz` per frame, or accumulate in memory for small simulations.
  - Ground truth 2D is the projected center `(u, v)` in each camera for debugging.

**Deliverables:**
- `render_frame()`, `render_all_cameras()`, `FrameBundle` with I/O.
- Integration test: render 1 frame with 3 cameras, verify eagle blob appears at expected pixel locations by comparing with `cv2.projectPoints` ground truth.

**Dependencies:** Tasks 2.3.1–2.3.5, Task 3 (Camera class).

---

## SECTION 3 — CAMERA SIMULATION

---

### 3.1 Intrinsic Parameters

#### Task 3.1.1 — Camera Intrinsics Representation

**Objective:** Define and implement the camera intrinsic model (pinhole + Brown-Conrady distortion).

**Sub-tasks:**

- **3.1.1a** Define `CameraIntrinsics` dataclass:
  - `width`: int — image width in pixels.
  - `height`: int — image height in pixels.
  - `fx`, `fy`: float — focal lengths in pixels.
  - `cx`, `cy`: float — principal point in pixels (typically near `(width/2, height/2)` but not necessarily exact).
  - `dist_coeffs`: `np.ndarray` shape `(5,)` — `[k1, k2, p1, p2, k3]` (Brown-Conrady). Default all zeros (no distortion) for Phase 1.

- **3.1.1b** Implement `CameraIntrinsics.K() -> np.ndarray`:
  - Returns the 3×3 intrinsic matrix:
    ```
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    ```

- **3.1.1c** Provide factory methods for common thermal sensor configurations:
  - `CameraIntrinsics.flir_640x512(hfov_deg: float)` — computes `fx = width / (2 * tan(hfov/2))`, sets `cx = width/2`, `cy = height/2`, zero distortion. Typical HFOV for FLIR thermal: 24°–45°.
  - `CameraIntrinsics.low_cost_320x256(hfov_deg: float)` — same logic, smaller resolution.

- **3.1.1d** Implement `CameraIntrinsics.compute_fov() -> tuple`:
  - Returns `(hfov, vfov)` in radians, computed from focal lengths and sensor dimensions.
  - Useful for frustum visualization (Section 11.4).

**Deliverables:**
- `CameraIntrinsics` dataclass with `K()`, factory methods, `compute_fov()`.
- Unit test: factory-constructed intrinsics have correct FOV round-trip.

**Dependencies:** None.

---

#### Task 3.1.2 — Distortion and Undistortion

**Objective:** Wrap OpenCV's distortion/undistortion for use in the projection pipeline.

**Sub-tasks:**

- **3.1.2a** Implement `distort_points(points_2d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray`:
  - Takes ideal (undistorted) normalized image coordinates and applies Brown-Conrady distortion.
  - This is needed if you compute projection analytically (pinhole) and then want to add distortion.
  - Note: `cv2.projectPoints` handles this internally when distortion coefficients are passed, so this function is primarily for cases where you bypass OpenCV (e.g., custom ray tracer).

- **3.1.2b** Implement `undistort_points(points_2d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray`:
  - Wraps `cv2.undistortPoints`. Needed in the tracking pipeline when converting detections back to rays.

- **3.1.2c** Implement `undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray`:
  - Wraps `cv2.undistort`. Precompute the undistortion maps (`cv2.initUndistortRectifyMap`) and cache them per camera for efficiency.

**Deliverables:**
- Distortion utility functions.
- Unit test: distort then undistort a set of points → recover original within tolerance. Test with non-trivial distortion coefficients.

**Dependencies:** Task 3.1.1 (CameraIntrinsics).

---

### 3.2 Extrinsic Parameters

#### Task 3.2.1 — Camera Extrinsics Representation

**Objective:** Define the camera pose (position + orientation) in world coordinates and derive the view/projection matrices.

**Sub-tasks:**

- **3.2.1a** Define `CameraExtrinsics` dataclass:
  - `position`: `np.ndarray` shape `(3,)` — camera center `C` in world coordinates (meters).
  - `orientation`: stored as a quaternion `(w, x, y, z)` internally for interpolation stability, with constructors from:
    - Euler angles `(roll, pitch, yaw)` — define the rotation order explicitly (e.g., ZYX extrinsic = XYZ intrinsic). Use `scipy.spatial.transform.Rotation` for conversions.
    - Rotation matrix `R` (3×3 SO(3)).
    - Look-at specification: `look_at(target: np.ndarray, up: np.ndarray = [0,0,1])` — constructs the rotation that points the camera's optical axis (−Z in camera frame for OpenCV convention) toward `target`.

- **3.2.1b** Implement derived properties:
  - `R() -> np.ndarray`: 3×3 rotation matrix (world-to-camera rotation).
  - `t() -> np.ndarray`: translation vector `t = −R @ C`.
  - `Rt() -> np.ndarray`: 3×4 `[R | t]` matrix.
  - `view_direction() -> np.ndarray`: unit vector along the camera's optical axis in world coordinates.

- **3.2.1c** Implement `world_to_camera(point_world: np.ndarray) -> np.ndarray`:
  - Transforms a 3D world point to camera coordinates: `p_cam = R @ p_world + t`.
  - Handle both single points `(3,)` and batches `(N, 3)`.

**Deliverables:**
- `CameraExtrinsics` dataclass with all constructors and derived properties.
- Unit tests:
  - `look_at` a known target → verify `view_direction()` is parallel to `(target - position)`.
  - Round-trip: Euler → quaternion → rotation matrix → Euler matches within tolerance.
  - `world_to_camera(position)` returns `(0, 0, 0)` (camera center maps to origin in camera frame).

**Dependencies:** Task 2.1.1 (coordinate conventions).

---

### 3.3 Camera Configuration and Full Projection Pipeline

#### Task 3.3.1 — Camera Class

**Objective:** Compose intrinsics and extrinsics into a single `Camera` object that provides the full 3D-to-2D projection pipeline.

**Sub-tasks:**

- **3.3.1a** Define `Camera` class:
  - `id`: string — unique identifier.
  - `intrinsics`: `CameraIntrinsics`.
  - `extrinsics`: `CameraExtrinsics`.
  - `vignetting_map`: `np.ndarray` — precomputed, shape `(H, W)`. Generated at construction time from `RenderingConfig.vignetting_strength`.
  - `fpn_map`: `np.ndarray` — precomputed fixed-pattern noise map. Generated at construction time.
  - `undistort_maps`: tuple — precomputed `cv2.initUndistortRectifyMap` output, cached.

- **3.3.1b** Implement `Camera.project(points_3d: np.ndarray) -> np.ndarray`:
  - Input: `(N, 3)` world-coordinate points.
  - Output: `(N, 2)` pixel coordinates.
  - Internally: use `cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)`.
  - `rvec` = Rodrigues vector from `R`, `tvec` = `t`.
  - Also return a visibility mask: `(N,)` bool array indicating which points project within `(0, width) × (0, height)` and are in front of the camera (`z_cam > 0`).

- **3.3.1c** Implement `Camera.unproject(pixel: np.ndarray, depth: float = 1.0) -> np.ndarray`:
  - Given a pixel `(u, v)` and a depth, return the 3D world point.
  - Steps: undistort the pixel → compute normalized camera-frame ray direction `d_cam = K⁻¹ @ [u, v, 1]ᵀ` → rotate to world: `d_world = Rᵀ @ d_cam` → `point = C + depth * d_world / ||d_world||`.
  - This is needed by the space-carving variant (Section 5.3) and for debug ray visualization.

- **3.3.1d** Implement `Camera.P() -> np.ndarray`:
  - Returns the full 3×4 projection matrix `P = K @ [R | t]`.
  - Useful for linear triangulation and other multi-view geometry operations downstream.

- **3.3.1e** Implement `Camera.frustum_corners(near: float, far: float) -> np.ndarray`:
  - Returns the 8 corners of the camera's viewing frustum in world coordinates (4 near-plane corners + 4 far-plane corners).
  - Computed by unprojecting the 4 image corners at `depth = near` and `depth = far`.
  - Used for frustum visualization and coverage analysis.

**Deliverables:**
- `Camera` class with `project()`, `unproject()`, `P()`, `frustum_corners()`.
- Unit tests:
  - Project a world point, unproject the resulting pixel at the correct depth → recover the original point within tolerance.
  - `P()` matrix applied to a homogeneous world point gives the same pixel as `project()`.
  - Frustum corners form a valid frustum (near plane smaller than far plane, all corners on correct side of camera).

**Dependencies:** Tasks 3.1.1, 3.1.2, 3.2.1, 2.3.4 (vignetting), 2.3.3 (FPN).

---

#### Task 3.3.2 — Camera Placement Configuration

**Objective:** Define and validate a multi-camera setup for the simulation.

**Sub-tasks:**

- **3.3.2a** Define `CameraConfig` dataclass (for serialization):
  - `id`: string.
  - `position`: `[x, y, z]`.
  - `orientation_mode`: enum `{EULER, LOOK_AT}`.
  - `euler_angles`: `[roll, pitch, yaw]` in degrees (for config files; converted to radians internally).
  - `look_at_target`: `[x, y, z]` — alternative to Euler.
  - `intrinsics_preset`: string (e.g., `"flir_640x512"`) or inline intrinsics parameters.
  - `hfov_deg`: float — horizontal field of view if using a preset.

- **3.3.2b** Implement `build_cameras(configs: List[CameraConfig], rendering_config: RenderingConfig) -> List[Camera]`:
  - Constructs `Camera` objects from config, including precomputing vignetting and FPN maps.

- **3.3.2c** Implement default camera placement generators:
  - `generate_ring_placement(num_cameras: int, ring_radius: float, pole_height: float, look_at_center: np.ndarray) -> List[CameraConfig]`:
    - Places `num_cameras` equally spaced around a circle of `ring_radius` at height `pole_height`, all pointing toward `look_at_center`.
    - This is a common initial configuration for coverage analysis.
  - `generate_perimeter_placement(park_bounds, num_cameras, pole_height) -> List[CameraConfig]`:
    - Distributes cameras along the perimeter of the park, pointing inward and upward.

- **3.3.2d** Implement coverage validation utility:
  - `compute_coverage_map(cameras: List[Camera], grid_resolution: float, z_planes: List[float]) -> np.ndarray`:
    - For each point on a regular grid at each altitude in `z_planes`, count how many cameras can see it (i.e., the point projects within the image bounds and is within the camera's effective range given SNR constraints).
    - Return a 3D array of coverage counts.
    - Minimum coverage threshold: 2 cameras (required for triangulation / MVS).
  - `visualize_coverage(coverage_map, z_plane)`: 2D heatmap of coverage at a given altitude.
  - `visualize_frustums_3d(cameras)`: 3D visualization of all camera frustums using matplotlib or open3d.

**Deliverables:**
- `CameraConfig` dataclass, `build_cameras()` factory.
- Placement generators.
- Coverage analysis utilities with visualization.
- Integration test: generate a ring of 4 cameras around a 500 m radius circle at 10 m height, verify that the center of the ring at 100 m altitude has coverage ≥ 4.

**Dependencies:** Task 3.3.1 (Camera class), Task 2.3.5 (SNR for effective range).

---

#### Task 3.3.3 — Calibration Verification Pipeline

**Objective:** Implement a simulated calibration step that validates the projection pipeline end-to-end before running tracking experiments.

**Sub-tasks:**

- **3.3.3a** Define a set of calibration targets: 3D points at known world positions (e.g., a grid of thermal emitters at fixed locations — could be a regular 3D grid within the world bounds).

- **3.3.3b** Implement `run_calibration_check(cameras: List[Camera], targets_3d: np.ndarray) -> dict`:
  - For each camera, project all targets to pixel coordinates.
  - Verify: no target that should be visible (within frustum) is missing, and no target outside the frustum appears.
  - Compute reprojection error: project targets, then unproject at the known depth, compare to original 3D coordinates. The round-trip error should be < 0.01 pixels for the pinhole model and < 0.1 pixels after distortion/undistortion.
  - Return per-camera statistics: mean/max reprojection error, number of visible targets, any anomalies.

- **3.3.3c** (Optional, Phase 4) Simulate a full calibration procedure: given noisy 2D observations of calibration targets, recover intrinsics and extrinsics using `cv2.calibrateCamera` and `cv2.solvePnP`, and compare recovered parameters to the ground-truth simulation parameters. This tests the system's ability to self-calibrate in a deployment scenario.

**Deliverables:**
- `run_calibration_check()` function.
- Report/assertion that reprojection errors are within tolerance.

**Dependencies:** Task 3.3.1 (Camera class with project/unproject).

---

## CROSS-CUTTING CONCERNS

### Task X.1 — Logging and Reproducibility (Sections 2 & 3 scope)

- Ensure every RNG in the rendering and motion pipelines is seeded from `SimulationRunConfig.random_seed` via a `np.random.SeedSequence` / `np.random.Generator` hierarchy (one child generator per component: motion, noise per camera, FPN per camera). This guarantees reproducibility even if the order of operations changes.
- Log the full `SimulationConfig` (including seeds) at the start of every run.

### Task X.2 — Unit and Integration Test Suite (Sections 2 & 3 scope)

- Collect all unit tests from individual tasks into a pytest suite.
- Add an integration test: load `default_config.yaml`, run 10 frames of the full pipeline (eagle motion → rendering → frame bundle storage), verify:
  - All images have correct shape and dtype.
  - Eagle blob appears at expected pixel locations (within 1 px tolerance) for a deterministic trajectory.
  - Ground truth 3D positions match the motion model output.
  - No NaN or Inf values in any image.

### Task X.3 — Performance Baseline

- Profile `render_frame()` for a single camera at 640×512 resolution.
- Target: < 10 ms per camera per frame on a modern CPU (sufficient for 30 fps with 3 cameras).
- If projection-loop bottleneck emerges, flag `numba` JIT compilation as a Phase 3 optimization path.

---

## TASK DEPENDENCY GRAPH (SUMMARY)

```
2.1.1 WorldConfig ──────────┬──→ 2.1.2 SimulationConfig
                            │
                            ├──→ 2.2.1 EagleState ──→ 2.2.2 MotionGenerators
                            │
                            └──→ 3.2.1 Extrinsics
                                      │
3.1.1 Intrinsics ──→ 3.1.2 Distortion ──→ 3.3.1 Camera Class
                                                    │
2.3.1 Background ──→ 2.3.2 Blob Rendering ──┐      │
2.3.3 Noise ─────────────────────────────────┤      │
2.3.4 Vignetting ───────────────────────────┤      │
2.3.5 Attenuation ──────────────────────────┤      │
                                             ▼      ▼
                                    2.3.6 Full Rendering Pipeline
                                             │
                                             ├──→ 3.3.2 Camera Placement & Coverage
                                             └──→ 3.3.3 Calibration Verification
                                             └──→ X.2 Integration Tests
```



# Task Document: Section 4 (Synthetic Data Generation Pipeline) & Section 5 (Voxel-Based MVS Tracking)

**Scope:** All implementation work required to orchestrate the simulation loop that produces synchronized synthetic thermal imagery with ground truth, and the full voxel-based multi-view stereo tracking system that consumes that imagery to estimate 3D eagle position over time.

**Audience:** Computer scientists, software engineers, system design engineers, physicists.

**Prerequisite:** Sections 2 & 3 task document (WorldConfig, EagleState, MotionGenerators, Camera class, full rendering pipeline, FrameBundle).

---

## SECTION 4 — SYNTHETIC DATA GENERATION PIPELINE

---

### 4.1 Simulation Loop Orchestrator

#### Task 4.1.1 — Top-Level Simulation Driver

**Objective:** Implement the main simulation loop that advances the eagle, renders all cameras, packages ground truth, and either stores or streams results to the tracker.

**Sub-tasks:**

- **4.1.1a** Implement `SimulationEngine` class:
  - Constructor accepts `SimulationConfig` (from Task 2.1.2).
  - On initialization:
    - Builds all `Camera` objects via `build_cameras()` (Task 3.3.2).
    - Instantiates the appropriate `MotionGenerator` based on `EagleConfig.motion_type`.
    - Seeds the master RNG from `SimulationRunConfig.random_seed` and spawns child generators (one per camera for noise, one for motion) via `np.random.SeedSequence.spawn()`.
    - Initializes eagle state via `motion_generator.reset(seed)`.
    - Precomputes per-camera static assets (vignetting maps, FPN maps, undistortion maps) — these should already be cached on the `Camera` objects from Section 3 tasks.

- **4.1.1b** Implement `SimulationEngine.run() -> SimulationResult`:
  - Main loop for `t in range(num_frames)`:
    1. `eagle_state = motion_generator.step(eagle_state, dt)`.
    2. `frame_bundle = render_all_cameras(eagle_state, cameras, rendering_config, camera_rngs)` (Task 2.3.6).
    3. Annotate `frame_bundle` with:
       - `timestamp = t * dt`.
       - `ground_truth_3d = eagle_state.position.copy()`.
       - `ground_truth_velocity = eagle_state.velocity.copy()`.
       - Per-camera ground truth 2D projections (from `camera.project(eagle_state.position)` for each camera).
       - Per-camera visibility flags.
    4. Yield or store the `frame_bundle` (see 4.1.1c).
  - After loop, return `SimulationResult` containing metadata (config snapshot, total frames, wall-clock duration).

- **4.1.1c** Implement two output modes via a strategy pattern:
  - **Batch mode** (`SimulationOutputMode.BATCH`): accumulate all `FrameBundle` objects in memory, return the full list. Suitable for small simulations (< ~1000 frames at 640×512 with 3 cameras ≈ 3.7 GB float32).
  - **Streaming mode** (`SimulationOutputMode.STREAMING`): yield each `FrameBundle` via a Python generator. The consumer (e.g., the tracker in online mode) pulls frames one at a time. Memory-constant regardless of simulation length.
  - **Disk mode** (`SimulationOutputMode.DISK`): write each frame to disk as it's produced (see Task 4.2), keep only metadata in memory.

- **4.1.1d** Implement `SimulationEngine.step_single() -> FrameBundle`:
  - Advances the simulation by exactly one time step and returns the frame bundle.
  - Maintains internal state (current eagle state, current timestamp, RNG states).
  - Enables interactive / debugger-friendly operation and is the primitive that `run()` calls internally.

**Deliverables:**
- `SimulationEngine` class with `run()`, `step_single()`, and three output modes.
- `SimulationResult` dataclass (metadata container).
- Integration test: run 10 frames, verify frame count, timestamp monotonicity, ground truth consistency with motion generator output.

**Dependencies:** Task 2.1.2 (SimulationConfig), Task 2.2.2 (MotionGenerators), Task 2.3.6 (render pipeline), Task 3.3.2 (build_cameras).

---

#### Task 4.1.2 — Multi-Eagle Extension Interface

**Objective:** Design the simulation loop to handle multiple eagles from the outset, even though Phase 1 uses a single eagle.

**Sub-tasks:**

- **4.1.2a** Generalize `SimulationEngine` internals:
  - Store a list of `(EagleState, MotionGenerator)` pairs.
  - `render_all_cameras` already accepts a single eagle; extend `render_frame` (Task 2.3.6) to accept `List[EagleState]` and accumulate each eagle's thermal blob onto the same image.
  - The rendering order doesn't matter for additive blobs (no occlusion in Phase 1). For future mesh-based rendering, define a z-buffer or painter's-algorithm pass.

- **4.1.2b** Extend `FrameBundle.ground_truth_3d` from a single `(3,)` vector to a `Dict[str, np.ndarray]` keyed by eagle ID, or an `(M, 3)` array with an associated ID list. Same for 2D projections.

- **4.1.2c** Extend `EagleConfig` in `SimulationConfig` to `List[EagleConfig]`, each with its own motion type, temperature, radius, and motion parameters.

**Deliverables:**
- Generalized loop and frame bundle supporting `M ≥ 1` eagles.
- Unit test: 2 eagles, verify both blobs appear in rendered image at their respective projected locations.

**Dependencies:** Task 4.1.1.

**Note:** This task can be deferred to Phase 4 but the interface design should be established now to avoid a refactor later.

---

### 4.2 Data Storage and I/O

#### Task 4.2.1 — Frame Bundle Serialization

**Objective:** Implement efficient, structured storage of synthetic data for offline processing and reproducibility.

**Sub-tasks:**

- **4.2.1a** Define the on-disk layout. Two options, choose one:

  **Option A — Directory of `.npz` files (simpler, good for moderate runs):**
  ```
  simulation_run_001/
    config.yaml                      # Full config snapshot
    trajectory.npz                   # (N, 7) array: [t, px, py, pz, vx, vy, vz]
    frames/
      frame_000000.npz               # Keys: cam_01, cam_02, ..., gt_3d, gt_2d_cam_01, ...
      frame_000001.npz
      ...
  ```
  Each `.npz` file contains the thermal images as named arrays and ground truth as metadata arrays.

  **Option B — HDF5 (better for large runs, supports compression, random access):**
  ```
  simulation_run_001.h5
    /config         (attribute: YAML string)
    /trajectory     (N, 7) dataset
    /frames/
      /000000/
        /cam_01     (H, W) float32 dataset
        /cam_02     ...
        /gt_3d      (3,) dataset
        /gt_2d      group of (2,) datasets per camera
      /000001/
        ...
  ```
  Use `h5py` with chunk-aligned compression (`gzip` or `lz4`).

  **Recommendation:** Start with Option A for simplicity; migrate to HDF5 if I/O becomes a bottleneck or dataset sizes exceed ~10 GB.

- **4.2.1b** Implement `SimulationWriter` interface:
  ```python
  class SimulationWriter(ABC):
      def open(self, output_dir: str, config: SimulationConfig) -> None: ...
      def write_frame(self, frame_bundle: FrameBundle) -> None: ...
      def close(self) -> None: ...
  ```
  Concrete implementations: `NpzWriter`, `HDF5Writer`.

- **4.2.1c** Implement `SimulationReader` interface:
  ```python
  class SimulationReader(ABC):
      def open(self, path: str) -> SimulationConfig: ...
      def num_frames(self) -> int: ...
      def read_frame(self, index: int) -> FrameBundle: ...
      def read_trajectory(self) -> np.ndarray: ...
      def __iter__(self) -> Iterator[FrameBundle]: ...
  ```
  Concrete implementations: `NpzReader`, `HDF5Reader`.

- **4.2.1d** Implement config snapshot: at `open()`, serialize the full `SimulationConfig` to YAML and store it alongside the data. This ensures every dataset is self-documenting and reproducible.

**Deliverables:**
- `SimulationWriter` / `SimulationReader` with at least the `Npz` backend.
- Round-trip test: write 5 frames, read them back, verify all arrays are bitwise identical.
- Trajectory read-back matches motion generator output.

**Dependencies:** Task 4.1.1 (FrameBundle), Task 2.1.2 (SimulationConfig serialization).

---

#### Task 4.2.2 — Ground Truth Annotation Utilities

**Objective:** Provide convenience functions for extracting, visualizing, and exporting ground truth from stored simulation data.

**Sub-tasks:**

- **4.2.2a** Implement `extract_trajectory(reader: SimulationReader) -> np.ndarray`:
  - Returns `(N, 7)` array of `[t, px, py, pz, vx, vy, vz]`.
  - If trajectory is stored monolithically (as in Task 4.2.1), this is a direct read. If only per-frame GT is stored, iterate and stack.

- **4.2.2b** Implement `extract_2d_tracks(reader: SimulationReader, camera_id: str) -> np.ndarray`:
  - Returns `(N, 3)` array of `[t, u, v]` — the ground truth pixel-space trajectory in a specific camera.
  - Useful for verifying detection algorithms and measuring 2D tracking error independently.

- **4.2.2c** Implement visualization helpers:
  - `plot_trajectory_3d(trajectory, cameras=None)`: 3D plot of the eagle path with optional camera positions and frustums. Use matplotlib or plotly.
  - `overlay_gt_on_image(image, gt_2d, radius)`: draw a circle at the ground truth projected position on the thermal image. Returns annotated image. Useful for visual debugging.
  - `animate_simulation(reader, camera_id, output_path)`: generate an MP4 or GIF of a camera's view over time with GT overlay. Use `matplotlib.animation` or `imageio`.

**Deliverables:**
- Trajectory and 2D track extraction utilities.
- Visualization functions.
- Example script: load a stored simulation, produce a 3D trajectory plot and an animated GIF of one camera's view.

**Dependencies:** Task 4.2.1 (SimulationReader).

---

### 4.3 Pipeline Validation

#### Task 4.3.1 — Projection Consistency Check

**Objective:** Verify that the ground truth 3D positions, when projected through each camera, land on the bright blob in the synthetic image.

**Sub-tasks:**

- **4.3.1a** Implement `validate_projection_consistency(reader: SimulationReader, tolerance_px: float = 2.0) -> dict`:
  - For each frame, for each camera where the eagle is visible:
    - Find the peak pixel (maximum temperature) in the image.
    - Compare peak pixel location to the stored ground truth 2D projection.
    - Compute Euclidean distance in pixels.
  - Return per-camera statistics: mean, max, and std of the discrepancy.
  - **Expected result:** In the noise-free case, the peak pixel should coincide exactly with the projected center (sub-pixel). With noise, the discrepancy should be bounded by the noise level relative to the blob's gradient — typically < 1 px for reasonable SNR.

- **4.3.1b** Implement `validate_blob_intensity(reader: SimulationReader) -> dict`:
  - For each frame, for each visible camera:
    - Read peak pixel value.
    - Compute expected peak value analytically: `T_sky + (T_e - T_sky) * exp(-α * d) * V(u, v)` where `V` is the vignetting factor at the projected center.
    - Compare. Discrepancy should be < noise std (or zero without noise).
  - Return per-camera statistics.

- **4.3.1c** Run these validations as part of the integration test suite. A deterministic trajectory (Lissajous) with noise disabled should produce zero projection discrepancy and exact intensity match.

**Deliverables:**
- `validate_projection_consistency()` and `validate_blob_intensity()` functions.
- Assertions integrated into the test suite.

**Dependencies:** Task 4.2.1 (reader), Task 2.3.6 (rendering pipeline).

---

## SECTION 5 — VOXEL-BASED MVS TRACKING

---

### 5.1 Voxel Grid

#### Task 5.1.1 — Voxel Data Structure

**Objective:** Define the voxel representation and the sparse storage backend.

**Sub-tasks:**

- **5.1.1a** Define `VoxelData` dataclass (per-voxel payload):
  - `log_odds`: float — log-odds of occupancy. Initialized to 0.0 (corresponding to probability 0.5, the prior).
  - `last_updated_frame`: int — frame index of the most recent Bayesian update that touched this voxel. Used for temporal decay and pruning.
  - Convenience property `probability -> float`: `1 / (1 + exp(-log_odds))` (logistic sigmoid).

- **5.1.1b** Define `VoxelGridConfig` dataclass:
  - `voxel_size`: float — side length of each cubic voxel in meters. Default 1.0 m.
  - `roi_min`: `np.ndarray` (3,) — minimum corner of the region of interest in world coordinates.
  - `roi_max`: `np.ndarray` (3,) — maximum corner.
  - `occupancy_threshold`: float — probability above which a voxel is considered occupied. Default 0.8.
  - `log_odds_clamp`: float — clamp `|log_odds|` to this value to prevent saturation. Default 10.0 (corresponds to p ≈ 0.99995).
  - `temporal_decay_rate`: float — per-frame decay applied to log_odds toward zero. Default 0.1.
  - `pruning_threshold`: float — voxels with `|log_odds| < pruning_threshold` are removed from the sparse map. Default 0.05.

- **5.1.1c** Implement the sparse voxel map. Two implementation strategies; build one, leave the other as a swap-in alternative:

  **Strategy A — Python dict (hash map), recommended for Phase 2:**
  ```python
  class SparseVoxelGrid:
      _voxels: Dict[Tuple[int, int, int], VoxelData]
  ```
  - Key: integer grid coordinates `(ix, iy, iz)` where `ix = floor((x - roi_min_x) / voxel_size)`.
  - Provides `O(1)` access, insertion, deletion.
  - Memory-efficient when the occupied volume is small relative to the ROI (which it is — one eagle ≈ a few voxels vs. millions in the full ROI).

  **Strategy B — Octree, for Phase 3 / large-scale:**
  - Use an octree with a branching factor of 8, where leaf nodes store `VoxelData`.
  - Supports hierarchical queries (e.g., quickly skip empty octants during projection).
  - Can be implemented with a library like `pyoctree` or custom.
  - Defer to Phase 3 unless the hash map proves too slow for the target voxel counts.

- **5.1.1d** Implement `SparseVoxelGrid` methods:
  - `world_to_grid(point: np.ndarray) -> Tuple[int, int, int]`: converts world coordinates to grid indices.
  - `grid_to_world(ix, iy, iz) -> np.ndarray`: returns the world-space center of the voxel.
  - `get(ix, iy, iz) -> VoxelData | None`: returns voxel data or `None` if not allocated.
  - `get_or_create(ix, iy, iz) -> VoxelData`: allocates if absent, initializes with prior.
  - `set(ix, iy, iz, data: VoxelData)`: writes voxel data.
  - `get_occupied(threshold: float = None) -> List[Tuple[Tuple[int,int,int], VoxelData]]`: returns all voxels with probability above threshold.
  - `active_voxels() -> Iterator`: iterates over all allocated voxels.
  - `num_active() -> int`.
  - `clear()`: removes all voxels.

- **5.1.1e** Implement grid-region utilities:
  - `get_neighborhood(ix, iy, iz, radius: int = 1) -> List[Tuple[int,int,int]]`: returns the 6-connected (radius=1) or 26-connected (radius=1, full cube) neighborhood. Used by connected-component analysis.
  - `enumerate_region(min_grid, max_grid) -> Iterator[Tuple[int,int,int]]`: iterates over all grid cells in a bounding box. Used for dense fusion over a small subregion.

**Deliverables:**
- `VoxelData`, `VoxelGridConfig`, `SparseVoxelGrid` with hash-map backend.
- Unit tests:
  - `world_to_grid` and `grid_to_world` round-trip: `grid_to_world(world_to_grid(p))` is within `voxel_size/2` of `p` in each axis.
  - Insert 100 voxels, retrieve all, verify count.
  - `get_occupied` returns only voxels above threshold.
  - Neighborhood enumeration returns correct count (6 for 6-connected, 26 for 26-connected, minus any out-of-ROI).

**Dependencies:** Task 2.1.1 (WorldConfig for ROI bounds).

---

#### Task 5.1.2 — Temporal Decay and Pruning

**Objective:** Prevent stale voxel detections from persisting and bound memory usage.

**Sub-tasks:**

- **5.1.2a** Implement `SparseVoxelGrid.apply_temporal_decay(current_frame: int)`:
  - For each active voxel:
    - Compute `frames_since_update = current_frame - voxel.last_updated_frame`.
    - Apply decay: `voxel.log_odds *= (1 - decay_rate) ^ frames_since_update`. Alternatively, apply a fixed additive pull toward zero: `voxel.log_odds -= sign(log_odds) * decay_rate * frames_since_update`, clamped to not cross zero.
    - The multiplicative form is smoother; the additive form is simpler. Choose multiplicative for initial implementation.
  - Update `last_updated_frame` to `current_frame` after decay (so decay doesn't double-apply on the next call).

- **5.1.2b** Implement `SparseVoxelGrid.prune()`:
  - Remove all voxels where `|log_odds| < pruning_threshold`.
  - Call after every `N` frames (e.g., `N = 10`) or when `num_active()` exceeds a memory budget.
  - Return the number of pruned voxels for logging.

- **5.1.2c** Implement a memory budget mechanism:
  - `VoxelGridConfig.max_active_voxels`: int (optional). If set, trigger aggressive pruning (increasing the threshold) when this limit is approached.
  - Log a warning if the budget is hit — it may indicate misconfigured detection thresholds or excessive noise.

**Deliverables:**
- Decay and pruning methods on `SparseVoxelGrid`.
- Unit test: create voxels with high log-odds, advance 50 frames without updating them, verify log-odds have decayed toward zero. Prune and verify removal.

**Dependencies:** Task 5.1.1.

---

### 5.2 Occupancy Fusion

#### Task 5.2.1 — Per-Frame Bayesian Fusion

**Objective:** Implement the core MVS fusion step: for each candidate voxel, project it into every camera, read the thermal evidence, and update occupancy via Bayesian log-odds.

**Sub-tasks:**

- **5.2.1a** Define `FusionConfig` dataclass:
  - `detection_threshold_sigma`: float — a pixel is considered "hot" if its temperature exceeds `T_sky_mean + detection_threshold_sigma * T_sky_std`. Default 2.0. The sky mean and std can be estimated per-frame from the image periphery or assumed from `RenderingConfig`.
  - `p_hot_given_occupied`: float — likelihood of a hot pixel if the voxel truly contains the eagle and the voxel projects onto that pixel. Default 0.9 (not 1.0, to account for atmospheric loss, partial occlusion, and blob spread).
  - `p_hot_given_empty`: float — likelihood of a hot pixel if the voxel is empty (false alarm rate due to noise). Default 0.05.
  - `min_cameras_for_update`: int — only update a voxel if it is visible in at least this many cameras. Default 2 (otherwise a single noisy camera can hallucinate detections).
  - These two likelihoods give the per-camera log-likelihood ratio: `L = log(p_hot_given_occupied / p_hot_given_empty)` for a hot observation, and `L = log((1 - p_hot_given_occupied) / (1 - p_hot_given_empty))` for a cold observation.

- **5.2.1b** Implement per-frame image preprocessing:
  - `preprocess_thermal_image(image: np.ndarray, config: FusionConfig) -> np.ndarray`:
    - Compute sky statistics: mean and std of temperature from the image (or from the known `T_sky` if available). For robustness, use a robust estimator (median and MAD) rather than mean/std, since the eagle blob is an outlier.
    - Threshold: produce a binary "hot mask" where `image > median + detection_threshold_sigma * MAD`.
    - Optionally, instead of binary, return a continuous "hotness" score: `score = (image - median) / MAD`. This allows the fusion to weight strong detections more. For Phase 2, binary is sufficient.

- **5.2.1c** Implement `fuse_frame(voxel_grid: SparseVoxelGrid, cameras: List[Camera], images: List[np.ndarray], fusion_config: FusionConfig, current_frame: int)`:

  This is the performance-critical inner loop. Algorithm:

  1. **Determine the active voxel set.** Three strategies (selectable via config):
     - **Dense subregion:** Enumerate all voxels within a bounding box. For Phase 2, use the full ROI or a user-specified subregion.
     - **Prediction-guided:** Use the previous frame's detection centroid ± a search margin (e.g., `max_speed * dt * 3`) to define the subregion. Dramatically reduces the voxel count.
     - **Existing + expansion:** Iterate over all currently active voxels plus their neighborhoods.

     For Phase 2, start with dense subregion (small ROI). Transition to prediction-guided in Phase 3.

  2. **For each voxel in the active set:**
     a. Compute world-space center: `p_world = grid_to_world(ix, iy, iz)`.
     b. Initialize per-voxel accumulators: `total_log_likelihood = 0.0`, `num_visible = 0`.
     c. **For each camera:**
        - Project `p_world` to pixel `(u, v)` via `camera.project()`.
        - If `(u, v)` is outside image bounds or the voxel is behind the camera, skip this camera for this voxel.
        - Read the preprocessed hot mask (or score) at `(u, v)`. Use nearest-neighbor lookup or bilinear interpolation on the score map.
        - If hot: `total_log_likelihood += log(p_hot_given_occupied / p_hot_given_empty)`.
        - If cold: `total_log_likelihood += log((1 - p_hot_given_occupied) / (1 - p_hot_given_empty))`.
        - Increment `num_visible`.
     d. If `num_visible >= min_cameras_for_update`:
        - `voxel = voxel_grid.get_or_create(ix, iy, iz)`.
        - `voxel.log_odds += total_log_likelihood`.
        - Clamp: `voxel.log_odds = clamp(voxel.log_odds, -log_odds_clamp, +log_odds_clamp)`.
        - `voxel.last_updated_frame = current_frame`.
     e. If `num_visible < min_cameras_for_update` and the voxel exists: do not update (or optionally apply a mild negative update to penalize voxels that have fallen out of multi-view coverage).

  3. **Post-fusion:** Apply temporal decay (Task 5.1.2). Optionally prune.

- **5.2.1d** Vectorization strategy for the inner loop:
  - The naive Python triple loop (voxels × cameras × pixels) is prohibitively slow for large voxel sets. Mitigation:
    - **Batch projection:** Collect all active voxel centers into an `(N, 3)` array. Call `camera.project()` once per camera with the full batch. This leverages OpenCV's vectorized `cv2.projectPoints` and avoids per-voxel Python overhead.
    - **Vectorized lookup:** Use the projected `(N, 2)` pixel coordinates to index into the hot mask via `mask[v_coords, u_coords]` (integer-rounded). This is a single numpy advanced-indexing operation.
    - **Vectorized log-odds update:** Compute the full `(N,)` log-likelihood vector per camera, sum across cameras, and write back to voxel data in a batch.
  - Target: the fusion step for 100,000 voxels × 4 cameras should complete in < 50 ms. Profile and introduce `numba` if numpy vectorization is insufficient.

**Deliverables:**
- `FusionConfig` dataclass.
- `preprocess_thermal_image()` function.
- `fuse_frame()` function with batch-vectorized projection.
- Unit tests:
  - Single voxel at the eagle's true position, single camera: after fusion, the voxel's probability should increase significantly (verify numerically against the analytical Bayesian update).
  - Single voxel at a position far from the eagle: probability should decrease.
  - Two cameras, voxel at eagle position: probability should increase faster than with one camera (information from two views compounds).
  - Voxel behind one camera but visible in another: only the visible camera contributes.

**Dependencies:** Task 5.1.1 (SparseVoxelGrid), Task 3.3.1 (Camera.project), Task 2.3.6 (rendered images).

---

#### Task 5.2.2 — Space Carving Alternative

**Objective:** Implement a faster, simpler alternative to Bayesian fusion that uses binary silhouette intersection.

**Sub-tasks:**

- **5.2.2a** Implement `space_carve_frame(voxel_grid: SparseVoxelGrid, cameras: List[Camera], hot_masks: List[np.ndarray], config: SpaceCarvingConfig, current_frame: int)`:

  **Algorithm:**
  1. For each camera, the hot mask is already computed (from Task 5.2.1b).
  2. **Vote accumulation:** For each voxel in the active set:
     - Project voxel center into each camera.
     - If projection is visible and the hot mask is `True` at that pixel: increment `vote_count` for this voxel.
     - If projection is visible and the hot mask is `False`: increment `veto_count`.
  3. **Decision rule:** A voxel is marked occupied if:
     - `vote_count >= min_cameras_vote` (default 2), AND
     - `veto_count == 0` (no camera explicitly sees empty space where the voxel projects).
  4. Update voxel grid: set occupied voxels to high log-odds (e.g., `log_odds = 5.0`), vetoed voxels to low log-odds (e.g., `log_odds = -5.0`), ambiguous voxels (insufficient visibility) left unchanged.

- **5.2.2b** Define `SpaceCarvingConfig`:
  - `min_cameras_vote`: int. Default 2.
  - `allow_vetoing`: bool. Default True. If False, voxels are occupied based on votes alone (more permissive, more false positives).
  - `occupied_log_odds`: float. Value assigned to carved-in voxels.
  - `empty_log_odds`: float. Value assigned to carved-out voxels.

- **5.2.2c** Implement the ray-based variant (from the plan's description):
  - Instead of projecting voxels into cameras, cast rays from each hot pixel into the world:
    - For each hot pixel `(u, v)` in camera `c`, compute the ray direction via `camera.unproject(pixel, depth=1.0)` (Task 3.3.1c).
    - March along the ray through the voxel grid (use a 3D DDA / Bresenham-like traversal or Amanatides-Woo algorithm) and increment vote counts for each traversed voxel.
  - This is more efficient when the number of hot pixels is much smaller than the number of active voxels (which is the typical case: a few hundred hot pixels vs. potentially millions of voxels).
  - The voxel-projection approach (5.2.2a) is better when the active voxel set is small (prediction-guided). Choose based on the ratio.

- **5.2.2d** Implement `select_fusion_method(num_active_voxels: int, num_hot_pixels: int) -> str`:
  - Heuristic to auto-select between Bayesian fusion, voxel-projection space carving, and ray-based space carving based on computational cost estimates.
  - This is a convenience; the user can also force a method via config.

**Deliverables:**
- `space_carve_frame()` (voxel-projection variant and ray-based variant).
- `SpaceCarvingConfig`.
- Unit test: 3 cameras, eagle at known position, verify that the eagle's voxel is carved in and neighboring empty voxels are carved out.

**Dependencies:** Task 5.1.1, Task 5.2.1b (hot mask computation), Task 3.3.1 (Camera.project and Camera.unproject).

---

### 5.3 Clustering and 3D Position Estimation

#### Task 5.3.1 — Connected Component Analysis on Sparse Voxels

**Objective:** Group occupied voxels into spatially connected clusters, each representing a candidate eagle detection.

**Sub-tasks:**

- **5.3.1a** Implement `cluster_occupied_voxels(voxel_grid: SparseVoxelGrid, connectivity: int = 26) -> List[VoxelCluster]`:
  - Extract all occupied voxels (probability > threshold).
  - Perform connected-component labeling. Two approaches:
    - **Dense subgrid extraction:** If the occupied voxels span a reasonably bounded region, extract a dense boolean 3D array over the bounding box of occupied voxels and use `scipy.ndimage.label(structure=connectivity_kernel)`. The connectivity kernel is 6-connected (faces only), 18-connected (faces + edges), or 26-connected (faces + edges + corners).
    - **Graph-based BFS/DFS on the sparse map:** For each unvisited occupied voxel, perform a flood-fill using `get_neighborhood()` (Task 5.1.1e), visiting only occupied neighbors. This avoids materializing a dense array when occupied voxels are sparse but widely scattered.
  - For Phase 2, the dense subgrid approach is simpler and `scipy` handles it efficiently.

- **5.3.1b** Define `VoxelCluster` dataclass:
  - `cluster_id`: int.
  - `voxel_indices`: `List[Tuple[int,int,int]]` — grid coordinates of member voxels.
  - `voxel_positions`: `np.ndarray` shape `(K, 3)` — world coordinates of member voxel centers.
  - `centroid`: `np.ndarray` shape `(3,)` — mean of `voxel_positions`, optionally weighted by occupancy probability.
  - `size`: int — number of voxels.
  - `max_probability`: float — highest occupancy probability in the cluster.
  - `bounding_box`: tuple `(min_corner, max_corner)` in world coordinates.

- **5.3.1c** Implement centroid computation:
  - Unweighted: `centroid = mean(voxel_positions)`.
  - Probability-weighted: `centroid = sum(p_i * pos_i) / sum(p_i)` where `p_i` is the occupancy probability of voxel `i`. This pulls the centroid toward high-confidence voxels and generally gives sub-voxel accuracy.
  - Make the weighting scheme configurable.

- **5.3.1d** Implement cluster filtering:
  - Reject clusters that are too small (e.g., `size < min_cluster_size`, default 1 — a single voxel can be a valid detection if the eagle is far away and spans only 1 voxel).
  - Reject clusters that are too large (e.g., `size > max_cluster_size`, default 50 — an eagle at 1 m voxel resolution should be ~1–3 voxels; a 50-voxel cluster likely indicates noise or a misconfigured threshold).
  - These bounds should be configurable via a `ClusteringConfig` dataclass.

**Deliverables:**
- `cluster_occupied_voxels()`, `VoxelCluster`, `ClusteringConfig`.
- Unit tests:
  - Two separate groups of occupied voxels → two clusters.
  - One contiguous group → one cluster.
  - Centroid of a symmetric cluster is at the geometric center.
  - Clusters below `min_cluster_size` are filtered out.

**Dependencies:** Task 5.1.1 (SparseVoxelGrid.get_occupied, get_neighborhood).

---

#### Task 5.3.2 — DBSCAN-Based Clustering Alternative

**Objective:** Provide a density-based clustering alternative that doesn't require a grid structure and handles irregular voxel distributions more robustly.

**Sub-tasks:**

- **5.3.2a** Implement `cluster_dbscan(voxel_grid: SparseVoxelGrid, eps: float = None, min_samples: int = 1) -> List[VoxelCluster]`:
  - Extract occupied voxel world-space positions as an `(N, 3)` array.
  - Run `sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)`.
  - Default `eps = 1.5 * voxel_size` (slightly larger than the diagonal of a voxel to connect face/edge/corner neighbors).
  - Map labels back to `VoxelCluster` objects.
  - Voxels labeled as noise (`label = -1`) by DBSCAN are discarded.

- **5.3.2b** This is the preferred method when the voxel grid transitions to an octree or when variable voxel sizes are used (multi-resolution), since `scipy.ndimage.label` assumes a regular grid.

**Deliverables:**
- `cluster_dbscan()` function.
- Unit test: same test cases as 5.3.1, verify equivalent results.

**Dependencies:** Task 5.1.1, `scikit-learn`.

---

### 5.4 Multi-Frame Tracking

#### Task 5.4.1 — Frame-to-Frame Data Association

**Objective:** Associate cluster detections across consecutive frames to form persistent tracks.

**Sub-tasks:**

- **5.4.1a** Define `Track` dataclass:
  - `track_id`: int — unique, auto-incrementing.
  - `history`: `List[TrackPoint]` where `TrackPoint = (frame_index: int, centroid: np.ndarray, cluster: VoxelCluster)`.
  - `state`: enum `{TENTATIVE, CONFIRMED, LOST}`.
  - `frames_since_last_detection`: int.
  - `kalman_state`: optional, see Task 5.4.2.

- **5.4.1b** Define `TrackingConfig`:
  - `max_association_distance`: float — maximum Euclidean distance (meters) between a predicted track position and a detection centroid to be considered a valid association. Default `max_speed * dt * 2`.
  - `min_hits_to_confirm`: int — a tentative track must be associated with detections in this many consecutive frames before becoming confirmed. Default 3.
  - `max_frames_to_coast`: int — a confirmed track is kept alive (coasting on prediction) for this many frames without a detection before being marked lost. Default 5.
  - `max_tracks`: int — hard cap on simultaneous tracks (prevent runaway in noisy conditions). Default 10.

- **5.4.1c** Implement the Hungarian (Munkres) assignment:
  - At each frame, compute a cost matrix `C` of shape `(num_tracks, num_detections)`:
    - `C[i, j] = ||predicted_position_i - detection_centroid_j||` (Euclidean distance).
    - Entries exceeding `max_association_distance` are set to a large value (or masked) to prevent impossible associations.
  - Augment the cost matrix with dummy rows/columns:
    - Add `num_detections` dummy track rows with cost `max_association_distance` (representing "new track" — the cost of starting a new track rather than associating).
    - Add `num_tracks` dummy detection columns with cost `max_association_distance` (representing "missed detection" — the cost of a track not being associated with any detection).
  - Solve using `scipy.optimize.linear_sum_assignment`.
  - Interpret results:
    - Track-detection pairs where both are real: update track with detection.
    - Track matched to dummy detection: track missed this frame, increment `frames_since_last_detection`.
    - Detection matched to dummy track: start a new tentative track.

- **5.4.1d** Implement track lifecycle management:
  - **Birth:** New tentative track created for unassociated detections.
  - **Confirmation:** Tentative → confirmed after `min_hits_to_confirm` consecutive associations.
  - **Coasting:** Confirmed track with no detection for ≤ `max_frames_to_coast` frames: maintain using prediction only (Kalman or constant velocity).
  - **Death:** Track lost if `frames_since_last_detection > max_frames_to_coast` or if a tentative track fails to confirm within `min_hits_to_confirm` frames.

- **5.4.1e** Implement `Tracker` class:
  ```python
  class Tracker:
      def __init__(self, config: TrackingConfig): ...
      def update(self, detections: List[VoxelCluster], frame_index: int) -> List[Track]: ...
      def get_active_tracks(self) -> List[Track]: ...
      def get_all_tracks(self) -> List[Track]: ...  # Including lost tracks
  ```

**Deliverables:**
- `Track`, `TrackingConfig`, `Tracker` class with Hungarian association and lifecycle management.
- Unit tests:
  - Single detection moving linearly across 10 frames → one confirmed track, position error < `voxel_size`.
  - Detection disappears for 3 frames then reappears → track coasts and re-associates (if within `max_frames_to_coast`).
  - Two detections at different positions → two separate tracks.
  - Spurious single-frame detection → tentative track that never confirms and is pruned.

**Dependencies:** Task 5.3.1 or 5.3.2 (VoxelCluster centroids), `scipy.optimize.linear_sum_assignment`.

---

#### Task 5.4.2 — Kalman Filter for Trajectory Smoothing

**Objective:** Apply a Kalman filter to each track to smooth the estimated trajectory, predict positions during missed detections, and improve association.

**Sub-tasks:**

- **5.4.2a** Define the state vector and motion model:
  - State: `x = [px, py, pz, vx, vy, vz]ᵀ` — 3D position and velocity.
  - State transition (constant velocity model):
    ```
    F = [[1, 0, 0, dt, 0,  0 ],
         [0, 1, 0, 0,  dt, 0 ],
         [0, 0, 1, 0,  0,  dt],
         [0, 0, 0, 1,  0,  0 ],
         [0, 0, 0, 0,  1,  0 ],
         [0, 0, 0, 0,  0,  1 ]]
    ```
  - Process noise covariance `Q`: models acceleration uncertainty. Parameterize as `Q = q * G @ Gᵀ` where `G = [dt²/2, dt²/2, dt²/2, dt, dt, dt]ᵀ` and `q` is a scalar tuning parameter (m²/s⁴). Default `q = 5.0` (corresponding to ~2.2 m/s² std acceleration, reasonable for eagle maneuvers).
  - Measurement model: `H = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]]` — we observe position directly from the voxel cluster centroid.
  - Measurement noise covariance `R`: diagonal, `R = diag(σ_x², σ_y², σ_z²)` where `σ ≈ voxel_size / 2` (the centroid quantization noise).

- **5.4.2b** Implement `KalmanFilter3D` class:
  - `__init__(self, dt, q, r)`: constructs `F`, `Q`, `H`, `R` matrices.
  - `initialize(self, measurement: np.ndarray)`: sets initial state `x = [mx, my, mz, 0, 0, 0]ᵀ`, initial covariance `P = diag(R, large_velocity_uncertainty)`.
  - `predict(self) -> np.ndarray`: standard Kalman predict step, returns predicted position `H @ x`.
  - `update(self, measurement: np.ndarray) -> np.ndarray`: standard Kalman update step, returns filtered position.
  - `get_state(self) -> np.ndarray`: returns full state vector.
  - `get_covariance(self) -> np.ndarray`: returns `P`.

- **5.4.2c** Integrate `KalmanFilter3D` into `Tracker`:
  - On track creation, initialize a Kalman filter with the first detection.
  - On association, call `predict()` then `update()` with the new centroid.
  - On missed detection (coasting), call `predict()` only. The covariance `P` will grow, naturally increasing the `max_association_distance` for re-association (gated by the Mahalanobis distance instead of Euclidean — see 5.4.2d).
  - Store the Kalman filter as `track.kalman_state`.

- **5.4.2d** (Optional enhancement) Replace Euclidean distance in the cost matrix with Mahalanobis distance:
  - `d_mahal = sqrt((z - H @ x_pred)ᵀ @ S⁻¹ @ (z - H @ x_pred))` where `S = H @ P_pred @ Hᵀ + R` is the innovation covariance.
  - This accounts for the anisotropic uncertainty of the prediction (e.g., a track moving fast in X has higher positional uncertainty along X after a missed frame).
  - Gate associations at a chi-squared threshold (e.g., 3σ → Mahalanobis distance < 3.0 for 3 DOF).

**Deliverables:**
- `KalmanFilter3D` class.
- Integration into `Tracker`.
- Unit tests:
  - Constant velocity trajectory with exact measurements → filter converges to true velocity within 5 frames.
  - Constant velocity trajectory with noisy measurements (noise std = 0.5 m) → filtered positions have lower error than raw measurements.
  - 5 frames of missed detections → predicted position drifts along the velocity vector, covariance grows.

**Dependencies:** Task 5.4.1 (Tracker), numpy (no external Kalman library needed; the 6D state is small enough for a direct implementation).

---

### 5.5 Full Tracking Pipeline Integration

#### Task 5.5.1 — Tracking Pipeline Orchestrator

**Objective:** Compose the fusion, clustering, and tracking steps into a single callable pipeline that consumes a `FrameBundle` and updates tracking state.

**Sub-tasks:**

- **5.5.1a** Implement `TrackingPipeline` class:
  ```python
  class TrackingPipeline:
      def __init__(self, cameras: List[Camera], voxel_config: VoxelGridConfig,
                   fusion_config: FusionConfig, clustering_config: ClusteringConfig,
                   tracking_config: TrackingConfig, kalman_params: dict): ...
      def process_frame(self, frame_bundle: FrameBundle) -> TrackingResult: ...
      def reset(self): ...
  ```

- **5.5.1b** `process_frame()` implementation:
  1. Preprocess images: `hot_masks = [preprocess_thermal_image(img, fusion_config) for img in frame_bundle.camera_images.values()]`.
  2. Fuse: `fuse_frame(voxel_grid, cameras, images, fusion_config, frame_index)` or `space_carve_frame(...)` depending on config.
  3. Decay & prune: `voxel_grid.apply_temporal_decay(frame_index)`, optionally `voxel_grid.prune()`.
  4. Cluster: `clusters = cluster_occupied_voxels(voxel_grid)` or `cluster_dbscan(...)`.
  5. Track: `active_tracks = tracker.update(clusters, frame_index)`.
  6. Package results into `TrackingResult`.

- **5.5.1c** Define `TrackingResult` dataclass:
  - `frame_index`: int.
  - `timestamp`: float.
  - `clusters`: `List[VoxelCluster]`.
  - `active_tracks`: `List[Track]`.
  - `primary_track`: `Track | None` — the highest-confidence confirmed track (convenience accessor for the single-eagle case).
  - `estimated_position`: `np.ndarray | None` — `primary_track`'s Kalman-filtered position.
  - `num_active_voxels`: int.
  - `processing_time_ms`: float — wall-clock time for this frame's processing.

- **5.5.1d** Implement online mode integration with `SimulationEngine`:
  - `run_online_simulation(engine: SimulationEngine, pipeline: TrackingPipeline) -> List[TrackingResult]`:
    - Uses the streaming output mode of the engine.
    - For each frame yielded by the engine, call `pipeline.process_frame()`.
    - Collect and return all tracking results.
  - This is the end-to-end validation loop.

**Deliverables:**
- `TrackingPipeline`, `TrackingResult`, `run_online_simulation()`.
- Integration test: 50-frame Lissajous trajectory, 3 cameras, no noise → tracking error < `voxel_size` for all confirmed frames.

**Dependencies:** Tasks 5.2.1, 5.3.1, 5.4.1, 5.4.2, 4.1.1.

---

### 5.6 Evaluation

#### Task 5.6.1 — Metrics Computation

**Objective:** Implement the evaluation metrics specified in Section 6 of the plan.

**Sub-tasks:**

- **5.6.1a** Implement `compute_position_error(tracking_results: List[TrackingResult], ground_truth: np.ndarray) -> np.ndarray`:
  - For each frame where a primary track exists, compute `||estimated_position - gt_position||`.
  - Return `(N,)` array of errors. Frames without a detection return `NaN`.
  - Compute summary statistics: mean, median, max, 95th percentile, RMSE.

- **5.6.1b** Implement `compute_detection_rate(tracking_results, ground_truth, distance_threshold: float) -> float`:
  - A frame counts as a correct detection if `||estimated_position - gt_position|| < distance_threshold`.
  - `detection_rate = num_correct / num_frames`.
  - Also compute: frames to first detection (latency), longest gap without detection.

- **5.6.1c** Implement `compute_false_positive_rate(tracking_results) -> float`:
  - A false positive is a confirmed track whose centroid is further than `distance_threshold` from any ground truth eagle.
  - For the single-eagle case: `FP_rate = num_frames_with_spurious_tracks / num_frames`.

- **5.6.1d** Implement `compute_timing_statistics(tracking_results) -> dict`:
  - Mean, max, std of `processing_time_ms`.
  - Breakdown by pipeline stage if instrumented (fusion time, clustering time, tracking time).

- **5.6.1e** Implement `EvaluationReport` dataclass and `generate_report(tracking_results, ground_truth, config) -> EvaluationReport`:
  - Aggregates all metrics.
  - Serializable to JSON/YAML for automated experiment tracking.
  - Includes the `SimulationConfig` snapshot for full reproducibility.

**Deliverables:**
- All metric functions and `EvaluationReport`.
- Visualization: error-over-time plot, detection timeline, processing time histogram.
- Unit test: perfect tracking (estimated = ground truth) → error = 0, detection rate = 1.0, FP rate = 0.

**Dependencies:** Task 5.5.1 (TrackingResult), Task 4.2.2 (ground truth extraction).

---

## CROSS-CUTTING CONCERNS (Sections 4 & 5)

### Task X.4 — Performance Profiling and Optimization Roadmap

- Profile `fuse_frame()` for the target voxel count and camera count.
- Identify the bottleneck (expected: the voxel-projection inner loop or the hot-mask indexing).
- If pure-numpy vectorization is insufficient, implement the inner loop in `numba` (`@njit` with parallel=True over voxels).
- Document the profiling results and the break-even point where `numba` JIT compilation overhead is amortized (typically > 100 frames).
- Establish per-frame latency targets:
  - Phase 2 (offline, dense): < 500 ms per frame acceptable.
  - Phase 3 (online, sparse): < 50 ms per frame for real-time at 20 fps.

### Task X.5 — Visualization Toolkit (Sections 4 & 5)

- Implement `visualize_voxel_grid(voxel_grid, cameras=None, eagle_gt=None)`:
  - 3D scatter plot of occupied voxels, color-coded by occupancy probability.
  - Overlay camera positions and frustums.
  - Overlay ground truth eagle position as a distinct marker.
  - Use `matplotlib` 3D (quick) or `open3d` / `pyvista` (interactive).
- Implement `visualize_tracking_result(tracking_result, frame_bundle)`:
  - Side-by-side: thermal image with detected blob + 3D voxel view with cluster and track history.
- Implement `animate_tracking(tracking_results, ground_truth, output_path)`:
  - Full simulation replay as a video: 3D view of voxels evolving, tracks forming, ground truth overlay.

### Task X.6 — End-to-End Regression Test

- Define a canonical test scenario: Lissajous trajectory, 4 cameras in a ring at 500 m radius, 10 m pole height, 640×512 resolution, no noise, 100 frames, 1 m voxel size.
- Run the full pipeline (SimulationEngine → TrackingPipeline → EvaluationReport).
- Assert:
  - Detection rate > 95%.
  - Mean position error < 1.5 m (within ~1 voxel).
  - No false positives.
  - Per-frame processing time < 500 ms (Phase 2 target).
- Store baseline metrics. CI runs this test on every commit; alert if any metric degrades.

### Task X.7 — Parameter Sweep Infrastructure (Section 5 specific, prep for Phase 5)

- Implement `run_parameter_sweep(base_config, sweep_params: Dict[str, List]) -> pd.DataFrame`:
  - Takes a base configuration and a dictionary of parameter names to lists of values.
  - Runs the full pipeline for every combination (Cartesian product) or a specified subset.
  - Returns a DataFrame with one row per run, columns for all swept parameters and all evaluation metrics.
- Example sweep: `voxel_size` in `[0.5, 1.0, 2.0, 5.0]` × `num_cameras` in `[2, 3, 4, 6, 8]` × `noise_std` in `[0.0, 0.05, 0.1, 0.2]`.
- This is the primary tool for Phase 5 validation and export.

---

## TASK DEPENDENCY GRAPH (SECTIONS 4 & 5)

```
Section 2&3 deliverables
  │
  ├──→ 4.1.1 SimulationEngine ──→ 4.1.2 Multi-Eagle Extension
  │         │
  │         ├──→ 4.2.1 Frame Storage (Writer/Reader)
  │         │         │
  │         │         ├──→ 4.2.2 GT Annotation Utilities
  │         │         └──→ 4.3.1 Projection Consistency Validation
  │         │
  │         └──→ 5.5.1d Online Simulation Integration
  │
  ├──→ 5.1.1 SparseVoxelGrid ──→ 5.1.2 Temporal Decay & Pruning
  │         │
  │         ├──→ 5.2.1 Bayesian Fusion ──┐
  │         │                             │
  │         ├──→ 5.2.2 Space Carving ────┤
  │         │                             │
  │         └──→ 5.3.1 Connected Components ──→ 5.3.2 DBSCAN Alternative
  │                       │
  │                       └──→ 5.4.1 Hungarian Association ──→ 5.4.2 Kalman Filter
  │                                      │
  │                                      └──→ 5.5.1 TrackingPipeline
  │                                                   │
  │                                                   ├──→ 5.6.1 Evaluation Metrics
  │                                                   ├──→ X.6 Regression Test
  │                                                   └──→ X.7 Parameter Sweep
  │
  └──→ X.4 Performance Profiling
       X.5 Visualization Toolkit
```

**Critical path:** SparseVoxelGrid → Bayesian Fusion → Connected Components → Hungarian Association → Kalman Filter → TrackingPipeline → Evaluation Metrics.

**Parallelizable with critical path:** SimulationEngine (4.1.1) and Frame Storage (4.2.1) can be developed concurrently with the voxel grid and fusion tasks, as long as the `FrameBundle` interface is agreed upon first.



# Task Document: Section 6 (Evaluation Metrics), Section 7 (Implementation Roadmap), Section 8 (Recommended Libraries), Section 9 (Data Flow Diagram)

**Scope:** Evaluation framework beyond what was covered in Section 5's tracking-level metrics, the phased implementation roadmap translated into concrete milestone definitions with gate criteria, the dependency/environment management strategy, and the formal data-flow architecture connecting all pipeline stages.

**Audience:** Computer scientists, software engineers, system design engineers, physicists.

**Prerequisite:** Sections 2–5 task documents.

**Note on overlap:** Task 5.6.1 (in the Section 4–5 document) implemented the core per-run metric computations. This document extends that into a full evaluation framework: statistical rigor, comparative analysis, automated benchmarking, and visualization standards.

---

## SECTION 6 — EVALUATION METRICS & FRAMEWORK

---

### 6.1 Metric Definitions and Statistical Rigor

#### Task 6.1.1 — Position Error Analysis Suite

**Objective:** Go beyond single-run scalar metrics to provide statistically meaningful evaluation of tracking accuracy.

**Sub-tasks:**

- **6.1.1a** Implement `PositionErrorAnalysis` class:
  - Accepts a `List[EvaluationReport]` (from Task 5.6.1e) — i.e., results from multiple runs (different seeds, parameter settings, or scenarios).
  - Computes per-run: mean, median, RMSE, 95th percentile, max error.
  - Computes across runs: mean of means, std of means, confidence intervals (95% CI via bootstrap or t-distribution assuming approximately normal error distributions).
  - Decomposes error into axis-aligned components: `(err_x, err_y, err_z)` separately. This reveals whether the system has directional bias — e.g., systematically worse depth estimation (along the axis with least camera baseline) vs. lateral estimation. This is a critical diagnostic for camera placement optimization.

- **6.1.1b** Implement temporal error analysis:
  - `error_vs_time(tracking_results, ground_truth) -> np.ndarray`: `(N, 2)` array of `(timestamp, error)`.
  - Detect transient error spikes (e.g., during sharp eagle maneuvers where the constant-velocity Kalman model lags). Correlate error spikes with eagle acceleration magnitude to quantify the motion model's limitations.
  - Compute settling time: after track initialization, how many frames until error drops below a threshold (e.g., `voxel_size`). This characterizes the Kalman filter convergence.

- **6.1.1c** Implement error vs. geometry analysis:
  - `error_vs_distance(tracking_results, ground_truth, cameras) -> np.ndarray`: error as a function of the eagle's mean distance to all cameras. Expected: error increases with distance due to larger voxel subtended angle, weaker SNR, and coarser projected resolution.
  - `error_vs_num_cameras_visible(tracking_results, ground_truth, cameras) -> np.ndarray`: error as a function of how many cameras observe the eagle at each frame. Expected: error decreases with more views.
  - `error_vs_baseline_angle(tracking_results, ground_truth, cameras) -> np.ndarray`: error as a function of the maximum pairwise baseline angle (the angle subtended by the two most widely separated cameras that see the eagle). Wider baselines should yield better depth accuracy. This is a direct input to camera placement optimization.

**Deliverables:**
- `PositionErrorAnalysis` class with all decompositions.
- Plotting functions for each analysis (error vs. time, error vs. distance, etc.).
- Unit test: synthetic perfect tracking → all errors zero; synthetic tracking with known constant offset → mean error equals offset magnitude.

**Dependencies:** Task 5.6.1 (EvaluationReport), Task 4.2.2 (ground truth extraction).

---

#### Task 6.1.2 — Detection and False Positive Metrics

**Objective:** Formalize detection performance beyond simple rates.

**Sub-tasks:**

- **6.1.2a** Implement detection event classification per frame:
  - **True Positive (TP):** A confirmed track exists with centroid within `distance_threshold` of ground truth.
  - **False Positive (FP):** A confirmed track exists with no ground truth eagle within `distance_threshold`. In the single-eagle case, this means an extra track.
  - **False Negative (FN):** Ground truth eagle exists but no confirmed track is within `distance_threshold`.
  - **True Negative (TN):** No track and no eagle. Only meaningful in multi-eagle scenarios where some frames may have no eagles in the ROI.

- **6.1.2b** Compute derived metrics:
  - **Precision:** `TP / (TP + FP)` — of all detections, what fraction are real.
  - **Recall (= detection rate):** `TP / (TP + FN)` — of all real eagles, what fraction are detected.
  - **F1 score:** harmonic mean of precision and recall.
  - **MOTA (Multiple Object Tracking Accuracy):** `1 - (FN + FP + ID_switches) / num_gt_frames`. Standard MOT benchmark metric. `ID_switches` = number of frames where a track changes identity assignment (relevant in multi-eagle scenarios; always 0 for single eagle).
  - **MOTP (Multiple Object Tracking Precision):** mean position error over all TP frames.

- **6.1.2c** Implement track-level metrics (beyond frame-level):
  - **Track completeness:** fraction of ground truth trajectory covered by the longest associated track.
  - **Track fragmentation:** number of track breaks (a single ground truth trajectory should ideally produce one continuous track; fragmentation > 1 indicates the tracker lost and re-acquired the target).
  - **Time to first detection:** frames from simulation start to the first TP.
  - **Longest detection gap:** maximum consecutive FN frames within an otherwise tracked segment.

- **6.1.2d** Implement `compute_mot_metrics(tracking_results, ground_truth, distance_threshold) -> MOTMetrics`:
  - Returns a `MOTMetrics` dataclass containing all of the above.
  - Compatible with the `py-motmetrics` library format if external benchmarking is desired, but self-contained for this project.

**Deliverables:**
- Per-frame TP/FP/FN classification.
- `MOTMetrics` dataclass with precision, recall, F1, MOTA, MOTP, track completeness, fragmentation.
- Unit tests: perfect tracking → precision=recall=F1=MOTA=1.0, fragmentation=1. Complete miss → recall=0. Single false track → precision computed correctly.

**Dependencies:** Task 5.6.1, Task 5.4.1 (Track objects).

---

#### Task 6.1.3 — Processing Time Analysis

**Objective:** Characterize computational performance to assess real-time feasibility.

**Sub-tasks:**

- **6.1.3a** Instrument the `TrackingPipeline.process_frame()` (Task 5.5.1) with per-stage timing:
  - `t_preprocess`: image preprocessing (thresholding).
  - `t_fusion`: voxel fusion step.
  - `t_decay_prune`: temporal decay and pruning.
  - `t_clustering`: connected-component or DBSCAN clustering.
  - `t_tracking`: Hungarian assignment + Kalman update.
  - `t_total`: end-to-end.
  - Use `time.perf_counter_ns()` for sub-millisecond precision. Store all timings in `TrackingResult`.

- **6.1.3b** Implement `TimingAnalysis` class:
  - Accepts `List[TrackingResult]`.
  - Computes per-stage: mean, std, max, 99th percentile.
  - Identifies the bottleneck stage.
  - Computes throughput: `1000 / mean_t_total` (frames per second).
  - Correlates `t_fusion` with `num_active_voxels` — this should be roughly linear. Fit a linear model to estimate per-voxel fusion cost (µs/voxel) for capacity planning.
  - Correlates `t_clustering` with number of occupied voxels.

- **6.1.3c** Implement latency budget visualization:
  - Stacked bar chart showing per-stage contribution to total latency.
  - Time series of `t_total` across frames, with a horizontal line at the real-time target (e.g., 50 ms for 20 fps).

**Deliverables:**
- Per-stage instrumentation in `TrackingPipeline`.
- `TimingAnalysis` class with summary statistics and bottleneck identification.
- Visualization functions.

**Dependencies:** Task 5.5.1 (TrackingPipeline with instrumented timings).

---

### 6.2 Evaluation Automation

#### Task 6.2.1 — Benchmark Runner

**Objective:** Automate the execution of standardized evaluation scenarios.

**Sub-tasks:**

- **6.2.1a** Define `BenchmarkScenario` dataclass:
  - `name`: string identifier (e.g., `"baseline_lissajous_4cam_no_noise"`).
  - `config`: `SimulationConfig` — fully specifies the scenario.
  - `expected_metrics`: optional `MOTMetrics` — baseline values for regression testing.
  - `tolerance`: per-metric tolerance for regression checks.

- **6.2.1b** Implement `BenchmarkSuite`:
  ```python
  class BenchmarkSuite:
      def __init__(self, scenarios: List[BenchmarkScenario]): ...
      def run_all(self, output_dir: str) -> List[BenchmarkResult]: ...
      def run_single(self, scenario_name: str) -> BenchmarkResult: ...
      def compare_to_baseline(self, results: List[BenchmarkResult]) -> RegressionReport: ...
  ```
  - `run_all()` executes each scenario end-to-end: `SimulationEngine.run()` → `TrackingPipeline` → `EvaluationReport` → `MOTMetrics`.
  - Stores results (metrics + config + timing) to `output_dir` as JSON.
  - `compare_to_baseline()` loads stored baseline metrics and flags regressions (any metric that degraded beyond tolerance).

- **6.2.1c** Define the canonical benchmark scenarios:

  | Scenario | Motion | Cameras | Noise | Voxel Size | Purpose |
  |---|---|---|---|---|---|
  | `baseline_clean` | Lissajous | 4 ring | None | 1.0 m | Sanity check, regression gate |
  | `baseline_noisy` | Lissajous | 4 ring | NETD 0.05°C | 1.0 m | Noise robustness |
  | `random_walk` | Random walk | 4 ring | NETD 0.05°C | 1.0 m | Maneuver handling |
  | `sparse_cameras` | Lissajous | 2 opposing | None | 1.0 m | Minimum camera coverage |
  | `fine_voxel` | Lissajous | 4 ring | None | 0.5 m | Accuracy vs. cost trade-off |
  | `coarse_voxel` | Lissajous | 4 ring | None | 2.0 m | Accuracy vs. cost trade-off |
  | `long_range` | Lissajous (large amplitude) | 4 ring | NETD 0.05°C | 1.0 m | SNR degradation at distance |
  | `high_noise` | Lissajous | 4 ring | NETD 0.2°C | 1.0 m | Robustness under extreme noise |

- **6.2.1d** Integrate with CI:
  - `benchmark_suite.run_all()` is called in the CI pipeline.
  - `compare_to_baseline()` runs against stored baseline JSON files in the repository.
  - CI fails if any metric regresses beyond tolerance.
  - Baseline files are updated explicitly via a manual command (`update_baselines.py`) after reviewing intentional changes.

**Deliverables:**
- `BenchmarkScenario`, `BenchmarkSuite`, `BenchmarkResult`, `RegressionReport`.
- 8 canonical scenario configs as YAML files.
- CI integration script.

**Dependencies:** Task 5.5.1 (TrackingPipeline), Task 5.6.1 (EvaluationReport), Task 6.1.2 (MOTMetrics).

---

#### Task 6.2.2 — Comparative Reporting

**Objective:** Generate human-readable reports comparing multiple runs or parameter sweep results.

**Sub-tasks:**

- **6.2.2a** Implement `generate_comparison_table(results: List[BenchmarkResult]) -> pd.DataFrame`:
  - One row per scenario/run, columns for all metrics.
  - Highlight cells that regressed vs. baseline (red) or improved (green).

- **6.2.2b** Implement `generate_html_report(results, output_path)`:
  - Self-contained HTML file with:
    - Summary table (from 6.2.2a).
    - Per-scenario: error-over-time plot, detection timeline, timing breakdown.
    - Embedded plots as base64 PNGs (no external dependencies).
  - This is the primary artifact for experiment review and stakeholder communication.

- **6.2.2c** Implement `generate_latex_table(results, output_path)`:
  - For inclusion in technical reports or publications.
  - Standard format: scenario name, detection rate, RMSE, MOTA, FPS.

**Deliverables:**
- Comparison table generator (DataFrame, HTML, LaTeX).
- Example report from the canonical benchmark suite.

**Dependencies:** Task 6.2.1 (BenchmarkResult), `pandas`, `matplotlib`.

---

## SECTION 7 — IMPLEMENTATION ROADMAP (PHASED)

The original plan defines 5 phases. This section translates each phase into a milestone with concrete entry criteria, deliverables, gate criteria (what must pass before proceeding to the next phase), and a mapping to tasks from the Section 2–6 task documents.

---

### 7.1 Phase Definitions

#### Task 7.1.1 — Phase 1: Core Simulation

**Objective:** Minimal end-to-end pipeline — eagle moves, cameras render, images are correct.

**Entry criteria:** Development environment set up (Section 8 tasks complete).

**Task mapping (from prior documents):**
- Task 2.1.1 — WorldConfig
- Task 2.1.2 — SimulationConfig + YAML infrastructure
- Task 2.2.1 — EagleState
- Task 2.2.2b — RandomWalkMotion (simplest motion generator)
- Task 3.1.1 — CameraIntrinsics
- Task 3.2.1 — CameraExtrinsics
- Task 3.3.1 — Camera class (project, unproject)
- Task 3.3.2 — Camera placement (ring generator, `build_cameras()`)
- Task 2.3.1 — Background model (uniform sky fill, no ground plane)
- Task 2.3.2 — Eagle projection + Gaussian blob rendering
- Task 2.3.6a — `render_frame()` composition (skip noise, vignetting)
- Task 4.1.1 — SimulationEngine with batch output mode
- Task 4.3.1 — Projection consistency validation

**Explicit scope exclusions:** No noise, no vignetting, no atmospheric attenuation, no distortion (zero distortion coefficients), no voxel tracking, no multi-eagle.

**Deliverables:**
- Running simulation producing thermal images for N cameras over T frames.
- Visual verification: animated GIF of one camera's view showing the eagle blob traversing the image.
- Projection consistency check passing with 0.0 pixel error (no noise, no distortion).

**Gate criteria:**
- All Phase 1 unit tests pass.
- `validate_projection_consistency()` returns max error < 0.1 px.
- `validate_blob_intensity()` returns max error < 0.01 °C.
- At least 3 cameras, 100 frames, random walk trajectory, visual inspection confirms plausible motion.

---

#### Task 7.1.2 — Phase 2: Voxel Reconstruction (Offline)

**Objective:** Implement the voxel grid and Bayesian fusion. Process stored images offline. Produce 3D occupied voxel clouds and measure tracking accuracy.

**Entry criteria:** Phase 1 gate criteria met.

**Task mapping:**
- Task 5.1.1 — SparseVoxelGrid (hash-map backend)
- Task 5.2.1 — Bayesian fusion (dense subregion strategy for active voxel set)
- Task 5.3.1 — Connected component clustering
- Task 5.6.1a–c — Position error, detection rate, false positive rate
- Task 4.2.1 — Frame storage (NpzWriter/Reader) for offline processing
- Task X.5 (partial) — `visualize_voxel_grid()` for debugging

**Explicit scope exclusions:** No sparse/prediction-guided voxel selection (use dense subregion), no temporal decay, no Kalman filter, no Hungarian assignment (single cluster = single detection, direct centroid comparison to GT), no space carving (Bayesian only).

**Deliverables:**
- Offline pipeline: load stored frames → fuse → cluster → compute centroid → compare to GT.
- 3D visualization of occupied voxels overlaid with ground truth.
- `EvaluationReport` for the `baseline_clean` scenario.

**Gate criteria:**
- `baseline_clean` scenario: detection rate > 90%, mean position error < 2× voxel_size (< 2.0 m at 1.0 m voxel size).
- No false positives in the clean (noise-free) scenario.
- Occupied voxel visualization qualitatively matches the eagle's trajectory.

---

#### Task 7.1.3 — Phase 3: Sparse & Real-Time

**Objective:** Make the tracker efficient enough for online processing. Add prediction-guided voxel selection, temporal decay, pruning, multi-frame tracking with Hungarian + Kalman.

**Entry criteria:** Phase 2 gate criteria met.

**Task mapping:**
- Task 5.1.2 — Temporal decay and pruning
- Task 5.2.1c — Prediction-guided active voxel selection
- Task 5.2.2 — Space carving alternative (optional, benchmark against Bayesian)
- Task 5.3.2 — DBSCAN clustering alternative (optional, benchmark against connected components)
- Task 5.4.1 — Hungarian association + track lifecycle
- Task 5.4.2 — Kalman filter
- Task 5.5.1 — TrackingPipeline orchestrator
- Task 5.5.1d — `run_online_simulation()` (streaming mode)
- Task 6.1.3 — Processing time analysis
- Task X.4 — Performance profiling, numba optimization if needed

**Explicit scope exclusions:** No multi-eagle, no realism enhancements to rendering, no parameter sweeps.

**Deliverables:**
- Online pipeline: simulation streams frames → tracker processes in real time.
- `baseline_clean` and `baseline_noisy` benchmark results.
- Timing analysis showing per-frame latency and bottleneck identification.

**Gate criteria:**
- `baseline_clean`: detection rate > 95%, mean position error < 1.5 m, no FPs.
- `baseline_noisy`: detection rate > 85%, mean position error < 2.0 m, FP rate < 5%.
- Mean per-frame latency < 100 ms (intermediate target; Phase 5 targets real-time at < 50 ms).
- Kalman filter reduces position error std by ≥ 30% compared to raw centroid (measured on `baseline_noisy`).

---

#### Task 7.1.4 — Phase 4: Realism Enhancements

**Objective:** Add noise, vignetting, atmospheric attenuation to the rendering pipeline. Add multi-eagle support. Add simple terrain obstacles. Stress-test the tracker under degraded conditions.

**Entry criteria:** Phase 3 gate criteria met.

**Task mapping:**
- Task 2.3.3 — Noise model (Gaussian + shot + FPN)
- Task 2.3.4 — Vignetting model
- Task 2.3.5 — Atmospheric attenuation (already partially in Phase 1 blob rendering, but now fully parameterized and validated with SNR utility)
- Task 3.1.2 — Distortion and undistortion (enable non-zero distortion coefficients)
- Task 4.1.2 — Multi-eagle extension
- Task 2.2.2c — SplineMotion (for more realistic trajectories)
- Task 2.2.2d — LissajousMotion (for repeatable multi-eagle separation tests)
- Task 2.1.1d — Ground plane (optional terrain)
- Task 3.3.3 — Calibration verification pipeline (validate distortion handling)
- Task 6.1.2 — Full MOT metrics (MOTA, MOTP, ID switches — now relevant with multi-eagle)

**New tasks specific to Phase 4:**

- **7.1.4a** Implement multi-eagle data association stress test:
  - Scenario: 2 eagles with crossing paths. Verify that the tracker maintains distinct track IDs through the crossing (no ID switch) or, if an ID switch occurs, that MOTA reflects the penalty.
  - Scenario: 2 eagles, one exits the ROI. Verify the remaining track continues and the lost track is properly terminated.

- **7.1.4b** Implement terrain occlusion test:
  - Place a simple obstacle (e.g., a tall rectangular prism representing a cliff or building) in the world.
  - Verify that when the eagle passes behind the obstacle relative to a camera, that camera's image shows no blob (the rendering pipeline handles occlusion).
  - For Phase 1 rendering (projection + Gaussian blob), occlusion is NOT automatically handled. Implement a simple occlusion check: before rendering the blob in a camera, ray-cast from camera to eagle; if the ray intersects any obstacle geometry, suppress the blob for that camera.
  - This requires: `intersect_ray_mesh(origin, direction, mesh) -> bool`. Use `trimesh.ray` if a trimesh-based obstacle is defined, or a simple analytical ray-box intersection for rectangular prisms.

- **7.1.4c** Implement distortion-aware rendering:
  - In Phase 1, rendering uses `cv2.projectPoints` which already applies distortion to the projected center. But the Gaussian blob is drawn in pixel space around the distorted center — this is an approximation that breaks for large distortion near image edges (the blob should be distorted, not just shifted).
  - For Phase 4: either (a) render the blob in undistorted space and then apply `cv2.remap` to distort the entire image (more correct, moderate cost), or (b) accept the approximation and document its limitations (the error is small for typical LWIR lenses with mild distortion).
  - Implement option (a) with a flag to fall back to (b) for performance.

**Deliverables:**
- Full rendering pipeline with all realism effects enabled.
- Multi-eagle tracking working for at least 2 eagles.
- All 8 canonical benchmark scenarios passing.
- Distortion-aware rendering (or documented approximation).
- Occlusion handling for simple obstacles.

**Gate criteria:**
- `baseline_noisy` with full rendering effects: detection rate > 80%, RMSE < 2.5 m.
- 2-eagle crossing scenario: MOTA > 0.7, ID switches ≤ 2.
- Calibration verification (Task 3.3.3) passes with reprojection error < 0.5 px including distortion.
- `high_noise` scenario: detection rate > 60% (tracker degrades gracefully, no crashes).

---

#### Task 7.1.5 — Phase 5: Validation and Export

**Objective:** Run systematic parameter sweeps, identify optimal configurations, and export results for real-world deployment planning.

**Entry criteria:** Phase 4 gate criteria met.

**Task mapping:**
- Task X.7 — Parameter sweep infrastructure
- Task 6.2.1 — Benchmark runner (automated)
- Task 6.2.2 — Comparative reporting (HTML, LaTeX)
- Task 6.1.1 — Full position error analysis with geometric decomposition

**New tasks specific to Phase 5:**

- **7.1.5a** Define the sweep parameter space:
  - **Camera placement:** number of cameras (2, 3, 4, 6, 8), ring radius (250, 500, 1000 m), pole height (5, 10, 20 m), FOV (24°, 32°, 45°).
  - **Voxel size:** 0.25, 0.5, 1.0, 2.0, 5.0 m.
  - **Noise level:** NETD 0.0, 0.02, 0.05, 0.1, 0.2 °C.
  - **Fusion method:** Bayesian vs. space carving.
  - **Atmospheric conditions:** α = 0.0001, 0.0005, 0.001, 0.005 m⁻¹.
  - Full Cartesian product is too large (~50,000 combinations). Use a Latin Hypercube Sampling (LHS) or a structured factorial design to sample ~500–1000 representative combinations.

- **7.1.5b** Implement `CameraPlacementOptimizer`:
  - Given the sweep results, identify the Pareto front: configurations that are not dominated in any metric (detection rate, position error, FP rate, cost proxy = number of cameras).
  - Visualize the Pareto front as a scatter matrix (detection rate vs. RMSE, colored by num_cameras).
  - Output a ranked list of recommended configurations with trade-off annotations.

- **7.1.5c** Implement configuration export:
  - `export_deployment_config(optimal_config: SimulationConfig, output_path: str)`:
    - Writes a deployment-ready configuration file containing:
      - Camera positions and orientations (in a format consumable by the hardware installation team — e.g., GPS coordinates, compass bearing, tilt angle).
      - Recommended voxel size and fusion parameters.
      - Expected performance metrics under simulated conditions.
      - Environmental assumptions (attenuation coefficient, noise level) used in the simulation.
    - Include a disclaimer section: simulated performance is an upper bound; real-world factors (calibration error, weather, wildlife behavior variability) may degrade results.

- **7.1.5d** Implement sensitivity analysis:
  - For each parameter, compute the partial derivative of each metric with respect to that parameter (finite differences from the sweep data).
  - Identify which parameters the system is most sensitive to. If tracking error is highly sensitive to `α` (atmospheric attenuation), that flags the need for accurate atmospheric characterization in the field.
  - Visualize as a sensitivity bar chart or tornado diagram.

**Deliverables:**
- Parameter sweep results (stored as a structured DataFrame / CSV).
- Pareto front visualization.
- Deployment configuration export.
- Sensitivity analysis report.
- Final HTML report summarizing the complete validation study.

**Gate criteria:**
- At least 500 sweep configurations completed.
- At least one configuration achieving: detection rate > 90%, RMSE < 1.5 m, FP rate < 2%, with ≤ 6 cameras.
- Deployment config exported and reviewed.
- Final report generated and passes manual review.

---

### 7.2 Milestone Tracking Infrastructure

#### Task 7.2.1 — Phase Gate Automation

**Objective:** Automate the phase gate checks so that progression from one phase to the next is a verifiable, repeatable decision.

**Sub-tasks:**

- **7.2.1a** Implement `PhaseGate` class:
  ```python
  class PhaseGate:
      def __init__(self, phase: int, criteria: List[GateCriterion]): ...
      def evaluate(self, benchmark_results: List[BenchmarkResult]) -> GateResult: ...
  ```
  - `GateCriterion`: `(metric_name: str, scenario_name: str, operator: str, threshold: float)` — e.g., `("detection_rate", "baseline_clean", ">=", 0.95)`.
  - `GateResult`: `{passed: bool, details: List[(criterion, actual_value, pass/fail)]}`.

- **7.2.1b** Define gate criteria for all 5 phases as YAML:
  ```yaml
  phase_1:
    - metric: projection_max_error_px
      scenario: baseline_clean
      op: "<="
      threshold: 0.1
    - metric: blob_intensity_max_error_C
      scenario: baseline_clean
      op: "<="
      threshold: 0.01
  phase_2:
    - metric: detection_rate
      scenario: baseline_clean
      op: ">="
      threshold: 0.90
    ...
  ```

- **7.2.1c** Integrate `PhaseGate.evaluate()` into the CI pipeline:
  - After running the benchmark suite, evaluate the gate for the current target phase.
  - Report pass/fail per criterion in CI output.
  - Block merges to the `main` branch if the current phase gate fails (optional, configurable).

**Deliverables:**
- `PhaseGate`, `GateCriterion`, `GateResult`.
- Gate criteria YAML for all 5 phases.
- CI integration.

**Dependencies:** Task 6.2.1 (BenchmarkSuite).

---

## SECTION 8 — RECOMMENDED PYTHON LIBRARIES & ENVIRONMENT

---

### 8.1 Dependency Management

#### Task 8.1.1 — Environment and Packaging Setup

**Objective:** Establish a reproducible Python environment with all required dependencies pinned and tested.

**Sub-tasks:**

- **8.1.1a** Choose a dependency management tool:
  - **Option A — `pyproject.toml` + `pip` + `pip-tools`:** Standard, minimal tooling. Use `pip-compile` to generate a locked `requirements.txt` from abstract dependencies. Recommended for projects that will be distributed as a package.
  - **Option B — `poetry`:** Integrated dependency resolution, virtual environment management, and packaging. Heavier but more ergonomic for multi-developer teams.
  - **Option C — `conda` / `mamba` + `environment.yml`:** Preferred if any dependencies have complex native builds (e.g., `open3d`, `pyrender` with EGL). Conda handles system-level libraries (OpenGL, BLAS) that pip cannot.
  - **Recommendation:** Use `pyproject.toml` with `pip-tools` for the core package. Provide an optional `environment.yml` for users who need conda-managed system libraries (OpenGL for pyrender, GPU drivers for numba CUDA).

- **8.1.1b** Define the dependency tiers:

  **Core (required for all phases):**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `numpy` | `>=1.24,<2.0` | Array operations, linear algebra. Pin below 2.0 until all downstream libraries (opencv, scipy) confirm numpy 2.x compatibility. |
  | `opencv-python-headless` | `>=4.8` | Camera projection, distortion, image processing. Use `-headless` to avoid pulling in GUI dependencies on servers. |
  | `scipy` | `>=1.11` | Spline interpolation, connected components (`ndimage.label`), Hungarian assignment (`optimize.linear_sum_assignment`), spatial transforms (`Rotation`). |
  | `pyyaml` | `>=6.0` | YAML config parsing. |
  | `pydantic` | `>=2.0` | Config schema validation. |

  **Tracking extras (Phase 2+):**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `scikit-learn` | `>=1.3` | DBSCAN clustering, nearest-neighbor queries. |

  **Visualization extras (optional):**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `matplotlib` | `>=3.7` | 2D/3D plotting, animation. |
  | `plotly` | `>=5.15` | Interactive 3D plots (alternative to matplotlib). |
  | `open3d` | `>=0.17` | Interactive 3D point cloud / voxel visualization. Install via conda if pip build fails. |
  | `pyvista` | `>=0.42` | Alternative 3D viz, mesh handling. |

  **Rendering extras (Phase 4, optional):**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `pyrender` | `>=0.1.45` | OpenGL-based mesh rendering for advanced thermal rendering. Requires EGL or osmesa on headless servers. |
  | `trimesh` | `>=4.0` | Mesh I/O, ray intersection for occlusion tests. |

  **Performance extras (Phase 3+, optional):**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `numba` | `>=0.58` | JIT compilation for inner loops (voxel projection, fusion). |

  **Analysis extras (Phase 5):**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `pandas` | `>=2.0` | Parameter sweep result DataFrames. |
  | `h5py` | `>=3.9` | HDF5 I/O for large simulation datasets. |

  **Testing:**
  | Package | Version Constraint | Purpose |
  |---|---|---|
  | `pytest` | `>=7.4` | Test runner. |
  | `pytest-cov` | `>=4.1` | Coverage reporting. |

- **8.1.1c** Implement `pyproject.toml` with optional dependency groups:
  ```toml
  [project]
  name = "thermal-tracking-sim"
  dependencies = [
      "numpy>=1.24,<2.0",
      "opencv-python-headless>=4.8",
      "scipy>=1.11",
      "pyyaml>=6.0",
      "pydantic>=2.0",
  ]

  [project.optional-dependencies]
  tracking = ["scikit-learn>=1.3"]
  viz = ["matplotlib>=3.7", "plotly>=5.15", "open3d>=0.17"]
  rendering = ["pyrender>=0.1.45", "trimesh>=4.0"]
  perf = ["numba>=0.58"]
  analysis = ["pandas>=2.0", "h5py>=3.9"]
  dev = ["pytest>=7.4", "pytest-cov>=4.1"]
  all = ["thermal-tracking-sim[tracking,viz,rendering,perf,analysis,dev]"]
  ```

- **8.1.1d** Create lock files:
  - `requirements/core.txt` — pip-compiled from core dependencies only.
  - `requirements/all.txt` — pip-compiled with all optional groups.
  - `environment.yml` — conda environment for users needing system libraries.
  - Pin all transitive dependencies for CI reproducibility.

- **8.1.1e** Implement a `verify_environment.py` script:
  - Imports all required packages.
  - Checks version constraints.
  - Runs a smoke test: create a camera, project a point, render a single frame. If this works, the environment is functional.
  - Reports any missing optional dependencies with installation instructions.

**Deliverables:**
- `pyproject.toml` with dependency groups.
- Lock files for core and all.
- `environment.yml` for conda.
- `verify_environment.py` smoke test.
- CI workflow that installs from lock file and runs the smoke test.

**Dependencies:** None. This is a root task, prerequisite for all development.

---

#### Task 8.1.2 — Library Compatibility Verification

**Objective:** Verify that all chosen libraries work together correctly, especially at API boundaries.

**Sub-tasks:**

- **8.1.2a** OpenCV projection pipeline verification:
  - `cv2.projectPoints` with non-zero distortion coefficients and the Rodrigues rotation vector representation.
  - Verify convention: OpenCV uses `tvec = -R @ C` (camera translation, not camera position). Ensure `CameraExtrinsics` (Task 3.2.1) produces the correct `tvec`.
  - Verify that `cv2.projectPoints` output ordering matches expected `(u, v)` = `(column, row)` convention.

- **8.1.2b** Scipy spatial transform verification:
  - `scipy.spatial.transform.Rotation` quaternion convention: scalar-last `(x, y, z, w)` vs. scalar-first `(w, x, y, z)`. Scipy uses scalar-last. If internal code stores quaternions as scalar-first (common in aerospace), implement the conversion and unit-test it.
  - `Rotation.from_euler` sequence specification: verify that `'ZYX'` extrinsic = `'xyz'` intrinsic (lowercase in scipy = intrinsic rotations). Document the chosen convention.

- **8.1.2c** NumPy / OpenCV dtype interop:
  - OpenCV expects specific dtypes for certain operations (e.g., `cv2.projectPoints` wants `float64` inputs).
  - Thermal images are `float32`. Verify that all pipeline stages handle dtype correctly and no silent precision loss occurs.
  - Implement a `ensure_dtype(array, dtype)` utility that logs a warning on implicit conversion.

- **8.1.2d** Open3D / Matplotlib / Plotly visualization backend:
  - Verify that `open3d` can display voxels on the target platform (headless servers need offscreen rendering via EGL or MESA).
  - If `open3d` is not available, fall back to `matplotlib` 3D scatter (with a warning about interactivity limitations).
  - Implement a `get_visualization_backend() -> str` utility that probes available packages and returns `"open3d"`, `"pyvista"`, or `"matplotlib"`.

**Deliverables:**
- Compatibility test suite (can be part of the smoke test or a dedicated `test_compatibility.py`).
- Convention documentation: quaternion ordering, Euler sequence, OpenCV tvec sign, dtype expectations.

**Dependencies:** Task 8.1.1 (environment setup).

---

## SECTION 9 — DATA FLOW ARCHITECTURE

---

### 9.1 Formal Data Flow Specification

#### Task 9.1.1 — Interface Contracts

**Objective:** Define the formal data types and interface contracts between all pipeline stages, ensuring that components developed in parallel integrate cleanly.

**Sub-tasks:**

- **9.1.1a** Define all inter-stage data types as frozen dataclasses or pydantic models. Collect them in a single `types.py` module (or a `types/` package) so that all components import from one source of truth:

  | Data Type | Producer | Consumer(s) | Key Fields |
  |---|---|---|---|
  | `SimulationConfig` | YAML loader | SimulationEngine, TrackingPipeline, BenchmarkSuite | world, eagle, cameras, rendering, voxel, fusion, tracking, run params |
  | `EagleState` | MotionGenerator | Renderer, GT storage | position, velocity, temperature, radius |
  | `Camera` | build_cameras() | Renderer, Fusion, Clustering (unproject), Coverage | id, intrinsics, extrinsics, project(), unproject() |
  | `FrameBundle` | SimulationEngine / SimulationReader | TrackingPipeline, Validation, Visualization | timestamp, camera_images (Dict[str, ndarray]), gt_3d, gt_2d |
  | `VoxelData` | SparseVoxelGrid | Fusion, Clustering, Visualization | log_odds, last_updated_frame, probability |
  | `VoxelCluster` | Clustering | Tracker | cluster_id, voxel_positions, centroid, size |
  | `Track` | Tracker | Evaluation, Visualization | track_id, history, state, kalman_state |
  | `TrackingResult` | TrackingPipeline | Evaluation, Visualization, Storage | frame_index, clusters, active_tracks, estimated_position, timing |
  | `EvaluationReport` | Metrics computation | BenchmarkSuite, Reporting | config, position_errors, detection_rate, FP_rate, MOT_metrics, timing |
  | `BenchmarkResult` | BenchmarkSuite | Comparison, PhaseGate | scenario_name, config, evaluation_report, wall_clock_time |

- **9.1.1b** Define method signatures for all pipeline-boundary functions as abstract base classes or Protocol types (PEP 544). This allows type checkers (`mypy`) to verify integration correctness at development time:
  ```python
  class FrameSource(Protocol):
      def __iter__(self) -> Iterator[FrameBundle]: ...
      def __len__(self) -> int: ...

  class FusionMethod(Protocol):
      def fuse(self, voxel_grid: SparseVoxelGrid, cameras: List[Camera],
               images: List[np.ndarray], frame_index: int) -> None: ...

  class ClusteringMethod(Protocol):
      def cluster(self, voxel_grid: SparseVoxelGrid) -> List[VoxelCluster]: ...
  ```

- **9.1.1c** Enforce immutability of data flowing between stages:
  - `FrameBundle` images should be read-only (`ndarray.flags.writeable = False`) after creation to prevent the tracker from accidentally modifying the renderer's output.
  - `EagleState.position` should be a copy, not a reference, when stored in ground truth.
  - Use `dataclasses(frozen=True)` or pydantic `model_config = ConfigDict(frozen=True)` where appropriate.
  - Where performance requires mutability (e.g., `SparseVoxelGrid` during fusion), document the ownership semantics: "the fusion step has exclusive write access to the voxel grid during `fuse_frame()`."

**Deliverables:**
- `types.py` module with all shared data types.
- Protocol definitions for pipeline boundaries.
- Immutability enforcement on inter-stage data.
- `mypy` configuration (`mypy.ini` or `pyproject.toml` section) and a passing `mypy` run on the codebase.

**Dependencies:** All prior task documents (this task codifies the interfaces they defined).

---

#### Task 9.1.2 — Pipeline Topology and Execution Modes

**Objective:** Formalize the two execution topologies (offline and online) and ensure the same pipeline components work in both.

**Sub-tasks:**

- **9.1.2a** Document the offline topology:
  ```
  [SimulationEngine] → [SimulationWriter] → disk
                                              ↓
                              [SimulationReader] → [TrackingPipeline] → [EvaluationReport]
  ```
  - Two-pass: generate all data first, then track. Allows re-running the tracker on the same data with different parameters without re-rendering.
  - The `SimulationReader` implements `FrameSource` (Protocol from 9.1.1b).

- **9.1.2b** Document the online topology:
  ```
  [SimulationEngine (streaming)] → [TrackingPipeline] → [EvaluationReport]
  ```
  - Single-pass: frames flow directly from engine to tracker via generator.
  - The `SimulationEngine` in streaming mode implements `FrameSource`.
  - No intermediate disk I/O; lower latency, lower storage, but tracking parameters are fixed for the run.

- **9.1.2c** Implement `run_pipeline(source: FrameSource, pipeline: TrackingPipeline, ground_truth: np.ndarray | None) -> EvaluationReport`:
  - Unified entry point that works with either topology.
  - If `ground_truth` is not provided (e.g., real camera feed in the future), skip metric computation and return a partial report with tracks only.

- **9.1.2d** Implement the real-world transition adapter (from plan Section 11.10):
  - Define `CameraFeedSource(FrameSource)`:
    - Abstract class for ingesting real camera feeds.
    - Provides `__iter__` yielding `FrameBundle` objects (with `ground_truth_3d = None`).
    - Concrete implementations (GStreamer, frame-grabber SDK) are out of scope for the simulation project but the interface must be defined now so that the `TrackingPipeline` is feed-agnostic.
  - This ensures zero code changes to the tracker when transitioning from simulation to live.

**Deliverables:**
- Pipeline topology documentation (can be in the project README or a `docs/architecture.md`).
- `run_pipeline()` unified entry point.
- `CameraFeedSource` abstract class (interface only, no implementation).

**Dependencies:** Task 4.1.1 (SimulationEngine), Task 4.2.1 (Reader/Writer), Task 5.5.1 (TrackingPipeline).

---

#### Task 9.1.3 — Data Flow Diagram Artifacts

**Objective:** Produce maintainable, version-controlled data flow diagrams that serve as living documentation.

**Sub-tasks:**

- **9.1.3a** Produce a component-level data flow diagram covering:
  - All major classes/modules (SimulationEngine, Camera, MotionGenerator, Renderer, VoxelGrid, Fusion, Clustering, Tracker, Evaluation).
  - Data types flowing between them (arrows labeled with type names from `types.py`).
  - The two execution topologies (offline and online) shown as variants.
  - Use Mermaid syntax for maintainability (renders in GitHub, GitLab, and most doc tools).

- **9.1.3b** Produce a per-frame sequence diagram showing the exact order of operations within one simulation + tracking step:
  1. `MotionGenerator.step()` → `EagleState`
  2. For each camera: `render_frame()` → `np.ndarray`
  3. Package into `FrameBundle`
  4. `preprocess_thermal_image()` per camera → hot masks
  5. `fuse_frame()` → updated `SparseVoxelGrid`
  6. `apply_temporal_decay()` + `prune()`
  7. `cluster_occupied_voxels()` → `List[VoxelCluster]`
  8. `tracker.update()` → `List[Track]`
  9. Package into `TrackingResult`
  10. `compute_metrics()` → `EvaluationReport` (accumulated)

- **9.1.3c** Produce a module dependency diagram:
  - Shows which Python modules import from which.
  - Verifies there are no circular dependencies.
  - Use `pydeps` or manual Mermaid diagram.
  - Enforce a layered architecture:
    - **Layer 0 (types):** `types.py` — no imports from other project modules.
    - **Layer 1 (primitives):** `camera.py`, `eagle.py`, `voxel_grid.py` — import only from types.
    - **Layer 2 (algorithms):** `renderer.py`, `fusion.py`, `clustering.py`, `tracker.py` — import from Layer 0–1.
    - **Layer 3 (orchestration):** `simulation_engine.py`, `tracking_pipeline.py` — import from Layer 0–2.
    - **Layer 4 (evaluation/IO):** `evaluation.py`, `storage.py`, `benchmark.py` — import from Layer 0–3.
    - **Layer 5 (visualization):** `visualization.py` — import from anything (leaf node).

- **9.1.3d** Store all diagrams in `docs/` as `.md` files with embedded Mermaid. Include a CI step that renders them to PNG/SVG for inclusion in reports (use `mermaid-cli` or similar).

**Deliverables:**
- Component-level data flow diagram (Mermaid + rendered).
- Per-frame sequence diagram (Mermaid + rendered).
- Module dependency diagram with layered architecture verification.
- `docs/` directory with all diagrams.

**Dependencies:** All prior tasks (this is a documentation task that reflects the implemented architecture).

---

### 9.2 Logging, Tracing, and Reproducibility (Pipeline-Wide)

#### Task 9.2.1 — Structured Logging

**Objective:** Implement structured logging across the pipeline for debugging, auditing, and performance analysis.

**Sub-tasks:**

- **9.2.1a** Configure Python `logging` with structured output:
  - Use `logging.config.dictConfig` loaded from the simulation config YAML.
  - Define loggers per module: `thermal_sim.engine`, `thermal_sim.renderer`, `thermal_sim.fusion`, `thermal_sim.tracker`, etc.
  - Default level: `INFO` for orchestration, `WARNING` for inner loops (to avoid log spam from per-voxel operations).
  - Output format: JSON lines for machine parsing (compatible with ELK stack / CloudWatch if the system is deployed), with a human-readable console handler for development.

- **9.2.1b** Define key log events:
  - `simulation.start`: logs full config (or config hash + path).
  - `simulation.frame`: logs frame index, eagle position, number of visible cameras.
  - `tracking.fusion`: logs number of active voxels, number of occupied voxels post-fusion, fusion time.
  - `tracking.clusters`: logs number of clusters, cluster sizes, centroid positions.
  - `tracking.association`: logs cost matrix shape, assignment result, new/lost/coasting tracks.
  - `tracking.result`: logs estimated position, error (if GT available), per-stage timing.
  - `evaluation.gate`: logs phase gate evaluation result.

- **9.2.1c** Implement `SimulationTrace`:
  - A lightweight in-memory trace that records per-frame metadata without the overhead of full logging to disk.
  - Stored as a list of dicts (or a DataFrame) accumulating one row per frame.
  - Fields: `frame_index, timestamp, eagle_pos, num_active_voxels, num_occupied, num_clusters, estimated_pos, error, t_fusion_ms, t_track_ms, t_total_ms`.
  - Serializable to CSV for post-hoc analysis.
  - This is distinct from `EvaluationReport` (which computes aggregates); the trace provides the raw per-frame data that feeds into the analysis.

**Deliverables:**
- Logging configuration (YAML, loadable from `SimulationConfig`).
- Key log events emitted from all pipeline stages.
- `SimulationTrace` with CSV export.

**Dependencies:** All pipeline components (this is wired in as a cross-cutting concern).

---

#### Task 9.2.2 — Reproducibility Guarantees

**Objective:** Ensure that any simulation run can be exactly reproduced given its configuration and seed.

**Sub-tasks:**

- **9.2.2a** Verify determinism:
  - Run the same configuration + seed twice. Diff all output images (bitwise) and all tracking results. They must be identical.
  - This requires: (a) all RNGs are seeded and used in a fixed order (Task X.1 from Section 2–3 document), (b) no use of Python `dict` iteration order for anything that affects computation (dict iteration order is insertion-ordered since Python 3.7, but verify no hash-order-dependent code exists), (c) no floating-point non-determinism from multithreaded BLAS or OpenCV (set `OMP_NUM_THREADS=1` and `cv2.setNumThreads(1)` during reproducibility checks).

- **9.2.2b** Implement `reproducibility_check(config_path: str) -> bool`:
  - Runs the full pipeline twice with the same config.
  - Compares all outputs.
  - Returns True if bitwise identical, False with a diagnostic of where divergence occurred.
  - Include this as a CI test (slow, run nightly rather than per-commit).

- **9.2.2c** Document known sources of non-determinism and their mitigations:
  - NumPy BLAS threading: mitigate with `OMP_NUM_THREADS=1` (at a performance cost) or accept within-epsilon floating-point differences.
  - Numba parallel: `parallel=True` may introduce non-determinism due to floating-point summation order. Mitigate with `parallel=False` for reproducibility runs, or use compensated summation (Kahan).
  - Document these in `docs/reproducibility.md`.

**Deliverables:**
- `reproducibility_check()` function.
- Nightly CI job running the check.
- `docs/reproducibility.md` documenting known issues and mitigations.

**Dependencies:** Task 4.1.1, Task 5.5.1 (full pipeline).

---

## CROSS-CUTTING CONCERNS (Sections 6–9)

### Task X.8 — Project Structure and Packaging

- Define the Python package layout:
  ```
  thermal_tracking_sim/
    __init__.py
    types.py                  # All shared data types (Layer 0)
    camera.py                 # Camera, CameraIntrinsics, CameraExtrinsics (Layer 1)
    eagle.py                  # EagleState, MotionGenerators (Layer 1)
    voxel_grid.py             # SparseVoxelGrid, VoxelData (Layer 1)
    renderer.py               # Rendering pipeline (Layer 2)
    fusion.py                 # Bayesian fusion, space carving (Layer 2)
    clustering.py             # Connected components, DBSCAN (Layer 2)
    tracker.py                # Kalman, Hungarian, Tracker (Layer 2)
    simulation_engine.py      # SimulationEngine (Layer 3)
    tracking_pipeline.py      # TrackingPipeline (Layer 3)
    evaluation.py             # Metrics, MOTMetrics, reports (Layer 4)
    storage.py                # Writer/Reader (Layer 4)
    benchmark.py              # BenchmarkSuite, PhaseGate (Layer 4)
    visualization.py          # All plotting/animation (Layer 5)
    config.py                 # Config loading, SimulationConfig (Layer 0–1)
    utils.py                  # ensure_dtype, bounds checking, etc.
  tests/
    test_camera.py
    test_eagle.py
    test_renderer.py
    test_fusion.py
    test_clustering.py
    test_tracker.py
    test_integration.py
    test_benchmark.py
    test_reproducibility.py
  configs/
    default_config.yaml
    scenarios/
      baseline_clean.yaml
      baseline_noisy.yaml
      ...
  docs/
    architecture.md           # Data flow diagrams
    conventions.md            # Coordinate systems, units, quaternion ordering
    reproducibility.md
  ```

### Task X.9 — CI/CD Pipeline

- Define a GitHub Actions (or equivalent) workflow:
  - **On every push:**
    - Install from lock file.
    - Run `verify_environment.py`.
    - Run `pytest tests/ -x --cov` (fail fast, with coverage).
    - Run `mypy thermal_tracking_sim/`.
  - **On PR to main:**
    - All of the above.
    - Run `BenchmarkSuite` on canonical scenarios.
    - Run `PhaseGate.evaluate()` for the current target phase.
    - Post benchmark results as a PR comment.
  - **Nightly:**
    - Run `reproducibility_check()`.
    - Run full parameter sweep (if Phase 5 is active).
    - Update baseline metrics if a new release is tagged.

---

## TASK DEPENDENCY GRAPH (SECTIONS 6–9)

```
Sections 2–5 deliverables
  │
  ├──→ 6.1.1 Position Error Analysis Suite
  │         │
  ├──→ 6.1.2 Detection & FP Metrics (MOTMetrics) ──┐
  │                                                  │
  ├──→ 6.1.3 Processing Time Analysis              │
  │                                                  │
  │    ┌─────────────────────────────────────────────┘
  │    ▼
  ├──→ 6.2.1 Benchmark Runner ──→ 6.2.2 Comparative Reporting
  │         │
  │         └──→ 7.2.1 Phase Gate Automation
  │
  ├──→ 7.1.1–7.1.5 Phase Definitions (documentation + gate criteria YAML)
  │
  ├──→ 8.1.1 Environment & Packaging ──→ 8.1.2 Library Compatibility
  │
  ├──→ 9.1.1 Interface Contracts (types.py, Protocols)
  │         │
  │         ├──→ 9.1.2 Pipeline Topology & Execution Modes
  │         │         │
  │         │         └──→ 9.1.2d CameraFeedSource (real-world interface)
  │         │
  │         └──→ 9.1.3 Data Flow Diagrams
  │
  ├──→ 9.2.1 Structured Logging
  │
  ├──→ 9.2.2 Reproducibility Guarantees
  │
  └──→ X.8 Project Structure ──→ X.9 CI/CD Pipeline
```

**Critical path for initial development:** Task 8.1.1 (environment setup) is the absolute first task — nothing else can begin without a working Python environment. Then 9.1.1 (interface contracts / `types.py`) should be established before parallel development of pipeline stages begins, as it defines the integration surface.

**Critical path for evaluation maturity:** 6.1.2 (MOT metrics) → 6.2.1 (Benchmark runner) → 7.2.1 (Phase gates) → CI integration. This chain must be complete before Phase 2 gate evaluation can be automated.

**Parallelizable:** Sections 6 (evaluation), 8 (environment), and 9 (architecture) tasks can largely proceed in parallel with each other, as they have minimal inter-dependencies. Section 7 (roadmap) is primarily a documentation and gate-criteria task that consumes outputs from 6 and 8.



# Task Document: Section 10 (Technical Notes) & Section 11 (Additional Considerations)

**Scope:** All implementation work covering synchronization, performance optimization, extensibility architecture (Section 10), and the eleven additional considerations spanning coordinate conventions, configuration management, sensor specs, FOV coverage, calibration, temporal decay, multi-target association, logging, testing, and real-world transition (Section 11).

**Audience:** Computer scientists, software engineers, system design engineers, physicists.

**Prerequisite:** Sections 2–9 task documents.

**Note on overlap:** Many Section 11 topics were partially addressed in earlier task documents as they naturally arose (e.g., temporal decay in Task 5.1.2, configuration management in Task 2.1.2, testing in various acceptance criteria). This document consolidates those threads, fills gaps, and adds the implementation depth that the plan's bullet points left implicit. Where a prior task already covers the topic, we reference it and define only the residual work.

---

## SECTION 10 — TECHNICAL NOTES

---

### 10.1 Synchronization

#### Task 10.1.1 — Simulation-Time Synchronization Model

**Objective:** Formalize the time model within the simulation and define the abstraction that will later accommodate real-world clock drift and network jitter.

**Sub-tasks:**

- **10.1.1a** Define `SimulationClock` class:
  - Maintains a discrete simulation time `t` advanced in fixed increments of `dt`.
  - All components (motion generator, renderer, fusion, tracker) receive `t` from the clock — they never maintain independent time counters.
  - Properties:
    - `current_time -> float`: continuous time in seconds.
    - `current_frame -> int`: frame index.
    - `dt -> float`: time step.
    - `advance() -> float`: increments and returns new time.
    - `reset()`.
  - In simulation mode, `advance()` is instantaneous (logical time). In real-world mode, it blocks until wall-clock time reaches the next tick (or reports drift).

- **10.1.1b** Define `SyncPolicy` enum and implement behaviors:
  - `PERFECT_SYNC` (simulation default): All cameras produce frames at exactly the same `t`. The fusion step receives a batch of images guaranteed to be temporally aligned. No interpolation needed.
  - `JITTERED_SYNC` (Phase 4 realism enhancement): Each camera's effective capture time is `t + δ_c` where `δ_c ~ Uniform(-jitter, +jitter)`. The jitter parameter (e.g., ±1 ms for hardware-triggered cameras, ±10 ms for software-triggered) is part of `SimulationConfig`. The eagle's position at camera `c`'s capture time is `p(t + δ_c)`, computed by interpolating the motion model.
    - Implement `MotionGenerator.interpolate(t: float) -> EagleState`: given an arbitrary continuous time, return the interpolated state. For `RandomWalkMotion`, this requires storing at least two consecutive states and lerping. For `SplineMotion` and `LissajousMotion`, the underlying parameterization already supports continuous evaluation.
    - The renderer uses the per-camera jittered time to compute the eagle's position for that camera's frame.
  - `REAL_WORLD` (production): Camera frames arrive with hardware timestamps (PTP/NTP synchronized). The fusion step must handle frames that are not exactly aligned. Implement a `FrameSynchronizer` that buffers incoming frames and releases a `FrameBundle` when all cameras have reported within a configurable time window (e.g., ±5 ms). Late frames are either dropped or the bundle is released with a partial set (configurable).

- **10.1.1c** Implement `FrameSynchronizer` class (for `REAL_WORLD` and `JITTERED_SYNC` modes):
  ```python
  class FrameSynchronizer:
      def __init__(self, camera_ids: List[str], sync_window_ms: float, timeout_ms: float): ...
      def submit_frame(self, camera_id: str, timestamp: float, image: np.ndarray) -> None: ...
      def get_bundle(self) -> FrameBundle | None: ...
  ```
  - `submit_frame()` adds a frame to the internal buffer.
  - `get_bundle()` returns a `FrameBundle` once all cameras (or a quorum) have submitted frames within the sync window. Returns `None` if the window hasn't closed yet.
  - The `FrameBundle.timestamp` is set to the median of the individual camera timestamps.
  - Track statistics: number of dropped frames, mean inter-camera time spread, number of partial bundles.

- **10.1.1d** Wire `SimulationClock` into `SimulationEngine`:
  - Replace the current `for t in range(num_frames)` loop with `while clock.current_frame < num_frames: clock.advance(); ...`.
  - All downstream components receive `clock.current_time` and `clock.current_frame` rather than computing time independently.

**Deliverables:**
- `SimulationClock`, `SyncPolicy`, `FrameSynchronizer`.
- `MotionGenerator.interpolate()` for all three motion types.
- Unit tests:
  - `PERFECT_SYNC`: all camera timestamps in a bundle are identical.
  - `JITTERED_SYNC`: camera timestamps differ by at most `jitter`, eagle positions differ accordingly.
  - `FrameSynchronizer`: submit 4 camera frames with 3 ms spread → bundle released. Submit 3 of 4 within timeout → partial bundle or timeout behavior per config.

**Dependencies:** Task 4.1.1 (SimulationEngine), Task 2.2.2 (MotionGenerators).

---

### 10.2 Performance

#### Task 10.2.1 — Performance Profiling Framework

**Objective:** Establish systematic profiling infrastructure so that bottlenecks are identified quantitatively, not by intuition.

**Sub-tasks:**

- **10.2.1a** Implement `Profiler` context manager:
  ```python
  class Profiler:
      def __init__(self, name: str, enabled: bool = True): ...
      def __enter__(self): ...
      def __exit__(self, *args): ...
      @staticmethod
      def report() -> pd.DataFrame: ...
  ```
  - Uses `time.perf_counter_ns()`.
  - Accumulates call count, total time, min/max/mean per named section.
  - Thread-safe (use `threading.local` for per-thread accumulators).
  - `report()` returns a DataFrame sorted by total time descending.
  - Can be globally disabled via a flag for production runs (zero overhead when disabled via a no-op context manager).

- **10.2.1b** Instrument all performance-critical sections:
  - `render_frame()`: total, projection, blob drawing, noise, vignetting.
  - `fuse_frame()`: total, batch projection per camera, mask lookup, log-odds update, decay, prune.
  - `cluster_occupied_voxels()`: total, component labeling, centroid computation.
  - `tracker.update()`: total, cost matrix construction, Hungarian solve, Kalman predict/update.

- **10.2.1c** Implement `profile_simulation(config, num_frames=100) -> ProfilingReport`:
  - Runs the full pipeline with profiling enabled.
  - Returns structured report with per-stage breakdown.
  - Identifies the top-3 bottleneck functions and their percentage of total frame time.
  - Estimates per-unit costs: µs per voxel per camera (fusion), µs per pixel (rendering), µs per track (association).

**Deliverables:**
- `Profiler` context manager with global accumulation and reporting.
- Instrumentation across all pipeline stages.
- `profile_simulation()` entry point.

**Dependencies:** Task 5.5.1 (TrackingPipeline), Task 4.1.1 (SimulationEngine).

---

#### Task 10.2.2 — Spatial Optimization: Prediction-Guided Voxel Selection

**Objective:** Reduce the number of voxels evaluated per frame by exploiting temporal coherence — the eagle doesn't teleport between frames.

**Sub-tasks:**

- **10.2.2a** Implement `PredictionGuidedSelector`:
  ```python
  class PredictionGuidedSelector:
      def __init__(self, voxel_grid: SparseVoxelGrid, max_speed: float, dt: float, margin_factor: float = 3.0): ...
      def get_active_region(self, previous_centroid: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]: ...
      def enumerate_active_voxels(self, previous_centroid: np.ndarray | None) -> Iterator[Tuple[int,int,int]]: ...
  ```
  - If `previous_centroid` is available: define a cubic region centered on it with half-width `max_speed * dt * margin_factor`. Convert to grid index ranges. Enumerate all voxels in this box.
  - If `previous_centroid` is `None` (first frame, or track lost): fall back to a larger search region (e.g., the full ROI, or a coarser grid with larger voxels for initial detection, then refine).
  - The `margin_factor` of 3.0 provides a safety buffer of 3× the maximum possible displacement per frame. This accounts for acceleration and centroid estimation error.

- **10.2.2b** Implement coarse-to-fine search for track initialization:
  - Phase 1 (coarse): evaluate a sparse grid (e.g., every 4th voxel in each dimension → 64× fewer voxels). Identify regions with high occupancy probability.
  - Phase 2 (fine): around each coarse detection, evaluate the full-resolution grid.
  - This reduces cold-start cost from `O(N³)` to `O(N³/64 + k³)` where `k` is the fine search radius around each coarse detection.

- **10.2.2c** Integrate `PredictionGuidedSelector` into `fuse_frame()`:
  - Replace the voxel enumeration strategy (Task 5.2.1c) with the selector's output.
  - When the Kalman filter is available (Phase 3+), use the Kalman-predicted position instead of the raw previous centroid. The Kalman prediction accounts for velocity, so the search region can be tighter (the margin accounts for acceleration uncertainty rather than the full velocity).

- **10.2.2d** Quantify the speedup:
  - Compare `fuse_frame()` latency with dense enumeration vs. prediction-guided for the `baseline_clean` scenario.
  - Expected: 10–100× reduction in voxel count (from ~50 million for a 500×500×200 m ROI at 1 m resolution to ~27,000 for a 30×30×30 m search cube).

**Deliverables:**
- `PredictionGuidedSelector` with coarse-to-fine search.
- Integration into `fuse_frame()`.
- Benchmark: latency reduction factor and tracking accuracy comparison (must be identical to dense).

**Dependencies:** Task 5.2.1 (fuse_frame), Task 5.4.2 (Kalman filter for predicted position).

---

#### Task 10.2.3 — Numba JIT Acceleration

**Objective:** Accelerate the inner loops that remain bottlenecks after numpy vectorization.

**Sub-tasks:**

- **10.2.3a** Identify JIT candidates by profiling (Task 10.2.1):
  - Primary candidate: the batch projection + mask lookup + log-odds update loop in `fuse_frame()`. Even with numpy vectorization, the per-camera loop and the conditional logic (visibility check, in-bounds check) may leave Python overhead.
  - Secondary candidate: the blob rendering inner loop in `render_thermal_blob()` if the Gaussian kernel is large.
  - Tertiary: 3D DDA ray traversal in space carving (Task 5.2.2c).

- **10.2.3b** Implement numba-accelerated `_fuse_frame_inner()`:
  ```python
  @numba.njit(parallel=True)
  def _fuse_frame_inner(
      voxel_centers: np.ndarray,    # (N, 3) float64
      camera_Ps: np.ndarray,        # (C, 3, 4) float64 — projection matrices
      hot_masks: np.ndarray,        # (C, H, W) bool
      image_shapes: np.ndarray,     # (C, 2) int — (H, W) per camera
      p_hot_occ: float,
      p_hot_empty: float,
      log_odds_in: np.ndarray,      # (N,) float64
  ) -> np.ndarray:                  # (N,) float64 — updated log_odds
  ```
  - Inner loop parallelized over voxels (`numba.prange`).
  - For each voxel, project through each camera's P matrix (manual 3×4 matmul, avoiding OpenCV overhead). Homogeneous division to get pixel coordinates. Bounds check. Mask lookup. Accumulate log-likelihood ratio.
  - This avoids per-voxel Python overhead entirely and enables SIMD vectorization via LLVM.
  - Note: using `camera.P()` (the 3×4 projection matrix) directly is slightly less accurate than `cv2.projectPoints` with distortion, but the distortion error at thermal camera resolutions is typically sub-pixel. For production accuracy, apply a pre-computed undistortion map to the hot masks instead, so the projection can use the simple pinhole model.

- **10.2.3c** Implement numba-accelerated `_ray_march_3d_dda()` for space carving:
  ```python
  @numba.njit
  def _ray_march_3d_dda(
      origin: np.ndarray,     # (3,) camera position
      direction: np.ndarray,  # (3,) ray direction (unit)
      grid_min: np.ndarray,   # (3,) ROI min corner
      voxel_size: float,
      grid_shape: np.ndarray, # (3,) int — grid dimensions
      max_steps: int,
  ) -> np.ndarray:            # (K, 3) int — traversed grid indices
  ```
  - Amanatides-Woo algorithm: compute initial tMax and tDelta for each axis, step through the grid incrementally.
  - Returns the list of visited grid cells. The caller accumulates votes.

- **10.2.3d** Benchmark JIT vs. pure-numpy:
  - Measure latency for 10,000 / 100,000 / 1,000,000 voxels × 4 cameras.
  - Measure JIT compilation time (first call overhead). Ensure it's amortized over the simulation run.
  - Document the break-even point (number of frames where JIT overhead is recovered).

- **10.2.3e** Implement a fallback path:
  - If `numba` is not installed (optional dependency), fall back to the pure-numpy implementation with a performance warning.
  - Use a dynamic dispatch: `if numba is available: use JIT version; else: use numpy version`.
  - Ensure both paths produce identical results (add a cross-validation test).

**Deliverables:**
- `_fuse_frame_inner()` and `_ray_march_3d_dda()` numba kernels.
- Fallback dispatch.
- Benchmark report: speedup factor, JIT compilation time, break-even frame count.
- Cross-validation test: JIT output == numpy output for the same inputs.

**Dependencies:** Task 10.2.1 (profiling to confirm candidates), Task 5.2.1 (numpy fuse_frame), `numba` (optional).

---

### 10.3 Extensibility

#### Task 10.3.1 — Plugin Architecture for Swappable Components

**Objective:** Ensure that rendering methods, fusion algorithms, clustering strategies, and motion models can be swapped without modifying orchestration code.

**Sub-tasks:**

- **10.3.1a** Formalize the Protocol-based plugin interfaces (partially defined in Task 9.1.1b). Complete set:

  | Protocol | Methods | Implementations |
  |---|---|---|
  | `MotionModel` | `reset(seed) -> EagleState`, `step(state, dt) -> EagleState`, `interpolate(t) -> EagleState` | RandomWalk, Spline, Lissajous |
  | `ThermalRenderer` | `render_frame(eagle_state, camera, config) -> np.ndarray` | ProjectionRenderer (Phase 1), RayCastRenderer (Phase 4), PyRenderRenderer (Phase 4) |
  | `FusionMethod` | `fuse(grid, cameras, images, frame_idx) -> None` | BayesianFusion, SpaceCarving |
  | `ClusteringMethod` | `cluster(grid) -> List[VoxelCluster]` | ConnectedComponents, DBSCAN |
  | `AssociationMethod` | `associate(tracks, detections) -> AssignmentResult` | Hungarian, GreedyNearest |
  | `StateEstimator` | `predict() -> np.ndarray`, `update(measurement) -> np.ndarray` | KalmanFilter3D, ExtendedKalmanFilter (future), ParticleFilter (future) |
  | `FrameSource` | `__iter__() -> Iterator[FrameBundle]` | SimulationEngine, SimulationReader, CameraFeedSource |

- **10.3.1b** Implement a registry pattern for selecting implementations by name:
  ```python
  FUSION_REGISTRY: Dict[str, Type[FusionMethod]] = {
      "bayesian": BayesianFusion,
      "space_carving": SpaceCarving,
  }

  def get_fusion_method(name: str, config: FusionConfig) -> FusionMethod:
      cls = FUSION_REGISTRY[name]
      return cls(config)
  ```
  - The `name` is specified in `SimulationConfig` (e.g., `fusion_method: "bayesian"`).
  - Adding a new implementation requires only: (a) implementing the Protocol, (b) adding an entry to the registry. No changes to `TrackingPipeline` or `SimulationEngine`.

- **10.3.1c** Implement `TrackingPipeline` constructor to accept Protocols rather than concrete classes:
  ```python
  class TrackingPipeline:
      def __init__(self,
                   fusion: FusionMethod,
                   clustering: ClusteringMethod,
                   association: AssociationMethod,
                   estimator_factory: Callable[[], StateEstimator],
                   ...): ...
  ```
  - A factory function (`build_tracking_pipeline(config)`) reads the config and wires the appropriate implementations.

- **10.3.1d** Implement `SimulationEngine` to accept `ThermalRenderer` protocol:
  - Currently, rendering is hardcoded to the projection + Gaussian blob method.
  - Refactor: `SimulationEngine.__init__` accepts a `renderer: ThermalRenderer` (defaulting to `ProjectionRenderer`).
  - For Phase 4, swap in `RayCastRenderer` or `PyRenderRenderer` by changing the config string.

- **10.3.1e** Document the extension guide:
  - `docs/extending.md`: step-by-step instructions for adding a new motion model, renderer, fusion method, etc.
  - Include a worked example: "Adding a ConstantVelocity motion model in 3 steps."

**Deliverables:**
- Complete Protocol definitions for all swappable components.
- Registry pattern with config-driven instantiation.
- Refactored `TrackingPipeline` and `SimulationEngine` accepting protocol instances.
- `docs/extending.md`.

**Dependencies:** Task 9.1.1 (interface contracts), all Phase 1–3 implementations.

---

#### Task 10.3.2 — Class Interface Stability and Versioning

**Objective:** Ensure that the public API of core classes (`Camera`, `Eagle`, `VoxelGrid`, `TrackingPipeline`) is stable and changes are managed explicitly.

**Sub-tasks:**

- **10.3.2a** Define what constitutes the public API:
  - All methods and properties that are consumed across module boundaries (as documented in the Protocol definitions from 10.3.1a).
  - Methods prefixed with `_` are internal and may change without notice.

- **10.3.2b** Implement `__all__` in each module to explicitly export the public API.

- **10.3.2c** Add deprecation utilities:
  ```python
  import warnings

  def deprecated(message: str):
      def decorator(func):
          @functools.wraps(func)
          def wrapper(*args, **kwargs):
              warnings.warn(f"{func.__name__} is deprecated: {message}", DeprecationWarning, stacklevel=2)
              return func(*args, **kwargs)
          return wrapper
      return decorator
  ```
  - Use when renaming or removing a public method: keep the old name as a deprecated wrapper for one release cycle.

- **10.3.2d** Maintain a `CHANGELOG.md` with semantic versioning:
  - MAJOR: breaking change to a public Protocol or data type.
  - MINOR: new feature (new motion model, new metric, etc.) without breaking existing interfaces.
  - PATCH: bug fix, performance improvement.

**Deliverables:**
- `__all__` definitions in all modules.
- Deprecation decorator utility.
- `CHANGELOG.md` template.

**Dependencies:** Task 10.3.1 (Protocol definitions finalized).

---

## SECTION 11 — ADDITIONAL CONSIDERATIONS

---

### 11.1 Coordinate System and Units Convention

**Prior coverage:** Task 2.1.1 (WorldConfig, axis orientation, units). Task 8.1.2b (scipy quaternion conventions). Task 3.2.1 (camera frame conventions).

#### Task 11.1.1 — Convention Document (Residual Work)

**Objective:** Produce a single, authoritative reference document that consolidates all convention decisions made across prior tasks.

**Sub-tasks:**

- **11.1.1a** Write `docs/conventions.md` covering:
  - **Coordinate frame:** Right-handed, Z-up. Origin at park center (or specified GPS reference).
  - **Camera frame:** OpenCV convention — X-right, Y-down, Z-forward (optical axis). The rotation matrix `R` transforms from world to camera frame: `p_cam = R @ p_world + t`.
  - **Quaternion ordering:** Internal storage is scalar-first `(w, x, y, z)` for aerospace consistency. Conversion to scipy's scalar-last `(x, y, z, w)` is handled by `CameraExtrinsics` internally and never exposed at the interface level.
  - **Euler angle convention:** ZYX extrinsic (equivalently XYZ intrinsic). `(roll, pitch, yaw)` = rotation about `(X, Y, Z)` axes respectively. Configuration files use degrees; internal computation uses radians.
  - **Units:** Distances in meters. Temperatures in Celsius at the config/display layer, Kelvin internally for radiometric calculations (`T_K = T_C + 273.15`). Angular quantities in radians internally, degrees in config files. Time in seconds. Pixel coordinates in `(u, v)` = `(column, row)` following OpenCV convention (origin at top-left).
  - **GPS/geodetic mapping (future):** Define the mapping from the simulation's Cartesian origin to a WGS84 geodetic coordinate. For flat-earth approximation valid at the 5 km scale: `x = (lon - lon0) * R_earth * cos(lat0)`, `y = (lat - lat0) * R_earth`, `z = altitude_above_ground`. Store `lat0, lon0, alt0` in `WorldConfig` for future use.

- **11.1.1b** Add assertion-level convention enforcement:
  - In `CameraExtrinsics.__init__()`, assert that `det(R) ≈ 1.0` (proper rotation, not reflection).
  - In `Camera.project()`, assert output is `(u, v)` not `(v, u)` by checking that `u < width` and `v < height` (not transposed).
  - In temperature conversion utilities, assert `T_K > 0` (no negative Kelvin).
  - These assertions run in debug mode and are stripped in optimized builds (`python -O`).

**Deliverables:**
- `docs/conventions.md` — comprehensive, authoritative.
- Assertion-level enforcement in core classes.

**Dependencies:** Tasks 2.1.1, 3.1.1, 3.2.1 (convention decisions already made).

---

### 11.2 Configuration Management

**Prior coverage:** Task 2.1.2 (SimulationConfig, YAML loading, pydantic schema).

#### Task 11.2.1 — Configuration Versioning and Migration (Residual Work)

**Objective:** Handle the inevitable evolution of the configuration schema across development phases without breaking old config files.

**Sub-tasks:**

- **11.2.1a** Add a `schema_version: int` field to the top level of `SimulationConfig`. Starting value: `1`.

- **11.2.1b** Implement `migrate_config(raw_yaml: dict) -> dict`:
  - Reads `schema_version` from the raw YAML dict.
  - Applies migration functions sequentially: `v1_to_v2`, `v2_to_v3`, etc.
  - Each migration function adds new fields with defaults, renames changed fields, removes deprecated fields.
  - After migration, the dict can be loaded into the current `SimulationConfig` pydantic model.

- **11.2.1c** Implement config diffing:
  - `diff_configs(a: SimulationConfig, b: SimulationConfig) -> List[str]`:
    - Returns a list of human-readable differences (e.g., `"noise_config.gaussian_std: 0.05 → 0.10"`).
    - Useful for experiment comparison: "what changed between run A and run B?"

- **11.2.1d** Implement config hashing:
  - `hash_config(config: SimulationConfig) -> str`:
    - Deterministic SHA-256 of the canonical JSON serialization of the config.
    - Used as a cache key for simulation results — if the config hash matches a stored run, skip re-computation.

**Deliverables:**
- `schema_version` field and migration framework.
- `diff_configs()` and `hash_config()` utilities.
- Unit test: load a v1 config, migrate to current version, verify all new fields have correct defaults.

**Dependencies:** Task 2.1.2 (SimulationConfig).

---

### 11.3 Sensor Resolution and Frame Rate

**Prior coverage:** Task 3.1.1 (CameraIntrinsics, factory methods for FLIR and low-cost presets). Task 2.2.2 (motion model time step).

#### Task 11.3.1 — Frame Rate and Motion Aliasing Analysis (Residual Work)

**Objective:** Formalize the relationship between frame rate, eagle speed, and voxel size to ensure the simulation captures motion without aliasing and the tracker doesn't miss fast maneuvers.

**Sub-tasks:**

- **11.3.1a** Implement `compute_minimum_frame_rate(max_speed: float, voxel_size: float) -> float`:
  - Nyquist-inspired criterion: the eagle must not traverse more than one voxel per frame, otherwise the prediction-guided search may miss it.
  - `fps_min = max_speed / voxel_size`.
  - At `max_speed = 20 m/s` and `voxel_size = 1.0 m`: `fps_min = 20 Hz`.
  - Return this as a recommendation; warn if `SimulationRunConfig.dt` implies a lower frame rate.

- **11.3.1b** Implement `compute_motion_blur_extent(max_speed: float, exposure_time: float, focal_length_px: float, distance: float) -> float`:
  - The motion blur in pixels for a given exposure time: `blur_px = max_speed * exposure_time * focal_length_px / distance`.
  - At typical values (20 m/s, 10 ms exposure, 1000 px focal length, 500 m distance): `blur_px ≈ 0.4 px` — negligible. But at closer range (50 m): `blur_px ≈ 4 px` — significant, smears the blob.
  - For Phase 4 realism: convolve the rendered blob with a directional motion-blur kernel aligned with the eagle's projected velocity vector.

- **11.3.1c** Implement `validate_sampling_configuration(config: SimulationConfig) -> List[str]`:
  - Checks:
    - Frame rate ≥ `compute_minimum_frame_rate()`.
    - Exposure time (if modeled) doesn't produce > 2 px motion blur for the fastest/closest scenario.
    - Sensor resolution is sufficient to detect the eagle: the projected eagle radius at maximum range must be ≥ 1 pixel. Compute `min_projected_radius = focal_length_px * eagle_radius / max_distance`. If < 1.0, warn that the eagle is sub-pixel at long range and detection may be unreliable.
  - Returns a list of warning strings (empty if all checks pass).
  - Call this during `SimulationEngine.__init__()` and log any warnings.

**Deliverables:**
- Sampling analysis utility functions.
- `validate_sampling_configuration()` integrated into engine startup.
- Documentation: table of recommended frame rates vs. voxel sizes vs. eagle speeds.

**Dependencies:** Task 2.1.2 (SimulationConfig), Task 3.1.1 (CameraIntrinsics).

---

### 11.4 Field of View and Coverage Analysis

**Prior coverage:** Task 3.3.2c-d (coverage map computation, frustum visualization).

#### Task 11.4.1 — Coverage Quality Metrics (Residual Work)

**Objective:** Go beyond "how many cameras see each point" to quantify the geometric quality of multi-view coverage — which determines triangulation accuracy.

**Sub-tasks:**

- **11.4.1a** Implement `compute_triangulation_quality_map(cameras, grid_resolution, z_planes) -> np.ndarray`:
  - For each grid point at each altitude, compute the **condition number** or **Dilution of Precision (DOP)** of the multi-view triangulation geometry.
  - DOP is a standard metric from GPS and surveying. For N cameras observing a point, the observation matrix is `A` where each row is the unit direction vector from the point to a camera. `DOP = sqrt(trace((AᵀA)⁻¹))`. Lower DOP = better geometry.
  - Decompose into HDOP (horizontal) and VDOP (vertical) to separately assess lateral vs. altitude precision.
  - Return a 3D array of DOP values (same grid as coverage map).

- **11.4.1b** Implement `compute_baseline_angle_map(cameras, grid_resolution, z_planes) -> np.ndarray`:
  - For each grid point, compute the maximum pairwise angle between camera view directions. Wider angles give better depth resolution.
  - Threshold: baseline angle < 10° yields poor depth estimation (near-degenerate geometry). Flag these regions.

- **11.4.1c** Implement combined coverage quality score:
  - `quality(point) = coverage_count * (1 / DOP) * SNR_factor`.
  - Where `SNR_factor = min over cameras of SNR(point, camera)` (from Task 2.3.5).
  - This combines geometric quality, redundancy, and signal strength into a single score for camera placement optimization (Phase 5).

- **11.4.1d** Implement `CoveragePlanner`:
  - Input: target region (ROI), minimum coverage quality, available mounting positions (e.g., a list of candidate pole locations).
  - Output: recommended camera selection and orientation to maximize minimum coverage quality over the ROI.
  - Algorithm: greedy or genetic — start with all cameras, iteratively remove the one whose removal causes the least quality degradation, until the minimum quality falls below the threshold. The remaining set is the minimal sufficient configuration.
  - This is an optimization tool for Phase 5, but defining the interface now informs the coverage metrics needed.

**Deliverables:**
- DOP map computation.
- Baseline angle map computation.
- Combined quality score.
- `CoveragePlanner` (basic greedy implementation).
- Visualization: DOP heatmap overlaid with camera positions.

**Dependencies:** Task 3.3.2 (Camera placement, coverage map), Task 2.3.5 (SNR utility).

---

### 11.5 Calibration Simulation

**Prior coverage:** Task 3.3.3 (calibration verification pipeline, reprojection error check).

#### Task 11.5.1 — Full Calibration Recovery Simulation (Residual Work)

**Objective:** Beyond verifying that the known projection pipeline is internally consistent, simulate the full calibration process that would be performed in the field, and measure how calibration errors propagate into tracking errors.

**Sub-tasks:**

- **11.5.1a** Implement `generate_calibration_targets(world_config, num_targets, pattern) -> np.ndarray`:
  - `pattern: str` — `"grid"` (regular 3D grid), `"random"` (uniformly distributed in the ROI), `"ground_plane"` (targets at known positions on z=0, simulating thermal emitters placed on the ground).
  - Returns `(M, 3)` array of 3D target positions in world coordinates.
  - Each target has a known temperature (e.g., 60 °C — hot enough to be clearly distinguishable from background).

- **11.5.1b** Implement `simulate_calibration_observations(targets, cameras, noise_config) -> Dict[str, np.ndarray]`:
  - For each camera, project all targets to pixel coordinates.
  - Add 2D detection noise: Gaussian with configurable std (e.g., 0.5 px for sub-pixel feature detection, 2 px for centroid-based detection on blobs).
  - Return per-camera `(M_visible, 2)` arrays of noisy 2D observations, plus the corresponding `(M_visible, 3)` world points.

- **11.5.1c** Implement `recover_calibration(observations, initial_guess) -> List[Camera]`:
  - Use `cv2.calibrateCamera` (if calibrating intrinsics) or `cv2.solvePnP` (if intrinsics are known and only extrinsics are being recovered — the more common scenario for fixed installations).
  - Use the noisy observations and known 3D targets.
  - Return `Camera` objects with recovered intrinsics/extrinsics.

- **11.5.1d** Implement `calibration_error_analysis(true_cameras, recovered_cameras) -> CalibrationErrorReport`:
  - Per-camera:
    - Extrinsic position error: `||C_true - C_recovered||` in meters.
    - Extrinsic orientation error: geodesic distance on SO(3) between `R_true` and `R_recovered` (in degrees): `θ = arccos((trace(R_true @ R_recovered.T) - 1) / 2)`.
    - Intrinsic errors: `|f_true - f_recovered|`, `|c_true - c_recovered|`, distortion coefficient differences.
    - Mean reprojection error over all targets.

- **11.5.1e** Implement `calibration_sensitivity_on_tracking(true_cameras, recovered_cameras, simulation_config) -> dict`:
  - Run the full tracking pipeline twice: once with ground-truth cameras, once with recovered (imperfect) cameras.
  - Compare tracking accuracy: position error increase, detection rate change.
  - This quantifies how much calibration error the system can tolerate — a critical input for field deployment planning.
  - Sweep detection noise levels (0.1, 0.5, 1.0, 2.0, 5.0 px) to build a curve of calibration error vs. tracking degradation.

**Deliverables:**
- Calibration target generation, observation simulation, recovery, error analysis.
- `calibration_sensitivity_on_tracking()` with swept detection noise.
- Report: "X px of 2D detection noise causes Y meters of calibration position error, which degrades tracking RMSE by Z meters."

**Dependencies:** Task 3.3.3 (basic calibration verification), Task 5.5.1 (TrackingPipeline for end-to-end impact assessment).

---

### 11.6 Temporal Decay and Voxel Pruning

**Prior coverage:** Task 5.1.2 (decay and pruning implementation).

#### Task 11.6.1 — Decay Parameter Tuning (Residual Work)

**Objective:** Determine optimal decay and pruning parameters and characterize their effect on tracking performance.

**Sub-tasks:**

- **11.6.1a** Implement `sweep_decay_parameters(config, decay_rates, pruning_thresholds) -> pd.DataFrame`:
  - For each combination of `decay_rate` and `pruning_threshold`, run the `baseline_noisy` scenario and measure:
    - Tracking accuracy (RMSE).
    - Detection rate.
    - False positive rate (stale voxels producing ghost detections).
    - Peak memory usage (max `num_active_voxels` over the run).
    - Mean per-frame fusion time.
  - Return results as a DataFrame.

- **11.6.1b** Analyze trade-offs:
  - Too fast decay: voxels disappear before the tracker can accumulate enough evidence → lower detection rate, track fragmentation.
  - Too slow decay: stale voxels persist → ghost detections, higher FP rate, higher memory usage.
  - Identify the Pareto-optimal decay_rate range.
  - Document recommended defaults with justification.

- **11.6.1c** Implement adaptive decay:
  - When the tracker is confident (low Kalman covariance, stable track), decay can be aggressive (the prediction-guided search will re-find the eagle quickly).
  - When the tracker is uncertain (high covariance, track coasting), decay should be slower to preserve evidence.
  - `effective_decay_rate = base_decay_rate * confidence_factor` where `confidence_factor = sigmoid(track_age - min_hits_to_confirm)` or similar.
  - This is an advanced feature; implement the infrastructure but keep the default as fixed-rate.

**Deliverables:**
- Decay parameter sweep utility.
- Analysis report with recommended defaults.
- Adaptive decay mechanism (optional, behind a config flag).

**Dependencies:** Task 5.1.2, Task X.7 (parameter sweep infrastructure).

---

### 11.7 Multi-Target Data Association

**Prior coverage:** Task 5.4.1 (Hungarian assignment with dummy rows/columns for birth/death), Task 4.1.2 (multi-eagle rendering).

#### Task 11.7.1 — Association Robustness Hardening (Residual Work)

**Objective:** Ensure the data association layer handles edge cases that arise in multi-target and noisy scenarios.

**Sub-tasks:**

- **11.7.1a** Implement cost matrix diagnostics:
  - Before solving the Hungarian, log:
    - Matrix dimensions (`num_tracks × num_detections`).
    - Min/max/mean costs.
    - Number of entries above `max_association_distance` (infeasible assignments).
    - Whether the matrix is square after augmentation.
  - If the cost matrix is degenerate (all entries above threshold → no feasible assignments), handle gracefully: all tracks coast, all detections start new tentative tracks.

- **11.7.1b** Implement gating strategies (beyond simple Euclidean distance):
  - **Appearance gating:** In thermal tracking, the "appearance" feature is the cluster's total thermal energy (sum of occupancy probabilities × voxel temperatures) or its spatial extent (bounding box volume). Two detections with wildly different thermal signatures are unlikely to be the same eagle. Add an appearance cost term to the cost matrix: `C[i,j] = α * spatial_distance + (1 - α) * appearance_distance`.
  - **Velocity gating:** A track moving at 15 m/s to the north is unlikely to associate with a detection 30 m to the south (even if within Euclidean range). Use the Mahalanobis distance (Task 5.4.2d) which naturally encodes this through the Kalman covariance.

- **11.7.1c** Implement track merge and split handling:
  - **Merge:** Two tracks whose centroids converge within `voxel_size` for `N` consecutive frames are merged into one. This handles the case where a single eagle was briefly fragmented into two clusters.
  - **Split:** A single cluster that suddenly becomes two (e.g., two eagles separating after a close encounter) should fork the parent track into two children, each inheriting the parent's Kalman state as initialization.
  - These are heuristics and should be configurable. Default: disabled in Phase 1–3, enabled in Phase 4+.

- **11.7.1d** Stress test with adversarial scenarios:
  - **Crossing trajectories:** 2 eagles cross paths, come within 2 m of each other, then diverge. Verify track identity is maintained (or MOTA correctly penalizes the switch).
  - **Parallel trajectories:** 2 eagles fly 5 m apart on parallel courses. Verify they are tracked as distinct entities.
  - **Appear/disappear:** Eagle enters ROI at frame 20, exits at frame 80. Verify track birth and death are handled correctly, no false tracks persist after exit.
  - **Noise storm:** High noise produces 5+ spurious clusters per frame for 10 frames, then noise subsides. Verify false tracks are pruned and the real track survives.

**Deliverables:**
- Cost matrix diagnostics logging.
- Appearance and velocity gating.
- Track merge/split heuristics (configurable).
- Adversarial scenario test suite.

**Dependencies:** Task 5.4.1 (Tracker), Task 5.4.2 (Kalman for Mahalanobis gating), Task 4.1.2 (multi-eagle simulation).

---

### 11.8 Logging and Reproducibility

**Prior coverage:** Task 9.2.1 (structured logging), Task 9.2.2 (reproducibility guarantees), Task X.1 (RNG seeding hierarchy).

#### Task 11.8.1 — Experiment Tracking Integration (Residual Work)

**Objective:** Go beyond raw logging to structured experiment tracking that enables comparison across hundreds of runs.

**Sub-tasks:**

- **11.8.1a** Define the experiment record format:
  ```yaml
  experiment_id: "exp_20260327_143022_a1b2c3"
  config_hash: "sha256:..."
  config_path: "configs/scenarios/baseline_noisy.yaml"
  git_commit: "abc1234"
  start_time: "2026-03-27T14:30:22Z"
  end_time: "2026-03-27T14:31:05Z"
  wall_clock_seconds: 43.2
  metrics:
    detection_rate: 0.93
    rmse_m: 1.42
    mota: 0.88
    fps: 47.3
  artifacts:
    trajectory: "results/exp_.../trajectory.npz"
    report: "results/exp_.../report.html"
  ```

- **11.8.1b** Implement `ExperimentTracker`:
  ```python
  class ExperimentTracker:
      def __init__(self, results_dir: str): ...
      def start_experiment(self, config: SimulationConfig) -> str: ...  # returns experiment_id
      def log_metrics(self, experiment_id: str, metrics: dict) -> None: ...
      def finish_experiment(self, experiment_id: str) -> None: ...
      def list_experiments(self, filter: dict = None) -> pd.DataFrame: ...
      def compare_experiments(self, ids: List[str]) -> pd.DataFrame: ...
  ```
  - Stores experiment records as YAML/JSON files in `results_dir`.
  - `list_experiments()` scans the directory and returns a summary DataFrame.
  - `compare_experiments()` loads selected records and produces a side-by-side comparison.

- **11.8.1c** (Optional) Integrate with MLflow or Weights & Biases:
  - Implement a `MLflowExperimentTracker` subclass that logs metrics and artifacts to an MLflow server.
  - This is valuable if the team grows or if the project transitions to a lab/production environment with shared experiment infrastructure.
  - Keep the file-based tracker as the default (zero infrastructure dependencies).

- **11.8.1d** Implement `generate_experiment_index(results_dir) -> str`:
  - Scans all experiment records and produces an HTML index page with sortable columns (experiment ID, config hash, key metrics, timestamp).
  - Enables quick browsing of all runs without specialized tools.

**Deliverables:**
- Experiment record format (YAML schema).
- `ExperimentTracker` with file-based backend.
- Optional MLflow adapter.
- HTML experiment index generator.

**Dependencies:** Task 9.2.1 (logging), Task 6.2.1 (BenchmarkResult as the metric source).

---

### 11.9 Testing Strategy

**Prior coverage:** Unit tests defined per-task throughout all prior documents. Task X.2 (integration test suite, Sections 2–3), Task X.6 (end-to-end regression test, Sections 4–5), Task X.9 (CI/CD pipeline, Sections 6–9).

#### Task 11.9.1 — Test Strategy Consolidation (Residual Work)

**Objective:** Define the complete testing taxonomy, coverage targets, and test infrastructure beyond the per-task tests already specified.

**Sub-tasks:**

- **11.9.1a** Classify all tests into tiers:

  | Tier | Name | Scope | Speed Target | When Run |
  |---|---|---|---|---|
  | T0 | Smoke | Environment, imports, config loading | < 5 s | Every commit |
  | T1 | Unit | Single function/class, mocked dependencies | < 60 s total | Every commit |
  | T2 | Integration | Multi-component (e.g., render + project + verify) | < 300 s total | Every PR |
  | T3 | System | Full pipeline, 100-frame scenarios | < 600 s total | Every PR |
  | T4 | Benchmark | Canonical scenarios with metric gates | < 1800 s total | Nightly / PR to main |
  | T5 | Stress | High noise, multi-eagle, long runs, parameter sweeps | > 3600 s | Weekly / manual |

- **11.9.1b** Define coverage targets:
  - Line coverage: ≥ 85% for core modules (camera, eagle, voxel_grid, renderer, fusion, clustering, tracker).
  - Branch coverage: ≥ 70% for core modules.
  - Protocol coverage: every Protocol method has at least one test exercising it through each concrete implementation.
  - Edge case coverage: explicitly test boundary conditions documented in prior task documents (eagle behind camera, eagle at image edge, zero-voxel clusters, empty frame bundles, NaN inputs).

- **11.9.1c** Implement test fixtures:
  - `conftest.py` with reusable fixtures:
    - `default_config`: loads `default_config.yaml`.
    - `simple_camera`: single camera at origin, looking along +Z, 640×512, 24° HFOV, no distortion.
    - `camera_ring_4`: 4 cameras in a ring, 500 m radius, 10 m height.
    - `eagle_at_origin`: eagle at `(0, 0, 100)`, stationary, 35 °C.
    - `lissajous_trajectory_100`: 100-frame Lissajous trajectory with default parameters.
    - `rendered_frame_bundle`: pre-rendered single frame from `simple_camera` + `eagle_at_origin`.
  - These fixtures avoid duplicated setup code across test files.

- **11.9.1d** Implement property-based tests (using `hypothesis`):
  - `Camera.project(camera.unproject(pixel, depth))` ≈ pixel for all valid pixels and positive depths.
  - `world_to_grid(grid_to_world(ix, iy, iz))` = `(ix, iy, iz)` for all valid grid indices.
  - `EagleState.position` always within world bounds after any number of `MotionGenerator.step()` calls.
  - `VoxelData.probability` always in `[0, 1]` after any sequence of Bayesian updates.
  - Add `hypothesis` to dev dependencies.

- **11.9.1e** Implement mutation testing infrastructure (optional, Phase 5):
  - Use `mutmut` or `cosmic-ray` to verify that tests actually catch regressions.
  - Target: mutation score ≥ 60% for core modules (i.e., ≥ 60% of artificially introduced bugs are caught by the test suite).

**Deliverables:**
- Test tier classification applied to all existing tests (via pytest markers: `@pytest.mark.tier0`, etc.).
- `conftest.py` with shared fixtures.
- Property-based tests for invariants.
- CI configured to run tiers T0–T3 per commit, T4 nightly.
- Coverage report generation and enforcement.

**Dependencies:** All prior task documents (tests are retroactively classified).

---

### 11.10 Real-World Transition Path

**Prior coverage:** Task 9.1.2d (CameraFeedSource abstract class), Task 10.3.1 (plugin architecture for swappable renderer / frame source).

#### Task 11.10.1 — Simulation-to-Production Boundary Documentation

**Objective:** Explicitly document which code is simulation-only vs. shared with production, and define the integration surface for real hardware.

**Sub-tasks:**

- **11.10.1a** Produce `docs/production_transition.md` with a component classification table:

  | Component | Simulation | Production | Shared? |
  |---|---|---|---|
  | `Camera` (intrinsics, extrinsics, projection) | ✓ | ✓ | **Yes** — identical code |
  | `SparseVoxelGrid` | ✓ | ✓ | **Yes** — identical code |
  | `BayesianFusion` / `SpaceCarving` | ✓ | ✓ | **Yes** — identical code |
  | `ConnectedComponents` / `DBSCAN` | ✓ | ✓ | **Yes** — identical code |
  | `Tracker` (Hungarian + Kalman) | ✓ | ✓ | **Yes** — identical code |
  | `Evaluation` metrics | ✓ | Partial | **Partial** — GT not available in production; only timing and track statistics are usable |
  | `MotionGenerator` | ✓ | ✗ | **No** — simulation only |
  | `ProjectionRenderer` | ✓ | ✗ | **No** — replaced by real camera feed |
  | `SimulationEngine` | ✓ | ✗ | **No** — replaced by real-time loop |
  | `FrameSynchronizer` | ✓ (JITTERED mode) | ✓ | **Yes** — critical for real-time |
  | `SimulationClock` | ✓ | Replaced | **Interface shared** — real clock implementation differs |

- **11.10.1b** Define the `CameraFeedSource` concrete implementation contract:
  - Input: hardware-specific frame-grabber SDK (e.g., FLIR Spinnaker, Basler pylon) or generic GStreamer pipeline.
  - Output: `FrameBundle` with hardware timestamps.
  - Requirements:
    - Must call `FrameSynchronizer.submit_frame()` for each camera frame.
    - Must handle dropped frames (camera timeout, network loss) by submitting a `None` image and flagging the camera as unavailable for that bundle.
    - Must convert raw sensor output (digital counts) to temperature using the camera's calibrated radiance-to-temperature mapping (LUT or polynomial fit provided by the sensor manufacturer).
  - Provide a reference implementation skeleton (`.py` file with method stubs and docstrings) so the hardware integration team has a clear target.

- **11.10.1c** Define the real-time execution loop contract:
  ```python
  class RealTimeLoop:
      def __init__(self, feed_source: CameraFeedSource, pipeline: TrackingPipeline,
                   clock: RealTimeClock, output_sink: TrackOutputSink): ...
      def run(self) -> None:
          while not self.stopped:
              bundle = self.feed_source.get_bundle(timeout=self.clock.dt * 2)
              if bundle is None:
                  log.warning("Frame bundle timeout")
                  continue
              result = self.pipeline.process_frame(bundle)
              self.output_sink.emit(result)
  ```
  - `TrackOutputSink` is another Protocol: could write to a database, a live dashboard, an MQTT broker, or a local log file.
  - `RealTimeClock` wraps `time.monotonic_ns()` and tracks drift vs. expected frame rate.

- **11.10.1d** Document deployment-specific configuration deltas:
  - Camera intrinsics: loaded from factory calibration files or field calibration (Task 11.5.1).
  - Camera extrinsics: determined by surveying camera positions (GPS + tilt sensor) or by the calibration recovery process.
  - Voxel size, fusion parameters, tracking parameters: carried over from simulation Phase 5 optimization.
  - Atmospheric attenuation: may need field measurement or lookup from weather data.
  - Noise model: not needed (real noise replaces simulated noise), but the detection threshold may need adjustment based on actual NETD.

**Deliverables:**
- `docs/production_transition.md` with component classification and deployment deltas.
- `CameraFeedSource` skeleton implementation.
- `RealTimeLoop` skeleton.
- `TrackOutputSink` Protocol definition.

**Dependencies:** Task 9.1.2d, Task 10.3.1 (plugin architecture), Task 10.1.1 (SimulationClock / FrameSynchronizer).

---

#### Task 11.10.2 — Clock Synchronization Planning

**Objective:** Define requirements and test infrastructure for sub-millisecond camera synchronization in deployment.

**Sub-tasks:**

- **11.10.2a** Document synchronization requirements:
  - At eagle speed 20 m/s, a 1 ms sync error causes 0.02 m positional ambiguity — well within voxel resolution at 1 m. At 10 ms sync error: 0.2 m — still manageable.
  - Target: < 1 ms inter-camera sync error.
  - Methods: hardware trigger (best, < 10 µs), PTP/IEEE 1588 (< 1 ms over Ethernet), NTP (< 10 ms, may be insufficient).

- **11.10.2b** Implement `JitteredSync` simulation mode (referenced in Task 10.1.1b) to stress-test the tracker under various sync error levels:
  - Sweep jitter: 0, 0.1, 1, 5, 10, 50 ms.
  - Measure tracking degradation vs. jitter.
  - Determine the maximum tolerable jitter for the system's accuracy requirements.

- **11.10.2c** Produce a deployment spec document:
  - Recommended synchronization method for the target camera count and cable run lengths.
  - Fallback options if PTP is not available.
  - Calibration procedure for measuring actual sync error in the field (e.g., simultaneously imaging a pulsed thermal source from all cameras and comparing timestamps of the observed pulse).

**Deliverables:**
- Sync requirement analysis document.
- Jitter sweep results and maximum tolerable jitter determination.
- Deployment sync spec.

**Dependencies:** Task 10.1.1 (SyncPolicy, FrameSynchronizer), Task X.7 (parameter sweep infrastructure).

---

## CROSS-CUTTING CONCERNS (Sections 10–11)

### Task X.10 — Documentation Consolidation

All prior task documents produced scattered documentation artifacts (`docs/conventions.md`, `docs/extending.md`, `docs/reproducibility.md`, `docs/production_transition.md`, `docs/architecture.md`). This task consolidates them.

**Sub-tasks:**

- **X.10a** Implement a `docs/` index with navigation:
  ```
  docs/
    index.md                    # Overview and navigation
    architecture.md             # Data flow diagrams (Task 9.1.3)
    conventions.md              # Coordinate systems, units (Task 11.1.1)
    configuration.md            # Config schema reference (Task 2.1.2, 11.2.1)
    extending.md                # Plugin guide (Task 10.3.1)
    reproducibility.md          # RNG, determinism (Task 9.2.2)
    production_transition.md    # Sim-to-real guide (Task 11.10.1)
    testing.md                  # Test strategy (Task 11.9.1)
    api_reference/              # Auto-generated from docstrings
      camera.md
      eagle.md
      voxel_grid.md
      ...
  ```

- **X.10b** Set up auto-generated API reference:
  - Use `sphinx` with `autodoc` and `napoleon` (for Google-style docstrings) or `mkdocs` with `mkdocstrings`.
  - CI step: build docs and publish to GitHub Pages (or equivalent).
  - Enforce docstring coverage: all public methods must have docstrings (enforced via `pydocstyle` or `ruff` lint rule).

- **X.10c** Write the project README:
  - Project overview (1 paragraph).
  - Quick start: install, run default simulation, view results.
  - Architecture overview (reference `docs/architecture.md`).
  - Phase status (which phase is currently being worked on).
  - Contributing guide (reference `docs/extending.md` and `docs/testing.md`).

### Task X.11 — Technical Debt Register

- Maintain a `TECH_DEBT.md` file that records known simplifications, approximations, and deferred improvements:

  | ID | Description | Impact | Phase to Address |
  |---|---|---|---|
  | TD-001 | Blob rendering uses circular projection, not elliptical | Sub-pixel error at oblique angles | Phase 4 |
  | TD-002 | Vignetting applied in temperature space, not radiance space | < 0.5 °C error at image corners | Phase 4 |
  | TD-003 | No occlusion handling in projection renderer | Ghost detections behind obstacles | Phase 4 |
  | TD-004 | Distortion-aware blob rendering uses full-image remap | Performance cost ~2× per frame | Phase 4/5 |
  | TD-005 | Kalman filter assumes constant velocity | Tracking lag during maneuvers | Phase 5 (consider IMM or EKF) |
  | TD-006 | Space carving DDA is pure Python | Slow for high hot-pixel counts | Phase 3 (numba) |
  | TD-007 | No multi-resolution voxel grid | Uniform resolution wastes memory at long range | Phase 5 |

- Update this register as new debt is incurred or resolved. Review quarterly.

---

## TASK DEPENDENCY GRAPH (SECTIONS 10–11)

```
Sections 2–9 deliverables
  │
  ├──→ 10.1.1 SimulationClock + SyncPolicy + FrameSynchronizer
  │         │
  │         └──→ 11.10.2 Clock Sync Planning (jitter sweep)
  │
  ├──→ 10.2.1 Profiling Framework
  │         │
  │         ├──→ 10.2.2 Prediction-Guided Voxel Selection
  │         │
  │         └──→ 10.2.3 Numba JIT Acceleration
  │
  ├──→ 10.3.1 Plugin Architecture ──→ 10.3.2 Interface Stability / Versioning
  │
  ├──→ 11.1.1 Convention Document (consolidation)
  │
  ├──→ 11.2.1 Config Versioning + Migration
  │
  ├──→ 11.3.1 Frame Rate / Aliasing Analysis
  │
  ├──→ 11.4.1 Coverage Quality (DOP, baseline angle, CoveragePlanner)
  │
  ├──→ 11.5.1 Full Calibration Recovery Simulation
  │         │
  │         └──→ 11.5.1e Calibration Sensitivity on Tracking
  │
  ├──→ 11.6.1 Decay Parameter Tuning (sweep + adaptive decay)
  │
  ├──→ 11.7.1 Association Robustness Hardening
  │         │
  │         └──→ 11.7.1d Adversarial Scenario Tests
  │
  ├──→ 11.8.1 Experiment Tracking
  │
  ├──→ 11.9.1 Test Strategy Consolidation (tiers, fixtures, property tests)
  │
  ├──→ 11.10.1 Production Transition Docs + Skeletons
  │
  ├──→ X.10 Documentation Consolidation + API Reference
  │
  └──→ X.11 Technical Debt Register
```

**Phase alignment:**

| Phase | Section 10–11 tasks activated |
|---|---|
| Phase 1 | 10.1.1a-b (PERFECT_SYNC only), 11.1.1, 11.3.1, X.11 (initial register) |
| Phase 2 | 10.2.1 (profiling), 11.9.1 (test consolidation) |
| Phase 3 | 10.2.2 (prediction-guided), 10.2.3 (numba), 10.3.1 (plugin architecture), 11.6.1 (decay tuning) |
| Phase 4 | 10.1.1b-c (JITTERED_SYNC, FrameSynchronizer), 11.2.1 (config migration), 11.5.1 (calibration recovery), 11.7.1 (association hardening), 11.4.1 (coverage quality) |
| Phase 5 | 11.10.1 (production transition), 11.10.2 (clock sync), 11.8.1 (experiment tracking), X.10 (docs consolidation), 10.3.2 (interface versioning) |

**Critical path for real-world transition:** 10.3.1 (plugin architecture) → 11.10.1 (production boundary docs + skeletons) → 11.10.2 (clock sync planning) → field deployment. This chain should be prioritized by the end of Phase 4 so that Phase 5 validation results can directly inform hardware procurement and installation.
