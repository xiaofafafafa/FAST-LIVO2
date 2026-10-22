# Validation status — VINS-style Ceres backend

Date: 2026-10-22. Refactor base: `df8ee2a3dfc59a4158a3adc4b19b9a1c17ae6c9e`.

This is a source-level research implementation, not a validated SLAM release.

| Check | Status | Scope |
|---|---|---|
| `git diff --check` | PASS | Whitespace check only |
| Local includes, CMake source paths, config fields and YAML/ROS readers | PASS | 29 source/header/test files inspected; not a compiler/type check |
| Obsolete optimizer/factory/preintegrator symbols and delimiter scan | PASS | Textual source checks only |
| Independent frontend-pose rotation residual check | PASS | NumPy right-increment log Jacobian comparison; maximum error 4.14e-10 |
| Independent scalar regularization / reference replacement | PASS | Expected solutions 0.020 m and 0.044 m for the synthetic conflicting measurement |
| C++ compilation and linking | NOT RUN | CMake, Eigen, Ceres and ROS unavailable in the editing environment |
| `adaptive_window_smoke` executable | NOT RUN | Test source is supplied; no passing C++ test result is claimed |
| ROS launch / rosbag / dataset replay | NOT RUN | Requires the user's configured ROS and sensor/data environment |
| Runtime, memory, ATE/RPE, extrinsic convergence | NOT MEASURED | No performance or accuracy claims |

The independent numerical checks verify the equations only. They do not compile
or execute the C++ factor, Ceres solver, pointer-lifetime handling or ROS bridge.
The 5 cm / 1 degree regularization scales and 20 cm / 5 degree acceptance gates
are starting configurations, not data-calibrated recommendations.

The earlier Ceres implementation recorded a NumPy Schur/square-root algebra check
on 2026-10-21 (worst relative reconstruction error about 2.44e-14 on three synthetic
systems). This refactor moves that computation into
`factor/marginalization_factor.cpp`; the old numerical check is not a new C++
regression-test result.

## Added C++ checks, awaiting execution

- Frontend-pose tangent Jacobian and quaternion sign invariance.
- Exactly one frontend pose factor per retained state, including same-time
  LIO/VIO updates and replacement of the latest frontend reference.
- A closed-form scalar problem that detects both a missing regularizer and an
  accidentally duplicated first-state pose anchor.
- Repeated solves that must not move the frontend reference to the backend result.
- Disabling the regularizer while retaining the first-pose gauge anchor.
- Translation and rotation acceptance gates, rejection counts and state rollback.
- Existing midpoint IMU, four sensor residuals, middle/oldest-frame Schur
  marginalization, inverse-depth removal, and synchronous/asynchronous synthetic
  trajectories remain in the test executable.

## Required next checks

1. Build and run the standalone CTest target described in
   [the integration guide](ADAPTIVE_WINDOW_INTEGRATION.md).
2. Build the full catkin package against Ceres >= 2.1 and the original FAST-LIVO2
   dependencies. Record exact compiler, ROS, Eigen and Ceres versions.
3. Replay a short static/slow-motion recording. Check IMU rejection/reset counts,
   factor counts, timestamps, transform directions, frontend correction magnitudes
   and correction rejection counts before evaluating trajectory accuracy.
4. Keep state feedback disabled. Compare the shadow output with the frontend;
   ablate the frontend-pose regularizer using the same data and configuration.
   Enable extrinsic optimization only on sufficiently excited data.
5. Add high-rotation-rate and recorded-IMU tests. The supplied synthetic checks
   are intentionally not a benchmark.

The frontend pose factor reuses information from the same sensor data and is
explicitly treated as correlated regularization. Neither compilation nor
synthetic tests establish statistical consistency or improved SLAM accuracy.
