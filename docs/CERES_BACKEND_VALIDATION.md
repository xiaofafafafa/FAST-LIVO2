# Validation status — Ceres backend

Date: 2026-10-21. Base: `90acf12dd7e94081e00a63a86393768b63444cc0`.

This is a source-level research implementation, not a validated SLAM release.

| Check | Status | Scope |
|---|---|---|
| `git diff --check` | PASS | Whitespace/conflict-marker check only |
| Local includes, CMake source paths, config fields and YAML readers | PASS | Source wiring; not a compiler/type check |
| Legacy frozen-linear and pseudo-reprojection entry points removed | PASS | Source inspection |
| Ceres 2.1 public API signatures | Inspected | Manifold ownership, parameter blocks, vector residual-block overload, dynamic autodiff |
| Independent numerical Schur/square-root reconstruction | PASS | NumPy algebra cross-check; worst relative reconstruction error about 2.44e-14 on three synthetic systems |
| C++ compilation and linking | NOT RUN | CMake, Eigen, Ceres and ROS unavailable in the editing environment |
| `adaptive_window_smoke` executable | NOT RUN | Test source is supplied; no passing C++ test result is claimed |
| ROS launch / rosbag / dataset replay | NOT RUN | Requires the user's configured ROS and sensor/data environment |
| Runtime, memory, ATE/RPE, extrinsic convergence | NOT MEASURED | No performance or accuracy claims |

The independent algebra check verifies reconstruction of the Schur Hessian and
gradient by the proposed square-root equations. It does not compile or execute
`fixed_lag_optimizer.cpp`, and cannot detect C++ integration errors.

## Required next checks

1. Build and run the standalone CTest target described in
   [the integration guide](ADAPTIVE_WINDOW_INTEGRATION.md).
2. Build the full catkin package against Ceres >= 2.1 and the original FAST-LIVO2
   dependencies. Record exact compiler, ROS, Eigen and Ceres versions.
3. Replay a short static/slow-motion recording. Check IMU rejection/reset counts,
   factor counts for all four residual families, frame timestamps and transform
   directions before evaluating trajectory accuracy.
4. Keep state feedback disabled. Compare the shadow output with the original
   frontend, then enable extrinsic optimization only on sufficiently excited data.
5. Add finite-difference/preintegration tests for high rotation rates and recorded
   IMU noise; the provided synthetic checks are intentionally not a benchmark.

Known methodological limits, failure handling, factor equations, configuration,
and diagnostics are documented in the integration guide. Do not interpret a
successful compile or synthetic test as evidence of estimator consistency,
asynchronous-frontend support, realtime performance or improved SLAM accuracy.
