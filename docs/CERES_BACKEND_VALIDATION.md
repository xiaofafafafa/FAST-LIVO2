# Validation status — VINS-style Ceres backend

Updated: 2026-11-13. Refactor base: `df8ee2a3dfc59a4158a3adc4b19b9a1c17ae6c9e`.

This table records the 2026-11-13 environment. A later full NTU VIRAL
`eee_01` ROS replay and accuracy comparison completed on 2026-11-16; see
[NTU adaptive evaluation](NTU_ADAPTIVE_EVALUATION.md).

This is a source-level research implementation, not a validated SLAM release.

| Check | Status | Scope |
|---|---|---|
| `git diff --check` | PASS | Whitespace check only |
| Local includes, CMake source paths, config fields and YAML/ROS readers | PASS | 29 source/header/test files inspected; not a compiler/type check |
| Obsolete optimizer/factory/preintegrator symbols and delimiter scan | PASS | Textual source checks only |
| Independent frontend-pose rotation residual check | PASS | NumPy right-increment log Jacobian comparison; maximum error 4.14e-10 |
| Independent scalar regularization / reference replacement | PASS | Expected solutions 0.020 m and 0.044 m for the synthetic conflicting measurement |
| C++ compilation and linking | PASS | ROS Noetic, GCC 9.4, Eigen 3.3.7, Ceres 1.14; isolated catkin build produced `fastlivo_mapping` |
| `adaptive_window_smoke` executable | PASS | Standalone executable and catkin CTest passed |
| Address/undefined-behavior sanitizers | PASS (without leak check) | Standalone smoke test passed with ASan and UBSan; LeakSanitizer cannot inspect processes in this sandbox |
| Livox PointCloud2 preprocessing | PASS | Actual C++ `Preprocess` processed the first 300 scans of a local bag: 4,613,740 finite, time-ordered points; the same bag's first 30 seconds had 6,000 finite IMU messages with at most 7.3 ms gaps |
| ROS launch configuration | PASS | `roslaunch --dump-params` and `roslaunch --nodes` resolved the PointCloud2 adaptive launch |
| ROS launch / rosbag / dataset replay | BLOCKED | `roscore` fails with `Unable to start XML-RPC server: Operation not permitted` in this execution environment |
| Runtime, memory, ATE/RPE, extrinsic convergence | NOT MEASURED | No performance or accuracy claims |

The passing synthetic test executes the C++ factors and Ceres solver. The ROS
bridge compiled and linked, but its live callback behavior remains unverified.
The 5 cm / 1 degree regularization scales and 20 cm / 5 degree acceptance gates
are starting configurations, not data-calibrated recommendations.

The earlier Ceres implementation recorded a NumPy Schur/square-root algebra check
on 2026-10-21 (worst relative reconstruction error about 2.44e-14 on three synthetic
systems). This refactor moves that computation into
`factor/marginalization_factor.cpp`; the old numerical check is not a new C++
regression-test result.

## C++ checks executed

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

1. Replay a short static/slow-motion recording outside the sandbox. Check IMU rejection/reset counts,
   factor counts, timestamps, transform directions, frontend correction magnitudes
   and correction rejection counts before evaluating trajectory accuracy.
2. Keep state feedback disabled. Compare the shadow output with the frontend;
   ablate the frontend-pose regularizer using the same data and configuration.
   Enable extrinsic optimization only on sufficiently excited data.
3. Add high-rotation-rate and recorded-IMU tests. The supplied synthetic checks
   are intentionally not a benchmark.

The frontend pose factor reuses information from the same sensor data and is
explicitly treated as correlated regularization. Neither compilation nor
synthetic tests establish statistical consistency or improved SLAM accuracy.
