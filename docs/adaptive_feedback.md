# Adaptive fixed-lag feedback into FAST-LIVO2

`mapping_ouster_ntu_adaptive.launch` enables the fixed-lag backend and state
feedback. `mapping_ouster_ntu.launch` keeps the original FAST-LIVO2 path. The
adaptive launch accepts `feedback_enabled:=false` for a shadow-mode comparison.

## Measurement and keyframe flow

Every LiDAR and image measurement still receives the normal frontend filter
update. The bridge retains its IMU samples, but a cheap upper bound on the
keyframe utility skips backend residual construction and optimization whenever
the policy cannot admit the measurement. Admitted LiDAR and image measurements
share one backend state when their timestamps coincide. The maximum keyframe
interval forces periodic admission even when motion or quality is low. The
current defaults are 0.15 s minimum, 0.55 s maximum, 0.25 m translation,
10 degrees rotation, and 0.60 utility threshold. The NTU launch exposes
`max_keyframe_dt` for replay comparisons.

After a successful solve, the optimized pose, velocity and IMU biases replace
the frontend state mean before the next frontend measurement. The frontend
covariance is symmetrized and receives bounded correction uncertainty. The
backend reuses frontend observations, so its marginal covariance is not fused
as an independent measurement.

## LiDAR map lifetime

The first scan seeds a provisional map so the filter can find its initial
planes. When the first backend keyframe is admitted, that seed is replaced by
the keyframe scan. Later ordinary scans do not enter either map. Each admitted
LiDAR keyframe has one active scan cache, keyed by the backend's stable frame
ID. Duplicate insertion of that ID is rejected. Active scan points are
reprojected from their IMU-frame coordinates using the **absolute optimized
pose** of their retained keyframe. This avoids accumulating incremental map
corrections from a changing frontend reference pose.

The backend emits the final optimized state of every marginalized frame,
including an interior frame chosen by adaptive marginalization. Its scan is
reprojected at that exact final pose, inserted into the frozen voxel map once,
and removed from the active cache. A cache size threshold only warns; it never
freezes a still-optimizable scan. The LiDAR residual search queries both the
active and frozen plane trees if they occupy the same voxel, and keeps the
highest likelihood association. The existing combined view remains available
to the visual frontend and RViz map publisher. Local-map sliding, if enabled,
prunes active cache points outside its bounds as well as frozen map cells.

## Visual map lifetime

Every image may track against the existing visual map and update the frontend
EKF. New landmarks, observation patches and reference patches are added only
once for an admitted image keyframe, after state feedback. Candidate LiDAR
points for those additions come from an active LiDAR keyframe scan, rather than
an ordinary scan. LiDAR-anchored landmarks and image reference poses still use
interpolated incremental corrections while their source keyframes are active.
After a source keyframe is marginalized, that anchor is marked frozen and
cannot be moved by later feedback. Online extrinsic transfer remains optional
and is disabled in the NTU default configuration.

The current LiDAR point vector is shared between LiDAR and visual processing;
the former full copy in `LIVMapper` has been removed. The active scan cache and
active octrees intentionally both retain points: the cache permits reprojection
after optimization, while the octrees support frontend plane queries. Frozen
scans are discarded from the cache after their one-time insertion.

Ground truth is used only for offline evaluation. A future global correction
to scans already frozen would require an additional submap or pose-graph layer.
