#pragma once
#include "adaptive_window/estimator.h"

namespace fast_livo { namespace adaptive_window {

// Compatibility adapter for the existing FAST-LIVO2 ROS bridge.
// All state, residual construction, optimization and marginalization live in
// Estimator. New integrations should use its VINS-style methods directly.
class AdaptiveWindowBackend : public Estimator {
 public:
  using Estimator::Estimator;
  void Initialize(const LidarExtrinsic &l, const ExtrinsicState &e, ProjectionFunction p) {
    setParameter(l, e, std::move(p));
  }
  void Reset() { clearState(); }
  void AddImuSamples(const ImuSampleVector &s) { processIMU(s); }
  bool ProcessLidar(const NavState &s, const MeasurementSummary &q, const LidarPlaneObservationVector &o) {
    return processLidar(s, q, o);
  }
  bool ProcessVision(const NavState &s, const MeasurementSummary &q, const VisualPatchObservationVector &o,
                     const ImageSampleFunction &image) { return processImage(s, q, o, image); }
  bool LatestEstimate(NavState *s) const { return latestEstimate(s); }
  ExtrinsicState CameraExtrinsicEstimate() const { return cameraExtrinsicEstimate(); }
};
}}
