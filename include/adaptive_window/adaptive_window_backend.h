#pragma once
#include "adaptive_window/adaptive_policy.h"
#include "adaptive_window/factor_builders.h"
#include "adaptive_window/fixed_lag_optimizer.h"
#include "adaptive_window/information_evaluator.h"
#include "adaptive_window/temporal_associator.h"
#include <map>

namespace fast_livo { namespace adaptive_window {
class AdaptiveWindowBackend {
 public:
  explicit AdaptiveWindowBackend(const AdaptiveWindowConfig &);
  void Initialize(const LidarExtrinsic &, const ExtrinsicState &, ProjectionFunction);
  void Reset();
  void AddImuSamples(const ImuSampleVector &);
  bool ProcessLidar(const NavState &, const MeasurementSummary &, const LidarPlaneObservationVector &);
  bool ProcessVision(const NavState &, const MeasurementSummary &, const VisualPatchObservationVector &,
                     const ImageSampleFunction &);
  bool LatestEstimate(NavState *) const;
  ExtrinsicState CameraExtrinsicEstimate() const { return optimizer_.EstimateCameraExtrinsic(); }
  const BackendDiagnostics &diagnostics() const { return diagnostics_; }
  const AdaptiveWindowConfig &config() const { return config_; }
 private:
  struct LidarFrame { NavState seed; LidarPlaneObservationVector observations; };
  struct Patch { std::uint64_t anchor; VisualPatchObservation reference; };
  PolicyDecision Decide(const NavState &, const MeasurementSummary &, Modality);
  std::uint64_t AddOrGetNode(const NavState &, Modality, const MeasurementSummary &, const PolicyDecision &);
  bool OptimizeAndAdapt();
  void ConsumeImuThrough(double);
  void Recover();
  AdaptiveWindowConfig config_;
  InformationEvaluator information_evaluator_;
  AdaptivePolicy policy_;
  TemporalAssociator associator_;
  FixedLagOptimizer optimizer_;
  LidarExtrinsic lidar_extrinsic_;
  ExtrinsicState camera_extrinsic_initial_;
  ProjectionFunction projection_;
  ImuSampleVector pending_imu_;
  MeasurementSummary latest_lidar_, latest_vision_;
  bool has_lidar_ = false, has_vision_ = false;
  std::uint64_t latest_node_ = 0;
  int target_window_ = 6;
  double last_lidar_time_ = -std::numeric_limits<double>::infinity();
  double last_image_time_ = -std::numeric_limits<double>::infinity();
  std::map<std::uint64_t, LidarFrame> lidar_frames_;
  std::map<std::uint64_t, Patch> patches_;
  BackendDiagnostics diagnostics_;
};
}}
