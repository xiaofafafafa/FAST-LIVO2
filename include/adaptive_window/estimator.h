#pragma once
#include "adaptive_window/adaptive_policy.h"
#include "adaptive_window/factor/factor_factory.h"
#include "adaptive_window/factor/pose_local_parameterization.h"
#include "adaptive_window/information_evaluator.h"
#include "adaptive_window/temporal_associator.h"
#include <map>

namespace fast_livo { namespace adaptive_window {

// VINS-style entry point for this variable-size, asynchronous sliding window.
// Stable frame IDs replace VINS' contiguous slots so arbitrary adaptive removal
// never invalidates retained parameter addresses or residual references.
class Estimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit Estimator(const AdaptiveWindowConfig &);
  Estimator(const Estimator &) = delete;
  Estimator &operator=(const Estimator &) = delete;
  void setParameter(const LidarExtrinsic &, const ExtrinsicState &, ProjectionFunction = {});
  void clearState();
  void processIMU(const ImuSampleVector &);
  bool processLidar(const NavState &, const MeasurementSummary &, const LidarPlaneObservationVector &);
  bool processImage(const NavState &, const MeasurementSummary &, const VisualPatchObservationVector &,
                    const ImageSampleFunction &);
  bool latestEstimate(NavState *) const;
  ExtrinsicState cameraExtrinsicEstimate() const;
  const BackendDiagnostics &diagnostics() const { return diagnostics_; }
  const AdaptiveWindowConfig &config() const { return config_; }

  // The solve follows vector2double -> Ceres -> failureDetection -> double2vector.
  bool solveOdometry();
  bool optimization(double *final_cost = nullptr);
  void vector2double();
  void double2vector();
  bool failureDetection();
  bool slideWindow();
  bool slideWindowOld();
  bool slideWindowNew();  // the second-newest retained state

  // Small graph API used by deterministic tests and non-ROS integrations.
  std::uint64_t addFrame(const NavState &);
  bool hasFrame(std::uint64_t id) const { return nodes_.count(id) != 0; }
  std::uint64_t findFrameNear(double timestamp, double tolerance) const;
  NavState stateEstimate(std::uint64_t id) const;
  void markFrame(std::uint64_t id, Modality, double information);
  bool addFeature(std::uint64_t track, std::uint64_t anchor, double inverse_depth);
  void removeUnusedFeature(std::uint64_t track);
  bool hasFeature(std::uint64_t track) const { return para_Feature.count(track) != 0; }
  bool addResidualBlock(const ResidualBlockInfo &);
  bool updateFrontendPose(std::uint64_t id, const NavState &);
  void setWeights(const ModalityWeights &w) { weights_ = w; }
  bool marginalizeFrame(std::uint64_t id);
  PolicyFrameVector policyFrames() const;
  std::vector<std::uint64_t> frameIds() const;
  std::size_t numFrames() const { return nodes_.size(); }
  std::size_t numFactors() const { return residual_blocks_.size() + (last_marginalization_factor.cost ? 1 : 0); }
  std::size_t countFactors(Modality) const;

 private:
  enum SolverFlag { INITIAL, NON_LINEAR };
  enum MarginalizationFlag { MARGIN_OLD, MARGIN_SECOND_NEW, MARGIN_SELECTED };
  struct LidarFrame { NavState seed; LidarPlaneObservationVector observations; };
  struct Patch { std::uint64_t anchor; VisualPatchObservation reference; };
  PolicyDecision decide(const NavState &, const MeasurementSummary &, Modality);
  std::uint64_t addOrGetFrame(const NavState &, Modality, const MeasurementSummary &, const PolicyDecision &);
  void consumeImuThrough(double);
  void recover();

  AdaptiveWindowConfig config_;
  InformationEvaluator information_evaluator_;
  AdaptivePolicy policy_;
  TemporalAssociator associator_;
  LidarExtrinsic lidar_extrinsic_;
  ExtrinsicState camera_extrinsic_initial_, camera_extrinsic_;
  ProjectionFunction projection_;
  ImuSampleVector pending_imu_;
  MeasurementSummary latest_lidar_, latest_vision_;
  bool has_lidar_ = false, has_vision_ = false;
  std::uint64_t latest_node_ = 0, next_id_ = 1;
  int target_window_ = 6;
  double last_lidar_time_ = -std::numeric_limits<double>::infinity();
  double last_image_time_ = -std::numeric_limits<double>::infinity();
  std::map<std::uint64_t, LidarFrame> lidar_frames_;
  std::map<std::uint64_t, Patch> patches_;
  BackendDiagnostics diagnostics_;
  ModalityWeights weights_;

  // VINS naming, keyed by stable ID. frame_count is the number of active states,
  // NOT VINS' newest slot index. Gravity and policy metadata stay in nodes_.
  std::map<std::uint64_t, Eigen::Matrix3d> Rs;
  std::map<std::uint64_t, Eigen::Vector3d> Ps, Vs, Bas, Bgs;
  std::map<std::uint64_t, double> Headers;
  std::map<std::uint64_t, PolicyFrame> nodes_;
  std::map<std::uint64_t, NavState> frontend_poses_;
  int frame_count = 0;
  SolverFlag solver_flag = INITIAL;
  MarginalizationFlag marginalization_flag = MARGIN_OLD;

  ParameterValues parameter_blocks_;  // owns all Ceres memory
  std::map<std::uint64_t, double *> para_Pose, para_SpeedBias, para_Feature;
  double *para_Ex_Pose[1] = {nullptr}; // T_CI, not VINS' opposite-direction T_IC
  std::map<std::uint64_t, std::uint64_t> feature_anchors_;
  std::vector<ResidualBlockInfo> residual_blocks_;
  PoseLocalParameterization local_parameterization_;
  // Destination ID -> the still-active IMU edge ending at that state.
  std::map<std::uint64_t, std::shared_ptr<IntegrationBase>> pre_integrations;
  std::shared_ptr<IntegrationBase> tmp_pre_integration;
  std::shared_ptr<MarginalizationInfo> last_marginalization_info;
  ResidualBlockInfo last_marginalization_factor;
  std::vector<double *> last_marginalization_parameter_blocks;
};
}}
