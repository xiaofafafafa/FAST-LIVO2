#ifndef FAST_LIVO_ADAPTIVE_WINDOW_TYPES_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_TYPES_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace fast_livo
{
namespace adaptive_window
{

constexpr int kNavStateDim = 15;
constexpr int kPoseDim = 6;
constexpr int kExtrinsicDim = 6;

enum class Modality
{
  kPrior = 0,
  kImu,
  kLidar,
  kVision,
  kCrossModal,
  kFrontendPose
};

inline const char *ToString(Modality modality)
{
  switch (modality)
  {
    case Modality::kPrior: return "prior";
    case Modality::kImu: return "imu";
    case Modality::kLidar: return "lidar";
    case Modality::kVision: return "vision";
    case Modality::kCrossModal: return "cross_modal";
    case Modality::kFrontendPose: return "frontend_pose";
  }
  return "unknown";
}

struct NavState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp = 0.0;
  Eigen::Quaterniond q_w_i = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_w_i = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_w_i = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_g = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_w = Eigen::Vector3d(0.0, 0.0, -9.81);
};

// Final pose of a keyframe removed from the fixed-lag graph. The frontend
// state is retained so its last map correction can be computed after removal.
struct RetiredFrame
{
  std::uint64_t id = 0;
  NavState optimized;
  NavState frontend;
};

struct ExtrinsicState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Active transform from the IMU frame to the camera frame:
  // p_C = q_c_i * p_I + p_c_i.
  Eigen::Quaterniond q_c_i = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_c_i = Eigen::Vector3d::Zero();
};

struct LidarExtrinsic
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Active transform from the LiDAR frame to the IMU frame:
  // p_I = q_i_l * p_L + p_i_l.
  Eigen::Quaterniond q_i_l = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_i_l = Eigen::Vector3d::Zero();
};

struct ImuSample
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp = 0.0;
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct LidarPlaneObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d point_l = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_w = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d center_w = Eigen::Vector3d::Zero();
  double signed_distance = 0.0;
  double variance = 1.0e-3;
};

// Raw patch sampled in the CURRENT image. The backend retains its own reference
// patch at an admitted keyframe, never a frontend-prewarped patch.
struct VisualPatchObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::uint64_t track_id = 0;
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero(); // initialization only
  Eigen::Vector3d center_ray = Eigen::Vector3d::UnitZ(); // z = 1
  Eigen::Vector3d normal_c = Eigen::Vector3d::UnitZ(); // fixed patch tangent
  double inverse_depth = 1.0;
  double tracking_score = 1.0;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> rays;
  std::vector<double> intensities;
  double inverse_exposure = 1.0;
  double photometric_sigma = 10.0;
};

struct MeasurementSummary
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Modality modality = Modality::kPrior;
  double timestamp = 0.0;
  Eigen::Matrix<double, kPoseDim, kPoseDim> information =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
  std::size_t residual_count = 0;
  double rmse = std::numeric_limits<double>::infinity();
  double inlier_ratio = 0.0;

  double log_det = -std::numeric_limits<double>::infinity();
  double min_eigenvalue = 0.0;
  double condition_number = std::numeric_limits<double>::infinity();
  double normalized_information = 0.0;
  double reliability = 0.0;
  bool valid = false;
};

struct ModalityWeights
{
  double lidar = 1.0;
  double vision = 1.0;
  double cross_modal = 1.0;
};

struct PolicyDecision
{
  ModalityWeights weights;
  int target_window_size = 6;
  bool select_keyframe = false;
  double keyframe_utility = 0.0;
  double complementarity_score = 0.0;
};

struct AdaptiveWindowConfig
{
  bool enabled = false;
  bool feedback_enabled = false;
  int feedback_max_active_scans = 20;  // warning only; retirement controls commits
  double feedback_map_min_translation = 0.01;
  double feedback_map_min_rotation_deg = 0.1;
  double feedback_covariance_inflation = 1.05;
  bool collect_factor_audit = false; // optional offline diagnostics; never uses GT

  int min_window_size = 4;
  int max_window_size = 12;
  int initial_window_size = 6;
  int max_lidar_factors = 160;
  int lidar_effective_factors = 160;
  double lidar_outlier_sigma = 3.0;
  int max_cross_modal_factors = 80;
  int max_active_patches = 160;
  bool preserve_visual_anchors = false;
  double cross_modal_temporal_max_dt = 0.0; // 0 disables an extra distinct-pose plane tie
  double cross_modal_temporal_time_sigma = 0.15;
  double cross_modal_temporal_confidence = 0.25;

  double merge_time_tolerance = 1.0e-4;
  double association_max_dt = 0.08;
  double association_time_sigma = 0.025;
  bool allow_pair_reuse = true;

  double min_keyframe_dt = 0.05;
  double max_keyframe_dt = 0.50;
  double keyframe_translation = 0.20;
  double keyframe_rotation_deg = 8.0;
  double keyframe_utility_threshold = 0.55;

  double min_modality_weight = 0.15;
  double max_modality_weight = 2.50;
  double weight_smoothing = 0.80;
  double information_smoothing = 0.90;
  double lidar_rmse_scale = 0.08;
  double vision_rmse_scale = 20.0;

  double information_eigen_floor = 1.0e-9;
  double information_logdet_floor = -35.0;
  double information_logdet_ceiling = 30.0;
  double degeneracy_eigen_threshold = 1.0e-4;
  bool normalized_information_policy = false;
  double information_translation_scale = 0.05; // metres for dimensionless pose increments
  double information_rotation_scale_deg = 1.0;

  int solver_iterations = 8;
  double solver_time_limit = 0.03; // seconds; a soft Ceres budget
  double huber_delta_lidar = 2.0;
  double huber_delta_vision = 3.0;
  double huber_delta_cross_modal = 3.0;
  bool optimize_extrinsics = false; // enable only with sufficient excitation
  double plane_search_radius = 0.75;
  double plane_max_rms = 0.05;
  double cross_modal_plane_sigma = 0.05;
  double min_inverse_depth = 0.005;
  double max_inverse_depth = 5.0;

  // Correlated regularization around FAST-LIVO2, not calibrated covariance.
  bool frontend_pose_prior_enabled = true;
  double frontend_pose_translation_sigma = 0.05; // metres
  double frontend_pose_rotation_deg = 1.0;
  double frontend_pose_prior_weight = 1.0;
  double weak_direction_prior_gain = 0.0; // 0 disables bounded spectral regularization
  double frontend_pose_max_translation = 0.20; // reject the whole solve beyond this
  double frontend_pose_max_rotation_deg = 5.0;

  // First-state pose anchor used only when frontend_pose_prior_enabled is false.
  double pose_prior_translation_sigma = 0.08;
  double pose_prior_rotation_deg = 3.0;
  double velocity_prior_sigma = 0.30;
  double bias_prior_sigma = 0.05;
  double extrinsic_translation_sigma = 0.05;
  double extrinsic_rotation_deg = 3.0;

  double imu_acc_noise = 0.10;
  double imu_gyr_noise = 0.01;
  double imu_acc_bias_rw = 0.001;
  double imu_gyr_bias_rw = 0.0001;
  double imu_max_gap = 0.05;
  double imu_repropagate_acc_bias = 0.05;
  double imu_repropagate_gyr_bias = 0.005;
};

struct BackendDiagnostics
{
  struct EventAudit
  {
    std::size_t input_count = 0;
    std::size_t invalid_count = 0;
    std::size_t gate_rejected = 0;
    std::size_t accepted_count = 0;
    std::size_t photo_count = 0;
    std::size_t cross_count = 0;
    std::size_t temporal_cross_count = 0;
    std::size_t temporal_plane_failed = 0;
    std::size_t plane_failed = 0;
    std::size_t no_match = 0;
    double seed_rms = 0.0;
    double photo_seed_rms = 0.0;
    double cross_seed_rms = 0.0;
    double imu_seed_rms = 0.0;
    double imu_post_rms = 0.0;
    double lidar_post_rms = 0.0;
    std::array<double, 5> imu_seed_blocks{};
    std::array<double, 5> imu_post_blocks{};
    double association_dt = std::numeric_limits<double>::quiet_NaN();
    // Accepted LiDAR-only Gauss-Newton information in [position, rotation]
    // tangent order. This is diagnostic and excludes nuisance coupling.
    Eigen::Matrix<double, kPoseDim, kPoseDim> lidar_information =
        Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
  } event;

  std::size_t window_size = 0;
  std::size_t factor_count = 0;
  int target_window_size = 0;
  ModalityWeights weights;
  double solve_time_ms = 0.0;
  double last_cost = 0.0;
  bool solve_success = false;
  std::uint64_t last_marginalized_id = 0;
  std::size_t cross_modal_pairs = 0;
  std::size_t lidar_factors = 0;
  std::size_t vision_factors = 0;
  std::size_t imu_factors = 0;
  std::size_t rejected_imu_intervals = 0;
  std::size_t reset_count = 0;
  std::size_t frontend_pose_factors = 0;
  std::size_t rejected_pose_refinements = 0;
  double max_frontend_translation_correction = 0.0;
  double max_frontend_rotation_correction_deg = 0.0;
  double camera_extrinsic_translation_change = 0.0;
  double camera_extrinsic_rotation_change_deg = 0.0;
  double keyframe_utility = 0.0;
  bool keyframe_selected = false;
  double keyframe_complementarity = 0.0;
  double last_marginalized_age = 0.0;
  double last_marginalized_information = 0.0;
};

using NavStateVector =
    std::vector<NavState, Eigen::aligned_allocator<NavState>>;
using ImuSampleVector =
    std::vector<ImuSample, Eigen::aligned_allocator<ImuSample>>;
using LidarPlaneObservationVector =
    std::vector<LidarPlaneObservation,
                Eigen::aligned_allocator<LidarPlaneObservation>>;
using VisualPatchObservationVector =
    std::vector<VisualPatchObservation,
                Eigen::aligned_allocator<VisualPatchObservation>>;

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_TYPES_H_
