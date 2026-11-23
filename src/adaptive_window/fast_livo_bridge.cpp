#include "adaptive_window/fast_livo_bridge.h"

#include "adaptive_window/math_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fast_livo
{
namespace adaptive_window
{
namespace
{

constexpr std::size_t kMaxVisualizationPathPoses = 10000;

template <typename T>
void ReadParameter(ros::NodeHandle &node_handle, const std::string &name,
                   T *value)
{
  node_handle.param<T>("adaptive_window/" + name, *value, *value);
}

bool BilinearIntensity(const cv::Mat &image, const Eigen::Vector2d &pixel,
                       double *intensity)
{
  if (intensity == nullptr || !pixel.allFinite() || image.empty() || image.type() != CV_8UC1)
    return false;
  const int x = static_cast<int>(std::floor(pixel.x()));
  const int y = static_cast<int>(std::floor(pixel.y()));
  if (x < 1 || y < 1 || x + 2 >= image.cols || y + 2 >= image.rows)
    return false;
  const double dx = pixel.x() - x;
  const double dy = pixel.y() - y;
  const double i00 = image.at<std::uint8_t>(y, x);
  const double i10 = image.at<std::uint8_t>(y, x + 1);
  const double i01 = image.at<std::uint8_t>(y + 1, x);
  const double i11 = image.at<std::uint8_t>(y + 1, x + 1);
  *intensity = (1.0 - dx) * (1.0 - dy) * i00 +
               dx * (1.0 - dy) * i10 +
               (1.0 - dx) * dy * i01 + dx * dy * i11;
  return true;
}

}  // namespace

AdaptiveWindowConfig FastLivoBridge::LoadConfig(
    ros::NodeHandle &node_handle)
{
  AdaptiveWindowConfig config;
  ReadParameter(node_handle, "enabled", &config.enabled);
  ReadParameter(node_handle, "feedback_enabled", &config.feedback_enabled);
  ReadParameter(node_handle, "feedback_max_active_scans", &config.feedback_max_active_scans);
  ReadParameter(node_handle, "feedback_map_min_translation", &config.feedback_map_min_translation);
  ReadParameter(node_handle, "feedback_map_min_rotation_deg", &config.feedback_map_min_rotation_deg);
  ReadParameter(node_handle, "feedback_covariance_inflation", &config.feedback_covariance_inflation);
  ReadParameter(node_handle, "min_window_size", &config.min_window_size);
  ReadParameter(node_handle, "max_window_size", &config.max_window_size);
  ReadParameter(node_handle, "initial_window_size",
                &config.initial_window_size);
  ReadParameter(node_handle, "max_lidar_factors",
                &config.max_lidar_factors);
  ReadParameter(node_handle, "lidar_effective_factors",
                &config.lidar_effective_factors);
  ReadParameter(node_handle, "lidar_outlier_sigma",
                &config.lidar_outlier_sigma);
  ReadParameter(node_handle, "max_cross_modal_factors",
                &config.max_cross_modal_factors);
  ReadParameter(node_handle, "merge_time_tolerance",
                &config.merge_time_tolerance);
  ReadParameter(node_handle, "association_max_dt",
                &config.association_max_dt);
  ReadParameter(node_handle, "association_time_sigma",
                &config.association_time_sigma);
  ReadParameter(node_handle, "allow_pair_reuse", &config.allow_pair_reuse);
  ReadParameter(node_handle, "min_keyframe_dt", &config.min_keyframe_dt);
  ReadParameter(node_handle, "max_keyframe_dt", &config.max_keyframe_dt);
  ReadParameter(node_handle, "keyframe_translation",
                &config.keyframe_translation);
  ReadParameter(node_handle, "keyframe_rotation_deg",
                &config.keyframe_rotation_deg);
  ReadParameter(node_handle, "keyframe_utility_threshold",
                &config.keyframe_utility_threshold);
  ReadParameter(node_handle, "min_modality_weight",
                &config.min_modality_weight);
  ReadParameter(node_handle, "max_modality_weight",
                &config.max_modality_weight);
  ReadParameter(node_handle, "weight_smoothing", &config.weight_smoothing);
  ReadParameter(node_handle, "information_smoothing",
                &config.information_smoothing);
  ReadParameter(node_handle, "information_logdet_floor",
                &config.information_logdet_floor);
  ReadParameter(node_handle, "information_logdet_ceiling",
                &config.information_logdet_ceiling);
  ReadParameter(node_handle, "normalized_information_policy",
                &config.normalized_information_policy);
  ReadParameter(node_handle, "information_translation_scale",
                &config.information_translation_scale);
  ReadParameter(node_handle, "information_rotation_scale_deg",
                &config.information_rotation_scale_deg);
  ReadParameter(node_handle, "degeneracy_eigen_threshold",
                &config.degeneracy_eigen_threshold);
  ReadParameter(node_handle, "lidar_rmse_scale", &config.lidar_rmse_scale);
  ReadParameter(node_handle, "vision_rmse_scale", &config.vision_rmse_scale);
  ReadParameter(node_handle, "solver_time_limit", &config.solver_time_limit);
  ReadParameter(node_handle, "solver_iterations", &config.solver_iterations);
  ReadParameter(node_handle, "huber_delta_lidar",
                &config.huber_delta_lidar);
  ReadParameter(node_handle, "huber_delta_cross_modal",
                &config.huber_delta_cross_modal);
  ReadParameter(node_handle, "frontend_pose_prior_enabled", &config.frontend_pose_prior_enabled);
  ReadParameter(node_handle, "frontend_pose_translation_sigma", &config.frontend_pose_translation_sigma);
  ReadParameter(node_handle, "frontend_pose_rotation_deg", &config.frontend_pose_rotation_deg);
  ReadParameter(node_handle, "frontend_pose_prior_weight", &config.frontend_pose_prior_weight);
  ReadParameter(node_handle, "weak_direction_prior_gain", &config.weak_direction_prior_gain);
  ReadParameter(node_handle, "frontend_pose_max_translation", &config.frontend_pose_max_translation);
  ReadParameter(node_handle, "frontend_pose_max_rotation_deg", &config.frontend_pose_max_rotation_deg);
  ReadParameter(node_handle, "pose_prior_translation_sigma",
                &config.pose_prior_translation_sigma);
  ReadParameter(node_handle, "pose_prior_rotation_deg",
                &config.pose_prior_rotation_deg);
  ReadParameter(node_handle, "extrinsic_translation_sigma",
                &config.extrinsic_translation_sigma);
  ReadParameter(node_handle, "extrinsic_rotation_deg",
                &config.extrinsic_rotation_deg);
  ReadParameter(node_handle, "imu_acc_noise", &config.imu_acc_noise);
  ReadParameter(node_handle, "imu_gyr_noise", &config.imu_gyr_noise);
  ReadParameter(node_handle, "imu_acc_bias_rw", &config.imu_acc_bias_rw);
  ReadParameter(node_handle, "imu_gyr_bias_rw", &config.imu_gyr_bias_rw);

  ReadParameter(node_handle, "max_active_patches", &config.max_active_patches);
  ReadParameter(node_handle, "cross_modal_temporal_max_dt",
                &config.cross_modal_temporal_max_dt);
  ReadParameter(node_handle, "cross_modal_temporal_time_sigma",
                &config.cross_modal_temporal_time_sigma);
  ReadParameter(node_handle, "cross_modal_temporal_confidence",
                &config.cross_modal_temporal_confidence);
  ReadParameter(node_handle, "preserve_visual_anchors", &config.preserve_visual_anchors);
  ReadParameter(node_handle, "huber_delta_vision", &config.huber_delta_vision);
  ReadParameter(node_handle, "optimize_extrinsics", &config.optimize_extrinsics);
  ReadParameter(node_handle, "plane_search_radius", &config.plane_search_radius);
  ReadParameter(node_handle, "plane_max_rms", &config.plane_max_rms);
  ReadParameter(node_handle, "cross_modal_plane_sigma", &config.cross_modal_plane_sigma);
  ReadParameter(node_handle, "min_inverse_depth", &config.min_inverse_depth);
  ReadParameter(node_handle, "max_inverse_depth", &config.max_inverse_depth);
  ReadParameter(node_handle, "imu_max_gap", &config.imu_max_gap);
  ReadParameter(node_handle, "imu_repropagate_acc_bias", &config.imu_repropagate_acc_bias);
  ReadParameter(node_handle, "imu_repropagate_gyr_bias", &config.imu_repropagate_gyr_bias);
  ReadParameter(node_handle, "velocity_prior_sigma", &config.velocity_prior_sigma);
  ReadParameter(node_handle, "bias_prior_sigma", &config.bias_prior_sigma);
  ReadParameter(node_handle, "information_eigen_floor", &config.information_eigen_floor);

  config.min_window_size = std::max(2, config.min_window_size);
  config.max_window_size =
      std::max(config.min_window_size, config.max_window_size);
  config.initial_window_size =
      std::max(config.min_window_size,
               std::min(config.max_window_size, config.initial_window_size));
  config.feedback_max_active_scans = std::max(2, config.feedback_max_active_scans);
  config.feedback_map_min_translation = std::max(0.0, config.feedback_map_min_translation);
  config.feedback_map_min_rotation_deg = std::max(0.0, config.feedback_map_min_rotation_deg);
  config.feedback_covariance_inflation = std::max(1.0, config.feedback_covariance_inflation);
  return config;
}

FastLivoBridge::FastLivoBridge(ros::NodeHandle &node_handle,
                               const M3D &rotation_i_l,
                               const V3D &translation_i_l,
                               const VIOManagerPtr &vio_manager)
    : config_(LoadConfig(node_handle))
{
  lidar_extrinsic_.q_i_l =
      Eigen::Quaterniond(rotation_i_l).normalized();
  lidar_extrinsic_.p_i_l = translation_i_l;

  if (!config_.enabled)
  {
    ROS_INFO("Adaptive fixed-lag backend is disabled.");
    return;
  }
  if (vio_manager == nullptr || vio_manager->cam == nullptr)
  {
    ROS_ERROR("Adaptive fixed-lag backend requires an initialized camera.");
    config_.enabled = false;
    return;
  }

  camera_extrinsic_initial_.q_c_i =
      Eigen::Quaterniond(vio_manager->Rci).normalized();
  camera_extrinsic_initial_.p_c_i = vio_manager->Pci;

  std::string diagnostic_csv_path;
  node_handle.param<std::string>("adaptive_window/diagnostic_csv_path",
                                 diagnostic_csv_path, "");
  if (!diagnostic_csv_path.empty())
  {
    diagnostic_csv_stream_.open(diagnostic_csv_path, std::ios::out);
    if (!diagnostic_csv_stream_.is_open())
      ROS_ERROR("Cannot open adaptive-window diagnostic CSV: %s",
                diagnostic_csv_path.c_str());
    else
    {
      config_.collect_factor_audit = true;
      diagnostic_csv_stream_ << std::setprecision(17)
          << "timestamp,source,solved,estimate_time,window,target,utility,complementarity,keyframe_selected,"
             "lidar_weight,vision_weight,cross_weight,solve_ms,cost,resets,rejected_imu,"
             "rejected_pose,correction_m,correction_deg,marginalized_id,marginalized_age,"
             "marginalized_info,factors,lidar_factors,vision_factors,imu_factors,"
             "cross_factors,frontend_pose_factors,input_count,invalid_count,gate_rejected,"
             "accepted_count,photo_count,cross_count,temporal_cross_count,"
             "temporal_plane_failed,plane_failed,no_match,seed_rms,"
             "photo_seed_rms,cross_seed_rms,imu_seed_rms,imu_post_rms,lidar_post_rms,"
             "imu_seed_position,imu_seed_rotation,imu_seed_velocity,imu_seed_acc_bias,imu_seed_gyro_bias,"
             "imu_post_position,imu_post_rotation,imu_post_velocity,imu_post_acc_bias,imu_post_gyro_bias,association_dt,"
             "lidar_summary_time,lidar_summary_valid,lidar_summary_count,"
             "lidar_summary_rmse,lidar_summary_inlier,lidar_logdet,lidar_min_eig,"
             "lidar_condition,lidar_quality,lidar_normalized_information,"
             "vision_summary_time,vision_summary_valid,vision_summary_count,"
             "vision_summary_rmse,vision_summary_inlier,vision_logdet,vision_min_eig,"
             "vision_condition,vision_quality,vision_normalized_information";
      for (const char *name : {"frontend_lidar_rp", "frontend_vision_rp", "accepted_lidar_pr"})
        for (int i = 0; i < kPoseDim; ++i)
          for (int j = i; j < kPoseDim; ++j)
            diagnostic_csv_stream_ << ',' << name << "_h" << i << j;
      diagnostic_csv_stream_ << ",estimate_x,estimate_y,estimate_z,"
                                "estimate_qx,estimate_qy,estimate_qz,estimate_qw\n";
      ROS_INFO("Adaptive-window diagnostic CSV: %s", diagnostic_csv_path.c_str());
    }
  }

  ProjectionFunction projection =
      [camera = vio_manager->cam, border = vio_manager->border](
          const Eigen::Vector3d &point, Eigen::Vector2d *pixel,
          Eigen::Matrix<double, 2, 3> *jacobian) {
        if (!point.allFinite() || point.z() <= 1.0e-4 || pixel == nullptr || jacobian == nullptr)
          return false;
        *pixel = camera->world2cam(point);
        if (!pixel->allFinite() ||
            !camera->isInFrame(pixel->cast<int>(), border))
          return false;
        const double epsilon = 1.0e-6;
        for (int column = 0; column < 3; ++column)
        {
          Eigen::Vector3d perturbed = point;
          perturbed(column) += epsilon;
          const Eigen::Vector2d plus = camera->world2cam(perturbed);
          perturbed(column) -= 2.0 * epsilon;
          jacobian->col(column) = (plus - camera->world2cam(perturbed)) / (2.0 * epsilon);
        }
        return jacobian->allFinite();
      };

  try { backend_.reset(new AdaptiveWindowBackend(config_)); }
  catch (const std::invalid_argument &error) {
    ROS_ERROR("Invalid adaptive-window configuration: %s", error.what());
    config_.enabled = false; return;
  }
  backend_->Initialize(lidar_extrinsic_, camera_extrinsic_initial_,
                       std::move(projection));

  odometry_publisher_ = node_handle.advertise<nav_msgs::Odometry>(
      "/adaptive_window/odometry", 20);
  optimized_odometry_publisher_ = node_handle.advertise<nav_msgs::Odometry>(
      "/adaptive_window/optimized_odometry", 20);
  optimized_path_publisher_ = node_handle.advertise<nav_msgs::Path>(
      "/adaptive_window/optimized_path", 1, true);
  diagnostics_publisher_ =
      node_handle.advertise<std_msgs::Float64MultiArray>(
          "/adaptive_window/diagnostics", 20);
  extrinsic_publisher_ =
      node_handle.advertise<geometry_msgs::TransformStamped>(
          "/adaptive_window/camera_imu_extrinsic", 20);
  bool pose_output_enabled = false;
  node_handle.param<bool>("adaptive_window/pose_output_en",
                          pose_output_enabled, false);
  if (pose_output_enabled)
  {
    std::string pose_output_path;
    node_handle.param<std::string>("adaptive_window/pose_output_path",
                                   pose_output_path, "");
    if (pose_output_path.empty())
    {
      std::string sequence_name;
      node_handle.param<std::string>("evo/seq_name", sequence_name, "01");
      pose_output_path = std::string(ROOT_DIR) + "Log/result/" +
                         sequence_name + "_adaptive.txt";
    }
    pose_output_stream_.open(pose_output_path, std::ios::out);
    if (!pose_output_stream_.is_open())
      ROS_ERROR("Cannot open adaptive-window trajectory: %s",
                pose_output_path.c_str());
    else
      ROS_INFO("Adaptive-window optimized trajectory: %s",
               pose_output_path.c_str());
  }
  ROS_INFO("Adaptive fixed-lag backend enabled: window [%d, %d].",
           config_.min_window_size, config_.max_window_size);
  if (config_.feedback_enabled)
    ROS_INFO("Adaptive fixed-lag feedback enabled: frontend state and active maps will be corrected.");
}

FastLivoBridge::~FastLivoBridge()
{
  FlushPendingPose();
}

NavState FastLivoBridge::ToNavState(const double timestamp,
                                    const StatesGroup &state)
{
  NavState result;
  result.timestamp = timestamp;
  result.q_w_i = Eigen::Quaterniond(state.rot_end).normalized();
  result.p_w_i = state.pos_end;
  result.v_w_i = state.vel_end;
  result.bias_g = state.bias_g;
  result.bias_a = state.bias_a;
  result.gravity_w = state.gravity;
  return result;
}

MeasurementSummary FastLivoBridge::BuildLidarSummary(
    const double timestamp, const StatesGroup &state,
    const std::vector<PointToPlane> &planes,
    const std::size_t raw_feature_count,
    LidarPlaneObservationVector *observations) const
{
  MeasurementSummary summary;
  summary.modality = Modality::kLidar;
  summary.timestamp = timestamp;
  observations->clear();
  observations->reserve(planes.size());

  double squared_error = 0.0;
  for (const PointToPlane &plane : planes)
  {
    const Eigen::Vector3d point_i =
        lidar_extrinsic_.q_i_l * plane.point_b_ +
        lidar_extrinsic_.p_i_l;
    const Eigen::Vector3d point_w = state.rot_end * point_i + state.pos_end;
    const Eigen::Matrix3d point_cross = Skew(point_i);
    const Eigen::Vector3d rotation_jacobian =
        point_cross * state.rot_end.transpose() * plane.normal_;
    Eigen::Matrix<double, 1, kPoseDim> jacobian;
    jacobian << rotation_jacobian.transpose(), plane.normal_.transpose();

    Eigen::Matrix<double, 1, 6> plane_jacobian;
    plane_jacobian.block<1, 3>(0, 0) = point_w - plane.center_;
    plane_jacobian.block<1, 3>(0, 3) = -plane.normal_;
    const Eigen::Matrix3d point_covariance =
        state.rot_end * lidar_extrinsic_.q_i_l.toRotationMatrix() *
        plane.body_cov_ *
        (state.rot_end * lidar_extrinsic_.q_i_l.toRotationMatrix())
            .transpose();
    const double variance =
        std::max(1.0e-9,
                 0.001 +
                     (plane_jacobian * plane.plane_var_ *
                      plane_jacobian.transpose())(0, 0) +
                     (plane.normal_.transpose() * point_covariance *
                      plane.normal_)(0, 0));
    summary.information.noalias() += jacobian.transpose() * jacobian / variance;
    squared_error += plane.dis_to_plane_ * plane.dis_to_plane_;

    LidarPlaneObservation observation;
    observation.point_l = plane.point_b_;
    observation.normal_w = plane.normal_;
    observation.center_w = plane.center_;
    observation.signed_distance = plane.dis_to_plane_;
    observation.variance = variance;
    observations->push_back(observation);
  }

  summary.residual_count = planes.size();
  summary.rmse = planes.empty()
                     ? std::numeric_limits<double>::infinity()
                     : std::sqrt(squared_error / planes.size());
  summary.inlier_ratio =
      raw_feature_count == 0
          ? 0.0
          : std::min(1.0, static_cast<double>(planes.size()) /
                              static_cast<double>(raw_feature_count));
  return summary;
}

MeasurementSummary FastLivoBridge::BuildVisionSummary(
    const double timestamp, const VIOManagerPtr &vio_manager,
    VisualPatchObservationVector *observations) const
{
  MeasurementSummary summary;
  summary.modality = Modality::kVision;
  summary.timestamp = timestamp;
  observations->clear();
  if (vio_manager == nullptr || vio_manager->visual_submap == nullptr ||
      vio_manager->total_points <= 0)
  {
    summary.rmse = std::numeric_limits<double>::infinity();
    return summary;
  }

  summary.information =
      vio_manager->H_T_H.topLeftCorner<kPoseDim, kPoseDim>() /
      std::max(1.0e-9, vio_manager->img_point_cov);
  summary.residual_count =
      static_cast<std::size_t>(vio_manager->total_points) *
      static_cast<std::size_t>(vio_manager->patch_size_total);

  double sum_squared_error = 0.0;
  for (const float error : vio_manager->visual_submap->errors)
    sum_squared_error += std::max(0.0f, error);
  summary.rmse = summary.residual_count == 0
                     ? std::numeric_limits<double>::infinity()
                     : std::sqrt(sum_squared_error / summary.residual_count);
  summary.inlier_ratio =
      std::min(1.0, static_cast<double>(vio_manager->total_points) /
                        std::max(1, vio_manager->length));

  if (!vio_manager->new_frame_ || !vio_manager->cam) return summary;
  const cv::Mat &image = vio_manager->new_frame_->img_;
  observations->reserve(vio_manager->visual_submap->voxel_points.size());
  for (std::size_t i = 0; i < vio_manager->visual_submap->voxel_points.size(); ++i)
  {
    const VisualPoint *point = vio_manager->visual_submap->voxel_points[i];
    if (!point || !point->is_normal_initialized_ || !point->normal_.allFinite() ||
        point->normal_.norm() < 1e-6) continue;
    const Eigen::Vector3d point_c = vio_manager->new_frame_->w2f(point->pos_);
    if (!point_c.allFinite() || point_c.z() <= 0.2) continue;
    const Eigen::Vector2d pixel = vio_manager->cam->world2cam(point_c);
    if (!pixel.allFinite()) continue;

    VisualPatchObservation observation;
    observation.track_id = point->backend_track_id_;
    observation.point_w = point->pos_;
    observation.center_ray = point_c / point_c.z();
    observation.inverse_depth = 1.0 / point_c.z();
    observation.normal_c = (vio_manager->new_frame_->T_f_w_.rotation_matrix() *
                            point->normal_).normalized();
    const double error = i < vio_manager->visual_submap->errors.size()
        ? std::max(0.0f, vio_manager->visual_submap->errors[i]) : 0.0;
    observation.tracking_score = std::exp(-std::sqrt(error /
        std::max(1, vio_manager->patch_size_total)) /
        std::max(1e-6, config_.vision_rmse_scale));
    observation.inverse_exposure = vio_manager->state
        ? vio_manager->state->inv_expo_time : 1.0;
    observation.photometric_sigma = std::sqrt(std::max(1e-6, vio_manager->img_point_cov));

    // A bounded 4 x 4 patch on pyramid level zero, sampled from the ORIGINAL
    // current image. The backend chooses references and rewarps at each solve.
    const int half = vio_manager->patch_size_half;
    bool valid = true;
    for (int row = 0; row < 4 && valid; ++row)
      for (int col = 0; col < 4; ++col)
      {
        const Eigen::Vector2d offset(-half + (2.0 * half - 1.0) * col / 3.0,
                                     -half + (2.0 * half - 1.0) * row / 3.0);
        const Eigen::Vector2d sample_pixel = pixel + offset;
        double value = 0.0;
        Eigen::Vector3d ray = vio_manager->cam->cam2world(sample_pixel);
        if (!ray.allFinite() || ray.z() <= 1e-6 ||
            !BilinearIntensity(image, sample_pixel, &value)) { valid = false; break; }
        observation.rays.push_back(ray / ray.z());
        observation.intensities.push_back(value);
      }
    if (valid) observations->push_back(std::move(observation));
  }
  return summary;
}

bool FastLivoBridge::ProcessLidar(
    const double timestamp, const StatesGroup &state,
    const std::deque<sensor_msgs::Imu::ConstPtr> &imu_messages,
    const double acceleration_scale, const std::vector<PointToPlane> &planes,
    const std::size_t raw_feature_count)
{
  if (!enabled()) return false;
  ImuSampleVector samples;
  samples.reserve(imu_messages.size());
  for (const auto &message : imu_messages)
  {
    if (message == nullptr) continue;
    ImuSample sample;
    sample.timestamp = message->header.stamp.toSec();
    sample.acceleration =
        acceleration_scale *
        Eigen::Vector3d(message->linear_acceleration.x,
                        message->linear_acceleration.y,
                        message->linear_acceleration.z);
    sample.angular_velocity =
        Eigen::Vector3d(message->angular_velocity.x,
                        message->angular_velocity.y,
                        message->angular_velocity.z);
    samples.push_back(sample);
  }
  backend_->AddImuSamples(samples);

  const NavState candidate = ToNavState(timestamp, state);
  if (!backend_->shouldAttemptFrame(candidate)) return false;

  LidarPlaneObservationVector observations;
  const MeasurementSummary summary = BuildLidarSummary(
      timestamp, state, planes, raw_feature_count, &observations);
  const bool result = backend_->ProcessLidar(
      candidate, summary, observations);
  Publish(timestamp, result, Modality::kLidar);
  return result;
}

bool FastLivoBridge::ProcessVision(const double timestamp,
                                   const StatesGroup &state,
                                   const VIOManagerPtr &vio_manager,
                                   const std::deque<sensor_msgs::Imu::ConstPtr>
                                       &imu_messages,
                                   const double acceleration_scale)
{
  if (!enabled()) return false;
  ImuSampleVector samples;
  samples.reserve(imu_messages.size());
  for (const auto &message : imu_messages)
  {
    if (message == nullptr) continue;
    ImuSample sample;
    sample.timestamp = message->header.stamp.toSec();
    sample.acceleration =
        acceleration_scale *
        Eigen::Vector3d(message->linear_acceleration.x,
                        message->linear_acceleration.y,
                        message->linear_acceleration.z);
    sample.angular_velocity =
        Eigen::Vector3d(message->angular_velocity.x,
                        message->angular_velocity.y,
                        message->angular_velocity.z);
    samples.push_back(sample);
  }
  backend_->AddImuSamples(samples);

  const NavState candidate = ToNavState(timestamp, state);
  if (!backend_->shouldAttemptFrame(candidate)) return false;

  VisualPatchObservationVector observations;
  const MeasurementSummary summary =
      BuildVisionSummary(timestamp, vio_manager, &observations);
  const cv::Mat current_image =
      (vio_manager != nullptr && vio_manager->new_frame_ != nullptr)
          ? vio_manager->new_frame_->img_.clone()
          : cv::Mat();
  const ImageSampleFunction image_sampler =
      [current_image](const Eigen::Vector2d &pixel, double *intensity,
                      Eigen::Vector2d *gradient) {
        if (gradient == nullptr ||
            !BilinearIntensity(current_image, pixel, intensity))
          return false;
        const int x = static_cast<int>(std::floor(pixel.x()));
        const int y = static_cast<int>(std::floor(pixel.y()));
        const double dx = pixel.x() - x, dy = pixel.y() - y;
        const double a = current_image.at<std::uint8_t>(y, x);
        const double b = current_image.at<std::uint8_t>(y, x + 1);
        const double c = current_image.at<std::uint8_t>(y + 1, x);
        const double d = current_image.at<std::uint8_t>(y + 1, x + 1);
        *gradient = Eigen::Vector2d((1.0 - dy) * (b - a) + dy * (d - c),
                                    (1.0 - dx) * (c - a) + dx * (d - b));
        return true;
      };
  const bool result = backend_->ProcessVision(
      candidate, summary, observations, image_sampler);
  Publish(timestamp, result, Modality::kVision);
  return result;
}

bool FastLivoBridge::CollectFeedback(
    double timestamp, const StatesGroup &frontend_state, NavState *estimate,
    std::vector<PoseCorrection> *corrections,
    std::vector<RetiredFrame> *retired)
{
  if (!feedback_enabled() || !estimate || !corrections || !retired) return false;
  *retired = backend_->takeRetiredFrames();
  if (!backend_->LatestEstimate(estimate) ||
      std::abs(estimate->timestamp - timestamp) >
          std::max(1.0e-3, config_.merge_time_tolerance)) return false;
  const Eigen::Quaterniond frontend_rotation(frontend_state.rot_end);
  const double distance = (estimate->p_w_i - frontend_state.pos_end).norm();
  const double angle = Eigen::AngleAxisd(
      frontend_rotation.conjugate() * estimate->q_w_i).angle();
  if (!std::isfinite(distance) || !std::isfinite(angle) ||
      distance > config_.frontend_pose_max_translation ||
      angle > config_.frontend_pose_max_rotation_deg * M_PI / 180.0 ||
      !estimate->p_w_i.allFinite() || !estimate->v_w_i.allFinite() ||
      !estimate->bias_g.allFinite() || !estimate->bias_a.allFinite()) return false;

  corrections->clear();
  const auto ids = backend_->frameIds();
  if (ids.empty()) return false;
  std::map<std::uint64_t, NavState> retained;
  bool any_material = false;
  for (const auto id : ids)
  {
    const NavState updated = backend_->stateEstimate(id);
    NavState previous;
    const auto applied = applied_states_.find(id);
    if (applied != applied_states_.end()) previous = applied->second;
    else if (!backend_->frontendState(id, &previous)) continue;
    PoseCorrection correction;
    correction.timestamp = updated.timestamp;
    correction.rotation = (updated.q_w_i * previous.q_w_i.conjugate()).normalized();
    correction.translation = updated.p_w_i - correction.rotation * previous.p_w_i;
    if (HasMaterialCorrection(correction,
        config_.feedback_map_min_translation,
        config_.feedback_map_min_rotation_deg * M_PI / 180.0))
    {
      retained[id] = updated;
      any_material = true;
    }
    else
    {
      correction.rotation.setIdentity();
      correction.translation.setZero();
      if (applied != applied_states_.end()) retained[id] = applied->second;
    }
    corrections->push_back(correction);
  }
  for (const auto &frame : *retired)
  {
    const auto applied = applied_states_.find(frame.id);
    const NavState &previous = applied != applied_states_.end()
        ? applied->second : frame.frontend;
    PoseCorrection correction;
    correction.timestamp = frame.optimized.timestamp;
    correction.rotation =
        (frame.optimized.q_w_i * previous.q_w_i.conjugate()).normalized();
    correction.translation = frame.optimized.p_w_i -
        correction.rotation * previous.p_w_i;
    if (HasMaterialCorrection(correction,
        config_.feedback_map_min_translation,
        config_.feedback_map_min_rotation_deg * M_PI / 180.0))
      any_material = true;
    else { correction.rotation.setIdentity(); correction.translation.setZero(); }
    corrections->push_back(correction);
  }
  std::sort(corrections->begin(), corrections->end(),
      [](const PoseCorrection &a, const PoseCorrection &b) {
        return a.timestamp < b.timestamp;
      });
  applied_states_.swap(retained);
  if (!any_material) corrections->clear();
  return true;
}

std::map<std::uint64_t, NavState> FastLivoBridge::ActiveKeyframeEstimates() const
{
  std::map<std::uint64_t, NavState> states;
  if (!enabled()) return states;
  for (const auto id : backend_->frameIds())
    states.emplace(id, backend_->stateEstimate(id));
  return states;
}

void FastLivoBridge::Publish(const double timestamp, const bool solved,
                            const Modality source)
{
  if (!enabled()) return;
  NavState estimate;
  if (backend_->LatestEstimate(&estimate))
  {
    nav_msgs::Odometry message;
    message.header.stamp.fromSec(estimate.timestamp);
    message.header.frame_id = "camera_init";
    message.child_frame_id = "body";
    message.pose.pose.position.x = estimate.p_w_i.x();
    message.pose.pose.position.y = estimate.p_w_i.y();
    message.pose.pose.position.z = estimate.p_w_i.z();
    message.pose.pose.orientation.w = estimate.q_w_i.w();
    message.pose.pose.orientation.x = estimate.q_w_i.x();
    message.pose.pose.orientation.y = estimate.q_w_i.y();
    message.pose.pose.orientation.z = estimate.q_w_i.z();
    const Eigen::Vector3d velocity_body = estimate.q_w_i.conjugate() * estimate.v_w_i;
    message.twist.twist.linear.x = velocity_body.x();
    message.twist.twist.linear.y = velocity_body.y();
    message.twist.twist.linear.z = velocity_body.z();
    odometry_publisher_.publish(message);
    if (solved)
    {
      optimized_odometry_publisher_.publish(message);
      geometry_msgs::PoseStamped pose;
      pose.header = message.header;
      pose.pose = message.pose.pose;
      optimized_path_.header = message.header;
      if (optimized_path_.poses.empty())
        optimized_path_.poses.push_back(pose);
      else if (pose.header.stamp == optimized_path_.poses.back().header.stamp)
        optimized_path_.poses.back() = pose;
      else if (pose.header.stamp > optimized_path_.poses.back().header.stamp)
        optimized_path_.poses.push_back(pose);
      if (optimized_path_.poses.size() > kMaxVisualizationPathPoses)
        optimized_path_.poses.erase(optimized_path_.poses.begin(),
            optimized_path_.poses.begin() +
            (optimized_path_.poses.size() - kMaxVisualizationPathPoses));
      optimized_path_publisher_.publish(optimized_path_);
      RecordOptimizedPose(estimate);
    }
  }

  const BackendDiagnostics &diagnostics = backend_->diagnostics();
  std_msgs::Float64MultiArray diagnostic_message;
  diagnostic_message.data = {
      static_cast<double>(diagnostics.window_size),
      static_cast<double>(diagnostics.target_window_size),
      diagnostics.weights.lidar,
      diagnostics.weights.vision,
      diagnostics.weights.cross_modal,
      diagnostics.solve_time_ms,
      diagnostics.last_cost,
      static_cast<double>(diagnostics.factor_count),
      static_cast<double>(diagnostics.cross_modal_pairs),
      diagnostics.camera_extrinsic_translation_change,
      diagnostics.camera_extrinsic_rotation_change_deg,
      static_cast<double>(diagnostics.lidar_factors),
      static_cast<double>(diagnostics.vision_factors),
      static_cast<double>(diagnostics.imu_factors),
      diagnostics.solve_success ? 1.0 : 0.0,
      static_cast<double>(diagnostics.rejected_imu_intervals),
      static_cast<double>(diagnostics.reset_count),
      static_cast<double>(diagnostics.frontend_pose_factors),
      diagnostics.max_frontend_translation_correction,
      diagnostics.max_frontend_rotation_correction_deg,
      static_cast<double>(diagnostics.rejected_pose_refinements)};
  diagnostics_publisher_.publish(diagnostic_message);

  ExtrinsicState extrinsic = backend_->CameraExtrinsicEstimate();
  // TransformStamped(parent=IMU, child=camera) carries T_IC, not T_CI.
  extrinsic.q_c_i = extrinsic.q_c_i.conjugate();
  extrinsic.p_c_i = -(extrinsic.q_c_i * extrinsic.p_c_i);
  geometry_msgs::TransformStamped extrinsic_message;
  extrinsic_message.header.stamp.fromSec(timestamp);
  extrinsic_message.header.frame_id = "imu";
  extrinsic_message.child_frame_id = "camera_adaptive";
  extrinsic_message.transform.translation.x = extrinsic.p_c_i.x();
  extrinsic_message.transform.translation.y = extrinsic.p_c_i.y();
  extrinsic_message.transform.translation.z = extrinsic.p_c_i.z();
  extrinsic_message.transform.rotation.w = extrinsic.q_c_i.w();
  extrinsic_message.transform.rotation.x = extrinsic.q_c_i.x();
  extrinsic_message.transform.rotation.y = extrinsic.q_c_i.y();
  extrinsic_message.transform.rotation.z = extrinsic.q_c_i.z();
  extrinsic_publisher_.publish(extrinsic_message);
  WriteDiagnosticCsv(timestamp, solved, source);
}

void FastLivoBridge::WriteDiagnosticCsv(const double timestamp, const bool solved,
                                        const Modality source)
{
  if (!diagnostic_csv_stream_.is_open()) return;
  const BackendDiagnostics &d = backend_->diagnostics();
  const auto &a = d.event;
  const auto &lidar = backend_->latestLidarSummary();
  const auto &vision = backend_->latestVisionSummary();
  NavState estimate;
  const double estimate_time = backend_->LatestEstimate(&estimate)
      ? estimate.timestamp : std::numeric_limits<double>::quiet_NaN();
  auto &out = diagnostic_csv_stream_;
  out << timestamp << ',' << ToString(source) << ',' << int(solved) << ','
      << estimate_time << ',' << d.window_size << ',' << d.target_window_size << ','
      << d.keyframe_utility << ',' << d.keyframe_complementarity << ','
      << int(d.keyframe_selected) << ','
      << d.weights.lidar << ',' << d.weights.vision << ',' << d.weights.cross_modal << ','
      << d.solve_time_ms << ',' << d.last_cost << ',' << d.reset_count << ','
      << d.rejected_imu_intervals << ',' << d.rejected_pose_refinements << ','
      << d.max_frontend_translation_correction << ','
      << d.max_frontend_rotation_correction_deg << ',' << d.last_marginalized_id << ','
      << d.last_marginalized_age << ',' << d.last_marginalized_information << ','
      << d.factor_count << ',' << d.lidar_factors << ',' << d.vision_factors << ','
      << d.imu_factors << ',' << d.cross_modal_pairs << ',' << d.frontend_pose_factors << ','
      << a.input_count << ',' << a.invalid_count << ',' << a.gate_rejected << ','
      << a.accepted_count << ',' << a.photo_count << ',' << a.cross_count << ','
      << a.temporal_cross_count << ',' << a.temporal_plane_failed << ','
      << a.plane_failed << ',' << a.no_match << ',' << a.seed_rms << ','
      << a.photo_seed_rms << ',' << a.cross_seed_rms << ',' << a.imu_seed_rms << ','
      << a.imu_post_rms << ',' << a.lidar_post_rms;
  for (double value : a.imu_seed_blocks) out << ',' << value;
  for (double value : a.imu_post_blocks) out << ',' << value;
  out << ',' << a.association_dt << ','
      << lidar.timestamp << ',' << int(lidar.valid) << ',' << lidar.residual_count << ','
      << lidar.rmse << ',' << lidar.inlier_ratio << ',' << lidar.log_det << ','
      << lidar.min_eigenvalue << ',' << lidar.condition_number << ','
      << lidar.reliability << ',' << lidar.normalized_information << ','
      << vision.timestamp << ',' << int(vision.valid) << ',' << vision.residual_count << ','
      << vision.rmse << ',' << vision.inlier_ratio << ',' << vision.log_det << ','
      << vision.min_eigenvalue << ',' << vision.condition_number << ','
      << vision.reliability << ',' << vision.normalized_information;
  for (const auto *matrix : {&lidar.information, &vision.information, &a.lidar_information})
    for (int i = 0; i < kPoseDim; ++i)
      for (int j = i; j < kPoseDim; ++j)
        out << ',' << (*matrix)(i, j);
  if (std::isfinite(estimate_time))
    out << ',' << estimate.p_w_i.x() << ',' << estimate.p_w_i.y() << ','
        << estimate.p_w_i.z() << ',' << estimate.q_w_i.x() << ','
        << estimate.q_w_i.y() << ',' << estimate.q_w_i.z() << ','
        << estimate.q_w_i.w();
  else
    for (int i = 0; i < 7; ++i) out << ',';
  out << '\n';
}

void FastLivoBridge::RecordOptimizedPose(const NavState &estimate)
{
  if (!pose_output_stream_.is_open()) return;
  if (!std::isfinite(estimate.timestamp) || !estimate.p_w_i.allFinite() ||
      !estimate.q_w_i.coeffs().allFinite())
  {
    ROS_ERROR("Skipping non-finite adaptive-window optimized pose.");
    return;
  }
  if (has_pending_pose_)
  {
    if (estimate.timestamp < pending_pose_.timestamp - 1e-6)
    {
      ROS_WARN("Skipping out-of-order adaptive-window optimized pose.");
      return;
    }
    if (estimate.timestamp > pending_pose_.timestamp + 1e-6)
      FlushPendingPose();
  }
  // A same-time image update can refine the latest LiDAR state. Keep the
  // final accepted estimate for that timestamp in the TUM trajectory.
  pending_pose_ = estimate;
  has_pending_pose_ = true;
}

void FastLivoBridge::FlushPendingPose()
{
  if (!has_pending_pose_ || !pose_output_stream_.is_open()) return;
  const Eigen::Quaterniond q = pending_pose_.q_w_i.normalized();
  pose_output_stream_ << std::fixed << std::setprecision(9)
                      << pending_pose_.timestamp << " "
                      << pending_pose_.p_w_i.x() << " "
                      << pending_pose_.p_w_i.y() << " "
                      << pending_pose_.p_w_i.z() << " "
                      << q.x() << " " << q.y() << " " << q.z() << " "
                      << q.w() << std::endl;
  has_pending_pose_ = false;
}

}  // namespace adaptive_window
}  // namespace fast_livo
