#ifndef FAST_LIVO_ADAPTIVE_WINDOW_FAST_LIVO_BRIDGE_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_FAST_LIVO_BRIDGE_H_

#include "adaptive_window/adaptive_window_backend.h"
#include "adaptive_window/feedback_correction.h"
#include "common_lib.h"
#include "vio.h"
#include "voxel_map.h"

#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64MultiArray.h>

#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <string>

namespace fast_livo
{
namespace adaptive_window
{

class FastLivoBridge
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FastLivoBridge(ros::NodeHandle &node_handle, const M3D &rotation_i_l,
                 const V3D &translation_i_l,
                 const VIOManagerPtr &vio_manager);
  ~FastLivoBridge();

  bool enabled() const { return config_.enabled && backend_ != nullptr; }

  bool ProcessLidar(
      double timestamp, const StatesGroup &state,
      const std::deque<sensor_msgs::Imu::ConstPtr> &imu_messages,
      double acceleration_scale, const std::vector<PointToPlane> &planes,
      std::size_t raw_feature_count);

  bool ProcessVision(
      double timestamp, const StatesGroup &state,
      const VIOManagerPtr &vio_manager,
      const std::deque<sensor_msgs::Imu::ConstPtr> &imu_messages,
      double acceleration_scale);

  bool feedback_enabled() const { return enabled() && config_.feedback_enabled; }
  bool CollectFeedback(double timestamp, const StatesGroup &frontend_state,
                       NavState *estimate,
                       std::vector<PoseCorrection> *corrections,
                       std::vector<RetiredFrame> *retired);
  std::uint64_t FrameIdAt(double timestamp) const {
    return backend_->findFrameNear(timestamp, config_.merge_time_tolerance);
  }
  std::map<std::uint64_t, NavState> ActiveKeyframeEstimates() const;
  const AdaptiveWindowConfig &config() const { return config_; }
  ExtrinsicState CameraExtrinsicEstimate() const {
    return backend_->CameraExtrinsicEstimate();
  }

private:
  static AdaptiveWindowConfig LoadConfig(ros::NodeHandle &node_handle);
  static NavState ToNavState(double timestamp, const StatesGroup &state);

  MeasurementSummary BuildLidarSummary(
      double timestamp, const StatesGroup &state,
      const std::vector<PointToPlane> &planes,
      std::size_t raw_feature_count,
      LidarPlaneObservationVector *observations) const;
  MeasurementSummary BuildVisionSummary(
      double timestamp, const VIOManagerPtr &vio_manager,
      VisualPatchObservationVector *observations) const;
  void Publish(double timestamp, bool solved, Modality source);
  void WriteDiagnosticCsv(double timestamp, bool solved, Modality source);
  void RecordOptimizedPose(const NavState &estimate);
  void FlushPendingPose();

  AdaptiveWindowConfig config_;
  LidarExtrinsic lidar_extrinsic_;
  ExtrinsicState camera_extrinsic_initial_;
  std::unique_ptr<AdaptiveWindowBackend> backend_;
  std::map<std::uint64_t, NavState> applied_states_;

  ros::Publisher odometry_publisher_;
  ros::Publisher optimized_odometry_publisher_;
  ros::Publisher optimized_path_publisher_;
  ros::Publisher diagnostics_publisher_;
  ros::Publisher extrinsic_publisher_;
  nav_msgs::Path optimized_path_;
  std::ofstream pose_output_stream_;
  std::ofstream diagnostic_csv_stream_;
  NavState pending_pose_;
  bool has_pending_pose_ = false;
};

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_FAST_LIVO_BRIDGE_H_
