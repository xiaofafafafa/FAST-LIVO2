/**
 * @file fast_livo_bridge.h
 * @brief FAST-LIVO2/ROS 与独立后端的边界：转换状态和观测、提供相机回调、发布结果。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#ifndef FAST_LIVO_ADAPTIVE_WINDOW_FAST_LIVO_BRIDGE_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_FAST_LIVO_BRIDGE_H_

#include "adaptive_window/adaptive_window_backend.h"
#include "common_lib.h"
#include "vio.h"
#include "voxel_map.h"

#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64MultiArray.h>

#include <deque>
#include <memory>

namespace fast_livo
{
namespace adaptive_window
{

// ROS 依赖集中于此；核心库可脱离 ROS 构建，桥接调用与前端默认在同一处理线程。
class FastLivoBridge
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 获取前端相机和标定，配置禁用/初始化失败时不创建后端。
  FastLivoBridge(ros::NodeHandle &node_handle, const M3D &rotation_i_l,
                 const V3D &translation_i_l,
                 const VIOManagerPtr &vio_manager);

  bool enabled() const { return config_.enabled && backend_ != nullptr; }

  // 前端 LIO 完成后调用，state 为滤波结果，planes 为本次点面匹配快照。
  bool ProcessLidar(
      double timestamp, const StatesGroup &state,
      const std::deque<sensor_msgs::Imu::ConstPtr> &imu_messages,
      double acceleration_scale, const std::vector<PointToPlane> &planes,
      std::size_t raw_feature_count);

  // 前端 VIO 完成后调用，复制当前图像以保证跨帧残差的图像生命周期。
  bool ProcessVision(
      double timestamp, const StatesGroup &state,
      const VIOManagerPtr &vio_manager,
      const std::deque<sensor_msgs::Imu::ConstPtr> &imu_messages,
      double acceleration_scale);

private:
  // 读取 adaptive_window/* 参数，未设值沿用 types.h 默认配置。
  static AdaptiveWindowConfig LoadConfig(ros::NodeHandle &node_handle);
  // 只复制/规范化状态，未传递前端协方差。
  static NavState ToNavState(double timestamp, const StatesGroup &state);

  // 同时导出策略统计和真实点面观测，二者用途分离。
  MeasurementSummary BuildLidarSummary(
      double timestamp, const StatesGroup &state,
      const std::vector<PointToPlane> &planes,
      std::size_t raw_feature_count,
      LidarPlaneObservationVector *observations) const;
  // 当前原始图像采样 patch，保留稳定 track ID 与前端曝光量。
  MeasurementSummary BuildVisionSummary(
      double timestamp, const VIOManagerPtr &vio_manager,
      VisualPatchObservationVector *observations) const;
  // 发布最新状态、21 项诊断及 T_IC 消息；无反馈，不自动广播 TF。
  void Publish(double timestamp) const;

  AdaptiveWindowConfig config_;
  LidarExtrinsic lidar_extrinsic_;
  ExtrinsicState camera_extrinsic_initial_;
  std::unique_ptr<AdaptiveWindowBackend> backend_;

  ros::Publisher odometry_publisher_;
  ros::Publisher diagnostics_publisher_;
  ros::Publisher extrinsic_publisher_;
};

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_FAST_LIVO_BRIDGE_H_
