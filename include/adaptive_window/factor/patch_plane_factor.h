/**
 * @file patch_plane_factor.h
 * @brief 参考 patch 中心逆深度与相邻 LiDAR 帧局部平面的几何一致性残差。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/factor/measurement_types.h"

namespace fast_livo { namespace adaptive_window {
struct PatchPlaneFactor {
  // 视觉参考与 LiDAR 关联为同一状态时，两次导航变换抵消，不能重复注册 pose 块。
  bool same_pose;
  // 视觉中心射线 z=1，ray/rho 得到参考相机系三维点。
  Eigen::Vector3d ray;
  LidarExtrinsic lidar;
  LocalPlane plane;
  template <typename T> bool operator()(T const *const *x, T *r) const {
    // 同状态参数为 [T_CI,rho]；不同状态为 [pose_anchor,pose_lidar,T_CI,rho]。
    const T *ext = x[same_pose ? 0 : 2];
    const T rho = x[same_pose ? 1 : 3][0];
    if (rho <= T(0)) return false;
    // T_IC 将相机点转入参考 IMU；必要时再经世界系转入关联 LiDAR 时刻的 IMU。
    Eigen::Matrix<T, 3, 1> point_i = Rotation(ext).conjugate() * (ray.cast<T>() / rho - Position(ext));
    if (!same_pose) point_i = (Rotation(x[1]).conjugate() *
        (Rotation(x[0]) * point_i + Position(x[0]) - Position(x[1]))).eval();
    // 固定 T_LI 转到关联扫描的 LiDAR 系，与该扫描拟合的局部平面比较。
    const Eigen::Matrix<T, 3, 1> point_l = lidar.q_i_l.cast<T>().conjugate() * (point_i - lidar.p_i_l.cast<T>());
    // 仅约束 patch 中心的点面距离，未实现完整 patch 法向对齐残差。
    r[0] = plane.normal_l.cast<T>().dot(point_l - plane.center_l.cast<T>()) / T(plane.sigma);
    return true;
  }
};

}}
