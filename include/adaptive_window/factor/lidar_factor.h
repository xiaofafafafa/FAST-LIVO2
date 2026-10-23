/**
 * @file lidar_factor.h
 * @brief LiDAR 点到世界系平面的距离残差；关联及平面固定，点位置随优化位姿重新计算。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"

namespace fast_livo { namespace adaptive_window {
struct LidarFactor {
  // point_i 已由固定 T_IL 变换到 IMU 系；法向和中心仍在世界系。
  Eigen::Vector3d point_i, normal_w, center_w;
  // 1/sqrt(观测方差)，将距离残差无量纲化。
  double inverse_sigma;
  template <typename T> bool operator()(const T *pose, T *r) const {
    // 每次迭代用当前 T_WI 重新投影点，再计算 n_W^T*(point_W-center_W)。
    const Eigen::Matrix<T, 3, 1> point_w = Rotation(pose) * point_i.cast<T>() + Position(pose);
    r[0] = T(inverse_sigma) * normal_w.cast<T>().dot(point_w - center_w.cast<T>());
    return true;
  }
};

}}
