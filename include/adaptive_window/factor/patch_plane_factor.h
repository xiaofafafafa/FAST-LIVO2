#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/factor/measurement_types.h"

namespace fast_livo { namespace adaptive_window {
struct PatchPlaneFactor {
  bool same_pose;
  Eigen::Vector3d ray;
  LidarExtrinsic lidar;
  LocalPlane plane;
  template <typename T> bool operator()(T const *const *x, T *r) const {
    const T *ext = x[same_pose ? 0 : 2];
    const T rho = x[same_pose ? 1 : 3][0];
    if (rho <= T(0)) return false;
    Eigen::Matrix<T, 3, 1> point_i = Rotation(ext).conjugate() * (ray.cast<T>() / rho - Position(ext));
    if (!same_pose) point_i = (Rotation(x[1]).conjugate() *
        (Rotation(x[0]) * point_i + Position(x[0]) - Position(x[1]))).eval();
    const Eigen::Matrix<T, 3, 1> point_l = lidar.q_i_l.cast<T>().conjugate() * (point_i - lidar.p_i_l.cast<T>());
    r[0] = plane.normal_l.cast<T>().dot(point_l - plane.center_l.cast<T>()) / T(plane.sigma);
    return true;
  }
};

}}
