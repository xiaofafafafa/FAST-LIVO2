#pragma once
#include "adaptive_window/ceres_types.h"

namespace fast_livo { namespace adaptive_window {
struct LidarFactor {
  Eigen::Vector3d point_i, normal_w, center_w;
  double inverse_sigma;
  template <typename T> bool operator()(const T *pose, T *r) const {
    const Eigen::Matrix<T, 3, 1> point_w = Rotation(pose) * point_i.cast<T>() + Position(pose);
    r[0] = T(inverse_sigma) * normal_w.cast<T>().dot(point_w - center_w.cast<T>());
    return true;
  }
};

}}
