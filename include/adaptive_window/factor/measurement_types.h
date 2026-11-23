#pragma once
#include "adaptive_window/types.h"
#include <functional>

namespace fast_livo { namespace adaptive_window {
using ProjectionFunction = std::function<bool(const Eigen::Vector3d &, Eigen::Vector2d *,
                                             Eigen::Matrix<double, 2, 3> *)>;
using ImageSampleFunction = std::function<bool(const Eigen::Vector2d &, double *, Eigen::Vector2d *)>;
struct LocalPlane {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center_l = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_l = Eigen::Vector3d::UnitZ();
  double sigma = 0.05;
};

}}
