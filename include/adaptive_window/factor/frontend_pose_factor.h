#pragma once
#include "adaptive_window/ceres_types.h"

namespace fast_livo { namespace adaptive_window {

// The filter and these factors use overlapping measurements. This is a tunable
// regularizer around FAST-LIVO2, not an independent pose sensor likelihood.
// Its reference is a frontend snapshot and must never follow the optimized pose.
struct FrontendPoseFactor {
  Eigen::Vector3d p_w_i;
  Eigen::Quaterniond q_w_i;
  double translation_sigma, rotation_sigma;
  template <typename T> bool operator()(const T *pose, T *out) const {
    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(out);
    r.template head<3>() = (Position(pose) - p_w_i.cast<T>()) / T(translation_sigma);
    r.template tail<3>() = QuaternionLog(q_w_i.cast<T>().conjugate() * Rotation(pose)) / T(rotation_sigma);
    return true;
  }
};

}}
