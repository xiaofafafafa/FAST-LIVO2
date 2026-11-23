#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/factor/integration_base.h"

namespace fast_livo { namespace adaptive_window {
struct IMUFactor {
  std::shared_ptr<IntegrationBase> pre_integration;
  Eigen::Vector3d gravity;
  template <typename T> bool operator()(const T *pose_i, const T *speedbias_i, const T *pose_j, const T *speedbias_j, T *out) const {
    const Eigen::Matrix<T, 3, 1> dba = Position(speedbias_i + 3) - pre_integration->linearized_ba.cast<T>();
    const Eigen::Matrix<T, 3, 1> dbg = Position(speedbias_i + 6) - pre_integration->linearized_bg.cast<T>();
    const Eigen::Matrix<T, 3, 1> corrected_p = pre_integration->delta_p.cast<T>() +
        pre_integration->jacobian.block<3, 3>(0, 9).cast<T>() * dba + pre_integration->jacobian.block<3, 3>(0, 12).cast<T>() * dbg;
    const Eigen::Matrix<T, 3, 1> corrected_v = pre_integration->delta_v.cast<T>() +
        pre_integration->jacobian.block<3, 3>(6, 9).cast<T>() * dba + pre_integration->jacobian.block<3, 3>(6, 12).cast<T>() * dbg;
    const Eigen::Matrix<T, 3, 1> correction = pre_integration->jacobian.block<3, 3>(3, 12).cast<T>() * dbg;
    const Eigen::Quaternion<T> small(T(1), T(0.5) * correction.x(), T(0.5) * correction.y(), T(0.5) * correction.z());
    const Eigen::Quaternion<T> corrected_q = (pre_integration->delta_q.cast<T>() * small).normalized();
    const auto qi = Rotation(pose_i);
    Eigen::Matrix<T, 15, 1> r;
    r.template segment<3>(0) = qi.conjugate() * (Position(pose_j) - Position(pose_i) - T(pre_integration->sum_dt) * Position(speedbias_i) -
        T(0.5 * pre_integration->sum_dt * pre_integration->sum_dt) * gravity.cast<T>()) - corrected_p;
    Eigen::Quaternion<T> error = corrected_q.conjugate() * qi.conjugate() * Rotation(pose_j);
    if (error.w() < T(0)) error.coeffs() *= T(-1);
    r.template segment<3>(3) = T(2) * error.vec();
    r.template segment<3>(6) = qi.conjugate() * (Position(speedbias_j) - Position(speedbias_i) - T(pre_integration->sum_dt) * gravity.cast<T>()) - corrected_v;
    r.template segment<3>(9) = Position(speedbias_j + 3) - Position(speedbias_i + 3);
    r.template segment<3>(12) = Position(speedbias_j + 6) - Position(speedbias_i + 6);
    Eigen::Map<Eigen::Matrix<T, 15, 1>> result(out);
    result = pre_integration->sqrt_information.cast<T>() * r;
    return true;
  }
};


}}
