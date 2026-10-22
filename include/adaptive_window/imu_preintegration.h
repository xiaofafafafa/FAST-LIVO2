#pragma once
#include "adaptive_window/types.h"

namespace fast_livo { namespace adaptive_window {

// Independent midpoint implementation of the VINS-Mono preintegration model.
// Error/covariance order [dp, dtheta, dv, dba, dbg], in the initial IMU frame.
class ImuPreintegration {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Matrix15 = Eigen::Matrix<double, 15, 15>;
  explicit ImuPreintegration(const AdaptiveWindowConfig &config) : config_(config) {}
  bool Integrate(const ImuSampleVector &samples, double start, double end,
                 const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);
  bool Repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);
  bool NeedsRepropagation(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg) const;
  double dt = 0.0;
  Eigen::Vector3d dp = Eigen::Vector3d::Zero(), dv = Eigen::Vector3d::Zero();
  Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linearized_ba = Eigen::Vector3d::Zero(), linearized_bg = Eigen::Vector3d::Zero();
  Matrix15 covariance = Matrix15::Zero(), jacobian = Matrix15::Identity();
  Matrix15 sqrt_information = Matrix15::Zero();
 private:
  void Step(const ImuSample &a, const ImuSample &b);
  AdaptiveWindowConfig config_;
  ImuSampleVector interval_;
};
}}
