#pragma once
#include "adaptive_window/types.h"

namespace fast_livo { namespace adaptive_window {

// Independent midpoint implementation of the VINS-Mono preintegration model.
// Error/covariance order [delta_p, dtheta, delta_v, dba, dbg], in the initial IMU frame.
class IntegrationBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Matrix15 = Eigen::Matrix<double, 15, 15>;
  explicit IntegrationBase(const AdaptiveWindowConfig &config) : config_(config) {}
  bool integrate(const ImuSampleVector &samples, double start, double end,
                 const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);
  bool repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);
  bool needsRepropagation(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg) const;
  double sum_dt = 0.0;
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero(), delta_v = Eigen::Vector3d::Zero();
  Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d linearized_ba = Eigen::Vector3d::Zero(), linearized_bg = Eigen::Vector3d::Zero();
  Matrix15 covariance = Matrix15::Zero(), jacobian = Matrix15::Identity();
  Matrix15 sqrt_information = Matrix15::Zero();
 private:
  void midPointIntegration(const ImuSample &a, const ImuSample &b);
  AdaptiveWindowConfig config_;
  ImuSampleVector imu_buf;
};
}}
