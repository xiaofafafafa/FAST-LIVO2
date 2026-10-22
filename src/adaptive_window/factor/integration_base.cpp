#include "adaptive_window/factor/integration_base.h"
#include "adaptive_window/math_utils.h"
#include <Eigen/Cholesky>
#include <algorithm>

namespace fast_livo { namespace adaptive_window {
namespace {
ImuSample Interpolate(const ImuSample &a, const ImuSample &b, double t) {
  const double u = (t - a.timestamp) / (b.timestamp - a.timestamp);
  ImuSample out;
  out.timestamp = t;
  out.acceleration = (1.0 - u) * a.acceleration + u * b.acceleration;
  out.angular_velocity = (1.0 - u) * a.angular_velocity + u * b.angular_velocity;
  return out;
}
}

bool IntegrationBase::integrate(const ImuSampleVector &samples, double start,
                                 double end, const Eigen::Vector3d &ba,
                                 const Eigen::Vector3d &bg) {
  imu_buf.clear();
  if (!std::isfinite(start) || !std::isfinite(end) || end <= start || samples.size() < 2 ||
      samples.front().timestamp > start || samples.back().timestamp < end) return false;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    const auto &a = samples[i - 1];
    const auto &b = samples[i];
    if (!std::isfinite(a.timestamp) || !std::isfinite(b.timestamp) ||
        b.timestamp <= a.timestamp || !a.acceleration.allFinite() ||
        !b.acceleration.allFinite() || !a.angular_velocity.allFinite() ||
        !b.angular_velocity.allFinite()) return false;
    if (b.timestamp <= start || a.timestamp >= end) continue;
    if (b.timestamp - a.timestamp > config_.imu_max_gap) return false;
    const double t0 = std::max(start, a.timestamp), t1 = std::min(end, b.timestamp);
    if (imu_buf.empty()) imu_buf.push_back(Interpolate(a, b, t0));
    imu_buf.push_back(Interpolate(a, b, t1));
  }
  if (imu_buf.size() < 2) return false;
  return repropagate(ba, bg);
}

bool IntegrationBase::needsRepropagation(const Eigen::Vector3d &ba,
                                          const Eigen::Vector3d &bg) const {
  return (ba - linearized_ba).norm() > config_.imu_repropagate_acc_bias ||
         (bg - linearized_bg).norm() > config_.imu_repropagate_gyr_bias;
}

bool IntegrationBase::repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg) {
  if (imu_buf.size() < 2 || !ba.allFinite() || !bg.allFinite()) return false;
  sum_dt = 0.0; delta_p.setZero(); delta_v.setZero(); delta_q.setIdentity();
  covariance.setZero(); jacobian.setIdentity();
  linearized_ba = ba; linearized_bg = bg;
  for (std::size_t i = 1; i < imu_buf.size(); ++i) midPointIntegration(imu_buf[i - 1], imu_buf[i]);
  // Tiny numerical floor, not a motion prior. Avoid forming covariance.inverse().
  Matrix15 regularized = 0.5 * (covariance + covariance.transpose());
  regularized.diagonal().array() += 1e-12;
  Eigen::LLT<Matrix15> llt(regularized);
  if (llt.info() != Eigen::Success) return false;
  sqrt_information = llt.matrixL().solve(Matrix15::Identity());
  return sqrt_information.allFinite() && delta_p.allFinite() && delta_v.allFinite();
}

void IntegrationBase::midPointIntegration(const ImuSample &a, const ImuSample &b) {
  const double h = b.timestamp - a.timestamp, h2 = h * h;
  const Eigen::Vector3d w = 0.5 * (a.angular_velocity + b.angular_velocity) - linearized_bg;
  const Eigen::Vector3d a0 = a.acceleration - linearized_ba, a1 = b.acceleration - linearized_ba;
  const Eigen::Matrix3d r0 = delta_q.toRotationMatrix();
  const Eigen::Quaterniond step(1.0, 0.5 * h * w.x(), 0.5 * h * w.y(), 0.5 * h * w.z());
  const Eigen::Quaterniond next = (delta_q * step).normalized();
  const Eigen::Matrix3d r1 = next.toRotationMatrix(), identity = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d acceleration = 0.5 * (r0 * a0 + r1 * a1);
  delta_p += delta_v * h + 0.5 * acceleration * h2;
  delta_v += acceleration * h;
  delta_q = next; sum_dt += h;

  // First-order error dynamics of the midpoint rule; endpoint gyro noise also
  // perturbs the rotated endpoint acceleration. Discrete noise convention
  // matches VINS (the four stddev parameters are NOT continuous densities).
  const Eigen::Matrix3d attitude = identity - Skew(w) * h;
  const Eigen::Matrix3d da_dtheta = -0.5 * (r0 * Skew(a0) + r1 * Skew(a1) * attitude);
  const Eigen::Matrix3d da_dba = -0.5 * (r0 + r1);
  const Eigen::Matrix3d da_dbg = 0.5 * r1 * Skew(a1) * h;
  Matrix15 f = Matrix15::Identity();
  f.block<3, 3>(0, 3) = 0.5 * h2 * da_dtheta;
  f.block<3, 3>(0, 6) = h * identity;
  f.block<3, 3>(0, 9) = 0.5 * h2 * da_dba;
  f.block<3, 3>(0, 12) = 0.5 * h2 * da_dbg;
  f.block<3, 3>(3, 3) = attitude;
  f.block<3, 3>(3, 12) = -h * identity;
  f.block<3, 3>(6, 3) = h * da_dtheta;
  f.block<3, 3>(6, 9) = h * da_dba;
  f.block<3, 3>(6, 12) = h * da_dbg;

  Eigen::Matrix<double, 15, 18> g = Eigen::Matrix<double, 15, 18>::Zero();
  for (int endpoint = 0; endpoint < 2; ++endpoint) {
    const int acc = 6 * endpoint, gyr = acc + 3;
    const Eigen::Matrix3d r = endpoint == 0 ? r0 : r1;
    g.block<3, 3>(0, acc) = 0.25 * h2 * r;
    g.block<3, 3>(6, acc) = 0.5 * h * r;
    g.block<3, 3>(3, gyr) = 0.5 * h * identity;
    g.block<3, 3>(0, gyr) = -0.125 * h2 * h * r1 * Skew(a1);
    g.block<3, 3>(6, gyr) = -0.25 * h2 * r1 * Skew(a1);
  }
  g.block<3, 3>(9, 12) = h * identity;
  g.block<3, 3>(12, 15) = h * identity;
  Eigen::Matrix<double, 18, 1> noise;
  noise.segment<3>(0).setConstant(config_.imu_acc_noise * config_.imu_acc_noise);
  noise.segment<3>(3).setConstant(config_.imu_gyr_noise * config_.imu_gyr_noise);
  noise.segment<6>(6) = noise.head<6>();
  noise.segment<3>(12).setConstant(config_.imu_acc_bias_rw * config_.imu_acc_bias_rw);
  noise.segment<3>(15).setConstant(config_.imu_gyr_bias_rw * config_.imu_gyr_bias_rw);
  jacobian = (f * jacobian).eval();
  covariance = (f * covariance * f.transpose() + g * noise.asDiagonal() * g.transpose()).eval();
}
}}
