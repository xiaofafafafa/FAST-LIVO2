/**
 * @file integration_base.cpp
 * @brief 裁剪实际 IMU 时间区间，执行中值积分、误差传播和协方差白化；端点必须有数据包围。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#include "adaptive_window/factor/integration_base.h"
#include "adaptive_window/math_utils.h"
#include <Eigen/Cholesky>
#include <algorithm>

namespace fast_livo { namespace adaptive_window {
namespace {
// 两个原始样本间按时间线性插值；调用者先保证 b.timestamp>a.timestamp。
ImuSample Interpolate(const ImuSample &a, const ImuSample &b, double t) {
  const double u = (t - a.timestamp) / (b.timestamp - a.timestamp);
  ImuSample out;
  out.timestamp = t;
  out.acceleration = (1.0 - u) * a.acceleration + u * b.acceleration;
  out.angular_velocity = (1.0 - u) * a.angular_velocity + u * b.angular_velocity;
  return out;
}
}

// 只保存 [start,end] 区间及插值端点，避免不同步帧间使用错误的积分时长。
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
    // 无重叠区间跳过；有重叠时按原始采样间隔检查缺测，不能用插值掩盖长间断。
    if (b.timestamp <= start || a.timestamp >= end) continue;
    if (b.timestamp - a.timestamp > config_.imu_max_gap) return false;
    // 区间首端仅放一次，之后每个小段只追加右端，保证时间严格递增。
    const double t0 = std::max(start, a.timestamp), t1 = std::min(end, b.timestamp);
    if (imu_buf.empty()) imu_buf.push_back(Interpolate(a, b, t0));
    imu_buf.push_back(Interpolate(a, b, t1));
  }
  if (imu_buf.size() < 2) return false;
  return repropagate(ba, bg);
}

// 判断偏置一阶修正是否仍处于允许范围；阈值之外使用保存数据整段重积分。
bool IntegrationBase::needsRepropagation(const Eigen::Vector3d &ba,
                                          const Eigen::Vector3d &bg) const {
  return (ba - linearized_ba).norm() > config_.imu_repropagate_acc_bias ||
         (bg - linearized_bg).norm() > config_.imu_repropagate_gyr_bias;
}

// 从零开始重置增量/协方差和单位 Jacobian，不在旧积分结果上累加第二遍。
bool IntegrationBase::repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg) {
  if (imu_buf.size() < 2 || !ba.allFinite() || !bg.allFinite()) return false;
  sum_dt = 0.0; delta_p.setZero(); delta_v.setZero(); delta_q.setIdentity();
  covariance.setZero(); jacobian.setIdentity();
  linearized_ba = ba; linearized_bg = bg;
  for (std::size_t i = 1; i < imu_buf.size(); ++i) midPointIntegration(imu_buf[i - 1], imu_buf[i]);
  // Tiny numerical floor, not a motion prior. Avoid forming covariance.inverse().
  // 对称化并加极小数值对角，改善 Cholesky 可分解性，不是额外运动约束。
  Matrix15 regularized = 0.5 * (covariance + covariance.transpose());
  regularized.diagonal().array() += 1e-12;
  Eigen::LLT<Matrix15> llt(regularized);
  if (llt.info() != Eigen::Success) return false;
  // 若 Sigma=L*L^T，则 W=L^-1 满足 W^T*W=Sigma^-1；用三角求解避免显式逆。
  sqrt_information = llt.matrixL().solve(Matrix15::Identity());
  return sqrt_information.allFinite() && delta_p.allFinite() && delta_v.allFinite();
}

// 一个中值步同时更新名义增量与一阶误差模型；不在积分器中减去世界系重力。
void IntegrationBase::midPointIntegration(const ImuSample &a, const ImuSample &b) {
  const double h = b.timestamp - a.timestamp, h2 = h * h;
  // 角速度取两端均值减偏置；加速度端点分别减同一线性化偏置。
  const Eigen::Vector3d w = 0.5 * (a.angular_velocity + b.angular_velocity) - linearized_bg;
  const Eigen::Vector3d a0 = a.acceleration - linearized_ba, a1 = b.acceleration - linearized_ba;
  const Eigen::Matrix3d r0 = delta_q.toRotationMatrix();
  // VINS 风格一阶四元数增量，经归一化后获得下一端点旋转。
  const Eigen::Quaterniond step(1.0, 0.5 * h * w.x(), 0.5 * h * w.y(), 0.5 * h * w.z());
  const Eigen::Quaterniond next = (delta_q * step).normalized();
  const Eigen::Matrix3d r1 = next.toRotationMatrix(), identity = Eigen::Matrix3d::Identity();
  // 两端加速度分别旋转到积分起点 IMU 系后求均值，再更新 Δp、Δv。
  const Eigen::Vector3d acceleration = 0.5 * (r0 * a0 + r1 * a1);
  delta_p += delta_v * h + 0.5 * acceleration * h2;
  delta_v += acceleration * h;
  delta_q = next; sum_dt += h;

  // First-order error dynamics of the midpoint rule; endpoint gyro noise also
  // perturbs the rotated endpoint acceleration. Discrete noise convention
  // matches VINS (the four stddev parameters are NOT continuous densities).
  // F 的误差排列 [dp,dtheta,dv,dba,dbg]；角速度误差还会影响终点加速度的旋转。
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

  // G 的 18 列：起点 acc/gyr 六维、终点 acc/gyr 六维、ba/bg 随机游走六维。
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
  // Q 对角由配置标准差的平方构成，当前传播已显式包含 dt 因子。
  Eigen::Matrix<double, 18, 1> noise;
  noise.segment<3>(0).setConstant(config_.imu_acc_noise * config_.imu_acc_noise);
  noise.segment<3>(3).setConstant(config_.imu_gyr_noise * config_.imu_gyr_noise);
  noise.segment<6>(6) = noise.head<6>();
  noise.segment<3>(12).setConstant(config_.imu_acc_bias_rw * config_.imu_acc_bias_rw);
  noise.segment<3>(15).setConstant(config_.imu_gyr_bias_rw * config_.imu_gyr_bias_rw);
  // J←FJ，Sigma←F*Sigma*F^T+G*Q*G^T；eval 防止 Eigen 表达式原位别名问题。
  jacobian = (f * jacobian).eval();
  covariance = (f * covariance * f.transpose() + g * noise.asDiagonal() * g.transpose()).eval();
}
}}
