#pragma once

#include "adaptive_window/math_utils.h"
#include <ceres/ceres.h>
#include <array>
#include <memory>
#include <tuple>

namespace fast_livo { namespace adaptive_window {

// VINS-compatible storage: pose [p, q_xyzw], speed/bias [v, ba, bg].
// Pose tangent is [dp_world, dtheta_body], NOT the policy's [rotation, position].
enum class BlockKind { kPose, kSpeedBias, kExtrinsic, kDepth };
struct BlockKey {
  BlockKind kind;
  std::uint64_t id;
  bool operator<(const BlockKey &b) const {
    return std::tie(kind, id) < std::tie(b.kind, b.id);
  }
  bool operator==(const BlockKey &b) const { return kind == b.kind && id == b.id; }
  bool is_pose() const { return kind == BlockKind::kPose || kind == BlockKind::kExtrinsic; }
  int size() const { return is_pose() ? 7 : (kind == BlockKind::kSpeedBias ? 9 : 1); }
  int local_size() const { return is_pose() ? 6 : size(); }
};
inline BlockKey PoseKey(std::uint64_t id) { return {BlockKind::kPose, id}; }
inline BlockKey MotionKey(std::uint64_t id) { return {BlockKind::kSpeedBias, id}; }
inline BlockKey DepthKey(std::uint64_t id) { return {BlockKind::kDepth, id}; }
inline BlockKey ExtrinsicKey() { return {BlockKind::kExtrinsic, 0}; }

template <typename T> Eigen::Quaternion<T> Rotation(const T *x) {
  return Eigen::Quaternion<T>(x[6], x[3], x[4], x[5]);
}
template <typename T> Eigen::Matrix<T, 3, 1> Position(const T *x) {
  return Eigen::Map<const Eigen::Matrix<T, 3, 1>>(x);
}
template <typename T> Eigen::Matrix<T, 3, 1> QuaternionLog(Eigen::Quaternion<T> q) {
  if (q.w() < T(0)) q.coeffs() *= T(-1);
  const T s2 = q.vec().squaredNorm();
  if (s2 < T(1e-16)) return T(2) * q.vec();
  using std::sqrt;
  using std::atan2;
  const T s = sqrt(s2);
  return (T(2) * atan2(s, q.w()) / s) * q.vec();
}

class PoseManifold final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 6; }
  bool Plus(const double *x, const double *d, double *out) const override {
    Eigen::Map<Eigen::Vector3d> p(out);
    Eigen::Map<Eigen::Quaterniond> q(out + 3);
    p = Position(x) + Position(d);
    q = (Rotation(x) * ExpQuaternion(Eigen::Map<const Eigen::Vector3d>(d + 3))).normalized();
    return p.allFinite() && q.coeffs().allFinite();
  }
  bool PlusJacobian(const double *x, double *out) const override {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(out);
    const auto q = Rotation(x);
    j.setZero();
    j.topLeftCorner<3, 3>().setIdentity();
    j.block<3, 3>(3, 3) = 0.5 * (q.w() * Eigen::Matrix3d::Identity() + Skew(q.vec()));
    j.block<1, 3>(6, 3) = -0.5 * q.vec().transpose();
    return true;
  }
  bool Minus(const double *y, const double *x, double *out) const override {
    Eigen::Map<Eigen::Vector3d> p(out), r(out + 3);
    p = Position(y) - Position(x);
    r = LogQuaternion(Rotation(x).conjugate() * Rotation(y));
    return p.allFinite() && r.allFinite();
  }
  bool MinusJacobian(const double *x, double *out) const override {
    Eigen::Matrix<double, 7, 6, Eigen::RowMajor> plus;
    PlusJacobian(x, plus.data());
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(out);
    j = plus.transpose();
    j.bottomRows<3>() *= 4.0;
    return true;
  }
};

inline std::vector<double> PackPose(const Eigen::Vector3d &p, const Eigen::Quaterniond &input) {
  const auto q = input.normalized();
  return {p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w()};
}

class ImuPreintegration;
struct NonlinearFactor {
  std::vector<BlockKey> keys;
  std::shared_ptr<ceres::CostFunction> cost;
  Modality modality = Modality::kPrior;
  double huber_delta = 0.0;
  double confidence = 1.0;
  std::shared_ptr<ImuPreintegration> preintegration;
};

}} // namespace fast_livo::adaptive_window
