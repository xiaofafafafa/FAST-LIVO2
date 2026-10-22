#pragma once

#include "adaptive_window/math_utils.h"
#include <ceres/ceres.h>
#include <array>
#include <memory>
#include <tuple>

namespace fast_livo { namespace adaptive_window {

constexpr int SIZE_POSE = 7;
constexpr int SIZE_SPEEDBIAS = 9;
constexpr int SIZE_FEATURE = 1;

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
  int size() const { return is_pose() ? SIZE_POSE : (kind == BlockKind::kSpeedBias ? SIZE_SPEEDBIAS : SIZE_FEATURE); }
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

inline std::vector<double> PackPose(const Eigen::Vector3d &p, const Eigen::Quaterniond &input) {
  const auto q = input.normalized();
  return {p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w()};
}


}} // namespace fast_livo::adaptive_window
