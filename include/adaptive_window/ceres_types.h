/**
 * @file ceres_types.h
 * @brief 参数键、环境/切空间维度与可自动微分的位姿运算；注意各状态排列的区别。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once

#include "adaptive_window/math_utils.h"
#include <ceres/ceres.h>
#include <array>
#include <memory>
#include <tuple>

namespace fast_livo { namespace adaptive_window {

// 环境维度为 7/9/1；位姿切空间是 6 维，顺序 [dp_world,dtheta_body]。
constexpr int SIZE_POSE = 7;
constexpr int SIZE_SPEEDBIAS = 9;
constexpr int SIZE_FEATURE = 1;

// VINS-compatible storage: pose [p, q_xyzw], speed/bias [v, ba, bg].
// Pose tangent is [dp_world, dtheta_body], NOT the policy's [rotation, position].
// 种类+ID 唯一定位参数；相同整数可分别作为帧 ID 和 track ID 而不冲突。
enum class BlockKind { kPose, kSpeedBias, kExtrinsic, kDepth };
// 构图与边缘化共享尺寸约定，is_pose 同时包含导航位姿和相机外参。
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
// 导航 pose/motion 共用帧 ID；depth 用 track ID；唯一外参使用固定键。
inline BlockKey PoseKey(std::uint64_t id) { return {BlockKind::kPose, id}; }
inline BlockKey MotionKey(std::uint64_t id) { return {BlockKind::kSpeedBias, id}; }
inline BlockKey DepthKey(std::uint64_t id) { return {BlockKind::kDepth, id}; }
inline BlockKey ExtrinsicKey() { return {BlockKind::kExtrinsic, 0}; }

// 内存按 q_xyzw，Eigen 构造按 q_wxyz；T 可为 double 或 Jet，不要强转丢失导数。
template <typename T> Eigen::Quaternion<T> Rotation(const T *x) {
  return Eigen::Quaternion<T>(x[6], x[3], x[4], x[5]);
}
// 读连续三个元素，也用于 speed-bias 的 v/ba/bg 子段。
template <typename T> Eigen::Matrix<T, 3, 1> Position(const T *x) {
  return Eigen::Map<const Eigen::Matrix<T, 3, 1>>(x);
}
// 单位四元数最短支 Log，输出弧度向量；统一 q/-q 符号，小角度用 2*vec。
template <typename T> Eigen::Matrix<T, 3, 1> QuaternionLog(Eigen::Quaternion<T> q) {
  if (q.w() < T(0)) q.coeffs() *= T(-1);
  const T s2 = q.vec().squaredNorm();
  if (s2 < T(1e-16)) return T(2) * q.vec();
  using std::sqrt;
  using std::atan2;
  const T s = sqrt(s2);
  return (T(2) * atan2(s, q.w()) / s) * q.vec();
}

// 按 Ceres 排列打包并归一化四元数，输入需非零且有限。
inline std::vector<double> PackPose(const Eigen::Vector3d &p, const Eigen::Quaterniond &input) {
  const auto q = input.normalized();
  return {p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w()};
}


}} // namespace fast_livo::adaptive_window
