/**
 * @file adaptive_window_backend.h
 * @brief 旧桥接 API 到 Estimator 的薄适配器，所有状态管理和优化都由基类完成。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/estimator.h"

namespace fast_livo { namespace adaptive_window {

// Compatibility adapter for the existing FAST-LIVO2 ROS bridge.
// All state, residual construction, optimization and marginalization live in
// Estimator. New integrations should use its VINS-style methods directly.
class AdaptiveWindowBackend : public Estimator {
 public:
  // 继承真正的后端构造，下面仅保留旧桥接层所用的大写函数名。
  using Estimator::Estimator;
  // 标定/回调设置后由 Estimator 重建窗口。
  void Initialize(const LidarExtrinsic &l, const ExtrinsicState &e, ProjectionFunction p) {
    setParameter(l, e, std::move(p));
  }
  // 对外显式重置会清空计数和 IMU，与自动 recover 的保留行为不同。
  void Reset() { clearState(); }
  // 数据包转发到 processIMU，未在适配层二次积分。
  void AddImuSamples(const ImuSampleVector &s) { processIMU(s); }
  // 转发点面观测；bool 语义与 Estimator::processLidar 一致。
  bool ProcessLidar(const NavState &s, const MeasurementSummary &q, const LidarPlaneObservationVector &o) {
    return processLidar(s, q, o);
  }
  // 转发当前图像采样回调及原始 patch，图优化完全由 Estimator 完成。
  bool ProcessVision(const NavState &s, const MeasurementSummary &q, const VisualPatchObservationVector &o,
                     const ImageSampleFunction &image) { return processImage(s, q, o, image); }
  bool LatestEstimate(NavState *s) const { return latestEstimate(s); }
  ExtrinsicState CameraExtrinsicEstimate() const { return cameraExtrinsicEstimate(); }
};
}}
