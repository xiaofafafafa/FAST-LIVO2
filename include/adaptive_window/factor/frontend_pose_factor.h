/**
 * @file frontend_pose_factor.h
 * @brief 滤波位姿正则：使优化结果在 FAST-LIVO2 快照附近微调；该项与原始观测信息相关。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"

namespace fast_livo { namespace adaptive_window {

// The filter and these factors use overlapping measurements. This is a tunable
// regularizer around FAST-LIVO2, not an independent pose sensor likelihood.
// Its reference is a frontend snapshot and must never follow the optimized pose.
struct FrontendPoseFactor {
  // 前端世界系位置快照，不能被本次优化结果覆盖。
  Eigen::Vector3d p_w_i;
  // 单位滤波旋转快照；同时间 VIO 更新由 Estimator 替换整份因子。
  Eigen::Quaterniond q_w_i;
  // 平移尺度单位 m，旋转尺度单位 rad；额外权重在 cost 外通过 ScaledLoss 施加。
  double translation_sigma, rotation_sigma;
  // r=[(p-p_front)/sigma_p, Log(q_front^-1*q)/sigma_R]，位姿扰动与 Ceres 流形一致。
  template <typename T> bool operator()(const T *pose, T *out) const {
    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(out);
    r.template head<3>() = (Position(pose) - p_w_i.cast<T>()) / T(translation_sigma);
    r.template tail<3>() = QuaternionLog(q_w_i.cast<T>().conjugate() * Rotation(pose)) / T(rotation_sigma);
    return true;
  }
};

}}
