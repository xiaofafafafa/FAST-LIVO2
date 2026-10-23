/**
 * @file initial_state_factor.h
 * @brief 首次状态和外参先验；启用逐帧滤波位姿正则时，首次状态不再额外约束位姿。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"

namespace fast_livo { namespace adaptive_window {
// 锚定首次速度/偏置，并在关闭逐帧滤波位姿正则时锚定首次位姿。
struct InitialStateFactor {
  NavState reference;
  AdaptiveWindowConfig config;
  template <typename T> bool operator()(const T *pose, const T *motion, T *out) const {
    Eigen::Map<Eigen::Matrix<T, 15, 1>> r(out);
    r.template segment<3>(0) = (Position(pose) - reference.p_w_i.cast<T>()) / T(config.pose_prior_translation_sigma);
    r.template segment<3>(3) = QuaternionLog(reference.q_w_i.cast<T>().conjugate() * Rotation(pose)) /
                               T(Radians(config.pose_prior_rotation_deg));
    // Each frontend pose is already constrained separately when enabled.
    // 每个状态已含一个滤波位姿正则时，删除首次状态的额外 6 维位姿贡献。
    if (config.frontend_pose_prior_enabled) r.template head<6>().setZero();
    // speed-bias 连续内存顺序为 v、ba、bg。
    r.template segment<3>(6) = (Position(motion) - reference.v_w_i.cast<T>()) / T(config.velocity_prior_sigma);
    r.template segment<3>(9) = (Position(motion + 3) - reference.bias_a.cast<T>()) / T(config.bias_prior_sigma);
    r.template segment<3>(12) = (Position(motion + 6) - reference.bias_g.cast<T>()) / T(config.bias_prior_sigma);
    return true;
  }
};
// T_CI 相对初始标定的软正则；外参固定模式下该块被 Ceres 设为常量。
struct ExtrinsicPriorFactor {
  ExtrinsicState reference;
  AdaptiveWindowConfig config;
  template <typename T> bool operator()(const T *x, T *out) const {
    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(out);
    r.template head<3>() = (Position(x) - reference.p_c_i.cast<T>()) / T(config.extrinsic_translation_sigma);
    r.template tail<3>() = QuaternionLog(reference.q_c_i.cast<T>().conjugate() * Rotation(x)) /
                          T(Radians(config.extrinsic_rotation_deg));
    return true;
  }
};

}}
