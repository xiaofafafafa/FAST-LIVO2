/**
 * @file imu_factor.h
 * @brief 15 维 IMU 预积分残差：位置、姿态、速度、加速度偏置、陀螺仪偏置，最后统一白化。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/factor/integration_base.h"

namespace fast_livo { namespace adaptive_window {
struct IMUFactor {
  // 与 Estimator 的 IMU 边共享同一积分结果，重积分后下一次求值自动使用更新值。
  std::shared_ptr<IntegrationBase> pre_integration;
  // 世界系重力，默认负 z；预测位置/速度中显式减去重力贡献。
  Eigen::Vector3d gravity;
  // 四个参数块分别为 [p_i,q_i]、[v_i,ba_i,bg_i]、[p_j,q_j]、[v_j,ba_j,bg_j]。
  template <typename T> bool operator()(const T *pose_i, const T *speedbias_i, const T *pose_j, const T *speedbias_j, T *out) const {
    // 先计算偏置相对线性化基点的变化，再校正 Δp、Δv、Δq。
    const Eigen::Matrix<T, 3, 1> dba = Position(speedbias_i + 3) - pre_integration->linearized_ba.cast<T>();
    const Eigen::Matrix<T, 3, 1> dbg = Position(speedbias_i + 6) - pre_integration->linearized_bg.cast<T>();
    const Eigen::Matrix<T, 3, 1> corrected_p = pre_integration->delta_p.cast<T>() +
        pre_integration->jacobian.block<3, 3>(0, 9).cast<T>() * dba + pre_integration->jacobian.block<3, 3>(0, 12).cast<T>() * dbg;
    const Eigen::Matrix<T, 3, 1> corrected_v = pre_integration->delta_v.cast<T>() +
        pre_integration->jacobian.block<3, 3>(6, 9).cast<T>() * dba + pre_integration->jacobian.block<3, 3>(6, 12).cast<T>() * dbg;
    const Eigen::Matrix<T, 3, 1> correction = pre_integration->jacobian.block<3, 3>(3, 12).cast<T>() * dbg;
    // 小角度一阶旋转修正；变化较大时 Estimator 会先触发整段重新积分。
    const Eigen::Quaternion<T> small(T(1), T(0.5) * correction.x(), T(0.5) * correction.y(), T(0.5) * correction.z());
    const Eigen::Quaternion<T> corrected_q = (pre_integration->delta_q.cast<T>() * small).normalized();
    const auto qi = Rotation(pose_i);
    Eigen::Matrix<T, 15, 1> r;
    // 将世界系两帧位置差扣除初速度和重力后旋转到起始 IMU 系，再减预积分位移。
    r.template segment<3>(0) = qi.conjugate() * (Position(pose_j) - Position(pose_i) - T(pre_integration->sum_dt) * Position(speedbias_i) -
        T(0.5 * pre_integration->sum_dt * pre_integration->sum_dt) * gravity.cast<T>()) - corrected_p;
    // 相对旋转误差统一四元数符号；VINS 风格姿态残差采用 2*vec，而非完整 Log。
    Eigen::Quaternion<T> error = corrected_q.conjugate() * qi.conjugate() * Rotation(pose_j);
    if (error.w() < T(0)) error.coeffs() *= T(-1);
    r.template segment<3>(3) = T(2) * error.vec();
    r.template segment<3>(6) = qi.conjugate() * (Position(speedbias_j) - Position(speedbias_i) - T(pre_integration->sum_dt) * gravity.cast<T>()) - corrected_v;
    // 随机游走模型约束两端偏置差，噪声已经进入预积分协方差。
    r.template segment<3>(9) = Position(speedbias_j + 3) - Position(speedbias_i + 3);
    r.template segment<3>(12) = Position(speedbias_j + 6) - Position(speedbias_i + 6);
    Eigen::Map<Eigen::Matrix<T, 15, 1>> result(out);
    // 协方差包含各分量交叉项，必须对完整 15 维向量统一白化。
    result = pre_integration->sqrt_information.cast<T>() * r;
    return true;
  }
};


}}
