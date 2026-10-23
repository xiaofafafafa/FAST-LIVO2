/**
 * @file integration_base.h
 * @brief VINS 风格中值预积分接口，保存增量、偏置 Jacobian、协方差和重积分所需原始样本。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
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
  // 对 [start,end] 精确裁剪，插值端点，检查递增时间及最大间隔，再调用 repropagate。
  bool integrate(const ImuSampleVector &samples, double start, double end,
                 const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);
  // 从保存样本重算增量/Jacobian/协方差，更新 linearized_ba/bg，偏置必须有限。
  bool repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);
  // 当前偏置相对积分基点的变化超过配置阈值时返回 true。
  bool needsRepropagation(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg) const;
  // 总积分时长；增量 delta_p/delta_v 表达在区间起始 IMU 坐标系。
  double sum_dt = 0.0;
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero(), delta_v = Eigen::Vector3d::Zero();
  // 区间相对旋转 q_i^-1*q_j 的无偏积分近似。
  Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
  // 当前预积分采用的偏置基点，因子用 Jacobian 一阶修正其与待估偏置的差。
  Eigen::Vector3d linearized_ba = Eigen::Vector3d::Zero(), linearized_bg = Eigen::Vector3d::Zero();
  // 误差排列 [dp,dtheta,dv,dba,dbg]；jacobian 是相对积分起点误差的传播矩阵。
  Matrix15 covariance = Matrix15::Zero(), jacobian = Matrix15::Identity();
  // L^-1，其中 covariance≈L*L^T；左乘残差后得到白化量。
  Matrix15 sqrt_information = Matrix15::Zero();
 private:
  // 对两个相邻已裁剪样本执行一次中值步，同时传播状态误差与端点测量噪声。
  void midPointIntegration(const ImuSample &a, const ImuSample &b);
  AdaptiveWindowConfig config_;
  // 保存裁剪后的区间样本，偏置变化时可重算，避免只线性修正过大的偏置变化。
  ImuSampleVector imu_buf;
};
}}
