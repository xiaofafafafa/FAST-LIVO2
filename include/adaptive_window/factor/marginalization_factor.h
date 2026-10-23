/**
 * @file marginalization_factor.h
 * @brief 非线性残差描述和 Schur 先验数据；稳定参数键与独立线性化基点避免悬空引用。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"
#include <map>
#include <set>

namespace fast_livo { namespace adaptive_window {
class IntegrationBase;
// 实际 Ceres 参数数组的拥有者类型，键决定每个 vector 的尺寸与局部自由度。
using ParameterValues = std::map<BlockKey, std::vector<double>>;

// Owns the nonlinear cost and its stable parameter keys. Used for both Ceres
// construction and marginalization, in the role of VINS ResidualBlockInfo.
struct ResidualBlockInfo {
  // 参数顺序必须与 cost->parameter_block_sizes() 完全一致。
  std::vector<BlockKey> keys;
  // 共享持有非线性 cost；Ceres Problem 不负责释放此对象。
  std::shared_ptr<ceres::CostFunction> cost;
  Modality modality = Modality::kPrior;
  // 对残差整块范数施加 Huber，0 表示普通平方损失。
  double huber_delta = 0.0;
  // 独立置信乘数；滤波位姿项将其用作显式正则权重。
  double confidence = 1.0;
  // 仅 IMU 因子非空，使求解前可按起点偏置重积分。
  std::shared_ptr<IntegrationBase> pre_integration;
  // 读取参数并求残差，可选返回切空间 Jacobian；姿态环境 Jacobian 右乘 PlusJacobian。
  bool Evaluate(const ParameterValues &, Eigen::VectorXd *,
                std::vector<Eigen::MatrixXd> *local_j = nullptr) const;
};
// 返回模态权重×confidence；先验和 IMU 默认模态乘数为 1。
double residualWeight(const ResidualBlockInfo &, const ModalityWeights &);

class MarginalizationInfo {
 public:
  // 收集本次要吸收的因子，调用者应只传触及待删变量的因子及旧先验。
  void addResidualBlockInfo(const ResidualBlockInfo &factor);
  // 建立 [待删,保留] 局部排列，保存保留变量基点，冻结当前鲁棒权重并累加 H、g。
  bool preMarginalize(const ParameterValues &, const std::set<BlockKey> &drop_set,
                      const ModalityWeights &);
  // 用对角均衡特征分解伪逆消元，再生成 A、b 使 A^T A≈H_prior，A^T b≈g_prior。
  bool marginalize();
  // 创建 r_prior=A*(x boxminus x0)+b；复制基点及线性系统，避免引用旧图内存。
  ResidualBlockInfo getFactor() const;
  std::vector<BlockKey> keep_block_keys;
  // 每个保留块在本次线性化点的环境维度数值副本。
  std::vector<std::vector<double>> keep_block_data;
  // A：列数等于保留块局部维度之和；不是完整图每次重算的 Jacobian。
  Eigen::MatrixXd linearized_jacobians;
  // b：先验在 x0 处的残差常量。
  Eigen::VectorXd linearized_residuals;
 private:
  std::vector<ResidualBlockInfo> factors_;
  Eigen::MatrixXd hessian_;
  Eigen::VectorXd gradient_;
  int marginalized_size_ = 0;
};
}}
