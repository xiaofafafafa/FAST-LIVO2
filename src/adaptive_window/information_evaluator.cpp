/**
 * @file information_evaluator.cpp
 * @brief 信息矩阵特征值统计、log-det 归一化和启发式可靠性；不是滑窗后验协方差计算。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#include "adaptive_window/information_evaluator.h"

#include "adaptive_window/math_utils.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace fast_livo
{
namespace adaptive_window
{

// 每个模态维护独立历史平滑量，初始状态由 History 的默认值给出。
InformationEvaluator::InformationEvaluator(const AdaptiveWindowConfig &config)
    : config_(config)
{
}

// Modality 连续枚举映射到历史数组；新增模态时必须同步扩容 history_。
std::size_t InformationEvaluator::Index(const Modality modality)
{
  return static_cast<std::size_t>(modality);
}

// 输入为前端统计矩阵与残差摘要，输出扩充后的质量摘要，不操作原始传感器因子。
MeasurementSummary InformationEvaluator::Evaluate(
    const MeasurementSummary &measurement)
{
  MeasurementSummary result = measurement;
  if (!measurement.information.allFinite() || !std::isfinite(measurement.inlier_ratio))
  {
    result.valid = false;
    result.normalized_information = result.reliability = 0.0;
    return result;
  }
  // 对称化后再分解，避免前端数值误差破坏自伴随特征分解假设。
  const auto symmetric = 0.5 * (measurement.information +
                                measurement.information.transpose());
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, kPoseDim, kPoseDim>> solver(symmetric);

  if (solver.info() != Eigen::Success || measurement.residual_count == 0 ||
      !std::isfinite(measurement.rmse))
  {
    result.valid = false;
    result.normalized_information = 0.0;
    result.reliability = 0.0;
    return result;
  }

  // 最小特征值衡量弱方向，条件数衡量各方向差异；均受坐标单位和信息缩放影响。
  const auto eigenvalues = solver.eigenvalues();
  result.min_eigenvalue = std::max(0.0, eigenvalues.minCoeff());
  const double maximum =
      std::max(config_.information_eigen_floor, eigenvalues.maxCoeff());
  result.condition_number =
      maximum /
      std::max(config_.information_eigen_floor, result.min_eigenvalue);

  // log-det 用各特征值的数值下限规避 log(0)，仅用于归一化质量打分。
  result.log_det = 0.0;
  for (int i = 0; i < eigenvalues.size(); ++i)
  {
    result.log_det += std::log(
        std::max(config_.information_eigen_floor, eigenvalues(i)));
  }

  const double logdet_range =
      std::max(1.0e-9, config_.information_logdet_ceiling -
                           config_.information_logdet_floor);
  // 固定区间线性归一化再限幅，跨数据集/尺度比较需留意区间设定。
  const double absolute_information = Clamp(
      (result.log_det - config_.information_logdet_floor) / logdet_range,
      0.0, 1.0);

  // 首次有效观测直接设历史；之后指数平滑，避免质量随单帧噪声剧烈变化。
  History &history = history_[Index(result.modality)];
  if (!history.initialized)
  {
    history.smoothed_information = absolute_information;
    history.initialized = true;
  }
  else
  {
    history.smoothed_information =
        config_.information_smoothing * history.smoothed_information +
        (1.0 - config_.information_smoothing) * absolute_information;
  }
  result.normalized_information = history.smoothed_information;

  // LiDAR 距离与图像灰度使用不同残差尺度，不能直接比较原始 RMSE。
  const double rmse_scale = result.modality == Modality::kLidar
                                ? config_.lidar_rmse_scale
                                : config_.vision_rmse_scale;
  const double residual_quality =
      std::exp(-measurement.rmse / std::max(1.0e-6, rmse_scale));
  const double observability = Clamp(
      result.min_eigenvalue /
          std::max(config_.degeneracy_eigen_threshold,
                   config_.information_eigen_floor),
      0.0, 1.0);
  // 当前经验组合：信息 0.45、残差质量 0.25、内点率 0.20、弱方向强度 0.10。
  result.reliability =
      Clamp(0.45 * result.normalized_information +
                0.25 * residual_quality +
                0.20 * Clamp(measurement.inlier_ratio, 0.0, 1.0) +
                0.10 * observability,
            0.0, 1.0);
  result.valid = true;
  return result;
}

// 只读查询；无有效历史返回零，而不是沿用其他模态的统计。
double InformationEvaluator::SmoothedInformation(const Modality modality) const
{
  const History &history = history_[Index(modality)];
  return history.initialized ? history.smoothed_information : 0.0;
}

}  // namespace adaptive_window
}  // namespace fast_livo
