#include "adaptive_window/information_evaluator.h"

#include "adaptive_window/math_utils.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace fast_livo
{
namespace adaptive_window
{

InformationEvaluator::InformationEvaluator(const AdaptiveWindowConfig &config)
    : config_(config)
{
}

Eigen::Matrix<double, kPoseDim, kPoseDim>
InformationEvaluator::NormalizedPoseInformation(
    const MeasurementSummary &measurement, const AdaptiveWindowConfig &config)
{
  using Matrix6 = Eigen::Matrix<double, kPoseDim, kPoseDim>;
  if (measurement.residual_count == 0 || !measurement.information.allFinite())
    return Matrix6::Zero();
  // Bridge summaries use [rotation, translation]. D maps a unitless state
  // perturbation to a physically meaningful rotation/translation increment.
  Matrix6 scale = Matrix6::Zero();
  scale.diagonal().head<3>().setConstant(
      Radians(config.information_rotation_scale_deg));
  scale.diagonal().tail<3>().setConstant(
      config.information_translation_scale);
  const Matrix6 symmetric = 0.5 * (measurement.information +
                                    measurement.information.transpose());
  const Matrix6 normalized = scale * symmetric * scale /
      static_cast<double>(measurement.residual_count);
  Eigen::SelfAdjointEigenSolver<Matrix6> eigen(normalized);
  if (eigen.info() != Eigen::Success) return Matrix6::Zero();
  // Tiny negative eigenvalues are numerical, never negative information.
  return eigen.eigenvectors() *
      eigen.eigenvalues().cwiseMax(0.0).asDiagonal() *
      eigen.eigenvectors().transpose();
}

std::size_t InformationEvaluator::Index(const Modality modality)
{
  return static_cast<std::size_t>(modality);
}

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
  const auto symmetric = config_.normalized_information_policy
      ? NormalizedPoseInformation(measurement, config_)
      : (0.5 * (measurement.information + measurement.information.transpose())).eval();
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

  const auto eigenvalues = solver.eigenvalues();
  result.min_eigenvalue = std::max(0.0, eigenvalues.minCoeff());
  const double maximum =
      std::max(config_.information_eigen_floor, eigenvalues.maxCoeff());
  result.condition_number =
      maximum /
      std::max(config_.information_eigen_floor, result.min_eigenvalue);

  double absolute_information = 0.0;
  result.log_det = 0.0;
  if (config_.normalized_information_policy)
  {
    // Mean fractional variance reduction under a unit reference precision:
    // tr(F (I+F)^-1)/6. It is bounded without dataset-specific logdet limits.
    for (int i = 0; i < eigenvalues.size(); ++i)
    {
      const double lambda = std::max(0.0, eigenvalues(i));
      result.log_det += std::log1p(lambda);
      absolute_information += lambda / (1.0 + lambda);
    }
    absolute_information /= static_cast<double>(kPoseDim);
  }
  else
  {
    for (int i = 0; i < eigenvalues.size(); ++i)
      result.log_det += std::log(
          std::max(config_.information_eigen_floor, eigenvalues(i)));
    const double logdet_range =
        std::max(1.0e-9, config_.information_logdet_ceiling -
                             config_.information_logdet_floor);
    absolute_information = Clamp(
        (result.log_det - config_.information_logdet_floor) / logdet_range,
        0.0, 1.0);
  }

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

  const double rmse_scale = result.modality == Modality::kLidar
                                ? config_.lidar_rmse_scale
                                : config_.vision_rmse_scale;
  const double residual_quality =
      std::exp(-measurement.rmse / std::max(1.0e-6, rmse_scale));
  const double observability = config_.normalized_information_policy
      ? result.min_eigenvalue / (1.0 + result.min_eigenvalue)
      : Clamp(result.min_eigenvalue /
                  std::max(config_.degeneracy_eigen_threshold,
                           config_.information_eigen_floor), 0.0, 1.0);
  result.reliability =
      Clamp(0.45 * result.normalized_information +
                0.25 * residual_quality +
                0.20 * Clamp(measurement.inlier_ratio, 0.0, 1.0) +
                0.10 * observability,
            0.0, 1.0);
  result.valid = true;
  return result;
}

double InformationEvaluator::SmoothedInformation(const Modality modality) const
{
  const History &history = history_[Index(modality)];
  return history.initialized ? history.smoothed_information : 0.0;
}

}  // namespace adaptive_window
}  // namespace fast_livo
