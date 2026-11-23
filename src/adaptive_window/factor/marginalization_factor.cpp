#include "adaptive_window/factor/marginalization_factor.h"
#include "adaptive_window/factor/pose_local_parameterization.h"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace fast_livo { namespace adaptive_window {
namespace {
// Only the marginalized information is fixed at its linearization point. All
// retained measurement factors remain nonlinear and are re-evaluated by Ceres.
struct MarginalizationFactor {
  std::vector<BlockKey> keys;
  std::vector<std::vector<double>> references;
  Eigen::MatrixXd a;
  Eigen::VectorXd b;
  template <typename T> bool operator()(T const *const *x, T *out) const {
    Eigen::Matrix<T, Eigen::Dynamic, 1> delta(a.cols());
    int offset = 0;
    for (std::size_t k = 0; k < keys.size(); ++k) {
      const auto &base = references[k];
      if (keys[k].is_pose()) {
        delta.template segment<3>(offset) = Position(x[k]) - Position(base.data()).cast<T>();
        delta.template segment<3>(offset + 3) = QuaternionLog(
            Rotation(base.data()).cast<T>().conjugate() * Rotation(x[k]));
      } else {
        for (int j = 0; j < keys[k].size(); ++j) delta[offset + j] = x[k][j] - T(base[j]);
      }
      offset += keys[k].local_size();
    }
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> r(out, b.size());
    r = a.cast<T>() * delta + b.cast<T>();
    return true;
  }
};

// Diagonal equilibration prevents position/bias unit differences from deciding
// the eigenvalue truncation. Null directions are discarded, never damped into
// artificial information. The 1e-9 cutoff applies to the scaled matrix.
bool Spectrum(const Eigen::MatrixXd &h, Eigen::VectorXd *scale,
              Eigen::VectorXd *values, Eigen::MatrixXd *vectors) {
  if (h.rows() == 0 || !h.allFinite()) return false;
  *scale = h.diagonal().cwiseAbs().cwiseMax(1e-12).cwiseSqrt().cwiseInverse();
  const Eigen::MatrixXd normalized = scale->asDiagonal() * (0.5 * (h + h.transpose())) * scale->asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(normalized);
  if (eigen.info() != Eigen::Success) return false;
  *values = eigen.eigenvalues(); *vectors = eigen.eigenvectors();
  for (int k = 0; k < values->size(); ++k) if ((*values)[k] < 1e-9) (*values)[k] = 0.0;
  return true;
}
}

bool ResidualBlockInfo::Evaluate(const ParameterValues &values, Eigen::VectorXd *r,
                                 std::vector<Eigen::MatrixXd> *local_j) const {
  const auto &f = *this;
  PoseLocalParameterization local_parameterization;
  if (!f.cost || !r) return false;
  const int rows = f.cost->num_residuals();
  std::vector<const double *> parameters;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ambient;
  std::vector<double *> jacobians;
  ambient.reserve(f.keys.size());
  for (const auto &key : f.keys) {
    const auto it = values.find(key); if (it == values.end()) return false;
    parameters.push_back(it->second.data());
    if (local_j) { ambient.emplace_back(rows, key.size()); jacobians.push_back(ambient.back().data()); }
  }
  r->resize(rows);
  if (!f.cost->Evaluate(parameters.data(), r->data(), local_j ? jacobians.data() : nullptr) || !r->allFinite()) return false;
  if (local_j) {
    local_j->clear();
    for (std::size_t k = 0; k < f.keys.size(); ++k) {
      if (!ambient[k].allFinite()) return false;
      if (f.keys[k].is_pose()) {
        Eigen::Matrix<double, 7, 6, Eigen::RowMajor> plus;
        local_parameterization.PlusJacobian(parameters[k], plus.data());
        local_j->push_back(ambient[k] * plus);
      } else local_j->push_back(ambient[k]);
    }
  }
  return true;
}

double residualWeight(const ResidualBlockInfo &f, const ModalityWeights &weights) {
  double weight = 1.0;
  if (f.modality == Modality::kLidar) weight = weights.lidar;
  if (f.modality == Modality::kVision) weight = weights.vision;
  if (f.modality == Modality::kCrossModal) weight = weights.cross_modal;
  // Frontend pose regularization has its own explicit weight in confidence.
  return weight * f.confidence;
}
void MarginalizationInfo::addResidualBlockInfo(const ResidualBlockInfo &factor) {
  factors_.push_back(factor);
}
bool MarginalizationInfo::preMarginalize(const ParameterValues &parameters,
    const std::set<BlockKey> &drop_set, const ModalityWeights &weights) {
  keep_block_keys.clear(); keep_block_data.clear();
  linearized_jacobians.resize(0, 0); linearized_residuals.resize(0);
  std::set<BlockKey> keep;
  for (const auto &f : factors_) for (const auto &key : f.keys)
    if (!drop_set.count(key)) keep.insert(key);
  if (keep.empty() || drop_set.empty()) return false;
  std::map<BlockKey, int> offsets;
  int dimension = 0;
  for (const auto &key : drop_set) { offsets[key] = dimension; dimension += key.local_size(); }
  marginalized_size_ = dimension;
  for (const auto &key : keep) {
    if (!parameters.count(key)) return false;
    offsets[key] = dimension; dimension += key.local_size();
    keep_block_keys.push_back(key); keep_block_data.push_back(parameters.at(key));
  }
  hessian_ = Eigen::MatrixXd::Zero(dimension, dimension);
  gradient_ = Eigen::VectorXd::Zero(dimension);
  for (const auto &f : factors_) {
    Eigen::VectorXd r; std::vector<Eigen::MatrixXd> jacobians;
    if (!f.Evaluate(parameters, &r, &jacobians)) return false;
    double weight = residualWeight(f, weights);
    if (f.huber_delta > 0.0 && r.norm() > f.huber_delta) weight *= f.huber_delta / r.norm();
    for (std::size_t i = 0; i < f.keys.size(); ++i) {
      const int oi = offsets.at(f.keys[i]), ni = f.keys[i].local_size();
      gradient_.segment(oi, ni).noalias() += weight * jacobians[i].transpose() * r;
      for (std::size_t j = 0; j < f.keys.size(); ++j) {
        const int oj = offsets.at(f.keys[j]), nj = f.keys[j].local_size();
        hessian_.block(oi, oj, ni, nj).noalias() += weight * jacobians[i].transpose() * jacobians[j];
      }
    }
  }
  return hessian_.allFinite() && gradient_.allFinite();
}
bool MarginalizationInfo::marginalize() {
  const int m = marginalized_size_, n = static_cast<int>(hessian_.rows()) - m;
  if (m <= 0 || n <= 0) return false;
  Eigen::VectorXd scale, values; Eigen::MatrixXd vectors;
  if (!Spectrum(hessian_.topLeftCorner(m, m), &scale, &values, &vectors)) return false;
  const Eigen::VectorXd inverse = values.unaryExpr([](double v) { return v > 0 ? 1.0 / v : 0.0; });
  const Eigen::MatrixXd pinv = scale.asDiagonal() * vectors * inverse.asDiagonal() * vectors.transpose() * scale.asDiagonal();
  const Eigen::MatrixXd cross = hessian_.topRightCorner(m, n);
  const Eigen::MatrixXd reduced = hessian_.bottomRightCorner(n, n) - cross.transpose() * pinv * cross;
  const Eigen::VectorXd gradient = gradient_.tail(n) - cross.transpose() * pinv * gradient_.head(m);
  if (!gradient.allFinite() || !Spectrum(reduced, &scale, &values, &vectors)) return false;
  const Eigen::VectorXd roots = values.cwiseSqrt();
  const Eigen::VectorXd invroots = roots.unaryExpr([](double v) { return v > 0 ? 1.0 / v : 0.0; });
  linearized_jacobians = roots.asDiagonal() * vectors.transpose() * scale.cwiseInverse().asDiagonal();
  linearized_residuals = invroots.asDiagonal() * vectors.transpose() * scale.asDiagonal() * gradient;
  // Drop original costs/images after absorption. The next prior must not retain
  // a chain of historical image buffers through last_marginalization_info.
  factors_.clear(); hessian_.resize(0, 0); gradient_.resize(0);
  return linearized_jacobians.allFinite() && linearized_residuals.allFinite();
}
ResidualBlockInfo MarginalizationInfo::getFactor() const {
  ResidualBlockInfo f;
  if (linearized_residuals.size() == 0) return f;
  auto *functor = new MarginalizationFactor;
  functor->keys = keep_block_keys; functor->references = keep_block_data;
  functor->a = linearized_jacobians; functor->b = linearized_residuals;
  auto *cost = new ceres::DynamicAutoDiffCostFunction<MarginalizationFactor, 4>(functor);
  for (const auto &key : keep_block_keys) cost->AddParameterBlock(key.size());
  cost->SetNumResiduals(static_cast<int>(linearized_residuals.size()));
  f.keys = keep_block_keys; f.cost.reset(cost);
  return f;
}
}}
