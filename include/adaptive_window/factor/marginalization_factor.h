#pragma once
#include "adaptive_window/ceres_types.h"
#include <map>
#include <set>

namespace fast_livo { namespace adaptive_window {
class IntegrationBase;
using ParameterValues = std::map<BlockKey, std::vector<double>>;

// Owns the nonlinear cost and its stable parameter keys. Used for both Ceres
// construction and marginalization, in the role of VINS ResidualBlockInfo.
struct ResidualBlockInfo {
  std::vector<BlockKey> keys;
  std::shared_ptr<ceres::CostFunction> cost;
  Modality modality = Modality::kPrior;
  double huber_delta = 0.0;
  double confidence = 1.0;
  std::shared_ptr<IntegrationBase> pre_integration;
  bool Evaluate(const ParameterValues &, Eigen::VectorXd *,
                std::vector<Eigen::MatrixXd> *local_j = nullptr) const;
};
double residualWeight(const ResidualBlockInfo &, const ModalityWeights &);

class MarginalizationInfo {
 public:
  void addResidualBlockInfo(const ResidualBlockInfo &factor);
  bool preMarginalize(const ParameterValues &, const std::set<BlockKey> &drop_set,
                      const ModalityWeights &);
  bool marginalize();
  ResidualBlockInfo getFactor() const;
  std::vector<BlockKey> keep_block_keys;
  std::vector<std::vector<double>> keep_block_data;
  Eigen::MatrixXd linearized_jacobians;
  Eigen::VectorXd linearized_residuals;
 private:
  std::vector<ResidualBlockInfo> factors_;
  Eigen::MatrixXd hessian_;
  Eigen::VectorXd gradient_;
  int marginalized_size_ = 0;
};
}}
