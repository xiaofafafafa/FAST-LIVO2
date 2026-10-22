#include "adaptive_window/fixed_lag_optimizer.h"
#include "adaptive_window/imu_preintegration.h"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace fast_livo { namespace adaptive_window {
namespace {
// Only the marginalized information is fixed at its linearization point. All
// retained measurement factors remain nonlinear and are re-evaluated by Ceres.
struct MarginalCost {
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

void FixedLagOptimizer::Reset(const ExtrinsicState &e) {
  nodes_.clear(); blocks_.clear(); depth_anchors_.clear(); factors_.clear();
  prior_ = NonlinearFactor(); next_id_ = 1; weights_ = ModalityWeights();
  blocks_[ExtrinsicKey()] = PackPose(e.p_c_i, e.q_c_i);
}
std::uint64_t FixedLagOptimizer::AddNode(const NavState &s) {
  if (!std::isfinite(s.timestamp) || !s.q_w_i.coeffs().allFinite() || s.q_w_i.norm() < 1e-6 ||
      !s.p_w_i.allFinite() || !s.v_w_i.allFinite() || !s.bias_a.allFinite() ||
      !s.bias_g.allFinite() || !s.gravity_w.allFinite()) return 0;
  if (!nodes_.empty() && s.timestamp <= nodes_.rbegin()->second.state.timestamp) return 0;
  const std::uint64_t id = next_id_++;
  PolicyFrame frame; frame.id = id; frame.state = s;
  nodes_.emplace(id, frame);
  blocks_[PoseKey(id)] = PackPose(s.p_w_i, s.q_w_i);
  blocks_[MotionKey(id)] = {s.v_w_i.x(), s.v_w_i.y(), s.v_w_i.z(),
      s.bias_a.x(), s.bias_a.y(), s.bias_a.z(), s.bias_g.x(), s.bias_g.y(), s.bias_g.z()};
  return id;
}
std::uint64_t FixedLagOptimizer::FindNodeNear(double t, double tolerance) const {
  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it)
    if (std::abs(it->second.state.timestamp - t) <= tolerance) return it->first;
  return 0;
}
NavState FixedLagOptimizer::EstimateNode(std::uint64_t id) const {
  NavState s = nodes_.at(id).state;
  const auto &pose = blocks_.at(PoseKey(id)), &motion = blocks_.at(MotionKey(id));
  s.p_w_i = Position(pose.data()); s.q_w_i = Rotation(pose.data());
  s.v_w_i = Position(motion.data()); s.bias_a = Position(motion.data() + 3); s.bias_g = Position(motion.data() + 6);
  return s;
}
ExtrinsicState FixedLagOptimizer::EstimateCameraExtrinsic() const {
  const auto &x = blocks_.at(ExtrinsicKey());
  ExtrinsicState e; e.p_c_i = Position(x.data()); e.q_c_i = Rotation(x.data()); return e;
}
void FixedLagOptimizer::MarkNode(std::uint64_t id, Modality m, double info) {
  if (!HasNode(id)) return;
  auto &f = nodes_.at(id); f.unique_information = std::max(f.unique_information, info);
  if (m == Modality::kLidar) f.has_lidar = true;
  if (m == Modality::kVision) f.has_vision = true;
  if (m == Modality::kCrossModal) f.has_cross_modal = true;
}
bool FixedLagOptimizer::AddDepth(std::uint64_t track, std::uint64_t anchor, double rho) {
  if (!HasNode(anchor) || HasDepth(track) || !std::isfinite(rho) ||
      rho < config_.min_inverse_depth || rho > config_.max_inverse_depth) return false;
  blocks_[DepthKey(track)] = {rho}; depth_anchors_[track] = anchor; return true;
}
void FixedLagOptimizer::RemoveUnusedDepth(std::uint64_t track) {
  const auto key = DepthKey(track);
  for (const auto &f : factors_) for (const auto &k : f.keys) if (k == key) return;
  for (const auto &k : prior_.keys) if (k == key) return;
  blocks_.erase(key); depth_anchors_.erase(track);
}
bool FixedLagOptimizer::Evaluate(const NonlinearFactor &f, Eigen::VectorXd *r,
                                std::vector<Eigen::MatrixXd> *local_j) const {
  if (!f.cost || !r) return false;
  const int rows = f.cost->num_residuals();
  std::vector<const double *> parameters;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ambient;
  std::vector<double *> jacobians;
  ambient.reserve(f.keys.size());
  for (const auto &key : f.keys) {
    const auto it = blocks_.find(key); if (it == blocks_.end()) return false;
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
        manifold_.PlusJacobian(parameters[k], plus.data());
        local_j->push_back(ambient[k] * plus);
      } else local_j->push_back(ambient[k]);
    }
  }
  return true;
}
bool FixedLagOptimizer::AddFactor(const NonlinearFactor &f) {
  if (!f.cost || !std::isfinite(f.confidence) || f.confidence <= 0 ||
      f.keys.size() != f.cost->parameter_block_sizes().size()) return false;
  std::set<BlockKey> unique;
  for (std::size_t k = 0; k < f.keys.size(); ++k)
    if (!unique.insert(f.keys[k]).second || f.keys[k].size() != f.cost->parameter_block_sizes()[k]) return false;
  Eigen::VectorXd r; std::vector<Eigen::MatrixXd> j;
  if (!Evaluate(f, &r, &j)) return false;
  factors_.push_back(f); return true;
}
double FixedLagOptimizer::Weight(const NonlinearFactor &f) const {
  double weight = 1.0;
  if (f.modality == Modality::kLidar) weight = weights_.lidar;
  if (f.modality == Modality::kVision) weight = weights_.vision;
  if (f.modality == Modality::kCrossModal) weight = weights_.cross_modal;
  return weight * f.confidence;
}
bool FixedLagOptimizer::Solve(double *final_cost) {
  if (nodes_.empty() || factors_.empty()) return false;
  for (auto &f : factors_) {
    if (!f.preintegration) continue;
    const auto &motion = blocks_.at(f.keys[1]);
    const auto ba = Position(motion.data() + 3), bg = Position(motion.data() + 6);
    if (f.preintegration->NeedsRepropagation(ba, bg) && !f.preintegration->Repropagate(ba, bg)) return false;
  }
  const auto backup = blocks_;
  ceres::Problem::Options po;
  po.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  po.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  po.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  std::vector<std::unique_ptr<ceres::LossFunction>> losses;
  ceres::Problem problem(po);
  for (auto &entry : blocks_) {
    const auto &key = entry.first; auto &data = entry.second;
    problem.AddParameterBlock(data.data(), key.size(), key.is_pose() ? &manifold_ : nullptr);
    if (key.kind == BlockKind::kDepth) {
      problem.SetParameterLowerBound(data.data(), 0, config_.min_inverse_depth);
      problem.SetParameterUpperBound(data.data(), 0, config_.max_inverse_depth);
    }
    if (key.kind == BlockKind::kExtrinsic && !config_.optimize_extrinsics) problem.SetParameterBlockConstant(data.data());
  }
  auto add = [&](const NonlinearFactor &f) {
    std::vector<double *> parameters;
    for (const auto &key : f.keys) parameters.push_back(blocks_.at(key).data());
    auto *robust = f.huber_delta > 0.0 ? new ceres::HuberLoss(f.huber_delta) : nullptr;
    losses.emplace_back(new ceres::ScaledLoss(robust, Weight(f), ceres::TAKE_OWNERSHIP));
    problem.AddResidualBlock(f.cost.get(), losses.back().get(), parameters);
  };
  for (const auto &f : factors_) add(f);
  if (prior_.cost) add(prior_);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = config_.solver_iterations;
  options.max_solver_time_in_seconds = config_.solver_time_limit;
  options.num_threads = 1;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  bool success = summary.IsSolutionUsable() && std::isfinite(summary.final_cost);
  for (const auto &entry : blocks_) for (double value : entry.second) success = success && std::isfinite(value);
  if (!success) {
    // Keep the addresses registered in the still-live Ceres problem valid.
    for (auto &entry : blocks_) std::copy(backup.at(entry.first).begin(), backup.at(entry.first).end(), entry.second.begin());
    return false;
  }
  if (final_cost) *final_cost = summary.final_cost;
  return true;
}

bool FixedLagOptimizer::Marginalize(std::uint64_t id) {
  if (!HasNode(id) || nodes_.size() < 2 || id == nodes_.rbegin()->first) return false;
  std::set<BlockKey> drop{PoseKey(id), MotionKey(id)};
  for (const auto &entry : depth_anchors_) if (entry.second == id) drop.insert(DepthKey(entry.first));
  std::vector<std::size_t> absorbed;
  std::vector<const NonlinearFactor *> selected;
  for (std::size_t i = 0; i < factors_.size(); ++i) {
    bool touches = false;
    for (const auto &key : factors_[i].keys) touches = touches || drop.count(key) != 0;
    if (touches) { absorbed.push_back(i); selected.push_back(&factors_[i]); }
  }
  if (prior_.cost) selected.push_back(&prior_);
  std::set<BlockKey> keep;
  for (const auto *f : selected) for (const auto &key : f->keys) if (!drop.count(key)) keep.insert(key);
  if (keep.empty()) return false;
  std::map<BlockKey, int> offsets;
  int dimension = 0;
  for (const auto &key : drop) { offsets[key] = dimension; dimension += key.local_size(); }
  const int marginalized = dimension;
  for (const auto &key : keep) { offsets[key] = dimension; dimension += key.local_size(); }
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dimension, dimension);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(dimension);
  for (const auto *f : selected) {
    Eigen::VectorXd r; std::vector<Eigen::MatrixXd> jacobians;
    if (!Evaluate(*f, &r, &jacobians)) return false;
    double weight = Weight(*f);
    if (f->huber_delta > 0.0 && r.norm() > f->huber_delta) weight *= f->huber_delta / r.norm();
    for (std::size_t i = 0; i < f->keys.size(); ++i) {
      const int oi = offsets.at(f->keys[i]), ni = f->keys[i].local_size();
      b.segment(oi, ni).noalias() += weight * jacobians[i].transpose() * r;
      for (std::size_t j = 0; j < f->keys.size(); ++j) {
        const int oj = offsets.at(f->keys[j]), nj = f->keys[j].local_size();
        h.block(oi, oj, ni, nj).noalias() += weight * jacobians[i].transpose() * jacobians[j];
      }
    }
  }
  Eigen::VectorXd scale, values; Eigen::MatrixXd vectors;
  if (!Spectrum(h.topLeftCorner(marginalized, marginalized), &scale, &values, &vectors)) return false;
  Eigen::VectorXd inverse = values.unaryExpr([](double v) { return v > 0 ? 1.0 / v : 0.0; });
  const Eigen::MatrixXd pinv = scale.asDiagonal() * vectors * inverse.asDiagonal() * vectors.transpose() * scale.asDiagonal();
  const int remaining = dimension - marginalized;
  const Eigen::MatrixXd cross = h.topRightCorner(marginalized, remaining);
  Eigen::MatrixXd reduced = h.bottomRightCorner(remaining, remaining) - cross.transpose() * pinv * cross;
  Eigen::VectorXd gradient = b.tail(remaining) - cross.transpose() * pinv * b.head(marginalized);
  if (!gradient.allFinite() || !Spectrum(reduced, &scale, &values, &vectors)) return false;
  const Eigen::VectorXd roots = values.cwiseSqrt();
  const Eigen::VectorXd invroots = roots.unaryExpr([](double v) { return v > 0 ? 1.0 / v : 0.0; });
  auto *functor = new MarginalCost;
  functor->keys.assign(keep.begin(), keep.end());
  for (const auto &key : keep) functor->references.push_back(blocks_.at(key));
  functor->a = roots.asDiagonal() * vectors.transpose() * scale.cwiseInverse().asDiagonal();
  functor->b = invroots.asDiagonal() * vectors.transpose() * scale.asDiagonal() * gradient;
  auto *cost = new ceres::DynamicAutoDiffCostFunction<MarginalCost, 4>(functor);
  for (const auto &key : keep) cost->AddParameterBlock(key.size());
  cost->SetNumResiduals(remaining);
  NonlinearFactor new_prior; new_prior.keys = functor->keys; new_prior.cost.reset(cost);
  Eigen::VectorXd check;
  if (!Evaluate(new_prior, &check, nullptr)) return false;
  // Commit only after the complete Schur prior is valid. Retained measurement
  // factors are not absorbed, avoiding double counting in subsequent solves.
  for (auto it = absorbed.rbegin(); it != absorbed.rend(); ++it) factors_.erase(factors_.begin() + *it);
  prior_ = std::move(new_prior);
  for (const auto &key : drop) {
    blocks_.erase(key);
    if (key.kind == BlockKind::kDepth) depth_anchors_.erase(key.id);
  }
  nodes_.erase(id);
  return true;
}
PolicyFrameVector FixedLagOptimizer::PolicyFrames() const {
  PolicyFrameVector out;
  for (const auto &entry : nodes_) { auto frame = entry.second; frame.state = EstimateNode(entry.first); out.push_back(frame); }
  return out;
}
std::vector<std::uint64_t> FixedLagOptimizer::NodeIds() const {
  std::vector<std::uint64_t> ids; for (const auto &entry : nodes_) ids.push_back(entry.first); return ids;
}
std::size_t FixedLagOptimizer::CountFactors(Modality m) const {
  return std::count_if(factors_.begin(), factors_.end(), [m](const NonlinearFactor &f) { return f.modality == m; });
}
}}
