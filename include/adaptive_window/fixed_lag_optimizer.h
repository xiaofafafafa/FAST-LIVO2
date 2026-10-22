#pragma once
#include "adaptive_window/adaptive_policy.h"
#include "adaptive_window/ceres_types.h"
#include <map>
#include <set>

namespace fast_livo { namespace adaptive_window {
class FixedLagOptimizer {
 public:
  explicit FixedLagOptimizer(const AdaptiveWindowConfig &config) : config_(config) {}
  void Reset(const ExtrinsicState &);
  std::uint64_t AddNode(const NavState &);
  bool HasNode(std::uint64_t id) const { return nodes_.count(id) != 0; }
  std::uint64_t FindNodeNear(double timestamp, double tolerance) const;
  NavState EstimateNode(std::uint64_t id) const;
  ExtrinsicState EstimateCameraExtrinsic() const;
  void MarkNode(std::uint64_t id, Modality, double information);
  bool AddDepth(std::uint64_t track, std::uint64_t anchor, double inverse_depth);
  void RemoveUnusedDepth(std::uint64_t track);
  bool HasDepth(std::uint64_t track) const { return blocks_.count(DepthKey(track)) != 0; }
  bool AddFactor(const NonlinearFactor &);
  void SetWeights(const ModalityWeights &w) { weights_ = w; }
  bool Solve(double *final_cost = nullptr);
  bool Marginalize(std::uint64_t id);
  PolicyFrameVector PolicyFrames() const;
  std::vector<std::uint64_t> NodeIds() const;
  std::size_t NumNodes() const { return nodes_.size(); }
  std::size_t NumFactors() const { return factors_.size() + (prior_.cost ? 1 : 0); }
  std::size_t CountFactors(Modality) const;
 private:
  double Weight(const NonlinearFactor &) const;
  bool Evaluate(const NonlinearFactor &, Eigen::VectorXd *, std::vector<Eigen::MatrixXd> *) const;
  AdaptiveWindowConfig config_;
  ModalityWeights weights_;
  PoseManifold manifold_;
  std::uint64_t next_id_ = 1;
  std::map<std::uint64_t, PolicyFrame> nodes_;
  std::map<BlockKey, std::vector<double>> blocks_;
  std::map<std::uint64_t, std::uint64_t> depth_anchors_;
  std::vector<NonlinearFactor> factors_;
  NonlinearFactor prior_;
};
}}
