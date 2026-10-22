#include "adaptive_window/adaptive_window_backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace fast_livo { namespace adaptive_window {
namespace {
void Validate(const AdaptiveWindowConfig &c) {
  if (c.min_window_size < 2 || c.max_window_size < c.min_window_size ||
      c.initial_window_size < c.min_window_size || c.initial_window_size > c.max_window_size ||
      c.max_lidar_factors < 0 || c.max_cross_modal_factors < 0 || c.max_active_patches < 0 || c.solver_iterations < 1)
    throw std::invalid_argument("Invalid adaptive-window counts");
  for (double v : {c.solver_time_limit, c.imu_max_gap, c.pose_prior_translation_sigma,
       c.pose_prior_rotation_deg, c.velocity_prior_sigma, c.bias_prior_sigma,
       c.extrinsic_translation_sigma, c.extrinsic_rotation_deg, c.imu_acc_noise,
       c.imu_gyr_noise, c.imu_acc_bias_rw, c.imu_gyr_bias_rw, c.min_inverse_depth,
       c.plane_search_radius, c.plane_max_rms, c.cross_modal_plane_sigma,
       c.association_time_sigma, c.max_keyframe_dt, c.min_modality_weight,
       c.imu_repropagate_acc_bias, c.imu_repropagate_gyr_bias, c.max_modality_weight,
       c.information_eigen_floor, c.keyframe_translation, c.keyframe_rotation_deg,
       c.lidar_rmse_scale, c.vision_rmse_scale, c.degeneracy_eigen_threshold})
    if (!std::isfinite(v) || v <= 0) throw std::invalid_argument("Adaptive-window scales must be finite and positive");
  if (!std::isfinite(c.max_inverse_depth) || c.max_inverse_depth <= c.min_inverse_depth ||
      !std::isfinite(c.merge_time_tolerance) || c.merge_time_tolerance < 0 ||
      !std::isfinite(c.association_max_dt) || c.association_max_dt < 0 ||
      c.max_modality_weight < c.min_modality_weight || !std::isfinite(c.min_keyframe_dt) ||
      c.min_keyframe_dt < 0 || c.max_keyframe_dt < c.min_keyframe_dt ||
      !std::isfinite(c.keyframe_utility_threshold) ||
      !std::isfinite(c.information_logdet_floor) || !std::isfinite(c.information_logdet_ceiling) ||
      c.information_logdet_ceiling <= c.information_logdet_floor)
    throw std::invalid_argument("Invalid adaptive-window bounds");
  for (double v : {c.information_smoothing, c.weight_smoothing})
    if (!std::isfinite(v) || v < 0 || v >= 1) throw std::invalid_argument("Invalid smoothing coefficient");
  for (double v : {c.huber_delta_lidar, c.huber_delta_vision, c.huber_delta_cross_modal})
    if (!std::isfinite(v) || v < 0) throw std::invalid_argument("Invalid Huber scale");
}
bool ValidPatch(const VisualPatchObservation &o) {
  if (o.track_id == 0 || o.rays.empty() || o.rays.size() != o.intensities.size() ||
      !o.point_w.allFinite() || !o.center_ray.allFinite() || !o.normal_c.allFinite() ||
      std::abs(o.center_ray.z() - 1.0) > 1e-6 || o.normal_c.norm() < 1e-6 ||
      !std::isfinite(o.inverse_exposure) || o.inverse_exposure <= 0 ||
      !std::isfinite(o.photometric_sigma) || o.photometric_sigma <= 0 ||
      !std::isfinite(o.tracking_score) || o.tracking_score <= 0) return false;
  for (std::size_t i = 0; i < o.rays.size(); ++i)
    if (!o.rays[i].allFinite() || !std::isfinite(o.intensities[i]) ||
        std::abs(o.normal_c.dot(o.rays[i])) < 1e-4 ||
        o.normal_c.dot(o.rays[i]) * o.normal_c.dot(o.center_ray) <= 0) return false;
  return true;
}
}

AdaptiveWindowBackend::AdaptiveWindowBackend(const AdaptiveWindowConfig &c)
    : config_(c), information_evaluator_(c), policy_(c), associator_(c), optimizer_(c),
      target_window_(c.initial_window_size) { Validate(c); }
void AdaptiveWindowBackend::Initialize(const LidarExtrinsic &l, const ExtrinsicState &e, ProjectionFunction p) {
  lidar_extrinsic_ = l; camera_extrinsic_initial_ = e; projection_ = std::move(p); Reset();
}
void AdaptiveWindowBackend::Reset() {
  pending_imu_.clear(); lidar_frames_.clear(); patches_.clear(); associator_.Clear();
  information_evaluator_ = InformationEvaluator(config_); policy_ = AdaptivePolicy(config_);
  has_lidar_ = false; has_vision_ = false; latest_node_ = 0;
  last_lidar_time_ = last_image_time_ = -std::numeric_limits<double>::infinity();
  target_window_ = config_.initial_window_size; diagnostics_ = BackendDiagnostics();
  optimizer_.Reset(camera_extrinsic_initial_);
  optimizer_.AddFactor(FactorBuilders::ExtrinsicPrior(camera_extrinsic_initial_, config_));
}
void AdaptiveWindowBackend::Recover() {
  const auto resets = diagnostics_.reset_count + 1, rejected = diagnostics_.rejected_imu_intervals;
  ImuSampleVector samples = std::move(pending_imu_);
  Reset(); pending_imu_ = std::move(samples);
  diagnostics_.reset_count = resets; diagnostics_.rejected_imu_intervals = rejected;
}
void AdaptiveWindowBackend::AddImuSamples(const ImuSampleVector &samples) {
  // Input packets overlap and may contain a future boundary sample. Insert by
  // timestamp, rather than dropping subsequent samples behind that boundary.
  for (const auto &s : samples) {
    if (!std::isfinite(s.timestamp) || !s.acceleration.allFinite() || !s.angular_velocity.allFinite()) continue;
    auto it = std::lower_bound(pending_imu_.begin(), pending_imu_.end(), s.timestamp,
        [](const ImuSample &a, double t) { return a.timestamp < t; });
    if (it != pending_imu_.end() && std::abs(it->timestamp - s.timestamp) < 1e-9) *it = s;
    else pending_imu_.insert(it, s);
  }
  if (!pending_imu_.empty()) ConsumeImuThrough(pending_imu_.back().timestamp - std::max(5.0, 5.0 * config_.max_keyframe_dt));
}
void AdaptiveWindowBackend::ConsumeImuThrough(double t) {
  auto it = std::lower_bound(pending_imu_.begin(), pending_imu_.end(), t,
      [](const ImuSample &a, double time) { return a.timestamp < time; });
  if (it != pending_imu_.begin()) --it;
  pending_imu_.erase(pending_imu_.begin(), it);
}
PolicyDecision AdaptiveWindowBackend::Decide(const NavState &s, const MeasurementSummary &input, Modality m) {
  auto summary = input; summary.modality = m; summary.timestamp = s.timestamp;
  summary = information_evaluator_.Evaluate(summary);
  if (m == Modality::kLidar) { latest_lidar_ = summary; has_lidar_ = true; }
  else { latest_vision_ = summary; has_vision_ = true; }
  // Stale quality statistics must not keep a missing modality at high weight.
  const double age = 2.0 * config_.max_keyframe_dt;
  auto decision = policy_.Decide(s,
      has_lidar_ && s.timestamp - latest_lidar_.timestamp <= age ? &latest_lidar_ : nullptr,
      has_vision_ && s.timestamp - latest_vision_.timestamp <= age ? &latest_vision_ : nullptr,
      optimizer_.PolicyFrames(), diagnostics_.solve_time_ms);
  target_window_ = decision.target_window_size; diagnostics_.weights = decision.weights;
  optimizer_.SetWeights(decision.weights);
  return decision;
}
std::uint64_t AdaptiveWindowBackend::AddOrGetNode(const NavState &s, Modality m,
    const MeasurementSummary &summary, const PolicyDecision &decision) {
  auto id = optimizer_.FindNodeNear(s.timestamp, config_.merge_time_tolerance);
  if (id) { optimizer_.MarkNode(id, m, summary.normalized_information); return id; }
  if (!decision.select_keyframe) return 0;
  std::shared_ptr<ImuPreintegration> preintegration;
  std::uint64_t previous = latest_node_;
  NavState previous_state;
  if (previous) {
    previous_state = optimizer_.EstimateNode(previous);
    if (s.timestamp <= previous_state.timestamp) return 0;
    preintegration = std::make_shared<ImuPreintegration>(config_);
    if (!preintegration->Integrate(pending_imu_, previous_state.timestamp, s.timestamp,
                                   previous_state.bias_a, previous_state.bias_g)) {
      ++diagnostics_.rejected_imu_intervals;
      // Do not create an unconnected node or extrapolate absent IMU data.
      if (s.timestamp - previous_state.timestamp <= std::max(1.0, 3.0 * config_.max_keyframe_dt)) return 0;
      Recover(); previous = 0; preintegration.reset();
    }
  }
  id = optimizer_.AddNode(s);
  if (!id) return 0;
  const auto factor = previous ? FactorBuilders::Imu(previous, id, previous_state.gravity_w, preintegration) :
                                 FactorBuilders::InitialState(id, s, config_);
  if (!optimizer_.AddFactor(factor)) { Recover(); return 0; }
  latest_node_ = id; optimizer_.MarkNode(id, m, summary.normalized_information);
  ConsumeImuThrough(s.timestamp);
  return id;
}

bool AdaptiveWindowBackend::ProcessLidar(const NavState &s, const MeasurementSummary &input,
                                        const LidarPlaneObservationVector &observations) {
  if (!std::isfinite(s.timestamp) || s.timestamp <= last_lidar_time_ ||
      (latest_node_ && s.timestamp < optimizer_.EstimateNode(latest_node_).timestamp - config_.merge_time_tolerance)) return false;
  last_lidar_time_ = s.timestamp;
  const auto decision = Decide(s, input, Modality::kLidar);
  const auto id = AddOrGetNode(s, Modality::kLidar, latest_lidar_, decision);
  if (!id) return false;
  associator_.Add(Modality::kLidar, s.timestamp, id);
  lidar_frames_[id] = {s, observations};
  const std::size_t limit = static_cast<std::size_t>(config_.max_lidar_factors);
  const std::size_t stride = limit ? std::max<std::size_t>(1, observations.size() / limit) : 1;
  std::size_t count = 0;
  for (std::size_t i = 0; i < observations.size() && count < limit; i += stride)
    if (optimizer_.AddFactor(FactorBuilders::Lidar(id, lidar_extrinsic_, observations[i], config_.huber_delta_lidar))) ++count;
  return OptimizeAndAdapt();
}
bool AdaptiveWindowBackend::ProcessVision(const NavState &s, const MeasurementSummary &input,
    const VisualPatchObservationVector &observations, const ImageSampleFunction &sampler) {
  if (!sampler || !std::isfinite(s.timestamp) || s.timestamp <= last_image_time_ ||
      (latest_node_ && s.timestamp < optimizer_.EstimateNode(latest_node_).timestamp - config_.merge_time_tolerance)) return false;
  last_image_time_ = s.timestamp;
  const auto decision = Decide(s, input, Modality::kVision);
  const auto id = AddOrGetNode(s, Modality::kVision, latest_vision_, decision);
  if (!id) return false;
  // One nearest retained LiDAR frame is selected for this image. Native times
  // remain attached to both states; no equality/retimestamping is assumed.
  const auto match = associator_.FindNearest(Modality::kLidar, s.timestamp);
  const std::size_t limit = static_cast<std::size_t>(config_.max_cross_modal_factors);
  const std::size_t stride = limit ? std::max<std::size_t>(1, observations.size() / limit) : 1;
  std::set<std::uint64_t> seen;
  std::size_t accepted = 0;
  for (std::size_t i = 0; i < observations.size() && accepted < limit; i += stride) {
    const auto &o = observations[i];
    if (!ValidPatch(o) || !seen.insert(o.track_id).second) continue;
    auto it = patches_.find(o.track_id);
    if (it != patches_.end()) {
      const auto &patch = it->second;
      if (patch.anchor != id && optimizer_.AddFactor(FactorBuilders::Photometric(
          patch.anchor, id, o.track_id, patch.reference, o.inverse_exposure, projection_, sampler,
          config_.huber_delta_vision, Clamp(o.tracking_score, 0.01, 1.0)))) ++accepted;
      continue;
    }
    if (!match.valid || !lidar_frames_.count(match.node_id) ||
        patches_.size() >= static_cast<std::size_t>(config_.max_active_patches)) continue;
    const auto &frame = lidar_frames_.at(match.node_id);
    const Eigen::Vector3d query_i = frame.seed.q_w_i.conjugate() * (o.point_w - frame.seed.p_w_i);
    const Eigen::Vector3d query_l = lidar_extrinsic_.q_i_l.conjugate() * (query_i - lidar_extrinsic_.p_i_l);
    LocalPlane plane;
    if (!FactorBuilders::FitLocalPlane(query_l, frame.observations, config_, &plane) ||
        !optimizer_.AddDepth(o.track_id, id, o.inverse_depth)) continue;
    if (!optimizer_.AddFactor(FactorBuilders::PatchPlane(id, match.node_id, o.track_id,
        o.center_ray, lidar_extrinsic_, plane, config_.huber_delta_cross_modal,
        match.confidence * Clamp(o.tracking_score, 0.01, 1.0)))) {
      optimizer_.RemoveUnusedDepth(o.track_id); continue;
    }
    patches_.emplace(o.track_id, Patch{id, o}); ++accepted;
    optimizer_.MarkNode(id, Modality::kCrossModal, latest_vision_.normalized_information);
    optimizer_.MarkNode(match.node_id, Modality::kCrossModal, latest_lidar_.normalized_information);
  }
  return OptimizeAndAdapt();
}
bool AdaptiveWindowBackend::OptimizeAndAdapt() {
  const auto start = std::chrono::steady_clock::now();
  double cost = 0.0;
  if (!optimizer_.Solve(&cost)) { Recover(); return false; }
  while (optimizer_.NumNodes() > static_cast<std::size_t>(target_window_)) {
    const auto id = policy_.SelectMarginalizationCandidate(optimizer_.PolicyFrames(), target_window_);
    if (!id || !optimizer_.Marginalize(id)) { Recover(); return false; }
    diagnostics_.last_marginalized_id = id;
    associator_.RemoveNode(id); lidar_frames_.erase(id);
    for (auto it = patches_.begin(); it != patches_.end(); ) {
      if (!optimizer_.HasDepth(it->first)) it = patches_.erase(it); else ++it;
    }
  }
  diagnostics_.solve_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  diagnostics_.solve_success = true; diagnostics_.last_cost = cost;
  diagnostics_.window_size = optimizer_.NumNodes(); diagnostics_.target_window_size = target_window_;
  diagnostics_.factor_count = optimizer_.NumFactors();
  diagnostics_.lidar_factors = optimizer_.CountFactors(Modality::kLidar);
  diagnostics_.vision_factors = optimizer_.CountFactors(Modality::kVision);
  diagnostics_.imu_factors = optimizer_.CountFactors(Modality::kImu);
  diagnostics_.cross_modal_pairs = optimizer_.CountFactors(Modality::kCrossModal);
  const auto ext = optimizer_.EstimateCameraExtrinsic();
  diagnostics_.camera_extrinsic_translation_change = (ext.p_c_i - camera_extrinsic_initial_.p_c_i).norm();
  diagnostics_.camera_extrinsic_rotation_change_deg = Degrees(LogQuaternion(
      camera_extrinsic_initial_.q_c_i.conjugate() * ext.q_c_i).norm());
  return true;
}
bool AdaptiveWindowBackend::LatestEstimate(NavState *state) const {
  if (!state || !latest_node_ || !optimizer_.HasNode(latest_node_)) return false;
  *state = optimizer_.EstimateNode(latest_node_); return true;
}
}}
