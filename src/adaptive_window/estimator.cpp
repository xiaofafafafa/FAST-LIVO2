#include "adaptive_window/estimator.h"
#include <Eigen/Eigenvalues>
#include <ceres/covariance.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace fast_livo { namespace adaptive_window {
namespace {
void Validate(const AdaptiveWindowConfig &c) {
  if (c.min_window_size < 2 || c.max_window_size < c.min_window_size ||
      c.initial_window_size < c.min_window_size || c.initial_window_size > c.max_window_size ||
      c.max_lidar_factors < 0 || c.lidar_effective_factors < 0 ||
      c.max_cross_modal_factors < 0 || c.max_active_patches < 0 || c.solver_iterations < 1)
    throw std::invalid_argument("Invalid adaptive-window counts");
  for (double v : {c.solver_time_limit, c.imu_max_gap, c.pose_prior_translation_sigma,
       c.pose_prior_rotation_deg, c.frontend_pose_translation_sigma, c.frontend_pose_rotation_deg,
       c.frontend_pose_prior_weight, c.frontend_pose_max_translation, c.frontend_pose_max_rotation_deg,
       c.velocity_prior_sigma, c.bias_prior_sigma,
       c.extrinsic_translation_sigma, c.extrinsic_rotation_deg, c.imu_acc_noise,
       c.imu_gyr_noise, c.imu_acc_bias_rw, c.imu_gyr_bias_rw, c.min_inverse_depth,
       c.plane_search_radius, c.plane_max_rms, c.cross_modal_plane_sigma,
       c.association_time_sigma, c.max_keyframe_dt, c.min_modality_weight,
       c.imu_repropagate_acc_bias, c.imu_repropagate_gyr_bias, c.max_modality_weight,
       c.information_eigen_floor, c.keyframe_translation, c.keyframe_rotation_deg,
       c.lidar_rmse_scale, c.vision_rmse_scale, c.degeneracy_eigen_threshold,
       c.lidar_outlier_sigma, c.information_translation_scale,
       c.information_rotation_scale_deg, c.cross_modal_temporal_time_sigma,
       c.cross_modal_temporal_confidence})
    if (!std::isfinite(v) || v <= 0) throw std::invalid_argument("Adaptive-window scales must be finite and positive");
  if (!std::isfinite(c.weak_direction_prior_gain) || c.weak_direction_prior_gain < 0)
    throw std::invalid_argument("Invalid weak-direction prior gain");
  if (!std::isfinite(c.max_inverse_depth) || c.max_inverse_depth <= c.min_inverse_depth ||
      !std::isfinite(c.merge_time_tolerance) || c.merge_time_tolerance < 0 ||
      !std::isfinite(c.association_max_dt) || c.association_max_dt < 0 ||
      !std::isfinite(c.cross_modal_temporal_max_dt) || c.cross_modal_temporal_max_dt < 0 ||
      c.cross_modal_temporal_confidence > 1.0 ||
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
bool ValidState(const NavState &s) {
  return std::isfinite(s.timestamp) && s.q_w_i.coeffs().allFinite() && s.q_w_i.norm() > 1e-6 &&
      s.p_w_i.allFinite() && s.v_w_i.allFinite() && s.bias_a.allFinite() &&
      s.bias_g.allFinite() && s.gravity_w.allFinite();
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

Estimator::Estimator(const AdaptiveWindowConfig &c)
    : config_(c), information_evaluator_(c), policy_(c), associator_(c),
      target_window_(c.initial_window_size) { Validate(c); clearState(); }
void Estimator::setParameter(const LidarExtrinsic &l, const ExtrinsicState &e, ProjectionFunction p) {
  lidar_extrinsic_ = l; camera_extrinsic_initial_ = e; projection_ = std::move(p); clearState();
}
void Estimator::clearState() {
  mapping_snapshot_.clear();
  pending_imu_.clear(); lidar_frames_.clear(); retired_frames_.clear(); patches_.clear(); associator_.Clear();
  information_evaluator_ = InformationEvaluator(config_); policy_ = AdaptivePolicy(config_);
  has_lidar_ = false; has_vision_ = false; latest_node_ = 0;
  latest_lidar_ = MeasurementSummary(); latest_vision_ = MeasurementSummary();
  last_lidar_time_ = last_image_time_ = -std::numeric_limits<double>::infinity();
  target_window_ = config_.initial_window_size; diagnostics_ = BackendDiagnostics();
  Rs.clear(); Ps.clear(); Vs.clear(); Bas.clear(); Bgs.clear(); Headers.clear();
  nodes_.clear(); frontend_poses_.clear(); parameter_blocks_.clear();
  para_Pose.clear(); para_SpeedBias.clear(); para_Feature.clear(); feature_anchors_.clear();
  residual_blocks_.clear(); pre_integrations.clear(); tmp_pre_integration.reset();
  last_marginalization_info.reset(); last_marginalization_factor = ResidualBlockInfo();
  last_marginalization_parameter_blocks.clear();
  next_id_ = 1; frame_count = 0; solver_flag = INITIAL; marginalization_flag = MARGIN_OLD;
  weights_ = ModalityWeights(); camera_extrinsic_ = camera_extrinsic_initial_;
  parameter_blocks_[ExtrinsicKey()] = PackPose(camera_extrinsic_.p_c_i, camera_extrinsic_.q_c_i);
  para_Ex_Pose[0] = parameter_blocks_.at(ExtrinsicKey()).data();
  addResidualBlock(FactorFactory::extrinsicPrior(camera_extrinsic_initial_, config_));
}
void Estimator::recover() {
  const auto resets = diagnostics_.reset_count + 1, rejected = diagnostics_.rejected_imu_intervals;
  const auto rejected_pose = diagnostics_.rejected_pose_refinements;
  const double translation = diagnostics_.max_frontend_translation_correction;
  const double rotation = diagnostics_.max_frontend_rotation_correction_deg;
  const auto next_id = next_id_;
  std::vector<RetiredFrame> retired = std::move(retired_frames_);
  for (const auto &entry : nodes_) {
    RetiredFrame frame;
    frame.id = entry.first;
    frame.optimized = stateEstimate(entry.first);
    const auto frontend = frontend_poses_.find(entry.first);
    frame.frontend = frontend != frontend_poses_.end()
        ? frontend->second : frame.optimized;
    if (!ValidState(frame.optimized)) frame.optimized = frame.frontend;
    retired.push_back(std::move(frame));
  }
  ImuSampleVector samples = std::move(pending_imu_);
  clearState(); pending_imu_ = std::move(samples);
  next_id_ = next_id;  // stable map keys must never be reused after a reset
  retired_frames_ = std::move(retired);
  diagnostics_.reset_count = resets; diagnostics_.rejected_imu_intervals = rejected;
  diagnostics_.rejected_pose_refinements = rejected_pose;
  diagnostics_.max_frontend_translation_correction = translation;
  diagnostics_.max_frontend_rotation_correction_deg = rotation;
}
void Estimator::processIMU(const ImuSampleVector &samples) {
  // Input packets overlap and may contain a future boundary sample. Insert by
  // timestamp, rather than dropping subsequent samples behind that boundary.
  for (const auto &s : samples) {
    if (!std::isfinite(s.timestamp) || !s.acceleration.allFinite() || !s.angular_velocity.allFinite()) continue;
    auto it = std::lower_bound(pending_imu_.begin(), pending_imu_.end(), s.timestamp,
        [](const ImuSample &a, double t) { return a.timestamp < t; });
    if (it != pending_imu_.end() && std::abs(it->timestamp - s.timestamp) < 1e-9) *it = s;
    else pending_imu_.insert(it, s);
  }
  if (!pending_imu_.empty()) consumeImuThrough(pending_imu_.back().timestamp - std::max(5.0, 5.0 * config_.max_keyframe_dt));
}
void Estimator::consumeImuThrough(double t) {
  auto it = std::lower_bound(pending_imu_.begin(), pending_imu_.end(), t,
      [](const ImuSample &a, double time) { return a.timestamp < time; });
  if (it != pending_imu_.begin()) --it;
  pending_imu_.erase(pending_imu_.begin(), it);
}
PolicyDecision Estimator::decide(const NavState &s, const MeasurementSummary &input, Modality m) {
  auto summary = input; summary.modality = m; summary.timestamp = s.timestamp;
  summary = information_evaluator_.Evaluate(summary);
  if (m == Modality::kLidar) { latest_lidar_ = summary; has_lidar_ = true; }
  else { latest_vision_ = summary; has_vision_ = true; }
  // Stale quality statistics must not keep a missing modality at high weight.
  const double age = 2.0 * config_.max_keyframe_dt;
  auto decision = policy_.Decide(s,
      has_lidar_ && s.timestamp - latest_lidar_.timestamp <= age ? &latest_lidar_ : nullptr,
      has_vision_ && s.timestamp - latest_vision_.timestamp <= age ? &latest_vision_ : nullptr,
      policyFrames(), diagnostics_.solve_time_ms);
  target_window_ = decision.target_window_size; diagnostics_.weights = decision.weights;
  diagnostics_.keyframe_utility = decision.keyframe_utility;
  diagnostics_.keyframe_selected = decision.select_keyframe;
  diagnostics_.keyframe_complementarity = decision.complementarity_score;
  setWeights(decision.weights);
  return decision;
}
std::uint64_t Estimator::addOrGetFrame(const NavState &s, Modality m,
    const MeasurementSummary &summary, const PolicyDecision &decision) {
  auto id = findFrameNear(s.timestamp, config_.merge_time_tolerance);
  if (id) {
    // LIO and VIO at the same state time replace one frontend prior. Never append
    // two correlated priors or turn the last backend result into a new reference.
    if (!updateFrontendPose(id, s)) return 0;
    markFrame(id, m, summary.normalized_information); return id;
  }
  if (!decision.select_keyframe) return 0;
  tmp_pre_integration.reset();
  std::uint64_t previous = latest_node_;
  NavState previous_state;
  if (previous) {
    previous_state = stateEstimate(previous);
    if (s.timestamp <= previous_state.timestamp) return 0;
    tmp_pre_integration = std::make_shared<IntegrationBase>(config_);
    if (!tmp_pre_integration->integrate(pending_imu_, previous_state.timestamp, s.timestamp,
                                   previous_state.bias_a, previous_state.bias_g)) {
      ++diagnostics_.rejected_imu_intervals;
      // Do not create an unconnected node or extrapolate absent IMU data.
      if (s.timestamp - previous_state.timestamp <= std::max(1.0, 3.0 * config_.max_keyframe_dt)) return 0;
      recover(); previous = 0; tmp_pre_integration.reset();
    }
  }
  id = addFrame(s);
  if (!id) return 0;
  const auto factor = previous ? FactorFactory::imu(previous, id, previous_state.gravity_w, tmp_pre_integration) :
                                 FactorFactory::initialState(id, s, config_);
  double imu_seed_rms = 0.0;
  if (!addResidualBlock(factor, config_.collect_factor_audit ? &imu_seed_rms : nullptr,
                        config_.collect_factor_audit && previous ?
                            &diagnostics_.event.imu_seed_blocks : nullptr)) {
    recover(); return 0;
  }
  if (previous) diagnostics_.event.imu_seed_rms = imu_seed_rms;
  tmp_pre_integration.reset();
  latest_node_ = id; markFrame(id, m, summary.normalized_information);
  consumeImuThrough(s.timestamp);
  return id;
}

bool Estimator::processLidar(const NavState &s, const MeasurementSummary &input,
                                        const LidarPlaneObservationVector &observations) {
  diagnostics_.event = BackendDiagnostics::EventAudit();
  diagnostics_.event.input_count = observations.size();
  if (!ValidState(s) || s.timestamp <= last_lidar_time_ ||
      (latest_node_ && s.timestamp < stateEstimate(latest_node_).timestamp - config_.merge_time_tolerance)) return false;
  last_lidar_time_ = s.timestamp;
  const auto decision = decide(s, input, Modality::kLidar);
  const auto id = addOrGetFrame(s, Modality::kLidar, latest_lidar_, decision);
  if (!id) return false;
  associator_.Add(Modality::kLidar, s.timestamp, id);
  lidar_frames_[id] = {s, observations};
  const std::size_t limit = static_cast<std::size_t>(config_.max_lidar_factors);
  const std::size_t stride = limit ? std::max<std::size_t>(1, observations.size() / limit) : 1;
  const std::size_t selected = limit ?
      std::min(limit, (observations.size() + stride - 1) / stride) : 0;
  // Plane estimates come from the same frontend map and nearby points share
  // plane uncertainty. Keep broad scan coverage without counting every point
  // as an independent map observation in the fixed-lag objective.
  const double lidar_confidence = selected ? std::min(
      1.0, static_cast<double>(config_.lidar_effective_factors) /
               static_cast<double>(selected)) : 0.0;
  std::size_t count = 0;
  for (std::size_t i = 0; i < observations.size() && count < limit; i += stride)
  {
    const auto &observation = observations[i];
    if (!observation.point_l.allFinite() ||
        !observation.normal_w.allFinite() ||
        !observation.center_w.allFinite() ||
        !std::isfinite(observation.variance) || observation.variance <= 0.0 ||
        observation.normal_w.norm() < 1.0e-6) {
      ++diagnostics_.event.invalid_count; continue;
    }
    const Eigen::Vector3d point_i = lidar_extrinsic_.q_i_l * observation.point_l +
                                    lidar_extrinsic_.p_i_l;
    const Eigen::Vector3d point_w = s.q_w_i * point_i + s.p_w_i;
    const double normalized_distance = std::abs(
        observation.normal_w.normalized().dot(point_w - observation.center_w)) /
        std::sqrt(observation.variance);
    // The frontend gates associations before its last EKF update. Recheck at
    // the final frontend pose so a stale map plane cannot pull the smoother.
    if (!std::isfinite(normalized_distance) ||
        normalized_distance > config_.lidar_outlier_sigma) {
      ++diagnostics_.event.gate_rejected; continue;
    }
    auto factor = FactorFactory::lidar(
        id, lidar_extrinsic_, observation, config_.huber_delta_lidar);
    factor.confidence = lidar_confidence;
    if (addResidualBlock(factor)) {
      ++count;
      if (config_.collect_factor_audit) {
        auto &audit = diagnostics_.event;
        audit.seed_rms += normalized_distance * normalized_distance;
        const Eigen::Vector3d normal = observation.normal_w.normalized();
        const double inverse_sigma = 1.0 / std::sqrt(observation.variance);
        Eigen::Matrix<double, 1, kPoseDim> jacobian;
        jacobian.head<3>() = inverse_sigma * normal.transpose();
        jacobian.tail<3>() = -inverse_sigma * normal.transpose() *
            s.q_w_i.toRotationMatrix() * Skew(point_i);
        const double robust_weight = config_.huber_delta_lidar > 0.0 ?
            std::min(1.0, config_.huber_delta_lidar /
                          std::max(1.0e-9, normalized_distance)) : 1.0;
        audit.lidar_information.noalias() += weights_.lidar *
            lidar_confidence * robust_weight * jacobian.transpose() * jacobian;
      }
    }
  }
  diagnostics_.event.accepted_count = count;
  if (config_.collect_factor_audit && count)
    diagnostics_.event.seed_rms = std::sqrt(diagnostics_.event.seed_rms / count);
  return solveOdometry();
}
bool Estimator::processImage(const NavState &s, const MeasurementSummary &input,
    const VisualPatchObservationVector &observations, const ImageSampleFunction &sampler) {
  diagnostics_.event = BackendDiagnostics::EventAudit();
  diagnostics_.event.input_count = observations.size();
  if (!sampler || !ValidState(s) || s.timestamp <= last_image_time_ ||
      (latest_node_ && s.timestamp < stateEstimate(latest_node_).timestamp - config_.merge_time_tolerance)) return false;
  last_image_time_ = s.timestamp;
  const auto decision = decide(s, input, Modality::kVision);
  const auto id = addOrGetFrame(s, Modality::kVision, latest_vision_, decision);
  if (!id) return false;
  // One nearest retained LiDAR frame is selected for this image. Native times
  // remain attached to both states; no equality/retimestamping is assumed.
  const auto match = associator_.FindNearest(Modality::kLidar, s.timestamp);
  if (match.valid) diagnostics_.event.association_dt = match.delta_time;
  const std::size_t limit = static_cast<std::size_t>(config_.max_cross_modal_factors);
  const std::size_t stride = limit ? std::max<std::size_t>(1, observations.size() / limit) : 1;
  std::set<std::uint64_t> seen;
  std::size_t accepted = 0;
  for (std::size_t i = 0; i < observations.size() && accepted < limit; i += stride) {
    const auto &o = observations[i];
    if (!ValidPatch(o) || !seen.insert(o.track_id).second) {
      ++diagnostics_.event.invalid_count; continue;
    }
    auto it = patches_.find(o.track_id);
    if (it != patches_.end()) {
      const auto &patch = it->second;
      double rms = 0.0;
      if (patch.anchor != id && addResidualBlock(FactorFactory::photometric(
          patch.anchor, id, o.track_id, patch.reference, o.inverse_exposure, projection_, sampler,
          config_.huber_delta_vision, Clamp(o.tracking_score, 0.01, 1.0)),
          config_.collect_factor_audit ? &rms : nullptr)) {
        ++accepted;
        ++diagnostics_.event.photo_count;
        if (config_.collect_factor_audit) diagnostics_.event.photo_seed_rms += rms * rms;
      }
      continue;
    }
    if (!match.valid || !lidar_frames_.count(match.node_id) ||
        patches_.size() >= static_cast<std::size_t>(config_.max_active_patches)) {
      ++diagnostics_.event.no_match; continue;
    }
    const auto &frame = lidar_frames_.at(match.node_id);
    const Eigen::Vector3d query_i = frame.seed.q_w_i.conjugate() * (o.point_w - frame.seed.p_w_i);
    const Eigen::Vector3d query_l = lidar_extrinsic_.q_i_l.conjugate() * (query_i - lidar_extrinsic_.p_i_l);
    LocalPlane plane;
    if (!FactorFactory::fitLocalPlane(query_l, frame.observations, config_, &plane)) {
      ++diagnostics_.event.plane_failed; continue;
    }
    if (!addFeature(o.track_id, id, o.inverse_depth)) continue;
    double rms = 0.0;
    if (!addResidualBlock(FactorFactory::patchPlane(id, match.node_id, o.track_id,
        o.center_ray, lidar_extrinsic_, plane, config_.huber_delta_cross_modal,
        match.confidence * Clamp(o.tracking_score, 0.01, 1.0)),
        config_.collect_factor_audit ? &rms : nullptr)) {
      removeUnusedFeature(o.track_id); continue;
    }
    ++diagnostics_.event.cross_count;
    if (config_.collect_factor_audit) diagnostics_.event.cross_seed_rms += rms * rms;
    // A same-pose image/LiDAR plane tie constrains inverse depth but its pose
    // Jacobians cancel. Optionally tie the same patch to a nearby *different*
    // retained scan so its independently fitted plane can constrain relative
    // pose. Keep the same-pose depth anchor and discount scan correlation.
    if (config_.cross_modal_temporal_max_dt > 0.0 && match.node_id == id) {
      std::uint64_t older_id = 0;
      const LidarFrame *older = nullptr;
      double older_dt = config_.cross_modal_temporal_max_dt;
      for (const auto &candidate : lidar_frames_) {
        const double dt = s.timestamp - candidate.second.seed.timestamp;
        if (candidate.first == id || !hasFrame(candidate.first) ||
            dt <= config_.merge_time_tolerance || dt > older_dt) continue;
        older_id = candidate.first;
        older = &candidate.second;
        older_dt = dt;
      }
      if (older) {
        const Eigen::Vector3d old_i = older->seed.q_w_i.conjugate() *
            (o.point_w - older->seed.p_w_i);
        const Eigen::Vector3d old_l = lidar_extrinsic_.q_i_l.conjugate() *
            (old_i - lidar_extrinsic_.p_i_l);
        LocalPlane older_plane;
        if (FactorFactory::fitLocalPlane(old_l, older->observations,
                                         config_, &older_plane)) {
          const double time_ratio = older_dt / config_.cross_modal_temporal_time_sigma;
          const double confidence = config_.cross_modal_temporal_confidence *
              std::exp(-0.5 * time_ratio * time_ratio) *
              Clamp(o.tracking_score, 0.01, 1.0);
          double temporal_rms = 0.0;
          if (addResidualBlock(FactorFactory::patchPlane(
                  id, older_id, o.track_id, o.center_ray, lidar_extrinsic_,
                  older_plane, config_.huber_delta_cross_modal, confidence),
                  config_.collect_factor_audit ? &temporal_rms : nullptr)) {
            ++diagnostics_.event.cross_count;
            ++diagnostics_.event.temporal_cross_count;
            if (config_.collect_factor_audit)
              diagnostics_.event.cross_seed_rms += temporal_rms * temporal_rms;
            markFrame(older_id, Modality::kCrossModal,
                      latest_lidar_.normalized_information);
          }
        } else {
          ++diagnostics_.event.temporal_plane_failed;
        }
      }
    }
    patches_.emplace(o.track_id, Patch{id, o}); ++accepted;
    markFrame(id, Modality::kCrossModal, latest_vision_.normalized_information);
    markFrame(match.node_id, Modality::kCrossModal, latest_lidar_.normalized_information);
  }
  diagnostics_.event.accepted_count = accepted;
  if (config_.collect_factor_audit) {
    if (diagnostics_.event.photo_count)
      diagnostics_.event.photo_seed_rms = std::sqrt(
          diagnostics_.event.photo_seed_rms / diagnostics_.event.photo_count);
    if (diagnostics_.event.cross_count)
      diagnostics_.event.cross_seed_rms = std::sqrt(
          diagnostics_.event.cross_seed_rms / diagnostics_.event.cross_count);
  }
  return solveOdometry();
}
bool Estimator::solveOdometry() {
  const auto start = std::chrono::steady_clock::now();
  double cost = 0.0;
  if (!optimization(&cost)) { recover(); return false; }
  if (config_.collect_factor_audit) {
    auto &audit = diagnostics_.event;
    double lidar_squared = 0.0;
    std::size_t lidar_count = 0;
    for (const auto &factor : residual_blocks_) {
      const bool is_current_lidar = factor.modality == Modality::kLidar &&
          factor.keys.size() == 1 && factor.keys[0] == PoseKey(latest_node_);
      const bool is_current_imu = factor.modality == Modality::kImu &&
          factor.keys.size() >= 3 && factor.keys[2] == PoseKey(latest_node_);
      if (!is_current_lidar && !is_current_imu) continue;
      Eigen::VectorXd residual;
      if (!factor.Evaluate(parameter_blocks_, &residual)) continue;
      if (is_current_lidar) {
        lidar_squared += residual.squaredNorm();
        lidar_count += residual.size();
      } else if (residual.size() == 15) {
        audit.imu_post_rms = std::sqrt(residual.squaredNorm() / residual.size());
        for (int block = 0; block < 5; ++block)
          audit.imu_post_blocks[block] = std::sqrt(
              residual.segment<3>(3 * block).squaredNorm() / 3.0);
      }
    }
    if (lidar_count) audit.lidar_post_rms = std::sqrt(lidar_squared / lidar_count);
  }
  if (!slideWindow()) { recover(); return false; }
  diagnostics_.solve_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  diagnostics_.solve_success = true; diagnostics_.last_cost = cost;
  diagnostics_.window_size = numFrames(); diagnostics_.target_window_size = target_window_;
  diagnostics_.factor_count = numFactors();
  diagnostics_.lidar_factors = countFactors(Modality::kLidar);
  diagnostics_.vision_factors = countFactors(Modality::kVision);
  diagnostics_.imu_factors = countFactors(Modality::kImu);
  diagnostics_.frontend_pose_factors = countFactors(Modality::kFrontendPose);
  diagnostics_.cross_modal_pairs = countFactors(Modality::kCrossModal);
  const auto ext = cameraExtrinsicEstimate();
  diagnostics_.camera_extrinsic_translation_change = (ext.p_c_i - camera_extrinsic_initial_.p_c_i).norm();
  diagnostics_.camera_extrinsic_rotation_change_deg = Degrees(LogQuaternion(
      camera_extrinsic_initial_.q_c_i.conjugate() * ext.q_c_i).norm());
  return true;
}
bool Estimator::latestEstimate(NavState *state) const {
  if (!state || !latest_node_ || !hasFrame(latest_node_)) return false;
  *state = stateEstimate(latest_node_); return true;
}

bool Estimator::shouldAttemptFrame(const NavState &candidate) const {
  if (!ValidState(candidate)) return false;
  if (nodes_.empty()) return true;
  if (findFrameNear(candidate.timestamp, config_.merge_time_tolerance)) return true;
  const NavState last = stateEstimate(latest_node_);
  const double dt = candidate.timestamp - last.timestamp;
  if (dt < config_.min_keyframe_dt) return false;
  if (dt >= config_.max_keyframe_dt) return true;
  const double translation = (candidate.p_w_i - last.p_w_i).norm();
  const double rotation = Degrees(LogQuaternion(last.q_w_i.conjugate() * candidate.q_w_i).norm());
  const double motion_score = std::max(
      Clamp(translation / config_.keyframe_translation, 0.0, 1.0),
      Clamp(rotation / config_.keyframe_rotation_deg, 0.0, 1.0));
  // Upper bound on Decide()'s utility: quality and complementarity each <= 1.
  // Skipping below this bound cannot reject a keyframe the full policy selects.
  const double upper_utility = 0.30 * motion_score + 0.25 +
      0.25 * Clamp(dt / config_.max_keyframe_dt, 0.0, 1.0) + 0.20;
  return upper_utility >= config_.keyframe_utility_threshold;
}

std::vector<RetiredFrame> Estimator::takeRetiredFrames() {
  std::vector<RetiredFrame> result;
  result.swap(retired_frames_);
  return result;
}

std::uint64_t Estimator::addFrame(const NavState &s) {
  if (!ValidState(s) || (!Headers.empty() && s.timestamp <= Headers.rbegin()->second)) return 0;
  const std::uint64_t id = next_id_++;
  PolicyFrame frame; frame.id = id; frame.state = s;
  nodes_.emplace(id, frame); Headers[id] = s.timestamp;
  Rs[id] = s.q_w_i.normalized().toRotationMatrix(); Ps[id] = s.p_w_i;
  Vs[id] = s.v_w_i; Bas[id] = s.bias_a; Bgs[id] = s.bias_g;
  parameter_blocks_[PoseKey(id)] = PackPose(s.p_w_i, s.q_w_i);
  parameter_blocks_[MotionKey(id)] = {s.v_w_i.x(), s.v_w_i.y(), s.v_w_i.z(),
      s.bias_a.x(), s.bias_a.y(), s.bias_a.z(), s.bias_g.x(), s.bias_g.y(), s.bias_g.z()};
  para_Pose[id] = parameter_blocks_.at(PoseKey(id)).data();
  para_SpeedBias[id] = parameter_blocks_.at(MotionKey(id)).data();
  if (!updateFrontendPose(id, s)) { recover(); return 0; }
  latest_node_ = id; frame_count = static_cast<int>(nodes_.size());
  return id;
}
std::uint64_t Estimator::findFrameNear(double t, double tolerance) const {
  for (auto it = Headers.rbegin(); it != Headers.rend(); ++it)
    if (std::abs(it->second - t) <= tolerance) return it->first;
  return 0;
}
NavState Estimator::stateEstimate(std::uint64_t id) const {
  NavState s = nodes_.at(id).state;
  s.timestamp = Headers.at(id); s.p_w_i = Ps.at(id); s.q_w_i = Eigen::Quaterniond(Rs.at(id));
  s.v_w_i = Vs.at(id); s.bias_a = Bas.at(id); s.bias_g = Bgs.at(id);
  return s;
}
bool Estimator::frontendState(std::uint64_t id, NavState *state) const {
  const auto it = frontend_poses_.find(id);
  if (!state || it == frontend_poses_.end()) return false;
  *state = it->second;
  return true;
}
ExtrinsicState Estimator::cameraExtrinsicEstimate() const { return camera_extrinsic_; }
void Estimator::markFrame(std::uint64_t id, Modality m, double info) {
  if (!hasFrame(id)) return;
  auto &f = nodes_.at(id); f.unique_information = std::max(f.unique_information, info);
  if (m == Modality::kLidar) f.has_lidar = true;
  if (m == Modality::kVision) f.has_vision = true;
  if (m == Modality::kCrossModal) f.has_cross_modal = true;
}
bool Estimator::updateFrontendPose(std::uint64_t id, const NavState &s) {
  if (!hasFrame(id) || !ValidState(s) ||
      std::abs(Headers.at(id) - s.timestamp) > config_.merge_time_tolerance) return false;
  if (config_.frontend_pose_prior_enabled) {
    auto factor = FactorFactory::frontendPose(id, s, config_, directionalPriorSqrt(s.timestamp));
    Eigen::VectorXd residual;
    if (!factor.Evaluate(parameter_blocks_, &residual)) return false;
    auto it = std::find_if(residual_blocks_.begin(), residual_blocks_.end(),
        [id](const ResidualBlockInfo &f) {
          return f.modality == Modality::kFrontendPose && f.keys.front() == PoseKey(id);
        });
    if (it == residual_blocks_.end()) residual_blocks_.push_back(std::move(factor));
    else *it = std::move(factor);
  }
  frontend_poses_[id] = s; frontend_poses_.at(id).q_w_i.normalize();
  // New states and same-time LIO/VIO updates start at the latest filter pose.
  // Already-retained, unrelated states keep their previous refined estimates.
  Ps.at(id) = s.p_w_i; Rs.at(id) = s.q_w_i.normalized().toRotationMatrix();
  const auto pose = PackPose(s.p_w_i, s.q_w_i);
  std::copy(pose.begin(), pose.end(), para_Pose.at(id));
  return true;
}
Eigen::Matrix<double, kPoseDim, kPoseDim> Estimator::directionalPriorSqrt(double timestamp) const {
  using Matrix6 = Eigen::Matrix<double, kPoseDim, kPoseDim>;
  if (config_.weak_direction_prior_gain == 0.0) return Matrix6::Identity();
  AdaptiveWindowConfig reference = config_;
  reference.information_translation_scale = config_.frontend_pose_translation_sigma;
  reference.information_rotation_scale_deg = config_.frontend_pose_rotation_deg;
  Matrix6 information_rp = Matrix6::Zero();
  int modalities = 0;
  const double maximum_age = 2.0 * config_.max_keyframe_dt;
  for (const auto *summary : {&latest_lidar_, &latest_vision_}) {
    if (!summary->valid || timestamp - summary->timestamp > maximum_age ||
        timestamp < summary->timestamp - config_.merge_time_tolerance) continue;
    information_rp += InformationEvaluator::NormalizedPoseInformation(*summary, reference);
    ++modalities;
  }
  if (modalities == 0) return Matrix6::Identity();
  Matrix6 swap = Matrix6::Zero();
  swap.topRightCorner<3, 3>().setIdentity();
  swap.bottomLeftCorner<3, 3>().setIdentity();
  const Matrix6 information_pr = swap * information_rp * swap;
  Eigen::SelfAdjointEigenSolver<Matrix6> eigen(information_pr);
  if (eigen.info() != Eigen::Success) return Matrix6::Identity();
  Eigen::Matrix<double, kPoseDim, 1> roots;
  for (int i = 0; i < kPoseDim; ++i)
    roots(i) = std::sqrt(1.0 + config_.weak_direction_prior_gain /
                              (1.0 + std::max(0.0, eigen.eigenvalues()(i))));
  return eigen.eigenvectors() * roots.asDiagonal() * eigen.eigenvectors().transpose();
}
bool Estimator::addFeature(std::uint64_t track, std::uint64_t anchor, double rho) {
  if (!hasFrame(anchor) || hasFeature(track) || !std::isfinite(rho) ||
      rho < config_.min_inverse_depth || rho > config_.max_inverse_depth) return false;
  parameter_blocks_[DepthKey(track)] = {rho}; feature_anchors_[track] = anchor;
  para_Feature[track] = parameter_blocks_.at(DepthKey(track)).data(); return true;
}
void Estimator::removeUnusedFeature(std::uint64_t track) {
  const auto key = DepthKey(track);
  for (const auto &f : residual_blocks_) for (const auto &k : f.keys) if (k == key) return;
  for (const auto &k : last_marginalization_factor.keys) if (k == key) return;
  para_Feature.erase(track); parameter_blocks_.erase(key); feature_anchors_.erase(track);
}
bool Estimator::addResidualBlock(const ResidualBlockInfo &f, double *initial_rms,
                                 std::array<double, 5> *initial_imu_blocks) {
  if (!f.cost || !std::isfinite(f.confidence) || f.confidence <= 0 ||
      f.keys.size() != f.cost->parameter_block_sizes().size()) return false;
  std::set<BlockKey> unique;
  for (std::size_t k = 0; k < f.keys.size(); ++k)
    if (!unique.insert(f.keys[k]).second || f.keys[k].size() != f.cost->parameter_block_sizes()[k]) return false;
  Eigen::VectorXd r; std::vector<Eigen::MatrixXd> j;
  if (!f.Evaluate(parameter_blocks_, &r, &j)) return false;
  if (initial_rms)
    *initial_rms = r.size() ? std::sqrt(r.squaredNorm() / r.size()) : 0.0;
  if (initial_imu_blocks && r.size() == 15)
    for (int block = 0; block < 5; ++block)
      (*initial_imu_blocks)[block] = std::sqrt(
          r.segment<3>(3 * block).squaredNorm() / 3.0);
  // Frontend priors must go through updateFrontendPose(), which replaces rather
  // than appends; this prevents accidental duplicate anchors through this API.
  if (f.modality == Modality::kFrontendPose) return false;
  residual_blocks_.push_back(f);
  if (f.pre_integration) pre_integrations[f.keys.at(2).id] = f.pre_integration;
  return true;
}
void Estimator::vector2double() {
  for (const auto &entry : nodes_) {
    const auto id = entry.first;
    const auto pose = PackPose(Ps.at(id), Eigen::Quaterniond(Rs.at(id)));
    std::copy(pose.begin(), pose.end(), para_Pose.at(id));
    double *speed_bias = para_SpeedBias.at(id);
    for (int k = 0; k < 3; ++k) {
      speed_bias[k] = Vs.at(id)[k];
      speed_bias[k + 3] = Bas.at(id)[k]; speed_bias[k + 6] = Bgs.at(id)[k];
    }
  }
  const auto ext = PackPose(camera_extrinsic_.p_c_i, camera_extrinsic_.q_c_i);
  std::copy(ext.begin(), ext.end(), para_Ex_Pose[0]);
  // Inverse depths already live directly in para_Feature.
}
void Estimator::double2vector() {
  for (const auto &entry : nodes_) {
    const auto id = entry.first;
    Ps.at(id) = Position(para_Pose.at(id)); Rs.at(id) = Rotation(para_Pose.at(id)).toRotationMatrix();
    Vs.at(id) = Position(para_SpeedBias.at(id));
    Bas.at(id) = Position(para_SpeedBias.at(id) + 3); Bgs.at(id) = Position(para_SpeedBias.at(id) + 6);
  }
  camera_extrinsic_.p_c_i = Position(para_Ex_Pose[0]);
  camera_extrinsic_.q_c_i = Rotation(para_Ex_Pose[0]);
  // No VINS-style gauge/yaw realignment: these poses use FAST-LIVO2's world frame.
}
bool Estimator::failureDetection() {
  double max_translation = 0.0, max_rotation_deg = 0.0;
  for (const auto &entry : frontend_poses_) {
    const double *pose = para_Pose.at(entry.first);
    const double translation = (Position(pose) - entry.second.p_w_i).norm();
    const double rotation = Degrees(QuaternionLog(entry.second.q_w_i.conjugate() * Rotation(pose)).norm());
    if (!std::isfinite(translation) || !std::isfinite(rotation)) return true;
    max_translation = std::max(max_translation, translation);
    max_rotation_deg = std::max(max_rotation_deg, rotation);
  }
  diagnostics_.max_frontend_translation_correction = max_translation;
  diagnostics_.max_frontend_rotation_correction_deg = max_rotation_deg;
  return config_.frontend_pose_prior_enabled &&
      (max_translation > config_.frontend_pose_max_translation ||
       max_rotation_deg > config_.frontend_pose_max_rotation_deg);
}
bool Estimator::optimization(double *final_cost) {
  if (frame_count == 0 || residual_blocks_.empty()) return false;
  vector2double();
  for (auto &f : residual_blocks_) {
    if (!f.pre_integration) continue;
    const auto &motion = parameter_blocks_.at(f.keys[1]);
    const auto ba = Position(motion.data() + 3), bg = Position(motion.data() + 6);
    if (f.pre_integration->needsRepropagation(ba, bg) && !f.pre_integration->repropagate(ba, bg)) return false;
  }
  const auto backup = parameter_blocks_;
  ceres::Problem::Options po;
  po.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  po.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  po.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
#else
  po.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
#endif
  std::vector<std::unique_ptr<ceres::LossFunction>> losses;
  ceres::Problem problem(po);
  for (auto &entry : parameter_blocks_) {
    const auto &key = entry.first; auto &data = entry.second;
    problem.AddParameterBlock(data.data(), key.size(), key.is_pose() ? &local_parameterization_ : nullptr);
    if (key.kind == BlockKind::kDepth) {
      problem.SetParameterLowerBound(data.data(), 0, config_.min_inverse_depth);
      problem.SetParameterUpperBound(data.data(), 0, config_.max_inverse_depth);
    }
    if (key.kind == BlockKind::kExtrinsic && !config_.optimize_extrinsics) problem.SetParameterBlockConstant(data.data());
  }
  auto add = [&](const ResidualBlockInfo &f, const std::vector<double *> &parameters) {
    auto *robust = f.huber_delta > 0.0 ? new ceres::HuberLoss(f.huber_delta) : nullptr;
    losses.emplace_back(new ceres::ScaledLoss(robust, residualWeight(f, weights_), ceres::TAKE_OWNERSHIP));
    problem.AddResidualBlock(f.cost.get(), losses.back().get(), parameters);
  };
  for (const auto &f : residual_blocks_) {
    std::vector<double *> parameters;
    for (const auto &key : f.keys) parameters.push_back(parameter_blocks_.at(key).data());
    add(f, parameters);
  }
  if (last_marginalization_factor.cost) add(last_marginalization_factor, last_marginalization_parameter_blocks);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = config_.solver_iterations;
  options.max_solver_time_in_seconds = config_.solver_time_limit;
  options.num_threads = 1;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  bool success = summary.IsSolutionUsable() && std::isfinite(summary.final_cost);
  for (const auto &entry : parameter_blocks_) for (double value : entry.second) success = success && std::isfinite(value);
  if (success && failureDetection()) { ++diagnostics_.rejected_pose_refinements; success = false; }
  if (!success) {
    // Restore in place; the still-live Ceres problem and para_* views own no
    // storage and require their parameter addresses to remain unchanged.
    for (auto &entry : parameter_blocks_)
      std::copy(backup.at(entry.first).begin(), backup.at(entry.first).end(), entry.second.begin());
    return false;
  }
  double2vector(); solver_flag = NON_LINEAR;
  if (snapshot_enabled_) {
    // Take the final pose of every node *before* slideWindow can delete it.
    // These are diagonal marginal blocks of the joint problem, not inverse
    // pose-only Hessians. A rank failure is explicit; it is not zero noise.
    mapping_snapshot_.clear();
    ceres::Covariance::Options opt;
    opt.algorithm_type = ceres::DENSE_SVD;
    opt.null_space_rank = 0;
    ceres::Covariance covariance(opt);
    std::vector<std::pair<const double*, const double*>> blocks;
    for (const auto& p : para_Pose) blocks.emplace_back(p.second, p.second);
    const bool available = covariance_enabled_ && covariance.Compute(blocks, &problem);
    for (const auto& p : para_Pose) {
      MappingPose f; f.id = p.first; f.state = stateEstimate(f.id);
      Eigen::Matrix<double,6,6,Eigen::RowMajor> cov;
      if (available && covariance.GetCovarianceBlockInTangentSpace(p.second, p.second, cov.data())
          && cov.allFinite()) { f.covariance = cov; f.covariance_valid = true; }
      mapping_snapshot_.push_back(f);
    }
  }
  if (final_cost) *final_cost = summary.final_cost;
  return true;
}
bool Estimator::slideWindow() {
  if (solver_flag != NON_LINEAR) return false;
  while (frame_count > target_window_) {
    const auto id = policy_.SelectMarginalizationCandidate(policyFrames(), target_window_);
    if (!id) return false;
    if (id == nodes_.begin()->first) marginalization_flag = MARGIN_OLD;
    else {
      auto second_newest = std::prev(nodes_.end(), 2);
      marginalization_flag = id == second_newest->first ? MARGIN_SECOND_NEW : MARGIN_SELECTED;
    }
    const bool success = marginalization_flag == MARGIN_OLD ? slideWindowOld() :
        (marginalization_flag == MARGIN_SECOND_NEW ? slideWindowNew() : marginalizeFrame(id));
    if (!success) return false;
  }
  return true;
}
bool Estimator::slideWindowOld() {
  return nodes_.size() > 1 && marginalizeFrame(nodes_.begin()->first);
}
bool Estimator::slideWindowNew() {
  return nodes_.size() > 1 && marginalizeFrame(std::prev(nodes_.end(), 2)->first);
}
bool Estimator::marginalizeFrame(std::uint64_t id) {
  if (!hasFrame(id) || nodes_.size() < 2 || id == nodes_.rbegin()->first) return false;
  diagnostics_.last_marginalized_age = Headers.at(nodes_.rbegin()->first) - Headers.at(id);
  diagnostics_.last_marginalized_information = nodes_.at(id).unique_information;
  std::set<BlockKey> drop{PoseKey(id), MotionKey(id)};
  for (const auto &entry : feature_anchors_) if (entry.second == id) drop.insert(DepthKey(entry.first));
  auto marginalization_info = std::make_shared<MarginalizationInfo>();
  std::vector<std::size_t> absorbed;
  for (std::size_t i = 0; i < residual_blocks_.size(); ++i) {
    bool touches = false;
    for (const auto &key : residual_blocks_[i].keys) touches = touches || drop.count(key) != 0;
    if (touches) { absorbed.push_back(i); marginalization_info->addResidualBlockInfo(residual_blocks_[i]); }
  }
  if (last_marginalization_factor.cost) marginalization_info->addResidualBlockInfo(last_marginalization_factor);
  if (!marginalization_info->preMarginalize(parameter_blocks_, drop, weights_) ||
      !marginalization_info->marginalize()) return false;
  auto prior = marginalization_info->getFactor();
  Eigen::VectorXd check;
  if (!prior.Evaluate(parameter_blocks_, &check)) return false;
  RetiredFrame retired;
  retired.id = id;
  retired.optimized = stateEstimate(id);
  const auto frontend = frontend_poses_.find(id);
  retired.frontend = frontend != frontend_poses_.end() ? frontend->second : retired.optimized;
  // Commit only the selected factors. Raw retained factors remain nonlinear and
  // are not counted a second time in the prior. Removed frontend anchors are
  // absorbed exactly once, along with the IMU/visual/LiDAR factors they touch.
  for (auto it = absorbed.rbegin(); it != absorbed.rend(); ++it) {
    const auto &f = residual_blocks_[*it];
    if (f.pre_integration) pre_integrations.erase(f.keys.at(2).id);
    residual_blocks_.erase(residual_blocks_.begin() + *it);
  }
  last_marginalization_info = std::move(marginalization_info);
  last_marginalization_factor = std::move(prior);
  last_marginalization_parameter_blocks.clear();
  for (const auto &key : last_marginalization_factor.keys)
    last_marginalization_parameter_blocks.push_back(parameter_blocks_.at(key).data());
  for (const auto &key : drop) {
    if (key.kind == BlockKind::kDepth) { para_Feature.erase(key.id); feature_anchors_.erase(key.id); }
    parameter_blocks_.erase(key);
  }
  para_Pose.erase(id); para_SpeedBias.erase(id);
  nodes_.erase(id); frontend_poses_.erase(id); Headers.erase(id);
  Rs.erase(id); Ps.erase(id); Vs.erase(id); Bas.erase(id); Bgs.erase(id);
  frame_count = static_cast<int>(nodes_.size()); diagnostics_.last_marginalized_id = id;
  retired_frames_.push_back(std::move(retired));
  associator_.RemoveNode(id); lidar_frames_.erase(id);
  for (auto it = patches_.begin(); it != patches_.end(); ) {
    if (!hasFeature(it->first)) it = patches_.erase(it); else ++it;
  }
  // Do not concatenate IMU intervals for an interior removal. Their information
  // is already carried by the Schur prior; readding an edge would double count.
  return true;
}
PolicyFrameVector Estimator::policyFrames() const {
  PolicyFrameVector out;
  for (const auto &entry : nodes_) {
    auto frame = entry.second;
    frame.state = stateEstimate(entry.first);
    frame.active_patch_tracks = 0;
    if (config_.preserve_visual_anchors)
      for (const auto &patch : patches_)
        if (patch.second.anchor == entry.first) ++frame.active_patch_tracks;
    out.push_back(frame);
  }
  return out;
}
std::vector<std::uint64_t> Estimator::frameIds() const {
  std::vector<std::uint64_t> ids; for (const auto &entry : nodes_) ids.push_back(entry.first); return ids;
}
std::size_t Estimator::countFactors(Modality m) const {
  return std::count_if(residual_blocks_.begin(), residual_blocks_.end(),
      [m](const ResidualBlockInfo &f) { return f.modality == m; });
}
}}
