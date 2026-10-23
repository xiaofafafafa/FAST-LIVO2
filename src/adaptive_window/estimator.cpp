/**
 * @file estimator.cpp
 * @brief 后端主流程：量测接纳、因子构建、Ceres 求解、修正检查、Schur 边缘化及恢复。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#include "adaptive_window/estimator.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace fast_livo { namespace adaptive_window {
namespace {
// 配置校验：拒绝非法窗口计数、非有限/非正尺度以及不一致上下界，避免求解器接收错误参数。
void Validate(const AdaptiveWindowConfig &c) {
  if (c.min_window_size < 2 || c.max_window_size < c.min_window_size ||
      c.initial_window_size < c.min_window_size || c.initial_window_size > c.max_window_size ||
      c.max_lidar_factors < 0 || c.max_cross_modal_factors < 0 || c.max_active_patches < 0 || c.solver_iterations < 1)
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
// 检查时间、姿态、位置、速度、偏置和重力有限性；四元数不能接近零。
bool ValidState(const NavState &s) {
  return std::isfinite(s.timestamp) && s.q_w_i.coeffs().allFinite() && s.q_w_i.norm() > 1e-6 &&
      s.p_w_i.allFinite() && s.v_w_i.allFinite() && s.bias_a.allFinite() &&
      s.bias_g.allFinite() && s.gravity_w.allFinite();
}
// 检查 patch 内容与切平面射线交点是否可用；逆深度边界在 addFeature 中另行检查。
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

// 构造独立后端；FAST-LIVO2 已负责前端初始化，此处直接准备空窗口。
Estimator::Estimator(const AdaptiveWindowConfig &c)
    : config_(c), information_evaluator_(c), policy_(c), associator_(c),
      target_window_(c.initial_window_size) { Validate(c); clearState(); }
// 保存固定标定/投影接口，并重新建立图；已有状态和统计会被清空。
void Estimator::setParameter(const LidarExtrinsic &l, const ExtrinsicState &e, ProjectionFunction p) {
  lidar_extrinsic_ = l; camera_extrinsic_initial_ = e; projection_ = std::move(p); clearState();
}
// 完全重置：释放因子、参数和图像引用，清空策略历史，再建立唯一外参块与先验。
void Estimator::clearState() {
  pending_imu_.clear(); lidar_frames_.clear(); patches_.clear(); associator_.Clear();
  information_evaluator_ = InformationEvaluator(config_); policy_ = AdaptivePolicy(config_);
  has_lidar_ = false; has_vision_ = false; latest_node_ = 0;
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
  // 图重置后参数所有权重新建立，para_Ex_Pose 立即指向新数组。
  parameter_blocks_[ExtrinsicKey()] = PackPose(camera_extrinsic_.p_c_i, camera_extrinsic_.q_c_i);
  para_Ex_Pose[0] = parameter_blocks_.at(ExtrinsicKey()).data();
  addResidualBlock(FactorFactory::extrinsicPrior(camera_extrinsic_initial_, config_));
}
// 异常恢复保留待用 IMU、累计拒绝/重置次数及修正幅度；下一次输入重新建立首帧。
void Estimator::recover() {
  const auto resets = diagnostics_.reset_count + 1, rejected = diagnostics_.rejected_imu_intervals;
  const auto rejected_pose = diagnostics_.rejected_pose_refinements;
  const double translation = diagnostics_.max_frontend_translation_correction;
  const double rotation = diagnostics_.max_frontend_rotation_correction_deg;
  ImuSampleVector samples = std::move(pending_imu_);
  clearState(); pending_imu_ = std::move(samples);
  diagnostics_.reset_count = resets; diagnostics_.rejected_imu_intervals = rejected;
  diagnostics_.rejected_pose_refinements = rejected_pose;
  diagnostics_.max_frontend_translation_correction = translation;
  diagnostics_.max_frontend_rotation_correction_deg = rotation;
}
// 不同传感器输入包可带有重叠和未来边界 IMU；按时间插入并去重，不能直接丢弃落后于末条的样本。
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
  // 防止长期无关键帧时原始 IMU 无界增长；保留至少与允许关键帧间隔相适应的时间跨度。
  if (!pending_imu_.empty()) consumeImuThrough(pending_imu_.back().timestamp - std::max(5.0, 5.0 * config_.max_keyframe_dt));
}
// 只删除已不再需要的早期样本，保留区间起点之前的一条用于线性插值。
void Estimator::consumeImuThrough(double t) {
  auto it = std::lower_bound(pending_imu_.begin(), pending_imu_.end(), t,
      [](const ImuSample &a, double time) { return a.timestamp < time; });
  if (it != pending_imu_.begin()) --it;
  pending_imu_.erase(pending_imu_.begin(), it);
}
// 先评价当前模态，再用最近有效统计产生策略；超过 2*max_keyframe_dt 的另一模态不再维持高质量。
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
  setWeights(decision.weights);
  return decision;
}
// 接纳状态的统一入口：同时间优先合并，新时间需通过关键帧选择及 IMU 连接验证。
std::uint64_t Estimator::addOrGetFrame(const NavState &s, Modality m,
    const MeasurementSummary &summary, const PolicyDecision &decision) {
  auto id = findFrameNear(s.timestamp, config_.merge_time_tolerance);
  // 即使本次策略不选新关键帧，同时间事件仍可更新同一个节点和滤波参考。
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
    // 两端均有 IMU 包围才允许创建新导航状态，不能在缺测时添加无连接节点。
    if (!tmp_pre_integration->integrate(pending_imu_, previous_state.timestamp, s.timestamp,
                                   previous_state.bias_a, previous_state.bias_g)) {
      ++diagnostics_.rejected_imu_intervals;
      // Do not create an unconnected node or extrapolate absent IMU data.
      // 短时缺测只拒绝本次建点；长时间断连清空旧图，以新输入重新作为首帧。
      if (s.timestamp - previous_state.timestamp <= std::max(1.0, 3.0 * config_.max_keyframe_dt)) return 0;
      recover(); previous = 0; tmp_pre_integration.reset();
    }
  }
  id = addFrame(s);
  if (!id) return 0;
  // 有前驱建立 IMU 因子，否则建立首次速度/偏置及可选位姿锚定。
  const auto factor = previous ? FactorFactory::imu(previous, id, previous_state.gravity_w, tmp_pre_integration) :
                                 FactorFactory::initialState(id, s, config_);
  if (!addResidualBlock(factor)) { recover(); return 0; }
  tmp_pre_integration.reset();
  latest_node_ = id; markFrame(id, m, summary.normalized_information);
  consumeImuThrough(s.timestamp);
  return id;
}

// LiDAR 事件：拒绝重复/倒序，选择或复用状态，保存扫描快照，限量添加点面因子，然后求解。
bool Estimator::processLidar(const NavState &s, const MeasurementSummary &input,
                                        const LidarPlaneObservationVector &observations) {
  if (!ValidState(s) || s.timestamp <= last_lidar_time_ ||
      (latest_node_ && s.timestamp < stateEstimate(latest_node_).timestamp - config_.merge_time_tolerance)) return false;
  last_lidar_time_ = s.timestamp;
  const auto decision = decide(s, input, Modality::kLidar);
  const auto id = addOrGetFrame(s, Modality::kLidar, latest_lidar_, decision);
  if (!id) return false;
  associator_.Add(Modality::kLidar, s.timestamp, id);
  lidar_frames_[id] = {s, observations};
  const std::size_t limit = static_cast<std::size_t>(config_.max_lidar_factors);
  // 近似均匀抽取并限制因子块数量，控制滑窗求解规模；不是重新匹配地图。
  const std::size_t stride = limit ? std::max<std::size_t>(1, observations.size() / limit) : 1;
  std::size_t count = 0;
  for (std::size_t i = 0; i < observations.size() && count < limit; i += stride)
    if (addResidualBlock(FactorFactory::lidar(id, lidar_extrinsic_, observations[i], config_.huber_delta_lidar))) ++count;
  return solveOdometry();
}
// 视觉事件：先接纳状态，再按稳定 track 分成已有参考/新参考两种路径。
bool Estimator::processImage(const NavState &s, const MeasurementSummary &input,
    const VisualPatchObservationVector &observations, const ImageSampleFunction &sampler) {
  if (!sampler || !ValidState(s) || s.timestamp <= last_image_time_ ||
      (latest_node_ && s.timestamp < stateEstimate(latest_node_).timestamp - config_.merge_time_tolerance)) return false;
  last_image_time_ = s.timestamp;
  const auto decision = decide(s, input, Modality::kVision);
  const auto id = addOrGetFrame(s, Modality::kVision, latest_vision_, decision);
  if (!id) return false;
  // One nearest retained LiDAR frame is selected for this image. Native times
  // remain attached to both states; no equality/retimestamping is assumed.
  // 只在保留的 LiDAR 帧中按时间找候选，使用其自身状态时间，不将扫描重标时间戳。
  const auto match = associator_.FindNearest(Modality::kLidar, s.timestamp);
  const std::size_t limit = static_cast<std::size_t>(config_.max_cross_modal_factors);
  const std::size_t stride = limit ? std::max<std::size_t>(1, observations.size() / limit) : 1;
  std::set<std::uint64_t> seen;
  std::size_t accepted = 0;
  for (std::size_t i = 0; i < observations.size() && accepted < limit; i += stride) {
    const auto &o = observations[i];
    if (!ValidPatch(o) || !seen.insert(o.track_id).second) continue;
    auto it = patches_.find(o.track_id);
    // 已有参考 patch：连接参考与当前不同状态，增加光度因子，不重复添加同一深度—平面观测。
    if (it != patches_.end()) {
      const auto &patch = it->second;
      if (patch.anchor != id && addResidualBlock(FactorFactory::photometric(
          patch.anchor, id, o.track_id, patch.reference, o.inverse_exposure, projection_, sampler,
          config_.huber_delta_vision, Clamp(o.tracking_score, 0.01, 1.0)))) ++accepted;
      continue;
    }
    // 新 patch：必须存在可用关联扫描、剩余容量和足够局部平面支持。
    if (!match.valid || !lidar_frames_.count(match.node_id) ||
        patches_.size() >= static_cast<std::size_t>(config_.max_active_patches)) continue;
    const auto &frame = lidar_frames_.at(match.node_id);
    // 用关联扫描的滤波快照将前端地图点变到该扫描局部系；仅作为找邻域的初始查询。
    const Eigen::Vector3d query_i = frame.seed.q_w_i.conjugate() * (o.point_w - frame.seed.p_w_i);
    const Eigen::Vector3d query_l = lidar_extrinsic_.q_i_l.conjugate() * (query_i - lidar_extrinsic_.p_i_l);
    LocalPlane plane;
    if (!FactorFactory::fitLocalPlane(query_l, frame.observations, config_, &plane) ||
        !addFeature(o.track_id, id, o.inverse_depth)) continue;
    // 几何约束成功才正式保留该 patch；若失败则撤回尚未被引用的逆深度块。
    if (!addResidualBlock(FactorFactory::patchPlane(id, match.node_id, o.track_id,
        o.center_ray, lidar_extrinsic_, plane, config_.huber_delta_cross_modal,
        match.confidence * Clamp(o.tracking_score, 0.01, 1.0)))) {
      removeUnusedFeature(o.track_id); continue;
    }
    patches_.emplace(o.track_id, Patch{id, o}); ++accepted;
    markFrame(id, Modality::kCrossModal, latest_vision_.normalized_information);
    markFrame(match.node_id, Modality::kCrossModal, latest_lidar_.normalized_information);
  }
  return solveOdometry();
}
// 一次完整后端更新：优化 → 自适应滑窗 → 统计输出；任何失败进入 recover，避免无界积累。
bool Estimator::solveOdometry() {
  const auto start = std::chrono::steady_clock::now();
  double cost = 0.0;
  if (!optimization(&cost)) { recover(); return false; }
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
// 输出最新保留状态，使用其真实时间；不把旧关键帧时间改写为当前输入时间。
bool Estimator::latestEstimate(NavState *state) const {
  if (!state || !latest_node_ || !hasFrame(latest_node_)) return false;
  *state = stateEstimate(latest_node_); return true;
}

// 分配稳定 ID 和固定尺寸参数存储，以滤波状态初始化；建立位姿参考但不自动补 IMU 边。
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
// 从最新帧向前找数值时间容差内节点；容差只用于合并同一状态事件。
std::uint64_t Estimator::findFrameNear(double t, double tolerance) const {
  for (auto it = Headers.rbegin(); it != Headers.rend(); ++it)
    if (std::abs(it->second - t) <= tolerance) return it->first;
  return 0;
}
// 从当前状态容器读取估计，nodes_ 只提供重力等未优化元数据。
NavState Estimator::stateEstimate(std::uint64_t id) const {
  NavState s = nodes_.at(id).state;
  s.timestamp = Headers.at(id); s.p_w_i = Ps.at(id); s.q_w_i = Eigen::Quaterniond(Rs.at(id));
  s.v_w_i = Vs.at(id); s.bias_a = Bas.at(id); s.bias_g = Bgs.at(id);
  return s;
}
// 读取当前已接受的 T_CI 外参；未开启外参优化时保持初始化标定。
ExtrinsicState Estimator::cameraExtrinsicEstimate() const { return camera_extrinsic_; }
// 质量仅取已接纳摘要的最大值，模态标记用于后续候选惩罚，不是新增测量残差。
void Estimator::markFrame(std::uint64_t id, Modality m, double info) {
  if (!hasFrame(id)) return;
  auto &f = nodes_.at(id); f.unique_information = std::max(f.unique_information, info);
  if (m == Modality::kLidar) f.has_lidar = true;
  if (m == Modality::kVision) f.has_vision = true;
  if (m == Modality::kCrossModal) f.has_cross_modal = true;
}
// 替换而非叠加同一状态的滤波位姿正则；参考快照和优化状态分别保存。
bool Estimator::updateFrontendPose(std::uint64_t id, const NavState &s) {
  if (!hasFrame(id) || !ValidState(s) ||
      std::abs(Headers.at(id) - s.timestamp) > config_.merge_time_tolerance) return false;
  if (config_.frontend_pose_prior_enabled) {
    auto factor = FactorFactory::frontendPose(id, s, config_);
    Eigen::VectorXd residual;
    if (!factor.Evaluate(parameter_blocks_, &residual)) return false;
    // 查找同一 pose 对应的旧正则，找到后替换，保证每个保留状态恰好一份。
    auto it = std::find_if(residual_blocks_.begin(), residual_blocks_.end(),
        [id](const ResidualBlockInfo &f) {
          return f.modality == Modality::kFrontendPose && f.keys.front() == PoseKey(id);
        });
    if (it == residual_blocks_.end()) residual_blocks_.push_back(std::move(factor));
    else *it = std::move(factor);
  }
  // 快照只能由输入滤波状态更新；后续 double2vector 不修改它。
  frontend_poses_[id] = s; frontend_poses_.at(id).q_w_i.normalize();
  // New states and same-time LIO/VIO updates start at the latest filter pose.
  // Already-retained, unrelated states keep their previous refined estimates.
  // 当前状态用新的滤波位姿起步；其他保留帧继续用上次优化值热启动。
  Ps.at(id) = s.p_w_i; Rs.at(id) = s.q_w_i.normalized().toRotationMatrix();
  const auto pose = PackPose(s.p_w_i, s.q_w_i);
  std::copy(pose.begin(), pose.end(), para_Pose.at(id));
  return true;
}
// 创建参考帧拥有的逆深度参数，先检查参考是否存在、track 是否重复及深度边界。
bool Estimator::addFeature(std::uint64_t track, std::uint64_t anchor, double rho) {
  if (!hasFrame(anchor) || hasFeature(track) || !std::isfinite(rho) ||
      rho < config_.min_inverse_depth || rho > config_.max_inverse_depth) return false;
  parameter_blocks_[DepthKey(track)] = {rho}; feature_anchors_[track] = anchor;
  para_Feature[track] = parameter_blocks_.at(DepthKey(track)).data(); return true;
}
// 只有原始因子和边缘化先验都不引用该深度，才能释放其数组和参考帧映射。
void Estimator::removeUnusedFeature(std::uint64_t track) {
  const auto key = DepthKey(track);
  for (const auto &f : residual_blocks_) for (const auto &k : f.keys) if (k == key) return;
  for (const auto &k : last_marginalization_factor.keys) if (k == key) return;
  para_Feature.erase(track); parameter_blocks_.erase(key); feature_anchors_.erase(track);
}
// 通用入图校验：键唯一、尺寸对应、残差/Jacobian 有限；避免在 Ceres 内部才触发错误。
bool Estimator::addResidualBlock(const ResidualBlockInfo &f) {
  if (!f.cost || !std::isfinite(f.confidence) || f.confidence <= 0 ||
      f.keys.size() != f.cost->parameter_block_sizes().size()) return false;
  std::set<BlockKey> unique;
  for (std::size_t k = 0; k < f.keys.size(); ++k)
    if (!unique.insert(f.keys[k]).second || f.keys[k].size() != f.cost->parameter_block_sizes()[k]) return false;
  Eigen::VectorXd r; std::vector<Eigen::MatrixXd> j;
  if (!f.Evaluate(parameter_blocks_, &r, &j)) return false;
  // Frontend priors must go through updateFrontendPose(), which replaces rather
  // than appends; this prevents accidental duplicate anchors through this API.
  // 专用位姿更新路径已检查残差并执行替换，禁止通用入口再加一份相关锚定。
  if (f.modality == Modality::kFrontendPose) return false;
  residual_blocks_.push_back(f);
  if (f.pre_integration) pre_integrations[f.keys.at(2).id] = f.pre_integration;
  return true;
}
// 原位同步当前 Eigen 状态到 Ceres 参数；不替换 vector 对象，以保持所有地址视图有效。
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
// 只在求解可用且修正通过检查时提交新状态；世界系沿用 FAST-LIVO2，不重新做全局 yaw 对齐。
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
// 对所有保留状态检查相对滤波快照的修正，旋转转为度后与配置比较；true 表示拒绝。
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
// 求解核心：状态打包、必要的偏置重积分、建 Ceres 问题、求解、检查、接受或原位回滚。
bool Estimator::optimization(double *final_cost) {
  if (frame_count == 0 || residual_blocks_.empty()) return false;
  vector2double();
  // 保留 IMU 原始样本，若起点偏置离开积分基点过多，先重积分再构造问题。
  for (auto &f : residual_blocks_) {
    if (!f.pre_integration) continue;
    const auto &motion = parameter_blocks_.at(f.keys[1]);
    const auto ba = Position(motion.data() + 3), bg = Position(motion.data() + 6);
    if (f.pre_integration->needsRepropagation(ba, bg) && !f.pre_integration->repropagate(ba, bg)) return false;
  }
  // 保留本次求解前全部参数，试探结果不可用/超限时原位恢复。
  const auto backup = parameter_blocks_;
  ceres::Problem::Options po;
  // cost 由 residual_blocks_/先验持有，loss 由局部 unique_ptr 持有，流形是成员；Problem 不接管。
  po.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  po.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  po.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  std::vector<std::unique_ptr<ceres::LossFunction>> losses;
  ceres::Problem problem(po);
  for (auto &entry : parameter_blocks_) {
    const auto &key = entry.first; auto &data = entry.second;
    problem.AddParameterBlock(data.data(), key.size(), key.is_pose() ? &local_parameterization_ : nullptr);
    // 逆深度有显式上下界；外参可配置为常量。位姿修正阈值则是求解后的接受检查。
    if (key.kind == BlockKind::kDepth) {
      problem.SetParameterLowerBound(data.data(), 0, config_.min_inverse_depth);
      problem.SetParameterUpperBound(data.data(), 0, config_.max_inverse_depth);
    }
    if (key.kind == BlockKind::kExtrinsic && !config_.optimize_extrinsics) problem.SetParameterBlockConstant(data.data());
  }
  auto add = [&](const ResidualBlockInfo &f, const std::vector<double *> &parameters) {
    // 先在白化残差整块范数上施加 Huber，再乘模态权重与 confidence。
    auto *robust = f.huber_delta > 0.0 ? new ceres::HuberLoss(f.huber_delta) : nullptr;
    losses.emplace_back(new ceres::ScaledLoss(robust, residualWeight(f, weights_), ceres::TAKE_OWNERSHIP));
    problem.AddResidualBlock(f.cost.get(), losses.back().get(), parameters);
  };
  for (const auto &f : residual_blocks_) {
    std::vector<double *> parameters;
    for (const auto &key : f.keys) parameters.push_back(parameter_blocks_.at(key).data());
    add(f, parameters);
  }
  // 历史信息只以一个 Schur 先验加入；已吸收的原始因子不再重复入图。
  if (last_marginalization_factor.cost) add(last_marginalization_factor, last_marginalization_parameter_blocks);
  ceres::Solver::Options options;
  // 当前使用单线程稠密 QR，适合有界小窗口；时间配置只是 Ceres 软预算。
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = config_.solver_iterations;
  options.max_solver_time_in_seconds = config_.solver_time_limit;
  options.num_threads = 1;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  bool success = summary.IsSolutionUsable() && std::isfinite(summary.final_cost);
  for (const auto &entry : parameter_blocks_) for (double value : entry.second) success = success && std::isfinite(value);
  // 软正则不保证硬幅度，因此先检查全窗口修正再决定是否调用 double2vector。
  if (success && failureDetection()) { ++diagnostics_.rejected_pose_refinements; success = false; }
  if (!success) {
    // Restore in place; the still-live Ceres problem and para_* views own no
    // storage and require their parameter addresses to remain unchanged.
    for (auto &entry : parameter_blocks_)
      std::copy(backup.at(entry.first).begin(), backup.at(entry.first).end(), entry.second.begin());
    return false;
  }
  double2vector(); solver_flag = NON_LINEAR;
  if (final_cost) *final_cost = summary.final_cost;
  return true;
}
// 根据候选 ID 区分最旧/次新/任意中间帧，循环删至目标状态数。
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
// 最旧帧路径：使用同一 Schur 消元实现。
bool Estimator::slideWindowOld() {
  return nodes_.size() > 1 && marginalizeFrame(nodes_.begin()->first);
}
// 次新帧路径：仍保护最新状态，不机械搬移连续槽位。
bool Estimator::slideWindowNew() {
  return nodes_.size() > 1 && marginalizeFrame(std::prev(nodes_.end(), 2)->first);
}
// 边缘化事务：先构造并验证新先验，再删旧因子和变量；失败时原图尚未提交删除。
bool Estimator::marginalizeFrame(std::uint64_t id) {
  if (!hasFrame(id) || nodes_.size() < 2 || id == nodes_.rbegin()->first) return false;
  // 同时删除 pose、speed-bias 和所有以该帧为参考的逆深度，避免留下无参考的 patch。
  std::set<BlockKey> drop{PoseKey(id), MotionKey(id)};
  for (const auto &entry : feature_anchors_) if (entry.second == id) drop.insert(DepthKey(entry.first));
  auto marginalization_info = std::make_shared<MarginalizationInfo>();
  std::vector<std::size_t> absorbed;
  // 仅选择触及 drop 的原始因子，完整保留与待删变量无关的非线性测量。
  for (std::size_t i = 0; i < residual_blocks_.size(); ++i) {
    bool touches = false;
    for (const auto &key : residual_blocks_[i].keys) touches = touches || drop.count(key) != 0;
    if (touches) { absorbed.push_back(i); marginalization_info->addResidualBlockInfo(residual_blocks_[i]); }
  }
  // 旧先验整体参与本次消元，成功后被新先验替换。
  if (last_marginalization_factor.cost) marginalization_info->addResidualBlockInfo(last_marginalization_factor);
  if (!marginalization_info->preMarginalize(parameter_blocks_, drop, weights_) ||
      !marginalization_info->marginalize()) return false;
  // Schur 数值结果还要作为实际 cost 求值一次，检查有效后才提交图结构修改。
  auto prior = marginalization_info->getFactor();
  Eigen::VectorXd check;
  if (!prior.Evaluate(parameter_blocks_, &check)) return false;
  // Commit only the selected factors. Raw retained factors remain nonlinear and
  // are not counted a second time in the prior. Removed frontend anchors are
  // absorbed exactly once, along with the IMU/visual/LiDAR factors they touch.
  // 按索引倒序删除，避免 vector 索引移动；同步移除已吸收 IMU 边的显式索引。
  for (auto it = absorbed.rbegin(); it != absorbed.rend(); ++it) {
    const auto &f = residual_blocks_[*it];
    if (f.pre_integration) pre_integrations.erase(f.keys.at(2).id);
    residual_blocks_.erase(residual_blocks_.begin() + *it);
  }
  last_marginalization_info = std::move(marginalization_info);
  last_marginalization_factor = std::move(prior);
  last_marginalization_parameter_blocks.clear();
  // 先记录所有保留参数地址，再删除 drop；稳定存储无需 VINS 式数组地址重映射。
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
  // 几何缓存、时间索引和 patch 生命周期必须与优化变量一起清理。
  associator_.RemoveNode(id); lidar_frames_.erase(id);
  for (auto it = patches_.begin(); it != patches_.end(); ) {
    if (!hasFeature(it->first)) it = patches_.erase(it); else ++it;
  }
  // Do not concatenate IMU intervals for an interior removal. Their information
  // is already carried by the Schur prior; readding an edge would double count.
  return true;
}
// 复制当前优化状态与策略标签，按稳定 ID 递增次序提供时间有序窗口。
PolicyFrameVector Estimator::policyFrames() const {
  PolicyFrameVector out;
  for (const auto &entry : nodes_) { auto frame = entry.second; frame.state = stateEstimate(entry.first); out.push_back(frame); }
  return out;
}
// 返回保留 ID 列表；删除中间状态后 ID 不再连续。
std::vector<std::uint64_t> Estimator::frameIds() const {
  std::vector<std::uint64_t> ids; for (const auto &entry : nodes_) ids.push_back(entry.first); return ids;
}
// 统计显式原始因子，不把已吸收的历史因子再次计数。
std::size_t Estimator::countFactors(Modality m) const {
  return std::count_if(residual_blocks_.begin(), residual_blocks_.end(),
      [m](const ResidualBlockInfo &f) { return f.modality == m; });
}
}}
