#include "adaptive_window/adaptive_window_backend.h"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace fast_livo::adaptive_window;
namespace {
void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message); // active in Release builds
}
ProjectionFunction Projection() {
  return [](const Eigen::Vector3d &p, Eigen::Vector2d *uv, Eigen::Matrix<double, 2, 3> *j) {
    if (!p.allFinite() || p.z() <= 0.1) return false;
    *uv = Eigen::Vector2d(200.0 * p.x() / p.z() + 320.0, 200.0 * p.y() / p.z() + 240.0);
    *j << 200.0 / p.z(), 0.0, -200.0 * p.x() / (p.z() * p.z()),
          0.0, 200.0 / p.z(), -200.0 * p.y() / (p.z() * p.z());
    return true;
  };
}
ImageSampleFunction Image(double camera_x = 0.0) {
  return [camera_x](const Eigen::Vector2d &uv, double *value, Eigen::Vector2d *gradient) {
    const double u = uv.x() + 50.0 * camera_x, v = uv.y();
    *value = 100.0 + 0.2 * u + 0.15 * v + 0.0001 * u * v;
    *gradient = Eigen::Vector2d(0.2 + 0.0001 * v, 0.15 + 0.0001 * u);
    return uv.allFinite();
  };
}
ImuSampleVector Samples(double start, double end, double step = 0.005) {
  ImuSampleVector samples;
  const int n = static_cast<int>(std::ceil((end - start) / step));
  for (int i = 0; i <= n; ++i) {
    ImuSample s;
    s.timestamp = start + (end - start) * i / n;
    s.acceleration = Eigen::Vector3d(0, 0, 9.81);
    samples.push_back(s);
  }
  return samples;
}
std::vector<double> Motion(const NavState &s) {
  return {s.v_w_i.x(), s.v_w_i.y(), s.v_w_i.z(),
          s.bias_a.x(), s.bias_a.y(), s.bias_a.z(), s.bias_g.x(), s.bias_g.y(), s.bias_g.z()};
}
Eigen::VectorXd Residual(const ResidualBlockInfo &f, const std::vector<std::vector<double>> &blocks) {
  Require(bool(f.cost), "null cost");
  std::vector<const double *> pointers;
  for (const auto &b : blocks) pointers.push_back(b.data());
  Eigen::VectorXd r(f.cost->num_residuals());
  Require(f.cost->Evaluate(pointers.data(), r.data(), nullptr), "cost evaluation failed");
  Require(r.allFinite(), "nonfinite residual");
  return r;
}
// Compare ambient autodiff * manifold PlusJacobian against central differences
// in the actual tangent space; this catches quaternion ordering/half-angle bugs.
void CheckJacobian(const ResidualBlockInfo &f, const std::vector<std::vector<double>> &blocks,
                   double tolerance = 2e-5) {
  const int rows = f.cost->num_residuals();
  std::vector<const double *> pointers;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> j;
  j.reserve(blocks.size());
  std::vector<double *> jp;
  for (std::size_t k = 0; k < blocks.size(); ++k) {
    pointers.push_back(blocks[k].data());
    j.emplace_back(rows, blocks[k].size()); jp.push_back(j.back().data());
  }
  Eigen::VectorXd r(rows);
  Require(f.cost->Evaluate(pointers.data(), r.data(), jp.data()), "jacobian evaluation");
  PoseLocalParameterization manifold;
  for (std::size_t k = 0; k < blocks.size(); ++k) {
    Eigen::MatrixXd analytic = j[k];
    if (f.keys[k].is_pose()) {
      Eigen::Matrix<double, 7, 6, Eigen::RowMajor> plus;
      manifold.PlusJacobian(blocks[k].data(), plus.data());
      analytic = (j[k] * plus).eval();
    }
    for (int col = 0; col < f.keys[k].local_size(); ++col) {
      auto high = blocks, low = blocks;
      const double epsilon = 1e-6;
      if (f.keys[k].is_pose()) {
        Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
        d[col] = epsilon; manifold.Plus(blocks[k].data(), d.data(), high[k].data());
        d[col] = -epsilon; manifold.Plus(blocks[k].data(), d.data(), low[k].data());
      } else { high[k][col] += epsilon; low[k][col] -= epsilon; }
      const Eigen::VectorXd numeric = (Residual(f, high) - Residual(f, low)) / (2.0 * epsilon);
      Require((numeric - analytic.col(col)).norm() / (1.0 + numeric.norm()) < tolerance,
              "tangent Jacobian mismatch: block " + std::to_string(k) + ", column " + std::to_string(col));
    }
  }
}
VisualPatchObservation Patch(std::uint64_t track, double x) {
  VisualPatchObservation o;
  o.track_id = track; o.point_w = Eigen::Vector3d(0.1 * track, 0, 4);
  o.center_ray = Eigen::Vector3d((o.point_w.x() - x) / 4, 0, 1);
  o.normal_c = Eigen::Vector3d::UnitZ(); o.inverse_depth = 0.25;
  for (int row : {-1, 1}) for (int col : {-1, 1}) {
    Eigen::Vector3d ray = o.center_ray + Eigen::Vector3d(col / 200.0, row / 200.0, 0);
    Eigen::Vector2d pixel, gradient; Eigen::Matrix<double, 2, 3> j; double intensity;
    Projection()(ray, &pixel, &j); Image(x)(pixel, &intensity, &gradient);
    o.rays.push_back(ray); o.intensities.push_back(intensity);
  }
  return o;
}
void TestManifold() {
  PoseLocalParameterization manifold;
  auto x = PackPose(Eigen::Vector3d(1, 2, 3), ExpQuaternion(Eigen::Vector3d(0.2, -0.1, 0.3)));
  std::array<double, 7> y;
  Eigen::Matrix<double, 6, 1> d, recovered;
  d << 0.1, -0.2, 0.3, 0.03, -0.02, 0.04;
  Require(manifold.Plus(x.data(), d.data(), y.data()), "pose plus");
  manifold.Minus(y.data(), x.data(), recovered.data());
  Require((d - recovered).norm() < 1e-10, "pose plus/minus inverse");
  Eigen::Matrix<double, 7, 6, Eigen::RowMajor> jp;
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> jm;
  manifold.PlusJacobian(x.data(), jp.data()); manifold.MinusJacobian(x.data(), jm.data());
  Require((jm * jp - Eigen::Matrix<double, 6, 6>::Identity()).norm() < 1e-10, "manifold Jacobian inverse");
  for (int col = 0; col < 6; ++col) {
    Eigen::Matrix<double, 6, 1> step = Eigen::Matrix<double, 6, 1>::Zero();
    std::array<double, 7> hi, lo;
    step[col] = 1e-7; manifold.Plus(x.data(), step.data(), hi.data());
    step[col] = -1e-7; manifold.Plus(x.data(), step.data(), lo.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> h(hi.data()), l(lo.data());
    Require(((h - l) / 2e-7 - jp.col(col)).norm() < 1e-7, "manifold numerical Jacobian");
  }
}
void TestPreintegrationAndFactors() {
  AdaptiveWindowConfig config;
  auto pre = std::make_shared<IntegrationBase>(config);
  const auto raw = Samples(-0.01, 1.01);
  Require(pre->integrate(raw, 0, 1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()), "integration");
  Require(std::abs(pre->sum_dt - 1) < 1e-10, "integration duration");
  Require((pre->delta_p - Eigen::Vector3d(0, 0, 4.905)).norm() < 1e-8, "midpoint position");
  Require((pre->delta_v - Eigen::Vector3d(0, 0, 9.81)).norm() < 1e-8, "midpoint velocity");
  Require((pre->covariance - pre->covariance.transpose()).norm() < 1e-10, "symmetric covariance");
  Require(pre->covariance.diagonal().minCoeff() > 0, "bias random-walk covariance");
  NavState a, b; b.timestamp = 1;
  auto imu = FactorFactory::imu(1, 2, a.gravity_w, pre);
  std::vector<std::vector<double>> ib{PackPose(a.p_w_i, a.q_w_i), Motion(a), PackPose(b.p_w_i, b.q_w_i), Motion(b)};
  Require(Residual(imu, ib).norm() < 1e-6, "stationary gravity sign");
  ib[2] = PackPose(Eigen::Vector3d(0.03, 0.02, -0.01), ExpQuaternion(Eigen::Vector3d(0.01, 0.02, -0.03)));
  CheckJacobian(imu, ib);
  IntegrationBase invalid(config);
  Require(!invalid.integrate(raw, -0.02, 1, a.bias_a, a.bias_g), "missing start bracket");
  Require(!invalid.integrate(raw, 0, 1.02, a.bias_a, a.bias_g), "missing end bracket");
  Require(!invalid.integrate(Samples(0, 1, 0.2), 0, 1, a.bias_a, a.bias_g), "IMU gap rejection");
  auto duplicate = raw; duplicate.insert(duplicate.begin() + 1, duplicate.front());
  Require(!invalid.integrate(duplicate, 0, 1, a.bias_a, a.bias_g), "duplicate time rejection");
  const auto base_dp = pre->delta_p;
  Eigen::Vector3d delta_ba(0.01, -0.02, 0.03);
  const Eigen::Vector3d predicted = base_dp + pre->jacobian.block<3, 3>(0, 9) * delta_ba;
  Require(pre->repropagate(delta_ba, Eigen::Vector3d::Zero()), "bias repropagation");
  Require((pre->delta_p - predicted).norm() < 1e-8, "bias correction vs repropagation");

  const auto pose = PackPose(Eigen::Vector3d(0.1, 0.2, 0.3), ExpQuaternion(Eigen::Vector3d(0.1, -0.05, 0.03)));
  const auto other = PackPose(Eigen::Vector3d(0.15, 0.1, 0.25), ExpQuaternion(Eigen::Vector3d(-0.04, 0.02, 0.06)));
  ExtrinsicState ext; ext.p_c_i = Eigen::Vector3d(0.02, -0.01, 0.03);
  ext.q_c_i = ExpQuaternion(Eigen::Vector3d(0.02, 0.03, -0.01));
  auto extrinsic = PackPose(ext.p_c_i, ext.q_c_i);
  LidarExtrinsic lidar; lidar.p_i_l = Eigen::Vector3d(-0.01, 0.03, 0.02);
  LidarPlaneObservation observation;
  observation.point_l = Eigen::Vector3d(0.1, 0.2, 4);
  observation.normal_w = Eigen::Vector3d(0.1, 0.2, 1).normalized();
  observation.center_w = Eigen::Vector3d(0, 0, 4);
  CheckJacobian(FactorFactory::lidar(1, lidar, observation, 2), {pose});
  auto patch = Patch(1, 0);
  patch.normal_c = Eigen::Vector3d(0.1, -0.2, 1).normalized();
  CheckJacobian(FactorFactory::photometric(1, 2, 1, patch, 1.1, Projection(), Image(), 3, 1),
                {pose, other, extrinsic, {0.25}});
  LocalPlane plane; plane.center_l = Eigen::Vector3d(0, 0, 4);
  CheckJacobian(FactorFactory::patchPlane(1, 2, 1, patch.center_ray, lidar, plane, 3, 1),
                {pose, other, extrinsic, {0.25}});
  CheckJacobian(FactorFactory::patchPlane(1, 1, 1, patch.center_ray, lidar, plane, 3, 1),
                {extrinsic, {0.25}});
  CheckJacobian(FactorFactory::initialState(1, a, config), {pose, Motion(a)});
  const auto anchor = FactorFactory::frontendPose(1, a, config);
  CheckJacobian(anchor, {pose});
  Require(Residual(anchor, {PackPose(a.p_w_i, a.q_w_i)}).norm() < 1e-12, "frontend reference residual");
  auto antipodal = PackPose(a.p_w_i, a.q_w_i);
  for (int k = 3; k < 7; ++k) antipodal[k] *= -1;
  Require(Residual(anchor, {antipodal}).norm() < 1e-12, "quaternion sign invariance");
  CheckJacobian(FactorFactory::extrinsicPrior(ext, config), {extrinsic});
}
void TestMarginalization() {
  AdaptiveWindowConfig c; c.solver_time_limit = 1;
  Estimator optimizer(c); optimizer.setParameter(LidarExtrinsic(), ExtrinsicState());
  NavState state;
  const auto first = optimizer.addFrame(state);
  Require(optimizer.addResidualBlock(FactorFactory::initialState(first, state, c)), "initial prior");
  auto previous = first;
  for (int i = 1; i < 4; ++i) {
    state.timestamp = 0.1 * i;
    const auto current = optimizer.addFrame(state);
    auto pre = std::make_shared<IntegrationBase>(c);
    Require(pre->integrate(Samples(0.1 * (i - 1) - 0.01, state.timestamp + 0.01),
        0.1 * (i - 1), state.timestamp, state.bias_a, state.bias_g), "marginal IMU");
    Require(optimizer.addResidualBlock(FactorFactory::imu(previous, current, state.gravity_w, pre)), "IMU factor");
    previous = current;
  }
  Require(optimizer.addFeature(99, 2, 0.25), "landmark depth");
  LocalPlane plane; plane.center_l.z() = 4;
  Require(optimizer.addResidualBlock(FactorFactory::patchPlane(2, 2, 99, Eigen::Vector3d::UnitZ(),
      LidarExtrinsic(), plane, 3, 1)), "landmark plane");
  double before = 0, after = 0;
  Require(optimizer.optimization(&before), "solve before marginalization");
  Require(optimizer.marginalizeFrame(2), "middle-frame Schur marginalization");
  Require(!optimizer.hasFeature(99), "anchor/depth removed together");
  Require(optimizer.optimization(&after), "solve after middle marginalization");
  Require(after < 1e-6 && optimizer.numFrames() == 3, "Schur zero residual preservation");
  Require(optimizer.countFactors(Modality::kImu) == 1, "only touching IMU factors absorbed");
  Require(optimizer.marginalizeFrame(first), "oldest-frame marginalization with existing prior");
  Require(optimizer.optimization(&after), "solve after oldest marginalization");
  Require(!optimizer.marginalizeFrame(previous), "newest state protected");
  Require(optimizer.numFrames() == 2, "bounded node count");
  Require(optimizer.countFactors(Modality::kFrontendPose) == 2, "only retained frontend priors remain");
}
void TestBackend(bool asynchronous) {
  AdaptiveWindowConfig c;
  c.enabled = true; c.min_window_size = 3; c.max_window_size = 5; c.initial_window_size = 3;
  c.min_keyframe_dt = 0.001; c.max_keyframe_dt = 0.09; c.keyframe_utility_threshold = 0;
  c.solver_time_limit = 1; c.max_lidar_factors = 20; c.max_cross_modal_factors = 8;
  AdaptiveWindowBackend backend(c); backend.Initialize(LidarExtrinsic(), ExtrinsicState(), Projection());
  MeasurementSummary summary;
  summary.information.setIdentity(); summary.information *= 100;
  summary.residual_count = 100; summary.rmse = 0.01; summary.inlier_ratio = 1;
  bool saw_photo = false, saw_plane = false, saw_marginalization = false;
  for (int frame = 0; frame < 12; ++frame) {
    const double time = 0.1 * frame;
    backend.AddImuSamples(Samples(time - 0.11, time + 0.04));
    NavState s; s.timestamp = time; s.p_w_i.x() = 0.2 * time; s.v_w_i.x() = 0.2;
    LidarPlaneObservationVector lidar;
    for (int row = -4; row <= 4; ++row) for (int col = -4; col <= 7; ++col) {
      LidarPlaneObservation o;
      o.point_l = Eigen::Vector3d(0.1 * col - s.p_w_i.x(), 0.1 * row, 4);
      o.center_w.z() = 4; o.variance = 0.01; lidar.push_back(o);
    }
    for (int index : {0, 5, 10})
      lidar[index].center_w.z() = 10; // stale planes at selected scan positions
    Require(backend.ProcessLidar(s, summary, lidar), "backend lidar");
    if (frame == 0)
      Require(backend.diagnostics().lidar_factors == 19, "stale LiDAR planes were admitted");
    if (asynchronous) { s.timestamp += 0.02; s.p_w_i.x() += 0.004; }
    VisualPatchObservationVector patches;
    for (std::uint64_t id = 1; id <= 4; ++id) patches.push_back(Patch(id, s.p_w_i.x()));
    Require(backend.ProcessVision(s, summary, patches, Image(s.p_w_i.x())), "backend vision");
    const auto &d = backend.diagnostics();
    Require(d.reset_count == 0 && d.rejected_imu_intervals == 0, "unexpected recovery or IMU rejection");
    Require(d.window_size <= static_cast<std::size_t>(c.max_window_size), "window bound");
    Require(d.lidar_factors > 0, "missing lidar factors");
    Require(d.frontend_pose_factors == d.window_size, "one frontend pose factor per state");
    Require(d.rejected_pose_refinements == 0, "unexpected refinement rejection");
    saw_photo = saw_photo || d.vision_factors > 0;
    saw_plane = saw_plane || d.cross_modal_pairs > 0;
    saw_marginalization = saw_marginalization || d.last_marginalized_id != 0;
    NavState estimate; Require(backend.LatestEstimate(&estimate), "latest state");
    Require((estimate.p_w_i - s.p_w_i).norm() < 1e-4, "synthetic exact trajectory");
    const auto factor_count = d.factor_count;
    Require(!backend.ProcessVision(s, summary, patches, Image()), "duplicate image rejection");
    Require(backend.diagnostics().factor_count == factor_count, "duplicate changed graph");
  }
  Require(saw_photo && saw_plane && saw_marginalization, "not all factor/marginalization paths covered");
  backend.Reset();
  NavState s; backend.ProcessLidar(s, summary, {});
  s.timestamp = 0.1;
  Require(!backend.ProcessLidar(s, summary, {}), "unbracketed IMU must not create a node");
  Require(backend.diagnostics().rejected_imu_intervals == 1, "IMU rejection diagnostic");
}

void TestMappingCovariance() {
  AdaptiveWindowConfig c; c.solver_time_limit=1; c.frontend_pose_prior_enabled=false;
  Estimator e(c); e.enableMappingSnapshots(true);
  NavState state; state.timestamp=0.3;
  const auto id=e.addFrame(state);
  Require(e.addResidualBlock(FactorFactory::initialState(id,state,c)), "full rank mapping prior");
  Require(e.solveOdometry(), "accepted snapshot solve");
  const auto& s=e.mappingSnapshot();
  Require(s.size()==1 && s[0].id==id && s[0].covariance_valid, "valid 6D marginal");
  Require(std::abs(s[0].covariance(0,0)-c.pose_prior_translation_sigma*c.pose_prior_translation_sigma)<1e-6,
          "position marginal agrees with known Gaussian prior");
  e.clearState();
  Require(e.mappingSnapshot().empty(), "reset invalidates previous snapshot");
}
void TestFrontendPoseRefinement() {
  AdaptiveWindowConfig c; c.solver_time_limit = 1;
  Estimator estimator(c);
  NavState frontend;
  const auto id = estimator.addFrame(frontend);
  Require(estimator.addResidualBlock(FactorFactory::initialState(id, frontend, c)), "motion prior");
  Require(!estimator.addResidualBlock(FactorFactory::frontendPose(id, frontend, c)), "duplicate pose-prior API blocked");
  LidarPlaneObservation lidar; lidar.normal_w = Eigen::Vector3d::UnitX();
  lidar.center_w = Eigen::Vector3d(0.1, 0, 0); lidar.variance = 0.01;
  Require(estimator.addResidualBlock(FactorFactory::lidar(id, LidarExtrinsic(), lidar, 0)), "conflicting lidar residual");
  Require(estimator.optimization(), "regularized refinement solve");
  // Independent 1-D solution: (400 * 0 + 100 * 0.1) / (400 + 100).
  Require(std::abs(estimator.stateEstimate(id).p_w_i.x() - 0.02) < 1e-5,
          "soft refinement strength: actual=" + std::to_string(estimator.stateEstimate(id).p_w_i.x()));
  Require(estimator.optimization(), "repeated refinement solve");
  Require(std::abs(estimator.stateEstimate(id).p_w_i.x() - 0.02) < 1e-5, "frontend reference must not follow backend");

  frontend.p_w_i.x() = 0.03;
  Require(estimator.updateFrontendPose(id, frontend), "same-state pose replacement");
  Require(estimator.countFactors(Modality::kFrontendPose) == 1, "pose replacement must not stack");
  Require(estimator.optimization(), "solve after pose replacement");
  Require(std::abs(estimator.stateEstimate(id).p_w_i.x() - 0.044) < 1e-5, "latest frontend reference used");

  AdaptiveWindowConfig off = c; off.frontend_pose_prior_enabled = false;
  Estimator unregularized(off);
  const auto first = unregularized.addFrame(frontend);
  Require(unregularized.countFactors(Modality::kFrontendPose) == 0, "pose-prior switch");
  auto displaced = PackPose(frontend.p_w_i + Eigen::Vector3d(0.1, 0, 0), frontend.q_w_i);
  Require(Residual(FactorFactory::initialState(first, frontend, off), {displaced, Motion(frontend)}).head<6>().norm() > 0,
          "first-pose gauge anchor when regularization disabled");
  Require(Residual(FactorFactory::initialState(first, frontend, c), {displaced, Motion(frontend)}).head<6>().norm() == 0,
          "no duplicate initial-pose constraint when regularization enabled");

  // A deliberately tight acceptance gate rejects a mathematically valid solve.
  AdaptiveWindowConfig tight = c; tight.frontend_pose_max_translation = 0.005;
  Estimator guarded(tight);
  NavState zero; const auto guarded_id = guarded.addFrame(zero);
  Require(guarded.addResidualBlock(FactorFactory::initialState(guarded_id, zero, tight)), "guard motion prior");
  Require(guarded.addResidualBlock(FactorFactory::lidar(guarded_id, LidarExtrinsic(), lidar, 0)), "guard lidar");
  Require(!guarded.optimization(), "excessive refinement rejected");
  Require(guarded.stateEstimate(guarded_id).p_w_i.norm() < 1e-12, "rejected result not committed");
  Require(guarded.diagnostics().rejected_pose_refinements == 1, "rejection counter");
  Require(!guarded.optimization(), "repeat solve still uses original state/reference");
  Require(guarded.diagnostics().rejected_pose_refinements == 2, "repeat rejection counter");

  AdaptiveWindowConfig turn = c; turn.frontend_pose_translation_sigma = 1e-4;
  Estimator rotation_guard(turn);
  const auto turn_id = rotation_guard.addFrame(NavState());
  Require(rotation_guard.addResidualBlock(FactorFactory::initialState(turn_id, NavState(), turn)), "rotation motion prior");
  LidarPlaneObservation turn_lidar;
  turn_lidar.point_l = Eigen::Vector3d::UnitX(); turn_lidar.normal_w = Eigen::Vector3d::UnitY();
  turn_lidar.center_w.y() = std::sin(Radians(10.0)); turn_lidar.variance = 1e-8;
  Require(rotation_guard.addResidualBlock(FactorFactory::lidar(turn_id, LidarExtrinsic(), turn_lidar, 0)), "rotation lidar");
  Require(!rotation_guard.optimization(), "rotation correction gate");
  Require(rotation_guard.diagnostics().max_frontend_rotation_correction_deg > turn.frontend_pose_max_rotation_deg,
          "rotation gate reports degrees");
  Require(rotation_guard.stateEstimate(turn_id).q_w_i.angularDistance(Eigen::Quaterniond::Identity()) < 1e-12,
          "rejected rotation not committed");

  // Exercise actual bridge-compatible event order, rather than only graph APIs.
  AdaptiveWindowBackend backend(c);
  backend.Initialize(LidarExtrinsic(), ExtrinsicState(), Projection());
  MeasurementSummary summary;
  Require(backend.ProcessLidar(zero, summary, {}), "LIO pose admission");
  zero.p_w_i.x() = 0.04;
  Require(backend.ProcessVision(zero, summary, {}, Image()), "same-time VIO pose update");
  NavState refined; Require(backend.LatestEstimate(&refined), "same-time output");
  Require(std::abs(refined.p_w_i.x() - 0.04) < 1e-6, "latest VIO pose is reference");
  Require(backend.diagnostics().frontend_pose_factors == 1 && backend.diagnostics().window_size == 1,
          "same-time LIO/VIO keep one pose anchor");
}
void TestNormalizedInformationPolicy() {
  AdaptiveWindowConfig config;
  config.normalized_information_policy = true;
  config.information_translation_scale = 1.0;
  config.information_rotation_scale_deg = 180.0 / M_PI;
  config.information_smoothing = 0.0;
  MeasurementSummary lidar;
  lidar.modality = Modality::kLidar;
  lidar.residual_count = 10;
  lidar.rmse = 0.01;
  lidar.inlier_ratio = 1.0;
  lidar.information(0, 0) = 100.0;
  InformationEvaluator evaluator(config);
  const auto first = evaluator.Evaluate(lidar);
  Require(first.valid && first.normalized_information > 0.0 &&
          first.normalized_information < 1.0,
          "dimensionless information must not saturate on a rank-one Hessian");
  lidar.information *= 10.0;
  lidar.residual_count *= 10;
  const auto repeated = evaluator.Evaluate(lidar);
  Require(std::abs(repeated.normalized_information - first.normalized_information) < 1e-12,
          "per-residual information should ignore duplicated support count");

  MeasurementSummary vision = lidar;
  vision.modality = Modality::kVision;
  vision.information.setZero();
  vision.information(3, 3) = 1000.0;
  auto complementary = evaluator.Evaluate(vision);
  PolicyFrame retained; retained.id = 1;
  PolicyFrameVector window{retained};
  NavState candidate; candidate.timestamp = 0.1;
  AdaptivePolicy distinct(config);
  const auto separate = distinct.Decide(candidate, &repeated, &complementary,
                                        window, 0.0);
  vision.information.setZero();
  vision.information(0, 0) = 1000.0;
  auto overlapping = evaluator.Evaluate(vision);
  AdaptivePolicy shared(config);
  const auto same = shared.Decide(candidate, &repeated, &overlapping,
                                 window, 0.0);
  Require(separate.complementarity_score > same.complementarity_score,
          "independent rotation/translation directions must add more information: " +
          std::to_string(separate.complementarity_score) + " vs " +
          std::to_string(same.complementarity_score));
}
void TestDirectionalPriorAndAnchorChoice() {
  AdaptiveWindowConfig config;
  NavState reference;
  Eigen::Matrix<double, 6, 6> direction = Eigen::Matrix<double, 6, 6>::Identity();
  direction(1, 1) = std::sqrt(2.0);
  const auto factor = FactorFactory::frontendPose(1, reference, config, direction);
  auto x = PackPose(Eigen::Vector3d(config.frontend_pose_translation_sigma, 0, 0),
                    Eigen::Quaterniond::Identity());
  auto y = PackPose(Eigen::Vector3d(0, config.frontend_pose_translation_sigma, 0),
                    Eigen::Quaterniond::Identity());
  Require(std::abs(Residual(factor, {x}).norm() - 1.0) < 1e-12 &&
          std::abs(Residual(factor, {y}).norm() - std::sqrt(2.0)) < 1e-12,
          "directional prior must strengthen only the selected pose axis");
  CheckJacobian(factor, {y});

  PolicyFrameVector frames(3);
  for (int i = 0; i < 3; ++i) {
    frames[i].id = i + 1;
    frames[i].state.timestamp = 0.1 * i;
    frames[i].state.p_w_i.x() = static_cast<double>(i);
  }
  frames[0].active_patch_tracks = 80;
  AdaptivePolicy original(config);
  Require(original.SelectMarginalizationCandidate(frames, 2) == 1,
          "unprotected baseline should remove oldest state");
  config.preserve_visual_anchors = true;
  AdaptivePolicy protected_policy(config);
  Require(protected_policy.SelectMarginalizationCandidate(frames, 2) == 2,
          "retain an active visual anchor while another state can be removed");
}
void TestDistinctPosePlaneTie() {
  AdaptiveWindowConfig config;
  config.enabled = true;
  config.solver_time_limit = 1;
  config.max_keyframe_dt = 0.09;
  config.keyframe_utility_threshold = 0;
  config.cross_modal_temporal_max_dt = 0.15;
  config.cross_modal_temporal_time_sigma = 0.15;
  AdaptiveWindowBackend backend(config);
  backend.Initialize(LidarExtrinsic(), ExtrinsicState(), Projection());
  MeasurementSummary summary;
  summary.information.setIdentity();
  summary.residual_count = 1;
  summary.rmse = 0.01;
  summary.inlier_ratio = 1.0;
  NavState state;
  for (int frame = 0; frame < 2; ++frame) {
    state.timestamp = 0.1 * frame;
    state.p_w_i.x() = 0.02 * frame;
    backend.AddImuSamples(Samples(state.timestamp - 0.11, state.timestamp + 0.04));
    LidarPlaneObservationVector points;
    for (int row = -4; row <= 4; ++row)
      for (int column = -4; column <= 7; ++column) {
        LidarPlaneObservation p;
        p.point_l = Eigen::Vector3d(0.1 * column - state.p_w_i.x(),
                                    0.1 * row, 4.0);
        p.center_w.z() = 4.0;
        p.variance = 0.01;
        points.push_back(p);
      }
    Require(backend.ProcessLidar(state, summary, points), "temporal plane LiDAR frame");
  }
  Require(backend.ProcessVision(state, summary, {Patch(1, state.p_w_i.x())},
                                Image(state.p_w_i.x())), "temporal plane image");
  Require(backend.diagnostics().event.temporal_cross_count == 1 &&
          backend.diagnostics().cross_modal_pairs == 2,
          "same-pose depth anchor plus distinct-pose geometric factor");
}
}
int main() {
  try {
    TestManifold(); TestPreintegrationAndFactors(); TestMarginalization(); TestMappingCovariance(); TestFrontendPoseRefinement();
    TestNormalizedInformationPolicy();
    TestDirectionalPriorAndAnchorChoice();
    TestDistinctPosePlaneTie();
    TestBackend(false); TestBackend(true);
    std::cout << "All Ceres adaptive-window checks passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAILED: " << error.what() << "\n";
    return 1;
  }
}
