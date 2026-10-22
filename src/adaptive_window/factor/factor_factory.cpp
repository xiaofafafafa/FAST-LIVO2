#include "adaptive_window/factor/factor_factory.h"
#include "adaptive_window/factor/initial_state_factor.h"
#include "adaptive_window/factor/lidar_factor.h"
#include "adaptive_window/factor/imu_factor.h"
#include "adaptive_window/factor/photometric_factor.h"
#include "adaptive_window/factor/patch_plane_factor.h"
#include "adaptive_window/factor/frontend_pose_factor.h"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace fast_livo { namespace adaptive_window {
namespace {
template <typename Functor>
std::shared_ptr<ceres::CostFunction> Dynamic(Functor *functor, const std::vector<BlockKey> &keys, int rows) {
  auto *cost = new ceres::DynamicAutoDiffCostFunction<Functor, 4>(functor);
  for (const auto &key : keys) cost->AddParameterBlock(key.size());
  cost->SetNumResiduals(rows);
  return std::shared_ptr<ceres::CostFunction>(cost);
}
} // namespace

ResidualBlockInfo FactorFactory::frontendPose(std::uint64_t id, const NavState &s, const AdaptiveWindowConfig &c) {
  ResidualBlockInfo f; f.keys = {PoseKey(id)}; f.modality = Modality::kFrontendPose;
  f.confidence = c.frontend_pose_prior_weight;
  if (!c.frontend_pose_prior_enabled) return f;
  f.cost.reset(new ceres::AutoDiffCostFunction<FrontendPoseFactor, 6, SIZE_POSE>(
      new FrontendPoseFactor{s.p_w_i, s.q_w_i.normalized(),
                             c.frontend_pose_translation_sigma, Radians(c.frontend_pose_rotation_deg)}));
  return f;
}
ResidualBlockInfo FactorFactory::initialState(std::uint64_t id, const NavState &s, const AdaptiveWindowConfig &c) {
  ResidualBlockInfo f; f.keys = {PoseKey(id), MotionKey(id)};
  f.cost.reset(new ceres::AutoDiffCostFunction<InitialStateFactor, 15, 7, 9>(new InitialStateFactor{s, c}));
  return f;
}
ResidualBlockInfo FactorFactory::extrinsicPrior(const ExtrinsicState &s, const AdaptiveWindowConfig &c) {
  ResidualBlockInfo f; f.keys = {ExtrinsicKey()};
  f.cost.reset(new ceres::AutoDiffCostFunction<ExtrinsicPriorFactor, 6, 7>(new ExtrinsicPriorFactor{s, c}));
  return f;
}
ResidualBlockInfo FactorFactory::imu(std::uint64_t i, std::uint64_t j, const Eigen::Vector3d &g,
                                   const std::shared_ptr<IntegrationBase> &p) {
  ResidualBlockInfo f; f.keys = {PoseKey(i), MotionKey(i), PoseKey(j), MotionKey(j)};
  f.modality = Modality::kImu; f.pre_integration = p;
  f.cost.reset(new ceres::AutoDiffCostFunction<IMUFactor, 15, 7, 9, 7, 9>(new IMUFactor{p, g}));
  return f;
}
ResidualBlockInfo FactorFactory::lidar(std::uint64_t id, const LidarExtrinsic &e,
                                     const LidarPlaneObservation &o, double huber) {
  ResidualBlockInfo f; f.keys = {PoseKey(id)}; f.modality = Modality::kLidar; f.huber_delta = huber;
  if (!o.point_l.allFinite() || !o.normal_w.allFinite() || !o.center_w.allFinite() ||
      !std::isfinite(o.variance) || o.variance <= 0 || o.normal_w.norm() < 1e-6) return f;
  f.cost.reset(new ceres::AutoDiffCostFunction<LidarFactor, 1, 7>(new LidarFactor{
      e.q_i_l * o.point_l + e.p_i_l, o.normal_w.normalized(), o.center_w, 1.0 / std::sqrt(o.variance)}));
  return f;
}
ResidualBlockInfo FactorFactory::photometric(std::uint64_t anchor, std::uint64_t current,
    std::uint64_t track, const VisualPatchObservation &ref, double exposure,
    const ProjectionFunction &p, const ImageSampleFunction &s, double huber, double confidence) {
  ResidualBlockInfo f; f.keys = {PoseKey(anchor), PoseKey(current), ExtrinsicKey(), DepthKey(track)};
  f.modality = Modality::kVision; f.huber_delta = huber; f.confidence = confidence;
  if (anchor == current || ref.rays.empty() || ref.rays.size() != ref.intensities.size() ||
      !p || !s || !std::isfinite(exposure) || exposure <= 0 ||
      !std::isfinite(ref.photometric_sigma) || ref.photometric_sigma <= 0) return f;
  f.cost = Dynamic(new PhotometricFactor(ref, exposure, p, s), f.keys, static_cast<int>(ref.rays.size()));
  return f;
}
ResidualBlockInfo FactorFactory::patchPlane(std::uint64_t anchor, std::uint64_t lidar,
    std::uint64_t track, const Eigen::Vector3d &ray, const LidarExtrinsic &e,
    const LocalPlane &plane, double huber, double confidence) {
  ResidualBlockInfo f; const bool same = anchor == lidar;
  f.keys = same ? std::vector<BlockKey>{ExtrinsicKey(), DepthKey(track)} :
                 std::vector<BlockKey>{PoseKey(anchor), PoseKey(lidar), ExtrinsicKey(), DepthKey(track)};
  f.modality = Modality::kCrossModal; f.huber_delta = huber; f.confidence = confidence;
  f.cost = Dynamic(new PatchPlaneFactor{same, ray, e, plane}, f.keys, 1);
  return f;
}
bool FactorFactory::fitLocalPlane(const Eigen::Vector3d &query, const LidarPlaneObservationVector &points,
                                   const AdaptiveWindowConfig &c, LocalPlane *out) {
  if (!out || !query.allFinite()) return false;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> support;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto &p : points) {
    if (p.point_l.allFinite() && (p.point_l - query).squaredNorm() < c.plane_search_radius * c.plane_search_radius) {
      support.push_back(p.point_l); center += p.point_l;
    }
  }
  if (support.size() < 6) return false;
  center /= static_cast<double>(support.size());
  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  for (const auto &p : support) scatter.noalias() += (p - center) * (p - center).transpose();
  scatter /= static_cast<double>(support.size());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(scatter);
  if (eigen.info() != Eigen::Success) return false;
  const auto values = eigen.eigenvalues();
  if (values[1] < 1e-4 || values[0] > 0.05 * values[1] ||
      values[0] > c.plane_max_rms * c.plane_max_rms) return false;
  out->center_l = center; out->normal_l = eigen.eigenvectors().col(0);
  out->sigma = std::sqrt(c.cross_modal_plane_sigma * c.cross_modal_plane_sigma + std::max(0.0, values[0]));
  return std::abs(out->normal_l.dot(query - center)) < 3.0 * out->sigma;
}
}}
