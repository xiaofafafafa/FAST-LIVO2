#include "adaptive_window/factor_builders.h"
#include <ceres/cost_function_to_functor.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace fast_livo { namespace adaptive_window {
namespace {
struct InitialCost {
  NavState reference;
  AdaptiveWindowConfig config;
  template <typename T> bool operator()(const T *pose, const T *motion, T *out) const {
    Eigen::Map<Eigen::Matrix<T, 15, 1>> r(out);
    r.template segment<3>(0) = (Position(pose) - reference.p_w_i.cast<T>()) / T(config.pose_prior_translation_sigma);
    r.template segment<3>(3) = QuaternionLog(reference.q_w_i.cast<T>().conjugate() * Rotation(pose)) /
                               T(Radians(config.pose_prior_rotation_deg));
    r.template segment<3>(6) = (Position(motion) - reference.v_w_i.cast<T>()) / T(config.velocity_prior_sigma);
    r.template segment<3>(9) = (Position(motion + 3) - reference.bias_a.cast<T>()) / T(config.bias_prior_sigma);
    r.template segment<3>(12) = (Position(motion + 6) - reference.bias_g.cast<T>()) / T(config.bias_prior_sigma);
    return true;
  }
};
struct ExtrinsicCost {
  ExtrinsicState reference;
  AdaptiveWindowConfig config;
  template <typename T> bool operator()(const T *x, T *out) const {
    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(out);
    r.template head<3>() = (Position(x) - reference.p_c_i.cast<T>()) / T(config.extrinsic_translation_sigma);
    r.template tail<3>() = QuaternionLog(reference.q_c_i.cast<T>().conjugate() * Rotation(x)) /
                          T(Radians(config.extrinsic_rotation_deg));
    return true;
  }
};
struct LidarCost {
  Eigen::Vector3d point_i, normal_w, center_w;
  double inverse_sigma;
  template <typename T> bool operator()(const T *pose, T *r) const {
    const Eigen::Matrix<T, 3, 1> point_w = Rotation(pose) * point_i.cast<T>() + Position(pose);
    r[0] = T(inverse_sigma) * normal_w.cast<T>().dot(point_w - center_w.cast<T>());
    return true;
  }
};
struct ImuCost {
  std::shared_ptr<ImuPreintegration> p;
  Eigen::Vector3d gravity;
  template <typename T> bool operator()(const T *xi, const T *mi, const T *xj, const T *mj, T *out) const {
    const Eigen::Matrix<T, 3, 1> dba = Position(mi + 3) - p->linearized_ba.cast<T>();
    const Eigen::Matrix<T, 3, 1> dbg = Position(mi + 6) - p->linearized_bg.cast<T>();
    const Eigen::Matrix<T, 3, 1> corrected_p = p->dp.cast<T>() +
        p->jacobian.block<3, 3>(0, 9).cast<T>() * dba + p->jacobian.block<3, 3>(0, 12).cast<T>() * dbg;
    const Eigen::Matrix<T, 3, 1> corrected_v = p->dv.cast<T>() +
        p->jacobian.block<3, 3>(6, 9).cast<T>() * dba + p->jacobian.block<3, 3>(6, 12).cast<T>() * dbg;
    const Eigen::Matrix<T, 3, 1> correction = p->jacobian.block<3, 3>(3, 12).cast<T>() * dbg;
    const Eigen::Quaternion<T> small(T(1), T(0.5) * correction.x(), T(0.5) * correction.y(), T(0.5) * correction.z());
    const Eigen::Quaternion<T> corrected_q = (p->dq.cast<T>() * small).normalized();
    const auto qi = Rotation(xi);
    Eigen::Matrix<T, 15, 1> r;
    r.template segment<3>(0) = qi.conjugate() * (Position(xj) - Position(xi) - T(p->dt) * Position(mi) -
        T(0.5 * p->dt * p->dt) * gravity.cast<T>()) - corrected_p;
    Eigen::Quaternion<T> error = corrected_q.conjugate() * qi.conjugate() * Rotation(xj);
    if (error.w() < T(0)) error.coeffs() *= T(-1);
    r.template segment<3>(3) = T(2) * error.vec();
    r.template segment<3>(6) = qi.conjugate() * (Position(mj) - Position(mi) - T(p->dt) * gravity.cast<T>()) - corrected_v;
    r.template segment<3>(9) = Position(mj + 3) - Position(mi + 3);
    r.template segment<3>(12) = Position(mj + 6) - Position(mi + 6);
    Eigen::Map<Eigen::Matrix<T, 15, 1>> result(out);
    result = p->sqrt_information.cast<T>() * r;
    return true;
  }
};

// The adapter differentiates intensity wrt camera-frame point. Autodiff then
// chains through poses, extrinsics and inverse depth, at EVERY Ceres iteration.
class IntensityCost final : public ceres::SizedCostFunction<1, 3> {
 public:
  IntensityCost(ProjectionFunction projection, ImageSampleFunction sample)
      : projection_(std::move(projection)), sample_(std::move(sample)) {}
  bool Evaluate(double const *const *x, double *r, double **j) const override {
    Eigen::Vector2d pixel, gradient;
    Eigen::Matrix<double, 2, 3> projection_j;
    if (!projection_(Position(x[0]), &pixel, &projection_j) ||
        !sample_(pixel, r, &gradient) || !std::isfinite(r[0])) return false;
    if (j && j[0]) {
      Eigen::Map<Eigen::RowVector3d> row(j[0]);
      row = gradient.transpose() * projection_j;
      if (!row.allFinite()) return false;
    }
    return true;
  }
 private:
  ProjectionFunction projection_;
  ImageSampleFunction sample_;
};
struct PhotoCost {
  VisualPatchObservation reference;
  double current_exposure;
  ceres::CostFunctionToFunctor<1, 3> intensity;
  PhotoCost(const VisualPatchObservation &ref, double exposure,
            const ProjectionFunction &p, const ImageSampleFunction &s)
      : reference(ref), current_exposure(exposure), intensity(new IntensityCost(p, s)) {}
  template <typename T> bool operator()(T const *const *x, T *r) const {
    const T *anchor = x[0], *current = x[1], *ext = x[2];
    const T rho = x[3][0];
    if (rho <= T(0)) return false;
    const auto qa = Rotation(anchor), qc = Rotation(current), qci = Rotation(ext);
    const double scale = 1.0 / (reference.photometric_sigma * std::sqrt(static_cast<double>(reference.rays.size())));
    for (std::size_t k = 0; k < reference.rays.size(); ++k) {
      const double denominator = reference.normal_c.dot(reference.rays[k]);
      if (std::abs(denominator) < 1e-4) return false;
      const double ratio = reference.normal_c.dot(reference.center_ray) / denominator;
      if (ratio <= 0.0) return false;
      const Eigen::Matrix<T, 3, 1> point_ref = reference.rays[k].cast<T>() * T(ratio) / rho;
      const Eigen::Matrix<T, 3, 1> world = qa * (qci.conjugate() * (point_ref - Position(ext))) + Position(anchor);
      const Eigen::Matrix<T, 3, 1> point_current = qci * (qc.conjugate() * (world - Position(current))) + Position(ext);
      if (!intensity(point_current.data(), r + k)) return false;
      r[k] = T(scale) * (T(current_exposure) * r[k] - T(reference.inverse_exposure * reference.intensities[k]));
    }
    return true;
  }
};
struct PlaneCost {
  bool same_pose;
  Eigen::Vector3d ray;
  LidarExtrinsic lidar;
  LocalPlane plane;
  template <typename T> bool operator()(T const *const *x, T *r) const {
    const T *ext = x[same_pose ? 0 : 2];
    const T rho = x[same_pose ? 1 : 3][0];
    if (rho <= T(0)) return false;
    Eigen::Matrix<T, 3, 1> point_i = Rotation(ext).conjugate() * (ray.cast<T>() / rho - Position(ext));
    if (!same_pose) point_i = (Rotation(x[1]).conjugate() *
        (Rotation(x[0]) * point_i + Position(x[0]) - Position(x[1]))).eval();
    const Eigen::Matrix<T, 3, 1> point_l = lidar.q_i_l.cast<T>().conjugate() * (point_i - lidar.p_i_l.cast<T>());
    r[0] = plane.normal_l.cast<T>().dot(point_l - plane.center_l.cast<T>()) / T(plane.sigma);
    return true;
  }
};
template <typename Functor>
std::shared_ptr<ceres::CostFunction> Dynamic(Functor *functor, const std::vector<BlockKey> &keys, int rows) {
  auto *cost = new ceres::DynamicAutoDiffCostFunction<Functor, 4>(functor);
  for (const auto &key : keys) cost->AddParameterBlock(key.size());
  cost->SetNumResiduals(rows);
  return std::shared_ptr<ceres::CostFunction>(cost);
}
} // namespace

NonlinearFactor FactorBuilders::InitialState(std::uint64_t id, const NavState &s, const AdaptiveWindowConfig &c) {
  NonlinearFactor f; f.keys = {PoseKey(id), MotionKey(id)};
  f.cost.reset(new ceres::AutoDiffCostFunction<InitialCost, 15, 7, 9>(new InitialCost{s, c}));
  return f;
}
NonlinearFactor FactorBuilders::ExtrinsicPrior(const ExtrinsicState &s, const AdaptiveWindowConfig &c) {
  NonlinearFactor f; f.keys = {ExtrinsicKey()};
  f.cost.reset(new ceres::AutoDiffCostFunction<ExtrinsicCost, 6, 7>(new ExtrinsicCost{s, c}));
  return f;
}
NonlinearFactor FactorBuilders::Imu(std::uint64_t i, std::uint64_t j, const Eigen::Vector3d &g,
                                   const std::shared_ptr<ImuPreintegration> &p) {
  NonlinearFactor f; f.keys = {PoseKey(i), MotionKey(i), PoseKey(j), MotionKey(j)};
  f.modality = Modality::kImu; f.preintegration = p;
  f.cost.reset(new ceres::AutoDiffCostFunction<ImuCost, 15, 7, 9, 7, 9>(new ImuCost{p, g}));
  return f;
}
NonlinearFactor FactorBuilders::Lidar(std::uint64_t id, const LidarExtrinsic &e,
                                     const LidarPlaneObservation &o, double huber) {
  NonlinearFactor f; f.keys = {PoseKey(id)}; f.modality = Modality::kLidar; f.huber_delta = huber;
  if (!o.point_l.allFinite() || !o.normal_w.allFinite() || !o.center_w.allFinite() ||
      !std::isfinite(o.variance) || o.variance <= 0 || o.normal_w.norm() < 1e-6) return f;
  f.cost.reset(new ceres::AutoDiffCostFunction<LidarCost, 1, 7>(new LidarCost{
      e.q_i_l * o.point_l + e.p_i_l, o.normal_w.normalized(), o.center_w, 1.0 / std::sqrt(o.variance)}));
  return f;
}
NonlinearFactor FactorBuilders::Photometric(std::uint64_t anchor, std::uint64_t current,
    std::uint64_t track, const VisualPatchObservation &ref, double exposure,
    const ProjectionFunction &p, const ImageSampleFunction &s, double huber, double confidence) {
  NonlinearFactor f; f.keys = {PoseKey(anchor), PoseKey(current), ExtrinsicKey(), DepthKey(track)};
  f.modality = Modality::kVision; f.huber_delta = huber; f.confidence = confidence;
  if (anchor == current || ref.rays.empty() || ref.rays.size() != ref.intensities.size() ||
      !p || !s || !std::isfinite(exposure) || exposure <= 0 ||
      !std::isfinite(ref.photometric_sigma) || ref.photometric_sigma <= 0) return f;
  f.cost = Dynamic(new PhotoCost(ref, exposure, p, s), f.keys, static_cast<int>(ref.rays.size()));
  return f;
}
NonlinearFactor FactorBuilders::PatchPlane(std::uint64_t anchor, std::uint64_t lidar,
    std::uint64_t track, const Eigen::Vector3d &ray, const LidarExtrinsic &e,
    const LocalPlane &plane, double huber, double confidence) {
  NonlinearFactor f; const bool same = anchor == lidar;
  f.keys = same ? std::vector<BlockKey>{ExtrinsicKey(), DepthKey(track)} :
                 std::vector<BlockKey>{PoseKey(anchor), PoseKey(lidar), ExtrinsicKey(), DepthKey(track)};
  f.modality = Modality::kCrossModal; f.huber_delta = huber; f.confidence = confidence;
  f.cost = Dynamic(new PlaneCost{same, ray, e, plane}, f.keys, 1);
  return f;
}
bool FactorBuilders::FitLocalPlane(const Eigen::Vector3d &query, const LidarPlaneObservationVector &points,
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
