#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/factor/measurement_types.h"
#include <ceres/cost_function_to_functor.h>

namespace fast_livo { namespace adaptive_window {
// The adapter differentiates intensity wrt camera-frame point. Autodiff then
// chains through poses, extrinsics and inverse depth, at EVERY Ceres iteration.
class IntensityFactor final : public ceres::SizedCostFunction<1, 3> {
 public:
  IntensityFactor(ProjectionFunction projection, ImageSampleFunction sample)
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
struct PhotometricFactor {
  VisualPatchObservation reference;
  double current_exposure;
  ceres::CostFunctionToFunctor<1, 3> intensity;
  PhotometricFactor(const VisualPatchObservation &ref, double exposure,
            const ProjectionFunction &p, const ImageSampleFunction &s)
      : reference(ref), current_exposure(exposure), intensity(new IntensityFactor(p, s)) {}
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

}}
