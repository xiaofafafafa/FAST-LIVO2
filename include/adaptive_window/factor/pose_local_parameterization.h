#pragma once
#include "adaptive_window/ceres_types.h"
#include <ceres/version.h>

namespace fast_livo { namespace adaptive_window {
// Right SO(3) perturbation with the API required by the installed Ceres version.
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
class PoseLocalParameterization final : public ceres::Manifold {
#else
class PoseLocalParameterization final : public ceres::LocalParameterization {
#endif
 public:
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 6; }
#else
  int GlobalSize() const override { return 7; }
  int LocalSize() const override { return 6; }
  bool ComputeJacobian(const double *x, double *out) const override {
    return PlusJacobian(x, out);
  }
#endif
  bool Plus(const double *x, const double *d, double *out) const override {
    Eigen::Map<Eigen::Vector3d> p(out);
    Eigen::Map<Eigen::Quaterniond> q(out + 3);
    p = Position(x) + Position(d);
    q = (Rotation(x) * ExpQuaternion(Eigen::Map<const Eigen::Vector3d>(d + 3))).normalized();
    return p.allFinite() && q.coeffs().allFinite();
  }
  bool PlusJacobian(const double *x, double *out) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(out);
    const auto q = Rotation(x);
    j.setZero();
    j.topLeftCorner<3, 3>().setIdentity();
    j.block<3, 3>(3, 3) = 0.5 * (q.w() * Eigen::Matrix3d::Identity() + Skew(q.vec()));
    j.block<1, 3>(6, 3) = -0.5 * q.vec().transpose();
    return true;
  }
  bool Minus(const double *y, const double *x, double *out) const {
    Eigen::Map<Eigen::Vector3d> p(out), r(out + 3);
    p = Position(y) - Position(x);
    r = LogQuaternion(Rotation(x).conjugate() * Rotation(y));
    return p.allFinite() && r.allFinite();
  }
  bool MinusJacobian(const double *x, double *out) const {
    Eigen::Matrix<double, 7, 6, Eigen::RowMajor> plus;
    PlusJacobian(x, plus.data());
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(out);
    j = plus.transpose();
    j.bottomRows<3>() *= 4.0;
    return true;
  }
};


}}
