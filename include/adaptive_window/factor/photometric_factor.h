/**
 * @file photometric_factor.h
 * @brief 参考 patch 向当前图像重投影的直接光度残差，每次求值重新采样并传播导数。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/factor/measurement_types.h"
#include <ceres/cost_function_to_functor.h>

namespace fast_livo { namespace adaptive_window {
// The adapter differentiates intensity wrt camera-frame point. Autodiff then
// chains through poses, extrinsics and inverse depth, at EVERY Ceres iteration.
// 解析/数值导数适配器：把外部投影与采样回调变成可由 Jet 链式传播的 Ceres cost。
class IntensityFactor final : public ceres::SizedCostFunction<1, 3> {
 public:
  IntensityFactor(ProjectionFunction projection, ImageSampleFunction sample)
      : projection_(std::move(projection)), sample_(std::move(sample)) {}
  // 输入一个相机系三维点；输出采样灰度和 1×3 的强度对三维点导数。
  bool Evaluate(double const *const *x, double *r, double **j) const override {
    Eigen::Vector2d pixel, gradient;
    Eigen::Matrix<double, 2, 3> projection_j;
    if (!projection_(Position(x[0]), &pixel, &projection_j) ||
        !sample_(pixel, r, &gradient) || !std::isfinite(r[0])) return false;
    if (j && j[0]) {
      Eigen::Map<Eigen::RowVector3d> row(j[0]);
      // 链式法则 dI/dP_C = dI/du * du/dP_C；桥接层的梯度与双线性采样一致。
      row = gradient.transpose() * projection_j;
      if (!row.allFinite()) return false;
    }
    return true;
  }
 private:
  ProjectionFunction projection_;
  ImageSampleFunction sample_;
};
// 一个 patch 一个残差块，行数 M 等于参考采样点数；曝光是固定前端输入。
struct PhotometricFactor {
  VisualPatchObservation reference;
  double current_exposure;
  ceres::CostFunctionToFunctor<1, 3> intensity;
  PhotometricFactor(const VisualPatchObservation &ref, double exposure,
            const ProjectionFunction &p, const ImageSampleFunction &s)
      : reference(ref), current_exposure(exposure), intensity(new IntensityFactor(p, s)) {}
  template <typename T> bool operator()(T const *const *x, T *r) const {
    // x=[参考导航位姿,当前导航位姿,T_CI,中心逆深度]；两帧不能使用相同参数地址。
    const T *anchor = x[0], *current = x[1], *ext = x[2];
    const T rho = x[3][0];
    if (rho <= T(0)) return false;
    const auto qa = Rotation(anchor), qc = Rotation(current), qci = Rotation(ext);
    // 除以 sigma*sqrt(M)，让 patch 平方损失对应白化像素均方，避免尺寸越大权重越大。
    const double scale = 1.0 / (reference.photometric_sigma * std::sqrt(static_cast<double>(reference.rays.size())));
    for (std::size_t k = 0; k < reference.rays.size(); ++k) {
      // 射线与参考切平面相交：P_k=f_k*(n^T f_0)/(rho*n^T f_k)。
      const double denominator = reference.normal_c.dot(reference.rays[k]);
      if (std::abs(denominator) < 1e-4) return false;
      const double ratio = reference.normal_c.dot(reference.center_ray) / denominator;
      // 排除平行/反向射线；无效试探步返回 false，由 Ceres 拒绝该步，不能将残差置零。
      if (ratio <= 0.0) return false;
      const Eigen::Matrix<T, 3, 1> point_ref = reference.rays[k].cast<T>() * T(ratio) / rho;
      // 参考相机 → 参考 IMU → 世界 → 当前 IMU → 当前相机。
      const Eigen::Matrix<T, 3, 1> world = qa * (qci.conjugate() * (point_ref - Position(ext))) + Position(anchor);
      const Eigen::Matrix<T, 3, 1> point_current = qci * (qc.conjugate() * (world - Position(current))) + Position(ext);
      if (!intensity(point_current.data(), r + k)) return false;
      // 残差=当前逆曝光×当前采样灰度 - 参考逆曝光×固定参考灰度。
      r[k] = T(scale) * (T(current_exposure) * r[k] - T(reference.inverse_exposure * reference.intensities[k]));
    }
    return true;
  }
};

}}
