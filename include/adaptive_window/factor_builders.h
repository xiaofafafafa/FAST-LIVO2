#pragma once
#include "adaptive_window/ceres_types.h"
#include "adaptive_window/imu_preintegration.h"
#include <functional>

namespace fast_livo { namespace adaptive_window {
using ProjectionFunction = std::function<bool(const Eigen::Vector3d &, Eigen::Vector2d *,
                                             Eigen::Matrix<double, 2, 3> *)>;
using ImageSampleFunction = std::function<bool(const Eigen::Vector2d &, double *, Eigen::Vector2d *)>;
struct LocalPlane {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center_l = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_l = Eigen::Vector3d::UnitZ();
  double sigma = 0.05;
};
class FactorBuilders {
 public:
  static NonlinearFactor InitialState(std::uint64_t id, const NavState &, const AdaptiveWindowConfig &);
  static NonlinearFactor ExtrinsicPrior(const ExtrinsicState &, const AdaptiveWindowConfig &);
  static NonlinearFactor Imu(std::uint64_t i, std::uint64_t j, const Eigen::Vector3d &gravity,
                             const std::shared_ptr<ImuPreintegration> &);
  static NonlinearFactor Lidar(std::uint64_t id, const LidarExtrinsic &,
                               const LidarPlaneObservation &, double huber);
  static NonlinearFactor Photometric(std::uint64_t anchor, std::uint64_t current,
                                     std::uint64_t track, const VisualPatchObservation &reference,
                                     double current_inverse_exposure,
                                     const ProjectionFunction &, const ImageSampleFunction &,
                                     double huber, double confidence);
  static NonlinearFactor PatchPlane(std::uint64_t anchor, std::uint64_t lidar,
                                    std::uint64_t track, const Eigen::Vector3d &ray,
                                    const LidarExtrinsic &, const LocalPlane &,
                                    double huber, double confidence);
  // Fit a scan-local plane from deskewed LiDAR measurements, not visual points.
  static bool FitLocalPlane(const Eigen::Vector3d &query_l,
                            const LidarPlaneObservationVector &points,
                            const AdaptiveWindowConfig &, LocalPlane *);
};
}}
