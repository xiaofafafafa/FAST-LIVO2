#pragma once
#include "adaptive_window/factor/measurement_types.h"
#include "adaptive_window/factor/integration_base.h"
#include "adaptive_window/factor/marginalization_factor.h"

namespace fast_livo { namespace adaptive_window {
class FactorFactory {
 public:
  static ResidualBlockInfo frontendPose(std::uint64_t id, const NavState &, const AdaptiveWindowConfig &);
  static ResidualBlockInfo initialState(std::uint64_t id, const NavState &, const AdaptiveWindowConfig &);
  static ResidualBlockInfo extrinsicPrior(const ExtrinsicState &, const AdaptiveWindowConfig &);
  static ResidualBlockInfo imu(std::uint64_t i, std::uint64_t j, const Eigen::Vector3d &gravity,
                             const std::shared_ptr<IntegrationBase> &);
  static ResidualBlockInfo lidar(std::uint64_t id, const LidarExtrinsic &,
                               const LidarPlaneObservation &, double huber);
  static ResidualBlockInfo photometric(std::uint64_t anchor, std::uint64_t current,
                                     std::uint64_t track, const VisualPatchObservation &reference,
                                     double current_inverse_exposure,
                                     const ProjectionFunction &, const ImageSampleFunction &,
                                     double huber, double confidence);
  static ResidualBlockInfo patchPlane(std::uint64_t anchor, std::uint64_t lidar,
                                    std::uint64_t track, const Eigen::Vector3d &ray,
                                    const LidarExtrinsic &, const LocalPlane &,
                                    double huber, double confidence);
  // Fit a scan-local plane from deskewed LiDAR measurements, not visual points.
  static bool fitLocalPlane(const Eigen::Vector3d &query_l,
                            const LidarPlaneObservationVector &points,
                            const AdaptiveWindowConfig &, LocalPlane *);
};
}}
