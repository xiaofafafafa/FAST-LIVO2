#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

namespace fast_livo { namespace adaptive_window {

// Incremental world-frame correction from the last map-consistent estimate to
// the newest fixed-lag estimate.  A point is updated as p' = q * p + t.
struct PoseCorrection {
  double timestamp = 0.0;
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

// Interpolate the correction field between active keyframes.  Refuse to move
// landmarks outside the active time span; they have already been frozen.
inline bool CorrectionAt(const std::vector<PoseCorrection> &corrections,
                         double timestamp, double max_extrapolation,
                         PoseCorrection *result) {
  if (!result || corrections.empty() || !std::isfinite(timestamp)) return false;
  const auto upper = std::lower_bound(corrections.begin(), corrections.end(), timestamp,
      [](const PoseCorrection &c, double t) { return c.timestamp < t; });
  if (upper == corrections.begin()) {
    if (upper->timestamp - timestamp > max_extrapolation) return false;
    *result = *upper;
  } else if (upper == corrections.end()) {
    if (timestamp - corrections.back().timestamp > max_extrapolation) return false;
    *result = corrections.back();
  } else {
    const auto &left = *(upper - 1);
    const double width = upper->timestamp - left.timestamp;
    if (width <= 0.0) return false;
    const double alpha = (timestamp - left.timestamp) / width;
    result->timestamp = timestamp;
    result->rotation = left.rotation.slerp(alpha, upper->rotation).normalized();
    result->translation = (1.0 - alpha) * left.translation +
                          alpha * upper->translation;
  }
  return result->rotation.coeffs().allFinite() &&
         result->translation.allFinite();
}

inline bool HasMaterialCorrection(const PoseCorrection &correction,
                                  double translation_threshold,
                                  double rotation_threshold_rad) {
  return correction.translation.norm() > translation_threshold ||
         Eigen::AngleAxisd(correction.rotation).angle() > rotation_threshold_rad;
}

}}  // namespace fast_livo::adaptive_window
