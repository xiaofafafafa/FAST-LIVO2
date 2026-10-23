/**
 * @file factor_factory.h
 * @brief 因子装配接口：设置参数键、模态、置信度和鲁棒核；公式位于各因子头文件。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/factor/measurement_types.h"
#include "adaptive_window/factor/integration_base.h"
#include "adaptive_window/factor/marginalization_factor.h"

namespace fast_livo { namespace adaptive_window {
class FactorFactory {
 public:
  // 一个 pose 块、6 维残差，权重独立；关闭位姿正则时返回空 cost。
  static ResidualBlockInfo frontendPose(std::uint64_t id, const NavState &, const AdaptiveWindowConfig &);
  // 首次 pose+speedbias 共 15 行；逐帧位姿正则启用时前 6 行为零。
  static ResidualBlockInfo initialState(std::uint64_t id, const NavState &, const AdaptiveWindowConfig &);
  // 唯一 T_CI 外参块相对初始标定的 6 维正则。
  static ResidualBlockInfo extrinsicPrior(const ExtrinsicState &, const AdaptiveWindowConfig &);
  // 参数顺序 pose_i, speedbias_i, pose_j, speedbias_j；共享可重积分的对象。
  static ResidualBlockInfo imu(std::uint64_t i, std::uint64_t j, const Eigen::Vector3d &gravity,
                             const std::shared_ptr<IntegrationBase> &);
  // 固定 T_IL 先把测量点变到 IMU 系，优化中只随 pose_i 变换。
  static ResidualBlockInfo lidar(std::uint64_t id, const LidarExtrinsic &,
                               const LidarPlaneObservation &, double huber);
  // 参数顺序 pose_anchor, pose_current, T_CI, rho_track；残差行数等于参考采样数。
  static ResidualBlockInfo photometric(std::uint64_t anchor, std::uint64_t current,
                                     std::uint64_t track, const VisualPatchObservation &reference,
                                     double current_inverse_exposure,
                                     const ProjectionFunction &, const ImageSampleFunction &,
                                     double huber, double confidence);
  // 不同状态使用两位姿+外参+深度；同一状态抵消位姿，仅使用外参+深度。
  static ResidualBlockInfo patchPlane(std::uint64_t anchor, std::uint64_t lidar,
                                    std::uint64_t track, const Eigen::Vector3d &ray,
                                    const LidarExtrinsic &, const LocalPlane &,
                                    double huber, double confidence);
  // Fit a scan-local plane from deskewed LiDAR measurements, not visual points.
  // query 和支持点都在同一扫描 LiDAR 系，检查邻域数量、平面性、厚度和查询距离。
  static bool fitLocalPlane(const Eigen::Vector3d &query_l,
                            const LidarPlaneObservationVector &points,
                            const AdaptiveWindowConfig &, LocalPlane *);
};
}}
