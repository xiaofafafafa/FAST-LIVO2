/**
 * @file estimator.h
 * @brief VINS 风格滑窗后端接口与成员定义；稳定帧 ID 支持任意非最新状态的自适应淘汰。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/adaptive_policy.h"
#include "adaptive_window/factor/factor_factory.h"
#include "adaptive_window/factor/pose_local_parameterization.h"
#include "adaptive_window/information_evaluator.h"
#include "adaptive_window/temporal_associator.h"
#include <map>

namespace fast_livo { namespace adaptive_window {

// VINS-style entry point for this variable-size, asynchronous sliding window.
// Stable frame IDs replace VINS' contiguous slots so arbitrary adaptive removal
// never invalidates retained parameter addresses or residual references.
class Estimator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // 校验配置并 clearState；构造后已准备外参参数块和外参先验。
  explicit Estimator(const AdaptiveWindowConfig &);
  // para_* 是内部存储的地址视图；禁止复制，避免参数地址指向其他实例。
  Estimator(const Estimator &) = delete;
  Estimator &operator=(const Estimator &) = delete;
  // 设置 T_IL、初始 T_CI 与投影回调，并清空窗口；不是运行中增量更新外参。
  void setParameter(const LidarExtrinsic &, const ExtrinsicState &, ProjectionFunction = {});
  // 清空全部状态、IMU、patch 和统计；自动恢复若需保留缓冲/计数则使用 recover。
  void clearState();
  // 缓存带实际时间戳的样本，排序去重，允许不同输入包重叠。
  void processIMU(const ImuSampleVector &);
  // 点云入口：策略 → 接纳/复用状态 → 点面因子 → 求解与滑窗。
  // false 也可能只是未选关键帧或 IMU 不足，应结合 diagnostics 判断。
  bool processLidar(const NavState &, const MeasurementSummary &, const LidarPlaneObservationVector &);
  // 图像入口：老 track 新增跨帧光度因子，新 track 需匹配局部平面才能建立深度。
  bool processImage(const NavState &, const MeasurementSummary &, const VisualPatchObservationVector &,
                    const ImageSampleFunction &);
  // 只返回最新保留状态，时间可早于当前输入；恢复后的空窗口返回 false。
  bool latestEstimate(NavState *) const;
  // 内部返回 T_CI，ROS 发布 T_IC 时由桥接层取逆。
  ExtrinsicState cameraExtrinsicEstimate() const;
  const BackendDiagnostics &diagnostics() const { return diagnostics_; }
  const AdaptiveWindowConfig &config() const { return config_; }

  // The solve follows vector2double -> Ceres -> failureDetection -> double2vector.
  // 常规求解入口：先优化再滑窗；失败则调用 recover，成功后刷新诊断。
  bool solveOdometry();
  // 低层单次求解：只负责参数回滚，不负责重置整图；final_cost 仅成功时写入。
  bool optimization(double *final_cost = nullptr);
  // Eigen 状态写入已有 Ceres 数组，原位复制以保持参数地址稳定。
  void vector2double();
  // 将被接受的解写回状态，不改 frontend_poses_ 的滤波参考快照。
  void double2vector();
  // true 表示拒绝；统计全窗口相对滤波参考的修正并检查门限。
  bool failureDetection();
  // 按策略循环边缘化直到目标大小，最新状态受到保护。
  bool slideWindow();
  // 消元最旧状态；先保留关联因子信息再删除存储。
  bool slideWindowOld();
  // New 沿用 VINS 命名，表示次新状态。
  bool slideWindowNew();  // the second-newest retained state

  // Small graph API used by deterministic tests and non-ROS integrations.
  // 低层建点接口，创建状态、参数和滤波位姿项；失败返回 0。
  // 单独调用不补 IMU/首次状态先验，常规使用应走传感器入口。
  std::uint64_t addFrame(const NavState &);
  bool hasFrame(std::uint64_t id) const { return nodes_.count(id) != 0; }
  // 由新到旧查找时间容差内的保留状态，返回稳定 ID 或 0。
  std::uint64_t findFrameNear(double timestamp, double tolerance) const;
  // 从当前优化状态和 nodes_ 中的重力等元数据组装 NavState，id 必须存在。
  NavState stateEstimate(std::uint64_t id) const;
  // 标记模态，并保存已接纳质量分数最大值；不是计算独占信息量。
  void markFrame(std::uint64_t id, Modality, double information);
  // track_id 创建一个逆深度块，anchor 是参考状态 ID；先检查范围和重复。
  bool addFeature(std::uint64_t track, std::uint64_t anchor, double inverse_depth);
  // 深度不再被原始因子和边缘化先验引用时才可释放。
  void removeUnusedFeature(std::uint64_t track);
  bool hasFeature(std::uint64_t track) const { return para_Feature.count(track) != 0; }
  // 检查参数键、尺寸和残差/Jacobian 后入图；逐帧位姿项须走专用替换接口。
  bool addResidualBlock(const ResidualBlockInfo &);
  // 每帧一份滤波参考；同时间 LIO/VIO 替换参考及位姿初值，不叠加第二份正则。
  bool updateFrontendPose(std::uint64_t id, const NavState &);
  // 更新存活因子权重；已吸入边缘化先验的历史权重不会追溯改变。
  void setWeights(const ModalityWeights &w) { weights_ = w; }
  // 删除非最新状态及以其为参考的深度；只吸收触及待删变量的因子和旧先验。
  bool marginalizeFrame(std::uint64_t id);
  // 按稳定 ID/时间顺序生成当前窗口策略快照。
  PolicyFrameVector policyFrames() const;
  std::vector<std::uint64_t> frameIds() const;
  std::size_t numFrames() const { return nodes_.size(); }
  std::size_t numFactors() const { return residual_blocks_.size() + (last_marginalization_factor.cost ? 1 : 0); }
  // 只数当前显式原始因子块，已进入先验的历史因子不重复计数。
  std::size_t countFactors(Modality) const;

 private:
  // INITIAL 仅表示后端尚无成功求解；这里没有 VINS 的视觉 SFM 初始化阶段。
  enum SolverFlag { INITIAL, NON_LINEAR };
  // 第三个选项支持自适应策略选中的任意中间帧。
  enum MarginalizationFlag { MARGIN_OLD, MARGIN_SECOND_NEW, MARGIN_SELECTED };
  // 扫描的前端位姿快照与匹配内点，用来在扫描坐标系搜索局部平面。
  struct LidarFrame { NavState seed; LidarPlaneObservationVector observations; };
  // patches_ 用 track_id 作键；anchor 指参考帧，reference 保存固定原始 patch。
  struct Patch { std::uint64_t anchor; VisualPatchObservation reference; };
  // 更新当前模态统计，屏蔽过期的另一模态，再产生本次策略。
  PolicyDecision decide(const NavState &, const MeasurementSummary &, Modality);
  // 同时间先复用；新时间须满足关键帧条件和严格的 IMU 连接条件。
  std::uint64_t addOrGetFrame(const NavState &, Modality, const MeasurementSummary &, const PolicyDecision &);
  // 清理历史 IMU，保留边界前一条用于下一段端点插值。
  void consumeImuThrough(double);
  // 清空图但保留 IMU 缓冲、累计异常和最近修正幅度，等待下一次输入重建。
  void recover();

  AdaptiveWindowConfig config_;
  InformationEvaluator information_evaluator_;
  AdaptivePolicy policy_;
  TemporalAssociator associator_;
  LidarExtrinsic lidar_extrinsic_;
  // initial_ 为固定标定参考，另一个成员保存已被接受的优化外参。
  ExtrinsicState camera_extrinsic_initial_, camera_extrinsic_;
  ProjectionFunction projection_;
  // 排序且去重的 IMU 缓冲，不允许用一个最新样本替代缺失时间段。
  ImuSampleVector pending_imu_;
  MeasurementSummary latest_lidar_, latest_vision_;
  bool has_lidar_ = false, has_vision_ = false;
  // 0 表示无效；next_id_ 在一次图生命周期内递增，重置后从 1 开始。
  std::uint64_t latest_node_ = 0, next_id_ = 1;
  int target_window_ = 6;
  // 各模态独立的重复事件检查时间，未接纳为关键帧也会更新。
  double last_lidar_time_ = -std::numeric_limits<double>::infinity();
  double last_image_time_ = -std::numeric_limits<double>::infinity();
  std::map<std::uint64_t, LidarFrame> lidar_frames_;
  std::map<std::uint64_t, Patch> patches_;
  BackendDiagnostics diagnostics_;
  ModalityWeights weights_;

  // VINS naming, keyed by stable ID. frame_count is the number of active states,
  // NOT VINS' newest slot index. Gravity and policy metadata stay in nodes_.
  // Rs[id]=R_WI；Ps/Vs 在世界系，Bas/Bgs 为 IMU 测量模型的偏置。
  std::map<std::uint64_t, Eigen::Matrix3d> Rs;
  std::map<std::uint64_t, Eigen::Vector3d> Ps, Vs, Bas, Bgs;
  // Headers 为秒时间戳；nodes_ 保存重力/标签等元数据，不是最新位姿的唯一来源。
  std::map<std::uint64_t, double> Headers;
  std::map<std::uint64_t, PolicyFrame> nodes_;
  // 独立保存滤波快照，只由前端新事件更新，避免正则参考随后端解漂移。
  std::map<std::uint64_t, NavState> frontend_poses_;
  int frame_count = 0;
  SolverFlag solver_flag = INITIAL;
  MarginalizationFlag marginalization_flag = MARGIN_OLD;

  // 唯一拥有 Ceres 参数数组内存的容器；参数存活期间不得重分配其 vector。
  ParameterValues parameter_blocks_;  // owns all Ceres memory
  // 借用上述内存：pose=[p,q_xyzw]，speedbias=[v,ba,bg]，feature=[rho]。
  std::map<std::uint64_t, double *> para_Pose, para_SpeedBias, para_Feature;
  double *para_Ex_Pose[1] = {nullptr}; // T_CI, not VINS' opposite-direction T_IC
  // track_id → 参考帧 ID，保证参考帧删除时其深度和全部关联 patch 信息共同消元。
  std::map<std::uint64_t, std::uint64_t> feature_anchors_;
  // 仍存活的原始非线性因子，每次优化重新求值。
  std::vector<ResidualBlockInfo> residual_blocks_;
  PoseLocalParameterization local_parameterization_;
  // Destination ID -> the still-active IMU edge ending at that state.
  std::map<std::uint64_t, std::shared_ptr<IntegrationBase>> pre_integrations;
  // 新 IMU 边入图前暂存，入图后由原始因子和 pre_integrations 共享持有。
  std::shared_ptr<IntegrationBase> tmp_pre_integration;
  // 仅保存当前先验的基点和平方根系统，避免形成持有所有历史图像的引用链。
  std::shared_ptr<MarginalizationInfo> last_marginalization_info;
  ResidualBlockInfo last_marginalization_factor;
  // 地址顺序必须与先验 keys 一致；保留参数地址不因删除其他帧而改变。
  std::vector<double *> last_marginalization_parameter_blocks;
};
}}
