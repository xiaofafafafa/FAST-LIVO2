/**
 * @file types.h
 * @brief 后端共享状态、观测、配置和诊断；W/I/C/L 依次表示世界、IMU、相机、LiDAR 坐标系。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#ifndef FAST_LIVO_ADAPTIVE_WINDOW_TYPES_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_TYPES_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fast_livo
{
namespace adaptive_window
{

constexpr int kNavStateDim = 15;
constexpr int kPoseDim = 6;
constexpr int kExtrinsicDim = 6;

// 模态标签用于损失权重与诊断分类；kFrontendPose 不参与 LiDAR/视觉自适应权重。
enum class Modality
{
  kPrior = 0,
  kImu,
  kLidar,
  kVision,
  kCrossModal,
  kFrontendPose
};

inline const char *ToString(Modality modality)
{
  switch (modality)
  {
    case Modality::kPrior: return "prior";
    case Modality::kImu: return "imu";
    case Modality::kLidar: return "lidar";
    case Modality::kVision: return "vision";
    case Modality::kCrossModal: return "cross_modal";
    case Modality::kFrontendPose: return "frontend_pose";
  }
  return "unknown";
}

// 单帧导航状态；T_AB 表示 B 系到 A 系的变换，重力由前端初始化并在后端固定。
struct NavState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 实际状态时间，单位秒；与各传感器及 IMU 必须使用同一时基。
  double timestamp = 0.0;
  // q_WI：IMU 系旋转到世界系的单位四元数。
  Eigen::Quaterniond q_w_i = Eigen::Quaterniond::Identity();
  // IMU 原点在世界系的位置，m。
  Eigen::Vector3d p_w_i = Eigen::Vector3d::Zero();
  // 世界系速度，m/s。
  Eigen::Vector3d v_w_i = Eigen::Vector3d::Zero();
  // 陀螺仪偏置，rad/s；Ceres speed-bias 中位于最后三维。
  Eigen::Vector3d bias_g = Eigen::Vector3d::Zero();
  // 加速度偏置，m/s²；Ceres speed-bias 中位于中间三维。
  Eigen::Vector3d bias_a = Eigen::Vector3d::Zero();
  // 世界系重力，m/s²，默认沿负 z；预积分残差使用此符号约定。
  Eigen::Vector3d gravity_w = Eigen::Vector3d(0.0, 0.0, -9.81);
};

// 可优化的 T_CI：p_C=R_CI*p_I+t_CI，方向与 VINS 常见的 T_IC 相反。
struct ExtrinsicState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Active transform from the IMU frame to the camera frame:
  // p_C = q_c_i * p_I + p_c_i.
  Eigen::Quaterniond q_c_i = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_c_i = Eigen::Vector3d::Zero();
};

// 固定的 T_IL：p_I=R_IL*p_L+t_IL；当前后端不优化 LiDAR—IMU 外参。
struct LidarExtrinsic
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Active transform from the LiDAR frame to the IMU frame:
  // p_I = q_i_l * p_L + p_i_l.
  Eigen::Quaterniond q_i_l = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_i_l = Eigen::Vector3d::Zero();
};

struct ImuSample
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // IMU 绝对采样时间（秒），不是相邻采样间隔 dt。
  double timestamp = 0.0;
  // IMU 系加速度，桥接层已按前端尺度换算为 m/s²，尚未减偏置。
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  // IMU 系角速度，rad/s，尚未减偏置。
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct LidarPlaneObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 去畸变后的 LiDAR 系点；进入后端后随位姿重新变换到世界系。
  Eigen::Vector3d point_l = Eigen::Vector3d::Zero();
  // 世界系地图平面法向，构建因子时归一化。
  Eigen::Vector3d normal_w = Eigen::Vector3d::UnitZ();
  // 世界系平面上的参考点。
  Eigen::Vector3d center_w = Eigen::Vector3d::Zero();
  // 前端匹配时的距离记录；Ceres 中重新计算距离，不用此固定值代替残差。
  double signed_distance = 0.0;
  // 测量和地图平面不确定性合成的距离方差；残差除以 sqrt(variance)。
  double variance = 1.0e-3;
};

// Raw patch sampled in the CURRENT image. The backend retains its own reference
// patch at an admitted keyframe, never a frontend-prewarped patch.
struct VisualPatchObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // 地图点的稳定编号，跨图像关联的键；0 不可用。
  std::uint64_t track_id = 0;
  // 前端地图点，仅用于初始化和搜索 LiDAR 平面邻域，非固定视觉位置残差。
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero(); // initialization only
  // 中心像素射线，z=1；rho 表示逆 z 深度，不是单位射线上的逆距离。
  Eigen::Vector3d center_ray = Eigen::Vector3d::UnitZ(); // z = 1
  // 参考相机系 patch 切平面法向，存活期间固定。
  Eigen::Vector3d normal_c = Eigen::Vector3d::UnitZ(); // fixed patch tangent
  // 中心逆 z 深度初值，1/m；入图后由独立标量参数优化。
  double inverse_depth = 1.0;
  // 前端追踪质量，构建因子时限幅成置信权重。
  double tracking_score = 1.0;
  // 原始图像上各采样像素的 z=1 射线；不是前端预先 warp 后的像素坐标。
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> rays;
  // 与 rays 一一对应的原始灰度；参考帧接纳后固定保存。
  std::vector<double> intensities;
  // 前端给出的逆曝光系数，后端暂不优化。
  double inverse_exposure = 1.0;
  // 灰度残差白化尺度；因子另除以 sqrt(像素数) 控制 patch 尺寸影响。
  double photometric_sigma = 10.0;
};

// 策略统计与真实优化量测分开：此结构用于打分，原始观测另行进入 Ceres 因子。
struct MeasurementSummary
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Modality modality = Modality::kPrior;
  double timestamp = 0.0;
  // 前端 6×6 位姿信息近似，顺序 [旋转,平移]；不能直接当作 Ceres [平移,旋转] 的信息矩阵。
  Eigen::Matrix<double, kPoseDim, kPoseDim> information =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
  // 统计所用标量残差数，和后端诊断的因子块数量含义不同。
  std::size_t residual_count = 0;
  // 各模态自身单位下的 RMS；LiDAR 与灰度必须分别按尺度归一化。
  double rmse = std::numeric_limits<double>::infinity();
  // 有效观测占候选观测的比例。
  double inlier_ratio = 0.0;

  // 对数行列式近似：加数值下限后的特征值对数和。
  double log_det = -std::numeric_limits<double>::infinity();
  // 最弱方向的信息强度代理量，不单独证明全系统可观测性。
  double min_eigenvalue = 0.0;
  // 最大/最小特征值比，分母加下限避免除零。
  double condition_number = std::numeric_limits<double>::infinity();
  // log-det 限幅归一化后按模态时间平滑。
  double normalized_information = 0.0;
  // 信息、残差、内点率、弱方向强度组合的启发式质量分数。
  double reliability = 0.0;
  // 无有效量测、非有限数或特征分解失败时为 false。
  bool valid = false;
};

// 白化残差平方/鲁棒损失外的乘数；IMU、初始/边缘化先验及位姿正则不使用这三个系数。
struct ModalityWeights
{
  double lidar = 1.0;
  double vision = 1.0;
  double cross_modal = 1.0;
};

// 一次输入的策略结果；未选新关键帧时，仍可合并更新已有的同时间节点。
struct PolicyDecision
{
  ModalityWeights weights;
  int target_window_size = 6;
  bool select_keyframe = false;
  double keyframe_utility = 0.0;
};

// 参数默认值集中于此；ROS 读取见 FastLivoBridge::LoadConfig，用户设置见 config/adaptive_window.yaml。
// 所有默认尺度为实验起点；构造 Estimator 时会检查有限性、范围和基本逻辑约束。
struct AdaptiveWindowConfig
{
  // 控制桥接层是否创建并调用后端。
  bool enabled = false;
  // 兼容旧参数；当前桥接层强制关闭反馈，尚无协方差/地图一致传递。
  bool feedback_enabled = false;

  // 目标窗口状态数下界；启动时实际状态可少于此值。
  int min_window_size = 4;
  // 自适应目标窗口上界，按导航状态个数计。
  int max_window_size = 12;
  // 初始化时的目标窗口大小。
  int initial_window_size = 6;
  // 每次 LiDAR 事件最多新增点面因子块数。
  int max_lidar_factors = 160;
  // 每幅图像最多接纳的 patch 观测数，既限制新增几何因子，也限制新增光度因子。
  int max_cross_modal_factors = 80;
  // 整个窗口存活的参考 patch/逆深度上限。
  int max_active_patches = 160;

  // 秒；数值同时间的 LIO/VIO 共享状态，不能替代真实时间偏移估计。
  double merge_time_tolerance = 1.0e-4;
  // 秒；最近邻视觉—LiDAR 关联的最大时间差。
  double association_max_dt = 0.08;
  // 秒；时间置信度 exp(-dt²/(2*sigma²)) 的尺度。
  double association_time_sigma = 0.025;
  // 是否允许同一 LiDAR 帧被多次选中；false 时查询成功立即标记已使用。
  bool allow_pair_reuse = true;

  // 秒；效用阈值接纳新帧的最小时间间隔。
  double min_keyframe_dt = 0.05;
  // 秒；到达此间隔时策略尝试接纳，仍需通过 IMU 端点/间隔检查。
  double max_keyframe_dt = 0.50;
  // m；平移效用的归一化尺度。
  double keyframe_translation = 0.20;
  // 度；旋转效用的归一化尺度。
  double keyframe_rotation_deg = 8.0;
  // 运动、时间、质量、模态互补组合分数的接纳阈值。
  double keyframe_utility_threshold = 0.55;

  // LiDAR/视觉/跨模态自适应权重下界。
  double min_modality_weight = 0.15;
  // 自适应权重上界。
  double max_modality_weight = 2.50;
  // 权重指数平滑中旧值占比，越大变化越慢。
  double weight_smoothing = 0.80;
  // 各模态归一化 log-det 的旧值平滑系数。
  double information_smoothing = 0.90;
  // m；LiDAR 残差质量指数衰减的尺度。
  double lidar_rmse_scale = 0.08;
  // 灰度；视觉残差质量指数衰减的尺度。
  double vision_rmse_scale = 20.0;

  // 信息统计的特征值数值下限，不是向 Schur 零空间注入信息。
  double information_eigen_floor = 1.0e-9;
  // log-det 归一化区间下端。
  double information_logdet_floor = -35.0;
  // log-det 归一化区间上端，区间外限幅到 [0,1]。
  double information_logdet_ceiling = 30.0;
  // 最小特征值质量项的归一化尺度。
  double degeneracy_eigen_threshold = 1.0e-4;

  // 每次 Ceres 求解的最大迭代次数。
  int solver_iterations = 8;
  // 秒；Ceres 软预算，不保证构图/求解/边缘化总时间的硬实时上限。
  double solver_time_limit = 0.03; // seconds; a soft Ceres budget
  // 白化点面残差的 Huber 转折点，0 表示不用鲁棒核。
  double huber_delta_lidar = 2.0;
  // 白化整块 patch 残差范数的 Huber 转折点。
  double huber_delta_vision = 3.0;
  // 白化几何残差的 Huber 转折点。
  double huber_delta_cross_modal = 3.0;
  // 是否联合优化 T_CI；打开开关本身不保证外参可观。
  bool optimize_extrinsics = false; // enable only with sufficient excitation
  // m；扫描内点拟合局部平面的邻域半径。
  double plane_search_radius = 0.75;
  // m；局部平面拟合厚度的 RMS 上限。
  double plane_max_rms = 0.05;
  // m；几何残差基础尺度，与平面拟合厚度合成。
  double cross_modal_plane_sigma = 0.05;
  // 1/m；Ceres 逆深度下界。
  double min_inverse_depth = 0.005;
  // 1/m；Ceres 逆深度上界。
  double max_inverse_depth = 5.0;

  // Correlated regularization around FAST-LIVO2, not calibrated covariance.
  // 同时启用逐帧滤波位姿正则和相对滤波结果的修正接受检查。
  bool frontend_pose_prior_enabled = true;
  // m；滤波位置正则尺度，越小越贴近滤波参考。
  double frontend_pose_translation_sigma = 0.05; // metres
  // 度；滤波旋转正则尺度，构建残差时转为弧度。
  double frontend_pose_rotation_deg = 1.0;
  // 位姿正则独立损失乘数，有效尺度为 sigma/sqrt(weight)。
  double frontend_pose_prior_weight = 1.0;
  // m；任一状态超限时拒绝整次求解。
  double frontend_pose_max_translation = 0.20; // reject the whole solve beyond this
  // 度；任一状态相对滤波快照的转角接受上限。
  double frontend_pose_max_rotation_deg = 5.0;

  // First-state pose anchor used only when frontend_pose_prior_enabled is false.
  // m；只在关闭逐帧位姿正则时使用的首次位置锚定尺度。
  double pose_prior_translation_sigma = 0.08;
  // 度；只在关闭逐帧位姿正则时使用的首次旋转锚定尺度。
  double pose_prior_rotation_deg = 3.0;
  // m/s；首次状态速度先验尺度。
  double velocity_prior_sigma = 0.30;
  // 首次状态偏置先验尺度；当前相同数值分别用于两种不同单位的偏置。
  double bias_prior_sigma = 0.05;
  // m；外参平移相对初始标定的正则尺度。
  double extrinsic_translation_sigma = 0.05;
  // 度；外参旋转相对初始标定的正则尺度。
  double extrinsic_rotation_deg = 3.0;

  // VINS 风格离散加速度噪声标准差，不能直接填连续噪声密度。
  double imu_acc_noise = 0.10;
  // 离散角速度噪声标准差。
  double imu_gyr_noise = 0.01;
  // 当前离散预积分模型的加速度偏置随机游走噪声参数。
  double imu_acc_bias_rw = 0.001;
  // 当前离散预积分模型的陀螺仪偏置随机游走噪声参数。
  double imu_gyr_bias_rw = 0.0001;
  // 秒；相邻原始 IMU 间隔超过此值时拒绝该区间。
  double imu_max_gap = 0.05;
  // m/s²；加速度偏置变化超过此阈值时从保存样本重积分。
  double imu_repropagate_acc_bias = 0.05;
  // rad/s；陀螺仪偏置重积分触发阈值。
  double imu_repropagate_gyr_bias = 0.005;
};

// 成功求解后更新的窗口统计和累计异常数。因子数按残差块计，已吸入先验的信息不重复计数。
struct BackendDiagnostics
{
  // 当前存活导航状态数，不含外参和深度。
  std::size_t window_size = 0;
  // 原始残差块加至多一个边缘化先验。
  std::size_t factor_count = 0;
  // 最近策略目标状态数。
  int target_window_size = 0;
  ModalityWeights weights;
  // 最近成功的求解加滑窗耗时，ms，供后续窗口策略使用。
  double solve_time_ms = 0.0;
  // Ceres 求解结束、边缘化之前的 cost。
  double last_cost = 0.0;
  // 最近求解成功标志；未选关键帧的输入不一定刷新此值。
  bool solve_success = false;
  // 最近淘汰的稳定帧 ID，0 表示尚无。
  std::uint64_t last_marginalized_id = 0;
  // 存活 patch–平面因子块数，不是全部历史时间匹配数。
  std::size_t cross_modal_pairs = 0;
  std::size_t lidar_factors = 0;
  std::size_t vision_factors = 0;
  std::size_t imu_factors = 0;
  // 累计 IMU 建边失败次数。
  std::size_t rejected_imu_intervals = 0;
  // 累计自动恢复次数；显式 clearState 会清空计数。
  std::size_t reset_count = 0;
  // 启用时正常等于窗口状态数。
  std::size_t frontend_pose_factors = 0;
  // 累计因修正检查拒绝的求解次数。
  std::size_t rejected_pose_refinements = 0;
  // 最近有限求解在全窗口的最大平移修正，m；拒绝后仍保留。
  double max_frontend_translation_correction = 0.0;
  // 全窗口最大旋转修正，度；边缘化前统计。
  double max_frontend_rotation_correction_deg = 0.0;
  // 当前外参相对初始标定的平移变化，m。
  double camera_extrinsic_translation_change = 0.0;
  // 当前外参相对初始标定的旋转变化，度。
  double camera_extrinsic_rotation_change_deg = 0.0;
};

using NavStateVector =
    std::vector<NavState, Eigen::aligned_allocator<NavState>>;
using ImuSampleVector =
    std::vector<ImuSample, Eigen::aligned_allocator<ImuSample>>;
using LidarPlaneObservationVector =
    std::vector<LidarPlaneObservation,
                Eigen::aligned_allocator<LidarPlaneObservation>>;
using VisualPatchObservationVector =
    std::vector<VisualPatchObservation,
                Eigen::aligned_allocator<VisualPatchObservation>>;

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_TYPES_H_
