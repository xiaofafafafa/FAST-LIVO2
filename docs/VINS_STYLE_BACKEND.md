# VINS 风格后端代码导航

本次调整基于 Ceres 分支的 `df8ee2a3dfc59a4158a3adc4b19b9a1c17ae6c9e`。
保留原来的四类传感器残差、时间关联、自适应策略和 Schur 算法；合并此前的
`AdaptiveWindowBackend` / `FixedLagOptimizer` 为一个真正持有状态并完成求解的
`Estimator`。旧桥接层只通过薄适配器转发调用。不是仅给旧类增加同名别名。

## 从熟悉的 VINS 名称开始阅读

| 位置 / 名称 | 在本项目中的作用 |
|---|---|
| [estimator.h](../include/adaptive_window/estimator.h) | 状态、参数、因子和生命周期接口 |
| [estimator.cpp](../src/adaptive_window/estimator.cpp) | 接收传感器输入、建图、求解与滑窗 |
| `setParameter()` / `clearState()` | 配置标定与相机回调；清空窗口、因子、缓存和先验 |
| `processIMU()` | 缓存实际时间戳的原始 IMU，排序并处理重叠数据包 |
| `processLidar()` / `processImage()` | 关键帧判定、接纳状态、建立传感器因子 |
| `solveOdometry()` | 调用优化、滑窗，并更新诊断 |
| `vector2double()` / `double2vector()` | Eigen 状态与 Ceres 参数转换；后者只接收通过检查的结果 |
| `optimization()` | IMU 必要重积分、构建 Ceres Problem、求解和回滚 |
| `failureDetection()` | 检查整个窗口相对滤波快照的平移/旋转修正 |
| `slideWindow()` / `slideWindowOld()` / `slideWindowNew()` | 调度自适应边缘化；Old 为最旧帧，New 为次新帧 |
| `marginalizeFrame(id)` | 任意选中非最新帧的 Schur 消元与缓存清理 |
| `Rs, Ps, Vs, Bas, Bgs, Headers` | 保留状态的姿态、位置、速度、偏置和时间戳 |
| `para_Pose, para_SpeedBias, para_Feature, para_Ex_Pose` | Ceres 位姿、速度/偏置、逆深度、外参内存视图 |
| `pre_integrations` / `tmp_pre_integration` | 存活 IMU 边 / 正在建立的 IMU 边 |
| `last_marginalization_info` / `last_marginalization_parameter_blocks` | 当前 Schur 先验及其保留变量地址 |

## 主调用流程

1. FAST-LIVO2 完成滤波后，桥接层传入滤波状态和原始观测。
2. `processIMU()` 缓存 IMU；`processLidar()` 或 `processImage()` 评估信息量和关键帧。
3. `addOrGetFrame()`：新状态使用滤波值初始化并建立 IMU 连接；相同时刻复用状态。
   `updateFrontendPose()` 维护一份滤波位姿参考，LIO 后的同一时刻 VIO 会替换该参考。
4. 通过 `FactorFactory` 添加 LiDAR、光度和 patch–平面约束。
5. `solveOdometry()` 内依次完成 `optimization()` 和 `slideWindow()`。
   优化内依次执行 `vector2double()`、Ceres 求解、`failureDetection()`、
   `double2vector()`；拒绝时原位恢复参数，随后上层重置后端图。
6. 发布独立后端里程计和诊断；前端状态、协方差和地图保持由 FAST-LIVO2 管理。

## factor 目录

| 文件 | 内容 |
|---|---|
| [integration_base.h](../include/adaptive_window/factor/integration_base.h) | `IntegrationBase`，包括 `sum_dt, delta_p, delta_q, delta_v, linearized_ba/bg, jacobian, covariance` |
| [integration_base.cpp](../src/adaptive_window/factor/integration_base.cpp) | `integrate()`、`midPointIntegration()`、`repropagate()` |
| [imu_factor.h](../include/adaptive_window/factor/imu_factor.h) | `IMUFactor`，15 维白化预积分残差 |
| [lidar_factor.h](../include/adaptive_window/factor/lidar_factor.h) | `LidarFactor`，FAST-LIVO2 点面观测 |
| [photometric_factor.h](../include/adaptive_window/factor/photometric_factor.h) | `PhotometricFactor`，跨帧原始 patch 重投影和采样 |
| [patch_plane_factor.h](../include/adaptive_window/factor/patch_plane_factor.h) | `PatchPlaneFactor`，patch 中心深度与 scan-local 平面 |
| [frontend_pose_factor.h](../include/adaptive_window/factor/frontend_pose_factor.h) | `FrontendPoseFactor`，围绕滤波快照的 6 维正则 |
| [marginalization_factor.cpp](../src/adaptive_window/factor/marginalization_factor.cpp) | `ResidualBlockInfo::Evaluate()`、`preMarginalize()`、`marginalize()` 和 `MarginalizationFactor` |
| [pose_local_parameterization.h](../include/adaptive_window/factor/pose_local_parameterization.h) | 位姿右扰动及真实环境维度 Jacobian |

## 与 VINS-Mono 有意保留的区别

- 窗口大小可变，状态容器按稳定 ID 索引。ID 不是数组下标，也不保证连续；
  `frame_count` 是保留状态数，不是 VINS 的最新槽位下标。
- `para_*` 是指向 `parameter_blocks_` 所有内存的视图，不能自行释放或更换地址。
  删除中间帧无需整体搬移数组或重映射地址；每次 Schur 先验保存自己的线性化基点。
- `Headers` 保存 double 秒时间戳；`processIMU()` 接收实际时间戳数据包。
  预积分先严格裁剪到两个状态的时间区间，再执行中值积分，不照搬 VINS 的回调签名。
- `pre_integrations` 按边的终点 ID 索引。中间帧移除后，两条相邻 IMU 边已吸收到
  Schur 先验中，不重新拼接并重复添加同一批惯性信息。
- `para_Ex_Pose[0]` 表示 **IMU 到相机的 T_CI**，沿用 FAST-LIVO2 方向，
  与 VINS 常用的相机到 IMU 外参方向相反；不要直接互换标定数组。
- `PoseLocalParameterization` 在 Ceres 2.1+ 使用 Manifold、在 1.x/2.0 使用 LocalParameterization，保留真实的 7×6
  Plus Jacobian。不能替换成旧 VINS 的占位 Jacobian 写法。
- 前端已经初始化。本后端不重复执行 VINS 的视觉 SFM / 重力初始化，也不在
  `double2vector()` 中做 VINS 式全局 yaw/位置重新对齐；世界坐标由前端位姿正则保持。
- 视觉使用光度 patch 因子，不引入 VINS 的特征重投影观测替换原目标。

这些接口及数学模型由本项目独立实现，未复制 VINS 源文件。

## 配置与验证

[adaptive_window.yaml](../config/adaptive_window.yaml) 中的 `frontend_pose_*` 控制正则：
默认尺度为 0.05 m / 1°，权重为 1；接受上限为 0.20 m / 5°。
减小尺度会加强约束。设 `frontend_pose_prior_enabled: false` 可同时关闭逐帧正则和
修正幅度门限，保留初始位姿锚定。位姿正则与原观测相关，不使用前端协方差来宣称独立测量。

[测试源码](../test/adaptive_window_smoke.cpp) 包含解析解、参考替换、Jacobian、
拒绝回滚、四类原有残差以及边缘化检查。
NTU VIRAL `eee_01` 的 C++ 编译、合成测试和完整数据集回放已完成，
精度对比见 [NTU 评估记录](NTU_ADAPTIVE_EVALUATION.md)。
构建命令、公式与全部诊断字段见
[集成指南](ADAPTIVE_WINDOW_INTEGRATION.md)，已完成检查见
[验证记录](CERES_BACKEND_VALIDATION.md)。
