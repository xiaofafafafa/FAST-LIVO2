# Ceres 自适应滑窗后端：实现、公式和验证

## 状态与适用范围

本分支从 `feature/adaptive-window-backend` 的
`90acf12dd7e94081e00a63a86393768b63444cc0` 重构，替换旧的 Eigen 冻结线性因子求解器。
随后在 Ceres 版本 `df8ee2a3dfc59a4158a3adc4b19b9a1c17ae6c9e` 上按 VINS-Mono 命名组织为
`Estimator` + `factor/`，并加入 FAST-LIVO2 位姿正则项。代码阅读顺序见
[VINS 风格代码导航](VINS_STYLE_BACKEND.md)。

**当前仍是研究实现。** 2026-11-13 在 ROS Noetic、Eigen 3.3.7、Ceres 1.14 上
完成 standalone 和 catkin 编译，合成后端测试通过；当时的执行环境阻止启动 ROS Master。
2026-11-16 已在当前 ROS 环境中完整回放 NTU VIRAL `eee_01`，精度结果见
[NTU 评估记录](NTU_ADAPTIVE_EVALUATION.md)：滑窗输出可以独立运行，但此配置下
其平移 APE 高于 FAST-LIVO2 前端，不能声称精度提高。

FAST-LIVO2 的 ESIKF 前端、去畸变和地图仍正常运行。本模块在同一处理线程内同步执行，
产生另一条后端里程计；“独立输出”不代表独立线程，也不是完整的后端地图重优化。
桥接层强制 shadow mode：不将后端状态或外参回写前端，因为尚未实现一致的协方差、
地图和历史线性化信息转移。旧的 `feedback:=true` 会警告并被忽略。

## 1. 模块边界

| 模块 | 职责 |
|---|---|
| `estimator.h/.cpp` | 状态管理、接收量测、构建因子、`optimization()`、`slideWindow()`、故障恢复 |
| `ceres_types.h` | 参数块标识、VINS 风格尺寸和四元数布局 |
| `factor/pose_local_parameterization.h` | Ceres Manifold API 下的位姿右扰动 |
| `factor/integration_base.h/.cpp` | 时间区间裁剪、中值积分、15 维协方差、偏置 Jacobian 和重积分 |
| `factor/{imu,lidar,photometric,patch_plane}_factor.h` | 保留的四类传感器残差 |
| `factor/frontend_pose_factor.h` | FAST-LIVO2 每个状态的位姿正则项 |
| `factor/factor_factory.h/.cpp` | 因子装配、参数块关联和 scan-local 平面拟合 |
| `factor/marginalization_factor.h/.cpp` | `ResidualBlockInfo`、`MarginalizationInfo` 和 Schur 先验 |
| `information_evaluator.*` / `adaptive_policy.*` | 信息评价、权重、关键帧、窗口和边缘化策略 |
| `temporal_associator.*` | 有限时间门限下的最近邻关联 |
| `adaptive_window_backend.h` | 兼容原桥接层的薄适配器，算法实现全部位于 Estimator |
| `fast_livo_bridge.*` | ROS 参数、patch 导出、相机/图像回调、诊断和发布 |

前端位姿同时用作新状态初值和可配置的正则参考，以实现滤波轨迹附近的局部微调。
它与四类传感器残差复用了观测，**不被解释为独立的位姿测量**，其尺度不是前端滤波协方差。
同一节点的 LIO/VIO 更新替换一份参考，不重复叠加；优化结果也不会覆盖参考快照。
关闭该项后保留首次位姿锚定、初始速度/偏置和外参先验，以及边缘化先验。

## 2. 状态、坐标系和目标函数

记 \(T_{AB}\) 把 B 系坐标变换到 A 系。导航状态为
\(T_{WI_i},v_i,b_{a_i},b_{g_i}\)，固定重力 \(g_W\) 来自前端初始化。
固定 LiDAR–IMU 标定是 \(T_{IL}\)，相机–IMU 标定是 \(T_{CI}\)。

Ceres 使用与 VINS 类似的存储布局：

- pose/extrinsic：\([p_x,p_y,p_z,q_x,q_y,q_z,q_w]\)，7 维存储、6 维切空间；
- speed-bias：\([v_x,v_y,v_z,b_{ax},b_{ay},b_{az},b_{gx},b_{gy},b_{gz}]\)；
- patch：参考帧中心射线 \(f_0\) 的逆 z 深度 \(\rho\)，每个 patch 一个标量。

位姿右扰动为 \(p^+=p+\delta p,\ q^+=q\operatorname{Exp}(\delta\theta)\)。
切空间顺序是 \([\delta p,\delta\theta]\)，与策略模块沿用的
\([\delta\theta,\delta p]\) 信息矩阵顺序不同；两者没有直接混用。
流形返回真实的 7×6 Plus Jacobian，而不是将局部 Jacobian 塞进环境 Jacobian 的前六列。

\[
\min_{\{x_i\},T_{CI},\{\rho_a\}}
 \|r_{\rm prior}\|^2+\lambda_F\sum_i\|r_{F,i}\|^2+
 \sum \|r_I\|^2+
 \sum w_L\,\mathcal H(\|r_L\|^2)+
 \sum w_V c_V\,\mathcal H(\|r_V\|^2)+
 \sum w_X c_t c_V\,\mathcal H(r_X^2).
\]

各残差先按噪声尺度白化；\(\mathcal H\) 是 Huber loss。
存活量测因子的模态权重在每次求解时更新；已经吸收进边缘化先验的权重固定，不追溯修改历史信息。
`optimize_extrinsics=false` 默认固定 \(T_{CI}\)；设为 true 后与导航状态及逆深度联合优化，
仍保留外参正则先验。仅启用该选项不保证外参可观。

### 2.1 LiDAR 点到平面

\[
r_L=\frac{n_W^\top(R_{WI_i}(R_{IL}p_L+t_{IL})+p_{WI_i}-c_W)}{\sigma_L}.
\]

使用 FAST-LIVO2 的 `ptpl_list_`、地图平面以及测量/平面不确定性加权形式。
`point_b_` 按原前端定义是去畸变后的 LiDAR 系点。
窗口内固定平面关联及其方差，但每次 Ceres 迭代重新计算变换和残差。
每个状态的前端位姿正则另行维护一份，与本点面因子分开。
本版本未在后端迭代中重新查体素地图或重新去畸变。

### 2.2 视觉光度 patch

桥接层导出当前原始灰度图上的 4×4 稀疏采样、射线和前端曝光估计。
首次接纳某个稳定 `backend_track_id_` 时，后端保存自己的参考 patch；
后续同一地图点的观测连接参考帧 \(a\)、当前帧 \(j\)、外参和逆深度。

参考 patch 的切平面法向 \(n_a\) 固定于参考相机坐标系。采样射线 \(f_k\) 的参考点为

\[
P_{a,k}=\frac{n_a^\top f_0}{\rho_a\,n_a^\top f_k}f_k,\qquad
P_{j,k}=T_{CI}T_{WI_j}^{-1}T_{WI_a}T_{IC}P_{a,k}.
\]
\[
r_{V,k}=\frac{\tau_j I_j(\pi(P_{j,k}))-\tau_a I_a(u_{a,k})}
                   {\sigma_V\sqrt{M}}.
\]

这里 \(\tau\) 是前端的逆曝光量，当前实现不将曝光作为优化变量；
除以 \(\sqrt M\) 使一个 patch 的 loss 与白化像素 RMS 对应，不随采样数直接放大。
该残差沿用 FAST-LIVO2 的直接光度模型，但后端采用参考帧射线–平面重投影，
**不是逐行复现前端的多层金字塔仿射 warp/ESIKF 更新**。

不使用已经 warp 到当前图像的固定 patch。每次迭代都会重新投影和双线性采样；
图像梯度是该双线性插值的精确分片导数。Vikit 投影采用中心数值差分，
后续变换由 Ceres 自动微分串接。图像采用 clone 持有，防止前端复用缓冲区。
初始投影无效的因子不加入图；试探步越界返回求值失败，让 Ceres 拒绝该步，而不是把残差置零。

### 2.3 VINS-Mono 风格 IMU 预积分

在相邻已接纳导航节点的实际时间区间 \([t_i,t_j]\) 上：

1. 两端都必须有 IMU 包围；端点做线性插值，不做无界外推。
2. 用去偏角速度中值更新 \(\Delta q\)，用两端旋转后的加速度均值更新 \(\Delta p,\Delta v\)。
3. 传播 \(J_{15\times15}\) 和 \(\Sigma_{15\times15}\)，噪声包含两端加速度、
   角速度及加速度/陀螺仪偏置随机游走。
4. 采用一阶偏置修正；偏置变化超过配置阈值时，从该区间保存的原始 IMU 重新积分。
5. 用 Cholesky 三角求解白化，避免显式求协方差逆。

残差顺序为位置、姿态、速度、加速度偏置、陀螺仪偏置：

\[
r_I=
\begin{bmatrix}
 R_i^\top(p_j-p_i-v_i\Delta t-\tfrac12g_W\Delta t^2)-\widehat{\Delta p}\\
 2\,{\rm vec}(\widehat{\Delta q}^{-1}q_i^{-1}q_j)\\
 R_i^\top(v_j-v_i-g_W\Delta t)-\widehat{\Delta v}\\
 b_{a_j}-b_{a_i}\\ b_{g_j}-b_{g_i}
\end{bmatrix}.
\]

代码中的 \(g_W\) 通常朝负 z；因此这里的重力符号不能直接照搬使用反号重力变量的实现。
噪声参数采用 VINS 风格的**离散标准差约定**，不能直接填入 IMU 数据手册的连续噪声密度。
桥接层按前端初始化的尺度将加速度转换到 m/s²，核心不再二次缩放。

`LIVMapper::backendImuMeasurements()` 只读获取帧末后的第一条 IMU，供后端插值，
不从前端队列弹出，也不改变前端测量包。异常间隔超过 `imu_max_gap` 时拒绝建边；
持续不能连接时重新初始化后端并增加 reset 计数，不构造无 IMU 连接的新状态。

这是参考数学模型的独立实现，不是将 VINS-Mono 源文件原样拷入仓库。

### 2.4 patch–LiDAR 平面几何约束

这是与光度残差分开的**patch 中心逆深度–局部平面残差**：

\[
P_L=T_{LI}T_{WI_\ell}^{-1}T_{WI_a}T_{IC}\frac{f_0}{\rho_a},\qquad
r_X=\frac{n_L^\top(P_L-c_L)}{\sigma_X}.
\]

局部平面从相邻 LiDAR 帧的去畸变 LiDAR 点拟合，不把优化后的视觉点当作平面量测。
先用前端地图点查找支持邻域，再用 PCA 拟合，检查至少 6 点、两个切向特征值、
平面性、拟合 RMS 和关联距离；不满足条件就不创建该 patch。该几何约束每个 patch
建立一次，后续图像增加光度因子，不反复重复同一个深度量测。

相同时间节点 \(a=\ell\) 时，导航位姿在变换链中抵消，只保留外参和深度参数块，
避免向 Ceres 重复传同一个参数块。其位姿约束通过后续跨帧光度因子产生。
本版本不优化平面法向，也没有实现完整 patch 表面法向对齐因子。

### 2.5 FAST-LIVO2 位姿正则与接受检查

记前端快照为 \((\bar p_i,\bar q_i)\)，则

\[
r_{F,i} =
\begin{bmatrix}
(p_i-\bar p_i)/\sigma_{p,F}\\
\operatorname{Log}(\bar q_i^{-1}q_i)/\sigma_{R,F}
\end{bmatrix},\qquad
\lambda_F=\texttt{frontend_pose_prior_weight}.
\]

这里的 Log 返回旋转向量，\(\sigma_{R,F}\) 在残差中使用弧度。默认
\(\sigma_{p,F}=0.05\) m、\(\sigma_{R,F}=1^\circ\)、\(\lambda_F=1\)。
每个已接纳状态一个 6 维正则项，不使用 Huber；权重独立于 LiDAR/视觉的自适应权重，
不参与信息量评价，避免把该人为正则当成新增传感器信息。有效尺度为
\(\sigma_F/\sqrt{\lambda_F}\)，减小尺度或增加权重会更贴近滤波结果。
这些默认值是可调起点，尚未通过数据集标定。

新状态由 FAST-LIVO2 初始化；同一时刻的视觉更新替换参考并更新该状态的位姿初值；
其他保留状态延续上次优化结果作为热启动。启用正则时，首次状态先验的前 6 维置零，
仅保留速度和偏置约束，避免额外重复锚定首次位姿。被移除帧的正则随关联因子一起
吸收到 Schur 先验中，存活帧的正则继续保持非线性。

软正则本身不构成硬界限，因此在 Ceres 求解后、写回状态前，检查整个窗口相对前端
快照的修正幅度：默认平移不超过 0.20 m、转角不超过 5°。超限则恢复本次求解前的
参数，拒绝结果；上层随后按已有恢复流程清空后端图，保留 IMU 缓冲，等待下一次输入。
前端照常运行。该策略是求解结果的接受检查，不是带硬边界的约束优化或自动缩放修正。

设置 `frontend_pose_prior_enabled: false` 同时关闭逐帧正则和这两个修正幅度门限，
恢复首次状态位姿锚定；新状态仍使用滤波初值。消融时应固定数据与其他配置，同时报告
该开关、正则尺度、拒绝次数和修正幅度。强正则会限制对前端漂移的修正能力；
本实现不能据此宣称统计一致性或必然提高 ATE/RPE。

## 3. 时间关联、自适应和边缘化

后端使用图像时刻与**保留的 LiDAR 帧**做最近邻匹配，满足
\(|t_a-t_\ell|\leq\texttt{association_max_dt}\) 后，以
\(c_t=\exp[-(t_a-t_\ell)^2/(2\sigma_t^2)]\) 降低时间间隔较大的关联权重。
两个节点由各自时间上的 IMU 约束连接，不把 LiDAR 帧直接改写成图像时间。
严格相同或数值容差内的时间戳可以共享节点。

这不是无限制的“无同步”：仍要求同一时基、有限时间差、静态局部表面和正确关联。
没有估计时间偏移、连续时间轨迹、rolling shutter 或重新去畸变。
此外，**原 FAST-LIVO2 前端的图像驱动点云切分/同步流程仍然存在**；
此分支的非硬同步能力是后端接口和因子模型的能力，不表示已经重写整个前端采样调度。

信息评价、模态权重、关键帧效用、窗口目标和候选评分沿用原分支的启发式设计；
新后端没有证明这些分数就是真实条件信息增益。过时模态统计不继续参与质量决策。
求解耗时包含求解和边缘化，反馈到窗口策略；Ceres 时间限制是软预算，密集分解、
构图和边缘化可能超时，不能直接据此宣称硬实时。

每次成功求解后，如超出窗口目标：

1. 选择非最新节点；同时淘汰以该帧为参考的 patch 逆深度。
2. 只收集接触待消元变量的量测因子，以及已有边缘化先验。
3. 在当前状态计算真正的切空间 Jacobian，冻结此次 Huber 权重，形成 \(H,b\)。
4. 使用对角均衡后的特征分解伪逆构造 Schur 补；
   舍弃数值零空间，不用 LM 阻尼“制造”边缘化信息。
5. 将先验写为 \(r_p=A(x\boxminus x_0)+b_p\)，保留 \(x_0\)；
   删除被吸收的因子，其他原始因子继续非线性重线性化。

因此可边缘化中间节点，而不必强行把其两段 IMU 简单拼接；
连接信息已通过 Schur 先验保留。参考帧被边缘化时，其 patch 的所有光度信息一起消元，
以后再次看到同一地图点可以建立新的参考 patch。
求解失败会恢复变量并重置后端图；边缘化失败也重置，不继续积累无界窗口。

## 4. 构建与运行

最低新增要求：C++17、CMake 3.16、Eigen 3.3、**Ceres 1.14**。
Ceres 1.x/2.0 使用 LocalParameterization，2.1+ 使用 Manifold；本工作空间已验证 1.14。
本分支即使不启用后端，构建时也需要 Ceres。选择版本时需兼容你的 ROS 发行版和工具链。

### 先运行不依赖 ROS 的核心测试

在仓库根目录执行：

~~~bash
cmake -S . -B build-aw -DADAPTIVE_WINDOW_STANDALONE=ON \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-aw -j2
(cd build-aw && ctest --output-on-failure)
~~~

非系统安装 Ceres 时，可在第一条命令追加 `-DCeres_DIR=/实际路径/lib/cmake/Ceres`。
测试不使用会被 Release/NDEBUG 关闭的 assert；任意检查失败返回非零退出码。
测试覆盖：流形正负运算和 Jacobian、静止预积分与重力符号、偏置重积分、
缺失端点/重复时间/IMU 间断拒绝、四类残差切空间数值差分、
中间帧/最旧帧边缘化、参考帧与深度共同淘汰、同步与异步合成轨迹、
重复图像不重复建因子以及窗口上界；另覆盖位姿正则 Jacobian、四元数符号不变性、
同一时刻参考替换、正则强度的一维解析解、开关、平移/旋转超限拒绝和状态回滚。
这些检查已在本工作空间的 Ceres 1.14 构建中通过。

### 再编译 ROS 集成

在本工作空间根目录构建随附的 Sophus，再编译三个 catkin 包：

~~~bash
cd /home/zyf/catkin_fast_adaptive
cmake -S third_party/Sophus -B build-sophus -DCMAKE_BUILD_TYPE=Release
cmake --build build-sophus -j2
catkin_make -j2 -DSophus_DIR="$PWD/build-sophus"
source devel/setup.bash
~~~

在允许本地套接字的终端回放 Livox PointCloud2 与 IMU bag，分别执行：

~~~bash
source /home/zyf/catkin_fast_adaptive/devel/setup.bash
roslaunch fast_livo mapping_livox_pointcloud2_adaptive_lio.launch rviz:=true use_sim_time:=true
~~~

~~~bash
rosbag play /home/zyf/2026-07-30-20-43-38.bag --clock --duration=30
~~~

该 bag 没有图像，因此此配置只验证 LiDAR–IMU 与后端的 LiDAR 路径；
精度评估前需填入真实 LiDAR–IMU 外参。Avia 自定义消息与图像数据可使用原启动文件：

~~~bash
source /home/zyf/catkin_fast_adaptive/devel/setup.bash
roslaunch fast_livo mapping_avia_adaptive.launch rviz:=true feedback:=false
~~~

NTU VIRAL `eee_01` 可用独立的 Ouster 启动文件；它将前端轨迹保存为
`Log/result/eee_01_adaptive_frontend.txt`，将成功求解的滑窗状态保存为
`Log/result/eee_01_adaptive_backend.txt`。该启动文件在通用后端参数之后加载
`config/adaptive_window_ntu.yaml`，使用与 NTU 前端方差尺度一致的 IMU 标准差，
限制重复地图点面观测的总权重。终端一：

~~~bash
source /home/zyf/catkin_fast_adaptive/devel/setup.bash
roslaunch fast_livo mapping_ouster_ntu_adaptive.launch rviz:=false
~~~

终端二：

~~~bash
rosbag play /home/zyf/catkin_fast_adaptive/eee_01/eee_01.bag --clock \
  --topics /left/image_raw /imu/imu /os1_cloud_node1/points
~~~

用 NTU 的 Leica 棱镜 GT 比较之前，先按 `eee_01/leica_prism.yaml` 中的
`T_Body_Prism` 将两条 IMU 轨迹转换到棱镜位置；脚本
`scripts/transform_ntu_tum_to_prism.py` 执行该转换。GT 四元数全是单位占位值，
因此只比较平移误差。

请按你实际相机/激光型号替换 launch 中的传感器 YAML 和话题；
当前示例继承 Avia 数据配置。可先减小 `max_lidar_factors`、
`max_cross_modal_factors` 和 `max_active_patches`，测量计算开销后再增加。
外参联合优化在配置文件中将 `optimize_extrinsics` 设为 true 后重启节点；
只在有足够旋转、平移激励与空间结构的数据上测试。

## 5. 输出与诊断

- `/adaptive_window/odometry`：最新保留状态的实际时间戳，不伪装成当前输入帧的时间。
  pose 在 `camera_init` 系，线速度按 ROS odometry 约定转到 `body` 系。
  本版不输出可信的后端协方差；不要将默认零 covariance 当成确定性。
- `/adaptive_window/optimized_odometry`：仅在滑窗完成一次被接受的求解后发布，
  位姿与 `/adaptive_window/odometry` 使用同一世界坐标系。`adaptive_window/pose_output_en`
  为 true 时，还按 TUM 格式写入 `adaptive_window/pose_output_path`；若路径为空，
  使用 `Log/result/<evo/seq_name>_adaptive.txt`。同一时间戳的后续求解会更新
  文件中待写入的位姿，因此文件保留该时间戳最后一次被接受的结果。
- `/adaptive_window/camera_imu_extrinsic`：parent=imu、child=camera_adaptive，
  消息内容为 \(T_{IC}\)，内部优化变量为其逆 \(T_{CI}\)。只是消息，不自动发布 TF。
- `/adaptive_window/diagnostics`：Float64MultiArray，前 17 项保持此前 Ceres 版本的字段顺序。

| 索引 | 含义 |
|---|---|
| 0–1 | 当前窗口节点数、目标节点数 |
| 2–4 | LiDAR、视觉、跨模态权重 |
| 5–6 | 最近成功的后端求解+边缘化耗时 ms、求解结束 cost（边缘化前） |
| 7–8 | 当前因子块总数、存活 patch–平面因子数 |
| 9–10 | 相机–IMU 外参相对初值的平移 m、转角 deg |
| 11–13 | 存活 LiDAR 因子块数、视觉 patch 因子块数、IMU 因子块数 |
| 14–16 | 最近一次求解成功标志、累计拒绝 IMU 区间数、累计后端 reset 数 |
| 17 | 存活前端位姿正则因子数，启用时等于窗口状态数 |
| 18–19 | 最近一次有限求解结果相对前端快照的窗口最大平移 m、最大转角 deg（边缘化前） |
| 20 | 累计因修正超限被拒绝的求解次数 |

因子块数不是残差标量数；已经吸收到先验中的历史因子不再计入各类存活计数。
首次 patch 只有几何约束，第二个可用关键帧观测才产生光度因子；检查 index 12 是否实际非零。
超限恢复后保留 index 18–20，index 14 为失败，窗口与存活因子数为零。
某次输入未选为关键帧时诊断保留上次求解状态，不代表本次输入触发了求解。

## 6. 数据集验证顺序及方法边界

建议先检查时间单位、加速度尺度、T_IL/T_CI 方向和 IMU 端点覆盖，再进行：

1. 原 FAST-LIVO2 对照；固定窗口/固定权重、固定外参的 Ceres 后端；
2. 同配置加入光度，再加入 patch–平面因子，确认计数及约束实际生效；
3. 分别启用权重、关键帧、窗口、边缘化自适应，并组合测试；
4. 在一致配置上开启/关闭前端位姿正则，比较修正幅度、拒绝率和轨迹精度；
5. 最后加入外参优化，报告不同初值、退化运动和充分激励下的收敛/失败情况。

`max_cross_modal_factors=0` 同时关闭 patch 生命周期及其两类因子，不是只关闭几何因子的
独立视觉消融开关。严格消融需保留一致的参考 patch/深度初始化规则并控制因子创建路径，
不能仅改变该上限就宣称完成“去除跨模态约束”消融。

至少报告 ATE/RPE、轨迹覆盖率、失败/reset、各类残差及因子数量、窗口与权重变化、
外参误差、CPU/内存和求解/构图耗时。不同配置必须使用相同数据、标定和评估对齐方式。

仍待研究/实现的内容：

- 去畸变、数据关联、地图和光度参考由前端提供，彼此及 LiDAR/跨模态复用点之间存在相关性。
  当前忽略联合相关协方差；逐帧位姿项明确作为相关正则使用，并未证明估计器一致性。
- 地图平面、patch 法向及关联在因子存活期间固定；没有后端地图更新或在线重关联。
- scan-local 平面只使用前端点面匹配的 LiDAR 内点，低密度/非平面环境可能无法建立 patch。
- 图像只有一级 4×4 稀疏采样；没有后端多尺度追踪、曝光优化或显式遮挡检测。
- 在线外参可观测性监测、协方差一致反馈、全局回环和完整时间偏移估计尚未实现。
- 自适应四项策略保留原启发式，尚未替换成严格的条件信息增益或最优实验设计。

## 7. 参考实现

- [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)：前端观测、地图及 ESIKF 接口。
- [VINS-Mono integration_base.h](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/vins_estimator/src/factor/integration_base.h)
  和 [imu_factor.h](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/vins_estimator/src/factor/imu_factor.h)：
  中值预积分、偏置修正和 IMU 残差模型参考。
- [VINS-Mono marginalization_factor.cpp](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/vins_estimator/src/factor/marginalization_factor.cpp)：
  边缘化思路参考；本实现采用自己的参数布局处理、流形和对角均衡数值流程。
- [Ceres 建模文档](https://ceres-solver.org/nnls_modeling.html)：
  Manifold、AutoDiff、CostFunctionToFunctor 和 robust loss。

未复制 VINS 源文件，也未修改仓库 LICENSE。首次对外发布或论文复现实验前，
请审查依赖和仓库现有许可证元数据，并补齐实测版本/提交号和验证记录。
