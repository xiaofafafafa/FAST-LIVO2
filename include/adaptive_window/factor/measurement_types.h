/**
 * @file measurement_types.h
 * @brief 与 ROS/OpenCV 解耦的相机投影、图像采样回调及 scan-local 平面数据。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#pragma once
#include "adaptive_window/types.h"
#include <functional>

namespace fast_livo { namespace adaptive_window {
// 输入相机系三维点，输出像素和 2×3 的 d(pixel)/d(point_C)；无效投影返回 false。
using ProjectionFunction = std::function<bool(const Eigen::Vector3d &, Eigen::Vector2d *,
                                             Eigen::Matrix<double, 2, 3> *)>;
// 输入亚像素坐标，输出灰度和二维图像梯度；越界/无效返回 false。
using ImageSampleFunction = std::function<bool(const Eigen::Vector2d &, double *, Eigen::Vector2d *)>;
// 单次 LiDAR 扫描坐标系下的局部平面，不随当前后端位姿在世界系重新拟合。
struct LocalPlane {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // 平面支持点质心，m。
  Eigen::Vector3d center_l = Eigen::Vector3d::Zero();
  // PCA 最小特征值对应单位法向；符号翻转只改变残差符号，不改变平方损失。
  Eigen::Vector3d normal_l = Eigen::Vector3d::UnitZ();
  // 距离白化尺度，m；由基础噪声和拟合厚度合成。
  double sigma = 0.05;
};

}}
