/**
 * @file math_utils.h
 * @brief SO(3)、限幅、角度与旧状态增量辅助函数；旧增量排列不能直接传给当前 Ceres 参数。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#ifndef FAST_LIVO_ADAPTIVE_WINDOW_MATH_UTILS_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_MATH_UTILS_H_

#include "adaptive_window/types.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace fast_livo
{
namespace adaptive_window
{

// 叉乘矩阵，满足 Skew(v)*u=v×u。
inline Eigen::Matrix3d Skew(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d result;
  result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return result;
}

// 弧度旋转向量的指数映射；近零使用一阶四元数再归一化。
inline Eigen::Quaterniond ExpQuaternion(const Eigen::Vector3d &theta)
{
  const double angle = theta.norm();
  if (angle < 1.0e-10)
  {
    Eigen::Quaterniond q(1.0, 0.5 * theta.x(), 0.5 * theta.y(),
                         0.5 * theta.z());
    return q.normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, theta / angle));
}

// 单位化并统一 q/-q 符号的最短支对数映射，输出弧度旋转向量。
inline Eigen::Vector3d LogQuaternion(const Eigen::Quaterniond &input)
{
  Eigen::Quaterniond q = input.normalized();
  if (q.w() < 0.0) q.coeffs() *= -1.0;
  const Eigen::Vector3d vector_part(q.x(), q.y(), q.z());
  const double vector_norm = vector_part.norm();
  if (vector_norm < 1.0e-10) return 2.0 * vector_part;
  const double angle = 2.0 * std::atan2(vector_norm, q.w());
  return angle * vector_part / vector_norm;
}

// 旧增量接口排列 [dtheta,dp,dv,dbg,dba]，旋转右乘。
// 当前 Ceres 不调用它更新状态；不能直接传入 [dp,dtheta,dv,dba,dbg]。
inline NavState ApplyIncrement(
    const NavState &state,
    const Eigen::Matrix<double, kNavStateDim, 1> &increment)
{
  NavState result = state;
  result.q_w_i = (state.q_w_i * ExpQuaternion(increment.segment<3>(0))).normalized();
  result.p_w_i += increment.segment<3>(3);
  result.v_w_i += increment.segment<3>(6);
  result.bias_g += increment.segment<3>(9);
  result.bias_a += increment.segment<3>(12);
  return result;
}

// 旧外参增量排列 [dtheta,dp]，与当前 Ceres 流形顺序相反。
inline ExtrinsicState ApplyIncrement(
    const ExtrinsicState &extrinsic,
    const Eigen::Matrix<double, kExtrinsicDim, 1> &increment)
{
  ExtrinsicState result = extrinsic;
  result.q_c_i =
      (extrinsic.q_c_i * ExpQuaternion(increment.segment<3>(0))).normalized();
  result.p_c_i += increment.segment<3>(3);
  return result;
}

// 与旧 NavState 增量成对，输出 [dtheta,dp,dv,dbg,dba]。
inline Eigen::Matrix<double, kNavStateDim, 1> StateDifference(
    const NavState &reference, const NavState &state)
{
  Eigen::Matrix<double, kNavStateDim, 1> difference;
  difference.segment<3>(0) =
      LogQuaternion(reference.q_w_i.conjugate() * state.q_w_i);
  difference.segment<3>(3) = state.p_w_i - reference.p_w_i;
  difference.segment<3>(6) = state.v_w_i - reference.v_w_i;
  difference.segment<3>(9) = state.bias_g - reference.bias_g;
  difference.segment<3>(12) = state.bias_a - reference.bias_a;
  return difference;
}

// 与旧外参增量成对，输出 [dtheta,dp]。
inline Eigen::Matrix<double, kExtrinsicDim, 1> ExtrinsicDifference(
    const ExtrinsicState &reference, const ExtrinsicState &state)
{
  Eigen::Matrix<double, kExtrinsicDim, 1> difference;
  difference.segment<3>(0) =
      LogQuaternion(reference.q_c_i.conjugate() * state.q_c_i);
  difference.segment<3>(3) = state.p_c_i - reference.p_c_i;
  return difference;
}

// 闭区间限幅，常用于质量分数和权重。
inline double Clamp(const double value, const double lower, const double upper)
{
  return std::max(lower, std::min(value, upper));
}

// 弧度转度，配置中的 *_deg 使用度。
inline double Degrees(const double radians)
{
  return radians * 180.0 / 3.14159265358979323846;
}

// 度转弧度，SO(3) 运算使用弧度。
inline double Radians(const double degrees)
{
  return degrees * 3.14159265358979323846 / 180.0;
}

// 旧通用平方根信息辅助：A=sqrt(lambda)*V^T，A^T A 近似输入信息。
// 该函数抬高小特征值且分解失败回退单位阵；当前 Schur 不调用它，
// 而是在 marginalization_factor.cpp 中独立舍弃数值零空间，不能直接替换。
inline Eigen::MatrixXd SymmetricSqrtInformation(
    const Eigen::MatrixXd &information, const double eigen_floor = 1.0e-12)
{
  if (information.rows() == 0 || information.rows() != information.cols())
    return Eigen::MatrixXd();

  const Eigen::MatrixXd symmetric = 0.5 * (information + information.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  if (solver.info() != Eigen::Success)
    return Eigen::MatrixXd::Identity(information.rows(), information.cols());

  Eigen::VectorXd eigenvalues = solver.eigenvalues();
  for (int i = 0; i < eigenvalues.size(); ++i)
    eigenvalues(i) = std::sqrt(std::max(eigen_floor, eigenvalues(i)));
  return eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
}

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_MATH_UTILS_H_
