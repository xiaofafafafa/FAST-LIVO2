/**
 * @file adaptive_policy.h
 * @brief 自适应决策接口：根据质量统计与窗口状态生成权重、关键帧、窗口和边缘化决策。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#ifndef FAST_LIVO_ADAPTIVE_WINDOW_ADAPTIVE_POLICY_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_ADAPTIVE_POLICY_H_

#include "adaptive_window/types.h"

#include <cstdint>
#include <vector>

namespace fast_livo
{
namespace adaptive_window
{

// 策略帧摘要，state 由 Estimator 读取当前优化值后填入。
struct PolicyFrame
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::uint64_t id = 0;
  NavState state;
  // 已接纳 normalized_information 的最大值，不是严格的独占/条件信息。
  double unique_information = 0.0;
  bool has_lidar = false;
  bool has_vision = false;
  bool has_cross_modal = false;
};

using PolicyFrameVector =
    std::vector<PolicyFrame, Eigen::aligned_allocator<PolicyFrame>>;

class AdaptivePolicy
{
public:
  explicit AdaptivePolicy(const AdaptiveWindowConfig &config);

  // 综合候选运动、近期质量、当前窗口和上次耗时生成三项接纳/权重/窗口决策。
  PolicyDecision Decide(const NavState &candidate,
                        const MeasurementSummary *lidar,
                        const MeasurementSummary *vision,
                        const PolicyFrameVector &window,
                        double last_solve_time_ms);

  // 窗口未超目标返回 0；否则在非最新帧中选择最高删除分数的稳定 ID。
  std::uint64_t SelectMarginalizationCandidate(
      const PolicyFrameVector &window, int target_window_size) const;

private:
  // 缺失/无效模态按零质量处理。
  double ModalityQuality(const MeasurementSummary *summary) const;
  // 相对质量归一化、限幅、指数平滑；跨模态目标取几何均值。
  ModalityWeights UpdateWeights(double lidar_quality,
                                double vision_quality);
  // 低质量倾向增加历史，超时倾向收缩；目标每次最多增减一帧。
  int UpdateWindowTarget(double combined_quality,
                         double last_solve_time_ms);

  AdaptiveWindowConfig config_;
  ModalityWeights weights_;
  int target_window_size_;
};

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_ADAPTIVE_POLICY_H_
