/**
 * @file information_evaluator.h
 * @brief 将前端位姿信息近似、残差与内点统计转换成各模态的平滑质量分数。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#ifndef FAST_LIVO_ADAPTIVE_WINDOW_INFORMATION_EVALUATOR_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_INFORMATION_EVALUATOR_H_

#include "adaptive_window/types.h"

#include <array>

namespace fast_livo
{
namespace adaptive_window
{

class InformationEvaluator
{
public:
  explicit InformationEvaluator(const AdaptiveWindowConfig &config);

  // 返回带统计和质量的新摘要，不修改传入对象，也不直接操作后端图。
  MeasurementSummary Evaluate(const MeasurementSummary &measurement);
  // 查询模态平滑历史，未初始化返回 0。
  double SmoothedInformation(Modality modality) const;

private:
  struct History
  {
    bool initialized = false;
    double smoothed_information = 0.0;
  };

  static std::size_t Index(Modality modality);

  AdaptiveWindowConfig config_;
  // 与 Modality 枚举对应；主流程通常只对 LiDAR/视觉调用评价。
  std::array<History, 6> history_;
};

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_INFORMATION_EVALUATOR_H_
