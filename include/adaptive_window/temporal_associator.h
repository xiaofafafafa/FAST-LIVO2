/**
 * @file temporal_associator.h
 * @brief 保留帧时间最近邻关联；无需时间戳相等，但要求同一时基和有限时间差。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#ifndef FAST_LIVO_ADAPTIVE_WINDOW_TEMPORAL_ASSOCIATOR_H_
#define FAST_LIVO_ADAPTIVE_WINDOW_TEMPORAL_ASSOCIATOR_H_

#include "adaptive_window/types.h"

#include <cstdint>
#include <deque>

namespace fast_livo
{
namespace adaptive_window
{

// valid=false 时不可用 node_id 建因子；置信度只表示时间接近，不保证几何匹配。
struct TemporalMatch
{
  bool valid = false;
  std::uint64_t node_id = 0;
  double frame_timestamp = 0.0;
  // 有符号时间差=候选帧时间-查询时间，单位秒。
  double delta_time = 0.0;
  double confidence = 0.0;
};

class TemporalAssociator
{
public:
  explicit TemporalAssociator(const AdaptiveWindowConfig &config);

  // 注册保留帧的模态、时间和 ID，几何数据由 Estimator 另行持有。
  void Add(Modality modality, double timestamp, std::uint64_t node_id);
  // 查询成功即标记已使用，即使后续平面拟合失败也不会撤销这一标记。
  TemporalMatch FindNearest(Modality modality, double timestamp);
  // 与状态边缘化同时删除时间索引，避免引用已释放的参数。
  void RemoveNode(std::uint64_t node_id);
  // 重置时清空全部候选和使用标记。
  void Clear();

private:
  struct Entry
  {
    Modality modality = Modality::kPrior;
    double timestamp = 0.0;
    std::uint64_t node_id = 0;
    bool used = false;
  };

  AdaptiveWindowConfig config_;
  std::deque<Entry> entries_;
};

}  // namespace adaptive_window
}  // namespace fast_livo

#endif  // FAST_LIVO_ADAPTIVE_WINDOW_TEMPORAL_ASSOCIATOR_H_
