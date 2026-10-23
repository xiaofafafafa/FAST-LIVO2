/**
 * @file temporal_associator.cpp
 * @brief 维护有界时间索引，选择最近候选并给出时间置信度；几何匹配另由因子构建验证。
 * 中文框架说明：docs/ADAPTIVE_WINDOW_CODE_GUIDE_CN.md。
 */
#include "adaptive_window/temporal_associator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fast_livo
{
namespace adaptive_window
{

// 保存时间门限与是否允许候选复用的配置。
TemporalAssociator::TemporalAssociator(const AdaptiveWindowConfig &config)
    : config_(config)
{
}

// 追加保留帧索引，容量限制为 max(16,4*最大窗口)，旧条目超限丢弃。
void TemporalAssociator::Add(const Modality modality, const double timestamp,
                             const std::uint64_t node_id)
{
  entries_.push_back({modality, timestamp, node_id, false});
  const std::size_t capacity =
      static_cast<std::size_t>(std::max(16, 4 * config_.max_window_size));
  while (entries_.size() > capacity) entries_.pop_front();
}

// 先按模态/使用标记过滤，再在候选中找绝对时间差最小者。
TemporalMatch TemporalAssociator::FindNearest(const Modality modality,
                                              const double timestamp)
{
  auto best = entries_.end();
  double best_dt = std::numeric_limits<double>::infinity();
  for (auto it = entries_.begin(); it != entries_.end(); ++it)
  {
    if (it->modality != modality || (!config_.allow_pair_reuse && it->used))
      continue;
    const double dt = std::abs(it->timestamp - timestamp);
    if (dt < best_dt)
    {
      best_dt = dt;
      best = it;
    }
  }

  TemporalMatch match;
  // 找不到候选或超过允许时间差时返回默认无效结果，不改变状态时间。
  if (best == entries_.end() || best_dt > config_.association_max_dt)
    return match;

  match.valid = true;
  match.node_id = best->node_id;
  match.frame_timestamp = best->timestamp;
  // 返回有符号时间差，但置信权重只取绝对时间差的平方。
  match.delta_time = best->timestamp - timestamp;
  const double sigma = std::max(1.0e-6, config_.association_time_sigma);
  match.confidence = std::exp(-0.5 * best_dt * best_dt / (sigma * sigma));
  // 查询成功立即消耗该条目；是否跳过 used 由 allow_pair_reuse 控制。
  best->used = true;
  return match;
}

// 图中删除节点时同步删除所有对应索引，避免后续查到已释放状态。
void TemporalAssociator::RemoveNode(const std::uint64_t node_id)
{
  entries_.erase(
      std::remove_if(entries_.begin(), entries_.end(),
                     [node_id](const Entry &entry) {
                       return entry.node_id == node_id;
                     }),
      entries_.end());
}

// 后端重置时清空使用标记和候选列表。
void TemporalAssociator::Clear()
{
  entries_.clear();
}

}  // namespace adaptive_window
}  // namespace fast_livo
