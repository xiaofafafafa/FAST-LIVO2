#pragma once
#include "adaptive_window/estimator.h"
#include "vio.h"
#include "voxel_map.h"
#include <fstream>

namespace fast_livo { namespace adaptive_window {
// Single producer, one new directory per run. Consumers need neither ROS nor
// the estimator's address space. Publish completed payloads before their line.
class GaussianJournal {
 public:
  explicit GaussianJournal(const std::string& directory);
  ~GaussianJournal();
  bool good() const { return out_.is_open() && out_.good(); }
  void scan(double t, const StatesGroup& proposal, const LidarExtrinsic&,
            const std::vector<PointToPlane>&);
  void image(double t, const VIOManagerPtr&);
  void solution(const Estimator&, bool accepted);
 private:
  void finishLine();
  std::string dir_;
  std::ofstream out_;
  size_t serial_ = 0, epoch_ = 0;
  cv::Mat map_x_, map_y_;
  cv::Size last_size_;
  std::string mask_id_;
};
}}
