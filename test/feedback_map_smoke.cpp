#include "voxel_map.h"
#include "vio.h"

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
  VoxelMapConfig config{};
  config.max_voxel_size_ = 1.0;
  config.max_layer_ = 0;
  config.layer_init_num_ = {3};
  config.max_points_num_ = 50;
  config.planner_threshold_ = 0.1;
  config.beam_err_ = 0.01;
  config.dept_err_ = 0.01;
  config.sigma_num_ = 3.0;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> empty;
  VoxelMapManager map(config, empty);
  map.extR_ = M3D::Identity();
  map.extT_ = V3D::Zero();
  map.state_.rot_end = M3D::Identity();
  map.state_.pos_end = V3D::Zero();
  map.EnableFeedbackMapping(10);

  std::vector<pointWithVar> scan(4);
  for (int i = 0; i < 4; ++i)
  {
    scan[i].point_b = V3D(0.1 + 0.1 * (i % 2),
                          0.1 + 0.1 * (i / 2), 0.1);
    scan[i].point_w = scan[i].point_b;
    scan[i].var = M3D::Identity() * 0.001;
    scan[i].source_lidar_time = 1.0;
  }
  map.UpdateVoxelMap(scan, 1.0, 7);
  if (!map.voxel_map_.count(VOXEL_LOCATION(0, 0, 0))) return 1;
  map.UpdateVoxelMap(scan, 1.0, 7);
  if (map.active_scan_count() != 1) return 7;
  auto later_scan = scan;
  for (auto &point : later_scan)
  {
    point.point_b.x() += 4.0;
    point.point_w.x() += 4.0;
    point.source_lidar_time = 2.0;
  }
  map.UpdateVoxelMap(later_scan, 2.0, 8);
  if (map.active_scan_count() != 2) return 8;

  fast_livo::adaptive_window::NavState active;
  active.timestamp = 1.0;
  active.p_w_i = V3D(2.0, 0.0, 0.0);
  if (map.SyncKeyframeScans({{7, active}}, {}, 0.0, 0.0).corrected != 1)
    return 2;
  if (map.voxel_map_.count(VOXEL_LOCATION(0, 0, 0)) ||
      !map.voxel_map_.count(VOXEL_LOCATION(2, 0, 0))) return 3;

  fast_livo::adaptive_window::RetiredFrame later_retired;
  later_retired.id = 8;
  later_retired.optimized.timestamp = 2.0;
  later_retired.optimized.p_w_i = V3D(1.0, 0.0, 0.0);
  if (map.SyncKeyframeScans({}, {later_retired}, 0.0, 0.0).committed != 1 ||
      map.active_scan_count() != 1 || map.committed_scan_count() != 1 ||
      !map.voxel_map_.count(VOXEL_LOCATION(5, 0, 0))) return 9;

  fast_livo::adaptive_window::RetiredFrame retired;
  retired.id = 7;
  retired.optimized.timestamp = 1.0;
  retired.optimized.p_w_i = V3D(2.0, 0.0, 0.0);
  if (map.SyncKeyframeScans({}, {retired}, 0.0, 0.0).committed != 1 ||
      map.SyncKeyframeScans({}, {retired}, 0.0, 0.0).committed != 0 ||
      map.active_scan_count() != 0 || map.committed_scan_count() != 2 ||
      !map.voxel_map_.count(VOXEL_LOCATION(2, 0, 0))) return 4;

  VIOManager visual;
  visual.visual_submap = new SubSparseMap;
  visual.Rli = M3D::Identity();
  visual.Pli = V3D::Zero();
  visual.Rci = M3D::Identity();
  visual.Pci = V3D::Zero();
  VisualPoint *landmark = new VisualPoint(V3D(0.1, 0.1, 0.1));
  landmark->world_anchor_time_ = 1.0;
  landmark->normal_ = V3D::UnitZ();
  landmark->previous_normal_ = V3D::UnitZ();
  landmark->covariance_ = M3D::Identity();
  Feature *feature = new Feature(landmark, new float[1]{1.0f},
                                 V2D::Zero(), V3D::UnitZ(),
                                 SE3(M3D::Identity(), V3D::Zero()), 0);
  feature->source_image_time_ = 1.0;
  landmark->addFrameRef(feature);
  visual.insertPointIntoVoxelMap(landmark);
  fast_livo::adaptive_window::PoseCorrection correction;
  correction.timestamp = 1.0;
  correction.translation = V3D(2.0, 0.0, 0.0);
  if (visual.ApplyFeedbackCorrections({correction}, 0.5) != 1 ||
      !visual.feat_map.count(VOXEL_LOCATION(4, 0, 0)) ||
      std::abs(feature->T_f_w_.translation().x() + 2.0) > 1.0e-9)
    return 5;
  visual.FreezeRetiredAnchors({retired});
  if (visual.ApplyFeedbackCorrections({correction}, 0.5) != 0 ||
      !visual.feat_map.count(VOXEL_LOCATION(4, 0, 0))) return 10;
  visual.ApplyOptimizedCameraExtrinsic(M3D::Identity(), V3D(0.1, 0, 0));
  if (std::abs(visual.Pci.x() - 0.1) > 1.0e-9 ||
      std::abs(feature->T_f_w_.translation().x() + 1.9) > 1.0e-9)
    return 6;

  std::cout << "Feedback LiDAR/visual map and extrinsic checks passed.\n";
  return 0;
}
