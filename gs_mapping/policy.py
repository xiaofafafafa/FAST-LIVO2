"""以空间块为单位估计新增方向约束；这是调度代理量，并非整幅地图的后验熵。"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    max_gap: float = .65          # 两个优化状态间允许插值的最长时间，秒
    pair_dt: float = .09          # 图像—激光的最大时间差，秒
    history: float = 2.0          # 建图候选时域（与状态滑窗独立）
    max_frames: int = 4
    max_candidates: int = 12
    max_pairs: int = 256          # 单个候选最多的投影点数
    max_active: int = 128         # 单轮最多更新的高斯
    max_map: int = 12000          # 全图持久高斯数量上限；满后停止新增
    max_work: int = 600           # 每轮观测关联工作量代理（不是毫秒）
    budget_ms: float = 100.0      # 时间软预算；不会强行中断正在运行的光栅化
    iterations: int = 2
    gain_min: float = .12         # 按新增信息/可见块计的阈值
    splat_gain: float = .035       # 单高斯方向增量与局部颜色差的调度阈值
    learn_shape: bool = True      # 多扫描、非共线回波条件下更新空间椭球
    voxel: float = .16
    block: float = .8
    radius: float = .095
    thickness: float = .025
    pose_sigma_m: float = .12     # 后验不可用时的显式保守配置；并非测得值
    pose_sigma_deg: float = 4.
    pixel_sigma: float = 1.5
    plane_sigma: float = .025
    coverage: float = .08
    use_pose_uncertainty: bool = True
    fixed_frames: bool = False
    fixed_splats: bool = False
    fixed: bool = False           # 对照组：同一渲染器、相同硬预算、最近帧优先

    def check(self):
        for key, value in vars(self).items():
            if key in ("fixed","learn_shape","use_pose_uncertainty","fixed_frames","fixed_splats"):
                if type(value) is not bool:
                    raise ValueError(key)
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(key)
        for k in ("max_frames","max_candidates","max_pairs","max_active","max_map","max_work","iterations"):
            if type(getattr(self,k)) is not int:
                raise ValueError(k)
        if self.coverage >= 1 or self.thickness > self.radius:
            raise ValueError("invalid shape/coverage")
        return self


def select(candidates, cfg):
    """按增量 logdet / 关联工作量贪心；跨帧共享的同一扫描几何只计一次。"""
    chosen, used_scan, h = [], set(), {}
    work = 0
    rest = list(candidates)
    while rest and len(chosen) < cfg.max_frames:
        best, priority = None, -np.inf
        for c in rest:
            cost = len(c.hits)
            if not cost or work+cost > cfg.max_work:
                continue
            gain = 0.
            for cell, (photo, geom) in c.evidence.items():
                old = h.get(cell, np.eye(3))
                inc = photo if (cell,c.scan_id) in used_scan else photo+geom
                gain += np.linalg.slogdet(old+inc)[1]-np.linalg.slogdet(old)[1]
            if not (cfg.fixed or cfg.fixed_frames) and gain/max(1,len(c.evidence)) < cfg.gain_min:
                continue
            value = c.t if (cfg.fixed or cfg.fixed_frames) else gain/cost
            if value > priority or (value == priority and best is not None and c.t > best.t):
                best, priority = c, value
        if best is None:
            break
        rest = [c for c in rest if c is not best]
        chosen.append(best)
        work += len(best.hits)
        for cell,(photo,geom) in best.evidence.items():
            key = (cell,best.scan_id)
            h[cell] = h.get(cell,np.eye(3))+photo+(0 if key in used_scan else geom)
            used_scan.add(key)
    return chosen
