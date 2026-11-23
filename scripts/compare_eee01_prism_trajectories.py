#!/usr/bin/env python3
"""Put two prism-frame frontend/backend runs on one position-GT time grid."""

import argparse
from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, ndmin=2)
    if poses.shape[1] != 8 or not np.isfinite(poses).all():
        raise ValueError(f"Invalid TUM trajectory: {path}")
    if not np.all(np.diff(poses[:, 0]) > 0):
        raise ValueError(f"Timestamps are not increasing: {path}")
    return poses


def covered(poses: np.ndarray, times: np.ndarray, max_gap: float) -> np.ndarray:
    right = np.searchsorted(poses[:, 0], times, side="right")
    inside = (right > 0) & (right < len(poses))
    gap = np.full(len(times), np.inf)
    gap[inside] = poses[right[inside], 0] - poses[right[inside] - 1, 0]
    return inside & (gap <= max_gap)


def write_position_tum(path: Path, times: np.ndarray, xyz: np.ndarray) -> None:
    identity = np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1))
    np.savetxt(path, np.column_stack((times, xyz, identity)), fmt="%.9f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline-front", type=Path, required=True)
    parser.add_argument("--baseline-back", type=Path, required=True)
    parser.add_argument("--candidate-front", type=Path, required=True)
    parser.add_argument("--candidate-back", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--front-gap", type=float, default=0.15)
    parser.add_argument("--back-gap", type=float, default=0.80)
    args = parser.parse_args()

    gt = load(args.gt)
    trajectories = {
        "baseline_front": load(args.baseline_front),
        "baseline_back": load(args.baseline_back),
        "candidate_front": load(args.candidate_front),
        "candidate_back": load(args.candidate_back),
    }
    times = gt[:, 0]
    keep = np.ones(len(times), dtype=bool)
    for name, trajectory in trajectories.items():
        gap = args.front_gap if name.endswith("front") else args.back_gap
        keep &= covered(trajectory, times, gap)
    times = times[keep]
    if not len(times):
        raise ValueError("No common GT timestamps")
    args.output.mkdir(parents=True, exist_ok=True)
    write_position_tum(args.output / "gt.tum", times, gt[keep, 1:4])
    for name, trajectory in trajectories.items():
        xyz = np.column_stack([
            np.interp(times, trajectory[:, 0], trajectory[:, column])
            for column in (1, 2, 3)
        ])
        write_position_tum(args.output / f"{name}.tum", times, xyz)
    print(f"{len(times)} shared GT positions, {times[0]:.6f} to {times[-1]:.6f}")


if __name__ == "__main__":
    main()
