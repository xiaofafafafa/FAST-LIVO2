#!/usr/bin/env python3
"""Create time-matched position-only TUM files for a fair evo APE comparison."""

import argparse
from pathlib import Path

import numpy as np


def load_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, ndmin=2)
    if poses.shape[1] != 8 or not np.isfinite(poses).all():
        raise ValueError(f"Invalid TUM trajectory: {path}")
    if not np.all(np.diff(poses[:, 0]) > 0):
        raise ValueError(f"Timestamps must be strictly increasing: {path}")
    return poses


def bracket_gaps(source: np.ndarray, times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source[:, 0], times, side="right")
    gaps = np.full(times.shape, np.inf)
    valid = (right > 0) & (right < len(source))
    gaps[valid] = source[right[valid], 0] - source[right[valid] - 1, 0]
    return gaps


def position_tum(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1))
    return np.column_stack((times, positions, quaternions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_tum", type=Path)
    parser.add_argument("frontend_prism_tum", type=Path)
    parser.add_argument("backend_prism_tum", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    reference = load_tum(args.reference_tum)
    frontend = load_tum(args.frontend_prism_tum)
    backend = load_tum(args.backend_prism_tum)
    times = backend[:, 0]
    # The reference is about 20 Hz and the frontend about 10 Hz. Exclude gaps
    # larger than one normal sample interval instead of bridging missing data.
    keep = (bracket_gaps(reference, times) <= 0.10) & (
        bracket_gaps(frontend, times) <= 0.15
    )
    times = times[keep]
    backend_positions = backend[keep, 1:4]
    if len(times) == 0:
        raise ValueError("No backend timestamps overlap both trajectories")

    def interpolate_positions(source: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.interp(times, source[:, 0], source[:, column]) for column in (1, 2, 3)]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, positions in (
        ("common_gt.tum", interpolate_positions(reference)),
        ("common_frontend.tum", interpolate_positions(frontend)),
        ("common_backend.tum", backend_positions),
    ):
        np.savetxt(args.output_dir / name, position_tum(times, positions), fmt="%.9f")
    print(f"Wrote {len(times)} common position samples from {times[0]:.6f} to {times[-1]:.6f}")


if __name__ == "__main__":
    main()
