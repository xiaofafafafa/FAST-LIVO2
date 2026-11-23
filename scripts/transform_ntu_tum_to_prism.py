#!/usr/bin/env python3
"""Transform a TUM body/IMU trajectory to the NTU VIRAL Leica prism frame."""

import argparse
from pathlib import Path
import re
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def read_body_prism(calibration_path: Path) -> Tuple[np.ndarray, Rotation]:
    text = calibration_path.read_text()
    match = re.search(r"T_Body_Prism:.*?data:\s*\[([^]]+)\]", text, re.S)
    if match is None:
        raise ValueError(f"T_Body_Prism matrix not found in {calibration_path}")
    values = np.fromstring(match.group(1), sep=",")
    if values.size != 16:
        raise ValueError("T_Body_Prism must contain 16 matrix entries")
    transform = values.reshape(4, 4)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("T_Body_Prism has an invalid homogeneous bottom row")
    return transform[:3, 3], Rotation.from_matrix(transform[:3, :3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tum", type=Path)
    parser.add_argument("calibration_yaml", type=Path)
    parser.add_argument("output_tum", type=Path)
    args = parser.parse_args()

    trajectory = np.loadtxt(args.input_tum, ndmin=2)
    if trajectory.shape[1] != 8 or not np.isfinite(trajectory).all():
        raise ValueError("Input must contain finite TUM rows: t x y z qx qy qz qw")
    translation_body_prism, rotation_body_prism = read_body_prism(
        args.calibration_yaml
    )
    rotation_world_body = Rotation.from_quat(trajectory[:, 4:8])
    result = trajectory.copy()
    result[:, 1:4] += rotation_world_body.apply(translation_body_prism)
    result[:, 4:8] = (rotation_world_body * rotation_body_prism).as_quat()
    np.savetxt(args.output_tum, result, fmt="%.9f")
    print(f"Wrote {len(result)} prism poses to {args.output_tum}")


if __name__ == "__main__":
    main()
