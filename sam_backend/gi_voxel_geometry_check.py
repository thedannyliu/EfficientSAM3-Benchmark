from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def voxel_means(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.int64), values.reshape(0, 3)
    cells = (values / voxel_size).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return unique, sums / np.bincount(inverse)[:, None]


def check_points(points: np.ndarray, voxel_size: float) -> dict[str, Any]:
    full_cells, full_means = voxel_means(points, voxel_size)
    compressed_cells, compressed_means = voxel_means(full_means, voxel_size)
    keys_identical = np.array_equal(full_cells, compressed_cells)
    max_mean_error = (
        float(np.max(np.abs(full_means - compressed_means)))
        if keys_identical and len(full_means)
        else float("inf") if not keys_identical else 0.0
    )
    return {
        "raw_points": len(points),
        "compressed_points": len(full_means),
        "cell_keys_identical": bool(keys_identical),
        "max_mean_error_m": max_mean_error,
    }


def run_check(input_path: Path, voxel_size: float) -> dict[str, Any]:
    messages = 0
    raw_points = 0
    compressed_points = 0
    all_keys_identical = True
    max_mean_error = 0.0
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            points = np.asarray(row.get("detection", {}).get("points", []), dtype=np.float64)
            check = check_points(points, voxel_size)
            messages += 1
            raw_points += check["raw_points"]
            compressed_points += check["compressed_points"]
            all_keys_identical &= check["cell_keys_identical"]
            max_mean_error = max(max_mean_error, check["max_mean_error_m"])
    return {
        "input": str(input_path),
        "voxel_size_m": voxel_size,
        "messages": messages,
        "raw_points": raw_points,
        "compressed_points": compressed_points,
        "point_reduction_fraction": 1.0 - compressed_points / max(raw_points, 1),
        "all_cell_keys_identical": all_keys_identical,
        "max_mean_error_m": max_mean_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check 1 cm Scene Graph voxel fidelity.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_check(args.input, args.voxel_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
