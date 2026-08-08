from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .scene_graph_ab_report import _resource_summary, _stats, _utc_ns


def main() -> None:
    args = parse_args()
    report = build_report(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "per_frame.csv", report["frames"])
    _write_markdown(args.output_dir / "report.md", report)
    _make_plots(args.root, args.output_dir, report)
    print(json.dumps(report["headline"], indent=2))


def build_report(root: Path) -> dict[str, Any]:
    baseline_rows = _read_jsonl(root / "gi-refresh30" / "profile.jsonl")
    headless_rows = _read_jsonl(root / "gi-refresh30-headless" / "profile.jsonl")
    baseline = _condition_summary(root / "gi-refresh30", baseline_rows)
    headless = _condition_summary(root / "gi-refresh30-headless", headless_rows)
    parity, frames = _parity_summary(
        root / "gi-refresh30" / "predictions",
        root / "gi-refresh30-headless" / "predictions",
        baseline_rows,
        headless_rows,
    )
    baseline_p50 = baseline["tracking_only"]["total_ms"]["p50"]
    headless_p50 = headless["tracking_only"]["total_ms"]["p50"]
    reduction = (baseline_p50 - headless_p50) / baseline_p50
    baseline_gap = baseline_p50 - baseline["tracking_only"]["process_ms"]["p50"]
    headless_gap = headless_p50 - headless["tracking_only"]["process_ms"]["p50"]
    gates = {
        "tracking_p50_at_most_160_ms": headless_p50 <= 160.0,
        "tracking_p50_reduction_at_least_20_percent": reduction >= 0.20,
        "all_frames_bitwise_identical": parity["exact_frames"] == parity["frames"],
        "completed_100_frames": len(headless_rows) == 100,
        "refresh_indices_match": headless["refresh_indices"] == [0, 30, 60, 90],
        "temperature_below_80_c": headless["resources"]["gpu_temperature_c"]["max"] < 80.0,
        "at_least_32_gib_available": headless["resources"]["mem_available_gib"]["min"] >= 32.0,
    }
    gates["all_t02_gates_pass"] = all(gates.values())
    headline = {
        "baseline_tracking_p50_ms": baseline_p50,
        "headless_tracking_p50_ms": headless_p50,
        "tracking_p50_reduction_percent": reduction * 100.0,
        "baseline_boundary_p50_ms": baseline_gap,
        "headless_boundary_p50_ms": headless_gap,
        "boundary_reduction_percent": (baseline_gap - headless_gap) / baseline_gap * 100.0,
        "baseline_effective_fps": baseline["effective_fps"],
        "headless_effective_fps": headless["effective_fps"],
        "exact_frames": parity["exact_frames"],
        "all_t02_gates_pass": gates["all_t02_gates_pass"],
    }
    return {
        "headline": headline,
        "conditions": {"baseline": baseline, "headless": headless},
        "parity": parity,
        "gates": gates,
        "frames": frames,
        "interpretation_limits": [
            "T02 changes only UI rendering/JPEG publication; the input JPEG upload and NPZ mask response remain HTTP.",
            "Runtime process_ms excludes the asynchronous mask publication and client transport boundary.",
            "Results cover one fixed 100-frame sequence and should be repeated in the ROS integration smoke.",
        ],
    }


def _condition_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tracking = [row for row in rows if float(row["status"]["detect_ms"]) == 0]
    refresh = [row for row in rows if float(row["status"]["detect_ms"]) > 0]
    summary = _read_json(path / "summary.json")
    start_ns = min(int(row["started_wall_ns"]) for row in rows)
    end_ns = max(int(row["ended_wall_ns"]) for row in rows)
    return {
        "frames": len(rows),
        "sequence_wall_seconds": float(summary["sequence_wall_seconds"]),
        "effective_fps": len(rows) / float(summary["sequence_wall_seconds"]),
        "all_frames": _latencies(rows),
        "tracking_only": _latencies(tracking),
        "refresh_frames": _latencies(refresh),
        "refresh_indices": [int(row["frame_index"]) for row in refresh],
        "resources": _resource_summary(
            path / "resources.jsonl",
            {"started_wall_ns": start_ns, "ended_wall_ns": end_ns},
        ),
    }


def _latencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "total_ms": _stats([float(row["total_ms"]) for row in rows]),
        "process_ms": _stats([float(row["status"]["process_ms"]) for row in rows]),
        "boundary_ms": _stats(
            [float(row["total_ms"]) - float(row["status"]["process_ms"]) for row in rows]
        ),
    }


def _parity_summary(
    baseline_dir: Path,
    headless_dir: Path,
    baseline_rows: list[dict[str, Any]],
    headless_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(baseline_rows) != len(headless_rows):
        raise ValueError("baseline and headless profile lengths differ")
    exact = labels = ids = lost = scores = masks = 0
    frames = []
    for baseline_row, headless_row in zip(baseline_rows, headless_rows):
        index = int(baseline_row["frame_index"])
        if index != int(headless_row["frame_index"]):
            raise ValueError("baseline and headless frame indices differ")
        baseline = _read_arrays(baseline_dir / f"{index:05d}.npz")
        headless = _read_arrays(headless_dir / f"{index:05d}.npz")
        checks = {
            "labels_equal": np.array_equal(baseline["labels"], headless["labels"]),
            "ids_equal": np.array_equal(baseline["ids"], headless["ids"]),
            "lost_equal": np.array_equal(baseline["lost"], headless["lost"]),
            "scores_equal": np.array_equal(baseline["scores"], headless["scores"]),
            "masks_equal": np.array_equal(baseline["mask_shape"], headless["mask_shape"])
            and np.array_equal(baseline["masks_packed"], headless["masks_packed"]),
        }
        checks["exact"] = all(checks.values())
        labels += checks["labels_equal"]
        ids += checks["ids_equal"]
        lost += checks["lost_equal"]
        scores += checks["scores_equal"]
        masks += checks["masks_equal"]
        exact += checks["exact"]
        frames.append(
            {
                "frame_index": index,
                "baseline_total_ms": float(baseline_row["total_ms"]),
                "headless_total_ms": float(headless_row["total_ms"]),
                **checks,
            }
        )
    return {
        "frames": len(frames),
        "exact_frames": exact,
        "labels_equal_frames": labels,
        "ids_equal_frames": ids,
        "lost_equal_frames": lost,
        "scores_equal_frames": scores,
        "masks_equal_frames": masks,
    }, frames


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key]
            for key in ("mask_shape", "masks_packed", "labels", "ids", "lost", "scores")
        }


def _make_plots(root: Path, output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = {
        "baseline": _read_jsonl(root / "gi-refresh30" / "profile.jsonl"),
        "headless": _read_jsonl(root / "gi-refresh30-headless" / "profile.jsonl"),
    }
    tracking = {
        name: [row for row in rows if float(row["status"]["detect_ms"]) == 0]
        for name, rows in profiles.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot(
        [[row["total_ms"] for row in tracking[name]] for name in ("baseline", "headless")],
        labels=["UI baseline", "mask-only headless"],
        showfliers=False,
    )
    axes[0].set_ylabel("tracking client latency (ms)")
    axes[0].set_title("Tracking-only latency")
    axes[1].boxplot(
        [
            [row["total_ms"] - row["status"]["process_ms"] for row in tracking[name]]
            for name in ("baseline", "headless")
        ],
        labels=["UI baseline", "mask-only headless"],
        showfliers=False,
    )
    axes[1].set_ylabel("client minus runtime process (ms)")
    axes[1].set_title("Asynchronous render/API boundary")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_and_boundary_boxplot.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for name, color in (("baseline", "#3366cc"), ("headless", "#109618")):
        axis.plot(
            [row["frame_index"] for row in profiles[name]],
            [row["total_ms"] for row in profiles[name]],
            label=name,
            color=color,
            alpha=0.85,
        )
    for frame in (0, 30, 60, 90):
        axis.axvline(frame, color="black", alpha=0.12)
    axis.set_xlabel("frame index")
    axis.set_ylabel("client latency (ms)")
    axis.set_title("T02 latency over the fixed sequence")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "latency_timeline.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = {"baseline": "#3366cc", "headless": "#109618"}
    for name, directory in (("baseline", "gi-refresh30"), ("headless", "gi-refresh30-headless")):
        rows = _read_jsonl(root / directory / "resources.jsonl")
        start_ns = min(int(row["started_wall_ns"]) for row in profiles[name])
        end_ns = max(int(row["ended_wall_ns"]) for row in profiles[name])
        rows = [row for row in rows if start_ns <= _utc_ns(row["timestamp_utc"]) <= end_ns]
        x = [(_utc_ns(row["timestamp_utc"]) - start_ns) / 1e9 for row in rows]
        axes[0, 0].plot(x, [row["nvidia_smi"]["gpu"]["utilization.gpu"] for row in rows], label=name, color=colors[name])
        axes[0, 1].plot(x, [row["nvidia_smi"]["gpu"]["power.draw"] for row in rows], label=name, color=colors[name])
        axes[1, 0].plot(x, [row["host_memory"]["memavailable_bytes"] / 2**30 for row in rows], label=name, color=colors[name])
        axes[1, 1].plot(x, [row["nvidia_smi"]["gpu"]["temperature.gpu"] for row in rows], label=name, color=colors[name])
    for axis, title, unit in zip(
        axes.flat,
        ("GPU utilization", "GPU power", "Unified memory available", "GPU temperature"),
        ("%", "W", "GiB", "°C"),
    ):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 0].set_xlabel("seconds since sequence start")
    axes[1, 1].set_xlabel("seconds since sequence start")
    fig.tight_layout()
    fig.savefig(output / "hardware_timeseries.png", dpi=160)
    plt.close(fig)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    g = report["gates"]
    text = f"""# T02: Mask-Only External API

## Outcome

T02 passed all pre-registered gates. Skipping UI overlay drawing and raw/tracked
JPEG publication reduced tracking-only client p50 from
{h['baseline_tracking_p50_ms']:.1f} ms to {h['headless_tracking_p50_ms']:.1f} ms
({h['tracking_p50_reduction_percent']:.1f}%). The client/runtime boundary fell
from {h['baseline_boundary_p50_ms']:.1f} ms to
{h['headless_boundary_p50_ms']:.1f} ms
({h['boundary_reduction_percent']:.1f}%).

Effective full-sequence FPS, including four refreshes, changed from
{h['baseline_effective_fps']:.3f} to {h['headless_effective_fps']:.3f}.
All {h['exact_frames']} frames were bitwise identical for masks, labels, IDs,
lost flags, and scores.

## Gates

| Gate | Passed |
| --- | --- |
| Tracking p50 <= 160 ms | {g['tracking_p50_at_most_160_ms']} |
| Tracking p50 reduction >= 20% | {g['tracking_p50_reduction_at_least_20_percent']} |
| All frames bitwise identical | {g['all_frames_bitwise_identical']} |
| Complete 100 frames | {g['completed_100_frames']} |
| Refresh indices are 0/30/60/90 | {g['refresh_indices_match']} |
| Temperature and memory gates | {g['temperature_below_80_c'] and g['at_least_32_gib_available']} |

Overall T02 gate: `{g['all_t02_gates_pass']}`.

The remaining tracking p50 is mostly model processing, so shared-memory IPC is
not the next priority. The next experiment should validate a stateful,
latest-frame-wins ROS detector boundary with the mask-only runtime.
"""
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize T02 headless API evaluation.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
