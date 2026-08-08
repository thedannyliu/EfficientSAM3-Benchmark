from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .scene_graph_ab_report import (
    LOG_RE,
    _edge_count,
    _read_json,
    _read_jsonl,
    _resource_summary,
    _stats,
    _utc_ns,
)


CONDITIONS = {
    "stateless": "sg5-stateless-pose-fixture",
    "stateful": "sg5-stateful-r30-pose-fixture",
}


def main() -> None:
    args = parse_args()
    report = build_report(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "report.md", report)
    _make_plots(args.root, args.output_dir, report)
    print(json.dumps(report["headline"], indent=2))


def build_report(root: Path) -> dict[str, Any]:
    conditions = {
        name: _condition_summary(root / directory, name)
        for name, directory in CONDITIONS.items()
    }
    control = conditions["stateless"]
    stateful = conditions["stateful"]
    completed_ratio = stateful["speed"]["completed_frames"] / max(
        control["speed"]["completed_frames"], 1
    )
    publication = stateful["detections"]["camera_to_last_detection_ms"]
    refresh_indices = stateful["speed"]["refresh_zero_based_indices"]
    expected_refresh = list(range(0, stateful["speed"]["detector_calls"], 30))
    gates = {
        "at_least_2x_completed_frames": completed_ratio >= 2.0,
        "publication_p50_below_500_ms": publication["p50"] is not None
        and publication["p50"] < 500.0,
        "publication_p95_below_1000_ms": publication["p95"] is not None
        and publication["p95"] < 1000.0,
        "latency_slope_at_most_10_ms_per_source_s": (
            stateful["detections"]["last_detection_latency_slope_ms_per_source_s"]
            <= 10.0
        ),
        "refresh_cadence_exact": refresh_indices == expected_refresh,
        "nonempty_3d_and_graph": stateful["detections"]["frames"] > 0
        and stateful["graph"]["nodes"] > 0,
        "no_pipeline_traceback": not stateful["pipeline_traceback"],
        "temperature_below_80_c": stateful["resources"]["gpu_temperature_c"]["max"]
        < 80.0,
        "at_least_32_gib_available": stateful["resources"]["mem_available_gib"]["min"]
        >= 32.0,
    }
    headline = {
        "stateless_completed_frames": control["speed"]["completed_frames"],
        "stateful_completed_frames": stateful["speed"]["completed_frames"],
        "completed_frame_ratio": completed_ratio,
        "stateless_completed_fps": control["speed"]["completed_frames_per_source_second"],
        "stateful_completed_fps": stateful["speed"]["completed_frames_per_source_second"],
        "stateless_http_p50_ms": control["speed"]["http_total_ms"]["p50"],
        "stateful_tracking_http_p50_ms": stateful["speed"]["tracking_http_total_ms"]["p50"],
        "stateful_publication_p50_ms": publication["p50"],
        "stateful_publication_p95_ms": publication["p95"],
        "stateful_latency_slope_ms_per_source_s": stateful["detections"]
        ["last_detection_latency_slope_ms_per_source_s"],
        "stateless_detection_frames": control["detections"]["frames"],
        "stateful_detection_frames": stateful["detections"]["frames"],
        "stateless_final_nodes": control["graph"]["nodes"],
        "stateful_final_nodes": stateful["graph"]["nodes"],
        "all_gates_pass": all(gates.values()),
    }
    return {"headline": headline, "gates": gates, "conditions": conditions}


def _condition_summary(path: Path, name: str) -> dict[str, Any]:
    recorder = _read_json(path / "recorder_summary.json")
    speed = _parse_detection_log(path / "detection.log")
    duration = float(recorder["source_duration_seconds"])
    speed["completed_frames_per_source_second"] = speed["completed_frames"] / duration
    speed["accepted_calls_per_source_second"] = speed["detector_calls"] / duration
    speed["busy_skip_ratio"] = speed["busy_skips"] / max(speed["sync_callbacks"], 1)
    resources = _resource_summary(path / "resources.jsonl", recorder)
    graph = _final_graph_summary(path / "final_graph.json")
    detections = _detection_summary(path / "detections.jsonl")
    return {
        "name": name,
        "recorder": recorder,
        "run_metadata": _read_json(path / "run_metadata.json"),
        "runtime_final_status": _read_json(path / "runtime_final_status.json"),
        "speed": speed,
        "detections": detections,
        "graph": graph,
        "resources": resources,
        "pipeline_traceback": "Traceback"
        in (path / "scene_graph.log").read_text(encoding="utf-8", errors="replace"),
    }


def _parse_detection_log(path: Path) -> dict[str, Any]:
    frame_started: dict[int, float] = {}
    frame_ms = []
    metrics = []
    busy_skips = 0
    sync_callbacks = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_RE.search(raw)
        if not match:
            continue
        timestamp = float(match.group(1))
        message = match.group(2)
        started = re.search(r"=== Processing frame (\d+) ===", message)
        if started:
            frame_started[int(started.group(1))] = timestamp
        completed = re.search(r"=== Completed frame (\d+) ===", message)
        if completed and int(completed.group(1)) in frame_started:
            frame_ms.append(
                (timestamp - frame_started[int(completed.group(1))]) * 1000.0
            )
        if message.startswith("Skipping frame"):
            busy_skips += 1
        if message.startswith("SYNC #"):
            sync_callbacks += 1
        if "DETECTOR_METRICS " in message:
            metrics.append(json.loads(message.split("DETECTOR_METRICS ", 1)[1]))

    refresh = [item for item in metrics if float(item.get("detect_ms") or 0.0) > 0.0]
    tracking = [item for item in metrics if float(item.get("detect_ms") or 0.0) == 0.0]
    return {
        "detector_calls": len(metrics),
        "completed_frames": len(frame_ms),
        "sync_callbacks": sync_callbacks,
        "busy_skips": busy_skips,
        "initialized_calls": sum(bool(item.get("initialized_this_frame")) for item in metrics),
        "refresh_zero_based_indices": [int(item["input_sequence"]) - 1 for item in refresh],
        "full_frame_ms": _stats(frame_ms),
        "http_total_ms": _metric_stats(metrics, "http_total_ms"),
        "runtime_process_ms": _metric_stats(metrics, "process_ms"),
        "refresh_http_total_ms": _metric_stats(refresh, "http_total_ms"),
        "tracking_http_total_ms": _metric_stats(tracking, "http_total_ms"),
        "tracking_runtime_process_ms": _metric_stats(tracking, "process_ms"),
        "runtime_mask_observations": sum(int(item.get("mask_count") or 0) for item in metrics),
    }


def _metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return _stats([float(row[key]) for row in rows])


def _detection_summary(path: Path) -> dict[str, Any]:
    by_stamp: dict[int, list[float]] = defaultdict(list)
    categories: Counter[str] = Counter()
    messages = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages += 1
            stamp = int(row["source_stamp_ns"])
            latency = row.get("camera_to_detection_ms")
            if latency is not None:
                by_stamp[stamp].append(float(latency))
            category = row.get("detection", {}).get("category")
            if category:
                categories[str(category)] += 1

    stamps = sorted(by_stamp)
    first = [min(by_stamp[stamp]) for stamp in stamps]
    last = [max(by_stamp[stamp]) for stamp in stamps]
    source_seconds = [(stamp - stamps[0]) / 1e9 for stamp in stamps] if stamps else []
    return {
        "messages": messages,
        "frames": len(stamps),
        "category_counts": dict(categories.most_common()),
        "source_stamps_ns": stamps,
        "source_seconds": source_seconds,
        "source_gap_seconds": _stats(
            [(right - left) / 1e9 for left, right in zip(stamps, stamps[1:])]
        ),
        "camera_to_first_detection_ms": _stats(first),
        "camera_to_last_detection_ms": _stats(last),
        "last_detection_latency_slope_ms_per_source_s": _linear_slope(
            source_seconds, last
        ),
    }


def _linear_slope(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    denominator = float(np.sum((x_array - x_array.mean()) ** 2))
    if denominator == 0.0:
        return 0.0
    return float(
        np.sum((x_array - x_array.mean()) * (y_array - y_array.mean()))
        / denominator
    )


def _final_graph_summary(path: Path) -> dict[str, Any]:
    graph = _read_json(path)
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    node_values = nodes.values() if isinstance(nodes, dict) else nodes
    categories = Counter(
        str(node.get("category") or node.get("object_type") or node.get("type"))
        for node in node_values
        if node.get("category") or node.get("object_type") or node.get("type")
    )
    return {
        "nodes": len(nodes),
        "edges": _edge_count(edges),
        "category_counts": dict(categories.most_common()),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    control = report["conditions"]["stateless"]
    stateful = report["conditions"]["stateful"]
    gate_rows = "\n".join(
        f"| `{name}` | {'Pass' if passed else 'Fail'} |"
        for name, passed in report["gates"].items()
    )
    text = f"""# T03b: Stateful GI Scheduling in Scene Graph

## Result

| Metric | Stateless control | Stateful R30 |
| --- | ---: | ---: |
| Completed full frames | {h['stateless_completed_frames']} | {h['stateful_completed_frames']} |
| Completed frames / source-second | {h['stateless_completed_fps']:.3f} | {h['stateful_completed_fps']:.3f} |
| Detector HTTP p50 | {h['stateless_http_p50_ms']:.1f} ms | {h['stateful_tracking_http_p50_ms']:.1f} ms tracking-only |
| Source frames with 3D detections | {h['stateless_detection_frames']} | {h['stateful_detection_frames']} |
| Final graph nodes | {h['stateless_final_nodes']} | {h['stateful_final_nodes']} |

Stateful scheduling completed **{h['completed_frame_ratio']:.2f}x** as many full
frames. Its complete source-to-3D-publication latency was p50
{h['stateful_publication_p50_ms']:.1f} ms and p95
{h['stateful_publication_p95_ms']:.1f} ms. The fitted latency slope was
{h['stateful_latency_slope_ms_per_source_s']:.2f} ms per source-second.

## Pre-registered gates

| Gate | Result |
| --- | --- |
{gate_rows}

Overall: **{'Pass' if h['all_gates_pass'] else 'Fail'}**.

## Hardware and capacity

| Metric | Stateless control | Stateful R30 |
| --- | ---: | ---: |
| Mean GPU utilization | {control['resources']['gpu_utilization_percent']['mean']:.1f}% | {stateful['resources']['gpu_utilization_percent']['mean']:.1f}% |
| GPU utilization p95 | {control['resources']['gpu_utilization_percent']['p95']:.1f}% | {stateful['resources']['gpu_utilization_percent']['p95']:.1f}% |
| Mean GPU power | {control['resources']['gpu_power_w']['mean']:.1f} W | {stateful['resources']['gpu_power_w']['mean']:.1f} W |
| Maximum GPU temperature | {control['resources']['gpu_temperature_c']['max']:.1f} C | {stateful['resources']['gpu_temperature_c']['max']:.1f} C |
| Minimum Linux `MemAvailable` | {control['resources']['mem_available_gib']['min']:.2f} GiB | {stateful['resources']['mem_available_gib']['min']:.2f} GiB |
| Mean Docker working set | {control['resources']['container_memory_gib']['mean']:.2f} GiB | {stateful['resources']['container_memory_gib']['mean']:.2f} GiB |
| Mean NVIDIA process memory | {control['resources']['nvidia_compute_memory_gib']['mean']:.2f} GiB | {stateful['resources']['nvidia_compute_memory_gib']['mean']:.2f} GiB |

## Interpretation limits

T03b isolates detector scheduling with recorded RGB/depth and a deterministic
camera-timestamp pose/TF fixture. It exercises mask inference, depth projection,
3D detection publication, and graph construction, but it is not a localization
benchmark. A final deployment run must pre-roll Cartographer from bag offset
zero. Post-playback external-frame idle timeout status is recorded separately
and is not treated as a playback crash when all measured calls completed.

## Figures

- `throughput_and_outputs.png`
- `latency_boxplot.png`
- `publication_latency_timeline.png`
- `hardware_timeseries.png`
"""
    path.write_text(text, encoding="utf-8")


def _make_plots(root: Path, output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"stateless": "#3366cc", "stateful": "#109618"}
    conditions = report["conditions"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = list(CONDITIONS)
    axes[0].bar(
        names,
        [conditions[name]["speed"]["completed_frames_per_source_second"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[0].set_ylabel("completed full frames / source-second")
    axes[0].set_title("Pipeline throughput")
    axes[1].bar(
        names,
        [conditions[name]["detections"]["frames"] for name in names],
        label="3D detection frames",
        color=[colors[name] for name in names],
    )
    axes[1].set_ylabel("source timestamps")
    axes[1].set_title("Non-empty 3D outputs")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "throughput_and_outputs.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    data = [
        conditions["stateless"]["speed"]["http_total_ms"]["values"],
        conditions["stateful"]["speed"]["tracking_http_total_ms"]["values"],
        conditions["stateless"]["speed"]["full_frame_ms"]["values"],
        conditions["stateful"]["speed"]["full_frame_ms"]["values"],
        conditions["stateful"]["detections"]["camera_to_last_detection_ms"]["values"],
    ]
    axis.boxplot(
        data,
        labels=[
            "Stateless HTTP",
            "Stateful tracking HTTP",
            "Stateless full frame",
            "Stateful full frame",
            "Stateful source→last 3D",
        ],
        showfliers=False,
    )
    axis.set_ylabel("milliseconds")
    axis.set_title("Detector, pipeline, and publication latency")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_boxplot.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    for name in names:
        detections = conditions[name]["detections"]
        axis.plot(
            detections["source_seconds"],
            detections["camera_to_last_detection_ms"]["values"],
            marker="o",
            markersize=3,
            label=name,
            color=colors[name],
        )
    axis.axhline(500, color="#ff9900", linestyle="--", label="p50 gate")
    axis.axhline(1000, color="#dc3912", linestyle="--", label="p95 gate")
    axis.set_xlabel("camera source time from first non-empty detection (s)")
    axis.set_ylabel("camera to last 3D message (ms)")
    axis.set_title("Source-aligned publication latency")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "publication_latency_timeline.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for name, directory in CONDITIONS.items():
        recorder = conditions[name]["recorder"]
        rows = _read_jsonl(root / directory / "resources.jsonl")
        rows = [
            row
            for row in rows
            if int(recorder["started_wall_ns"])
            <= _utc_ns(row["timestamp_utc"])
            <= int(recorder["ended_wall_ns"])
        ]
        x = [
            (_utc_ns(row["timestamp_utc"]) - int(recorder["started_wall_ns"])) / 1e9
            for row in rows
        ]
        axes[0, 0].plot(
            x,
            [row["nvidia_smi"]["gpu"]["utilization.gpu"] for row in rows],
            label=name,
            color=colors[name],
        )
        axes[0, 1].plot(
            x,
            [row["nvidia_smi"]["gpu"]["power.draw"] for row in rows],
            label=name,
            color=colors[name],
        )
        axes[1, 0].plot(
            x,
            [row["host_memory"]["memavailable_bytes"] / 2**30 for row in rows],
            label=name,
            color=colors[name],
        )
        axes[1, 1].plot(
            x,
            [row["nvidia_smi"]["gpu"]["temperature.gpu"] for row in rows],
            label=name,
            color=colors[name],
        )
    labels = [
        ("GPU utilization", "%"),
        ("GPU power", "W"),
        ("Unified memory available", "GiB"),
        ("GPU temperature", "C"),
    ]
    for axis, (title, unit) in zip(axes.flat, labels):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 0].set_xlabel("seconds since recorder start")
    axes[1, 1].set_xlabel("seconds since recorder start")
    fig.tight_layout()
    fig.savefig(output / "hardware_timeseries.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the controlled T03 report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
