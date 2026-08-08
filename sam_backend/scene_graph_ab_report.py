from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


LOG_RE = re.compile(r"\[(?:INFO|WARN|ERROR|FATAL)\] \[([0-9.]+)\].*?: (.*)$")


def main() -> None:
    args = parse_args()
    report = build_report(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_mask_csv(args.output_dir / "mask_agreement.csv", report["mask_agreement"]["frames"])
    _write_markdown(args.output_dir / "report.md", report)
    _make_plots(args.root, args.output_dir, report)
    print(json.dumps(report["headline"], indent=2))


def build_report(root: Path) -> dict[str, Any]:
    original = _condition_summary(root / "original", "original")
    gi = _condition_summary(root / "gi", "gi")
    masks = _mask_agreement(
        root / "quality" / "original-bag35",
        root / "quality" / "gi-bag35",
    )
    coco = {
        "original": _read_json(root / "quality" / "original-coco10" / "summary.json"),
        "gi": _read_json(root / "quality" / "gi-coco10" / "summary.json"),
    }
    graph = _graph_agreement(
        root / "original" / "final_graph.json", root / "gi" / "final_graph.json"
    )
    original_coco = coco["original"]["prompt_modes"]["text"]
    gi_coco = coco["gi"]["prompt_modes"]["text"]
    headline = {
        "source_duration_seconds": original["recorder"]["source_duration_seconds"],
        "original_processed_fps": original["speed"]["completed_frames_per_source_second"],
        "gi_processed_fps": gi["speed"]["completed_frames_per_source_second"],
        "original_grounding_p50_ms": original["speed"]["grounding_ms"]["p50"],
        "gi_http_p50_ms": gi["speed"]["http_total_ms"]["p50"],
        "original_detection_frames": original["recorder"]["detection_frames"],
        "gi_detection_frames": gi["recorder"]["detection_frames"],
        "original_final_nodes": original["graph"]["nodes"],
        "gi_final_nodes": gi["graph"]["nodes"],
        "bag_teacher_instance_miou": masks["teacher_instance_miou"],
        "bag_teacher_recall_at_50": masks["teacher_recall_at_50"],
        "original_coco_best_miou": original_coco["miou_best"],
        "gi_coco_best_miou": gi_coco["miou_best"],
        "original_coco_mean_ms": original_coco["mean_total_ms"],
        "gi_coco_mean_ms": gi_coco["mean_total_ms"],
        "original_gpu_util_mean": original["resources"]["gpu_utilization_percent"]["mean"],
        "gi_gpu_util_mean": gi["resources"]["gpu_utilization_percent"]["mean"],
        "original_mem_available_min_gib": original["resources"]["mem_available_gib"]["min"],
        "gi_mem_available_min_gib": gi["resources"]["mem_available_gib"]["min"],
    }
    return {
        "headline": headline,
        "conditions": {"original": original, "gi": gi},
        "mask_agreement": masks,
        "coco_gt": coco,
        "graph_agreement": graph,
        "interpretation_limits": [
            "Bag mask IoU treats Original SAM as a teacher; it is agreement, not ground truth.",
            "COCO-10 polygon annotations provide the true segmentation reference.",
            "GI runs in two containers and Original runs in one, matching the intended deployment architectures.",
            "Thor uses unified memory, so Linux MemAvailable and per-container/process metrics must be read together.",
            "GI threshold 0.8 produced zero masks in the preserved sensitivity run; the formal GI run uses 0.5.",
        ],
    }


def _condition_summary(path: Path, name: str) -> dict[str, Any]:
    recorder = _read_json(path / "recorder_summary.json")
    log = _parse_detection_log(path / "detection.log", name)
    duration = float(recorder["source_duration_seconds"])
    resources = _resource_summary(path / "resources.jsonl", recorder)
    graph = _final_graph_summary(path / "final_graph.json")
    detections = _detection_summary(path / "detections.jsonl")
    speed = {
        **log,
        "completed_frames_per_source_second": log["completed_frames"] / duration,
        "camera_messages_per_completed_frame": (
            recorder["camera_messages"] / log["completed_frames"]
            if log["completed_frames"]
            else math.inf
        ),
        "skip_ratio": log["skipped_messages"] / max(recorder["camera_messages"], 1),
    }
    return {
        "recorder": recorder,
        "speed": speed,
        "resources": resources,
        "detections": detections,
        "graph": graph,
    }


def _parse_detection_log(path: Path, condition: str) -> dict[str, Any]:
    grounding_started = None
    grounding_ms = []
    frame_started: dict[int, float] = {}
    frame_ms = []
    skipped = 0
    metrics = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_RE.search(raw)
        if not match:
            continue
        timestamp = float(match.group(1))
        message = match.group(2)
        if message.startswith("Starting ") and message.endswith(" grounding..."):
            grounding_started = timestamp
        elif message.startswith("Grounding completed") and grounding_started is not None:
            grounding_ms.append((timestamp - grounding_started) * 1000.0)
            grounding_started = None
        frame_match = re.search(r"=== Processing frame (\d+) ===", message)
        if frame_match:
            frame_started[int(frame_match.group(1))] = timestamp
        frame_match = re.search(r"=== Completed frame (\d+) ===", message)
        if frame_match:
            frame_number = int(frame_match.group(1))
            if frame_number in frame_started:
                frame_ms.append((timestamp - frame_started[frame_number]) * 1000.0)
        if message.startswith("Skipping frame"):
            skipped += 1
        if "DETECTOR_METRICS " in message:
            metrics.append(json.loads(message.split("DETECTOR_METRICS ", 1)[1]))
    result = {
        "completed_frames": len(frame_ms),
        "skipped_messages": skipped,
        "grounding_ms": _stats(grounding_ms),
        "full_frame_ms": _stats(frame_ms),
    }
    if condition == "gi":
        result.update(
            {
                "http_total_ms": _stats([float(item["http_total_ms"]) for item in metrics]),
                "runtime_detect_ms": _stats([float(item["detect_ms"]) for item in metrics]),
                "runtime_process_ms": _stats([float(item["process_ms"]) for item in metrics]),
                "runtime_nonzero_mask_frames": sum(int(item["mask_count"]) > 0 for item in metrics),
                "runtime_masks": sum(int(item["mask_count"]) for item in metrics),
            }
        )
    else:
        result["http_total_ms"] = _stats([])
    return result


def _resource_summary(path: Path, recorder: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(path)
    start_ns = int(recorder["started_wall_ns"])
    end_ns = int(recorder["ended_wall_ns"])
    steady = [row for row in rows if start_ns <= _utc_ns(row["timestamp_utc"]) <= end_ns]
    if not steady:
        steady = rows
    container_memory = []
    container_cpu = []
    compute_memory = []
    for row in steady:
        container_memory.append(sum(_memory_gib(item["MemUsage"].split(" / ")[0]) for item in row["containers"]))
        container_cpu.append(sum(float(item["CPUPerc"].rstrip("%")) for item in row["containers"]))
        compute_memory.append(sum(float(item["used_memory_mib"]) for item in row["nvidia_smi"]["compute_applications"]) / 1024.0)
    return {
        "samples": len(steady),
        "mem_available_gib": _stats([row["host_memory"]["memavailable_bytes"] / 2**30 for row in steady]),
        "tegrastats_ram_used_gib": _stats([row["tegrastats"]["ram_used_mb"] / 1024.0 for row in steady]),
        "container_memory_gib": _stats(container_memory),
        "container_cpu_percent": _stats(container_cpu),
        "nvidia_compute_memory_gib": _stats(compute_memory),
        "gpu_utilization_percent": _stats([row["nvidia_smi"]["gpu"]["utilization.gpu"] for row in steady]),
        "gpu_temperature_c": _stats([row["nvidia_smi"]["gpu"]["temperature.gpu"] for row in steady]),
        "gpu_power_w": _stats([row["nvidia_smi"]["gpu"]["power.draw"] for row in steady]),
        "system_power_w": _stats([row["tegrastats"]["power_mw"]["VIN"]["current"] / 1000.0 for row in steady]),
    }


def _detection_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    by_stamp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    categories = Counter()
    for row in rows:
        by_stamp[int(row["source_stamp_ns"])].append(row)
        category = row.get("detection", {}).get("category")
        if category:
            categories[category] += 1
    lags = []
    for values in by_stamp.values():
        numeric = [row["camera_to_detection_ms"] for row in values if row["camera_to_detection_ms"] is not None]
        if numeric:
            lags.append(min(numeric))
    return {
        "messages": len(rows),
        "frames": len(by_stamp),
        "camera_to_first_detection_ms": _stats(lags),
        "category_counts": dict(categories.most_common()),
    }


def _final_graph_summary(path: Path) -> dict[str, Any]:
    graph = _read_json(path)
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    categories = Counter()
    values = nodes.values() if isinstance(nodes, dict) else nodes
    for node in values:
        category = node.get("category") or node.get("object_type") or node.get("type")
        if category:
            categories[category] += 1
    return {
        "nodes": len(nodes),
        "edges": _edge_count(edges),
        "category_counts": dict(categories.most_common()),
    }


def _graph_agreement(original_path: Path, gi_path: Path) -> dict[str, Any]:
    original = _read_json(original_path)
    gi = _read_json(gi_path)
    original_counts = Counter(_node_categories(original))
    gi_counts = Counter(_node_categories(gi))
    overlap = sum((original_counts & gi_counts).values())
    precision = overlap / max(sum(gi_counts.values()), 1)
    recall = overlap / max(sum(original_counts.values()), 1)
    return {
        "category_multiset_overlap": overlap,
        "node_category_precision_vs_original": precision,
        "node_category_recall_vs_original": recall,
        "node_category_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "original_categories": dict(original_counts),
        "gi_categories": dict(gi_counts),
        "original_edges": _edge_count(original.get("edges", {})),
        "gi_edges": _edge_count(gi.get("edges", {})),
    }


def _mask_agreement(original_dir: Path, gi_dir: Path) -> dict[str, Any]:
    original_profile = _read_jsonl(original_dir / "profile.jsonl")
    gi_profile = {int(row["source_stamp_ns"]): row for row in _read_jsonl(gi_dir / "profile.jsonl")}
    teacher_ious = []
    reverse_ious = []
    union_ious = []
    frames = []
    for original_row in original_profile:
        stamp = int(original_row["source_stamp_ns"])
        gi_row = gi_profile[stamp]
        original_masks, original_labels, _ = _load_masks(original_dir / "predictions" / f"{stamp}.npz")
        gi_masks, gi_labels, _ = _load_masks(gi_dir / "predictions" / f"{stamp}.npz")
        frame_teacher = _directed_mask_ious(original_masks, original_labels, gi_masks, gi_labels)
        frame_reverse = _directed_mask_ious(gi_masks, gi_labels, original_masks, original_labels)
        frame_union = _label_union_ious(original_masks, original_labels, gi_masks, gi_labels)
        teacher_ious.extend(frame_teacher)
        reverse_ious.extend(frame_reverse)
        union_ious.extend(frame_union)
        frames.append(
            {
                "source_stamp_ns": stamp,
                "source_seconds": (stamp - int(original_profile[0]["source_stamp_ns"])) / 1e9,
                "original_masks": len(original_masks),
                "gi_masks": len(gi_masks),
                "teacher_instance_miou": mean(frame_teacher) if frame_teacher else None,
                "gi_to_teacher_instance_miou": mean(frame_reverse) if frame_reverse else None,
                "label_union_miou": mean(frame_union) if frame_union else None,
                "original_total_ms": original_row["total_ms"],
                "gi_total_ms": gi_row["total_ms"],
            }
        )
    return {
        "frames_total": len(frames),
        "frames_original_nonzero": sum(row["original_masks"] > 0 for row in frames),
        "frames_gi_nonzero": sum(row["gi_masks"] > 0 for row in frames),
        "frames_both_nonzero": sum(row["original_masks"] > 0 and row["gi_masks"] > 0 for row in frames),
        "original_masks": sum(row["original_masks"] for row in frames),
        "gi_masks": sum(row["gi_masks"] for row in frames),
        "teacher_instance_miou": mean(teacher_ious) if teacher_ious else 0.0,
        "teacher_recall_at_50": sum(value >= 0.5 for value in teacher_ious) / max(len(teacher_ious), 1),
        "gi_to_teacher_instance_miou": mean(reverse_ious) if reverse_ious else 0.0,
        "label_union_miou": mean(union_ious) if union_ious else 0.0,
        "original_latency_ms": _stats([float(row["total_ms"]) for row in original_profile]),
        "gi_latency_ms": _stats([float(row["total_ms"]) for row in gi_profile.values()]),
        "frames": frames,
    }


def _load_masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["mask_shape"])
        values = np.unpackbits(
            archive["masks_packed"], bitorder=str(archive["bitorder"].item())
        )[: int(np.prod(shape))]
        return (
            values.astype(bool).reshape(shape),
            archive["labels"].astype(str),
            archive["scores"].astype(float),
        )


def _directed_mask_ious(
    source_masks: np.ndarray,
    source_labels: np.ndarray,
    target_masks: np.ndarray,
    target_labels: np.ndarray,
) -> list[float]:
    values = []
    for mask, label in zip(source_masks, source_labels):
        candidates = target_masks[target_labels == label]
        values.append(max((_iou(mask, candidate) for candidate in candidates), default=0.0))
    return values


def _label_union_ious(
    original_masks: np.ndarray,
    original_labels: np.ndarray,
    gi_masks: np.ndarray,
    gi_labels: np.ndarray,
) -> list[float]:
    values = []
    for label in sorted(set(original_labels.tolist())):
        original_union = np.any(original_masks[original_labels == label], axis=0)
        candidates = gi_masks[gi_labels == label]
        gi_union = np.any(candidates, axis=0) if len(candidates) else np.zeros_like(original_union)
        values.append(_iou(original_union, gi_union))
    return values


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def _make_plots(root: Path, output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"original": "#3366cc", "gi": "#dc3912"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for name in ("original", "gi"):
        recorder = report["conditions"][name]["recorder"]
        rows = _read_jsonl(root / name / "resources.jsonl")
        rows = [row for row in rows if int(recorder["started_wall_ns"]) <= _utc_ns(row["timestamp_utc"]) <= int(recorder["ended_wall_ns"])]
        if not rows:
            continue
        x = [(_utc_ns(row["timestamp_utc"]) - int(recorder["started_wall_ns"])) / 1e9 for row in rows]
        axes[0, 0].plot(x, [row["nvidia_smi"]["gpu"]["utilization.gpu"] for row in rows], label=name, color=colors[name], alpha=0.85)
        axes[0, 1].plot(x, [row["nvidia_smi"]["gpu"]["power.draw"] for row in rows], label=name, color=colors[name], alpha=0.85)
        axes[1, 0].plot(x, [row["host_memory"]["memavailable_bytes"] / 2**30 for row in rows], label=name, color=colors[name], alpha=0.85)
        axes[1, 1].plot(x, [row["nvidia_smi"]["gpu"]["temperature.gpu"] for row in rows], label=name, color=colors[name], alpha=0.85)
    labels = [("GPU utilization", "%"), ("GPU power", "W"), ("Unified memory available", "GiB"), ("GPU temperature", "°C")]
    for axis, (title, unit) in zip(axes.flat, labels):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 0].set_xlabel("seconds since camera recording started")
    axes[1, 1].set_xlabel("seconds since camera recording started")
    fig.tight_layout()
    fig.savefig(output / "hardware_timeseries.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    data = [
        _series_from_log(root / "original" / "detection.log", "grounding"),
        _series_from_metrics(root / "gi" / "detection.log", "http_total_ms"),
        _series_from_log(root / "original" / "detection.log", "frame"),
        _series_from_log(root / "gi" / "detection.log", "frame"),
    ]
    axis.boxplot(data, labels=["Original grounding", "GI HTTP total", "Original full frame", "GI full frame"], showfliers=False)
    axis.set_ylabel("milliseconds")
    axis.set_title("End-to-end latency distribution")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_boxplot.png", dpi=160)
    plt.close(fig)

    frames = report["mask_agreement"]["frames"]
    fig, left = plt.subplots(figsize=(12, 5))
    x = [row["source_seconds"] for row in frames]
    y = [np.nan if row["teacher_instance_miou"] is None else row["teacher_instance_miou"] for row in frames]
    left.plot(x, y, color="#109618", marker="o", label="Original→GI instance mIoU")
    left.set_ylim(0, 1)
    left.set_xlabel("camera source timestamp offset (s)")
    left.set_ylabel("teacher agreement IoU")
    right = left.twinx()
    right.plot(x, [row["original_masks"] for row in frames], color=colors["original"], alpha=0.5, label="Original masks")
    right.plot(x, [row["gi_masks"] for row in frames], color=colors["gi"], alpha=0.5, label="GI masks")
    right.set_ylabel("mask count")
    left.grid(alpha=0.25)
    left.set_title("Source-time-aligned mask agreement")
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "mask_agreement_timeline.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for name in ("original", "gi"):
        values = report["coco_gt"][name]["prompt_modes"]["text"]
        axis.scatter(values["mean_total_ms"], values["miou_best"], s=120, color=colors[name], label=name)
        axis.annotate(name, (values["mean_total_ms"], values["miou_best"]), xytext=(7, 5), textcoords="offset points")
    axis.set_xlabel("COCO-10 mean latency (ms)")
    axis.set_ylabel("COCO-10 best-mask mIoU")
    axis.set_ylim(0, 1)
    axis.set_title("Quality versus latency, single text prompt")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_latency.png", dpi=160)
    plt.close(fig)


def _series_from_log(path: Path, kind: str) -> list[float]:
    parsed = _parse_detection_log(path, "original")
    key = "grounding_ms" if kind == "grounding" else "full_frame_ms"
    return parsed[key]["values"]


def _series_from_metrics(path: Path, key: str) -> list[float]:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "DETECTOR_METRICS " in line:
            values.append(float(json.loads(line.split("DETECTOR_METRICS ", 1)[1])[key]))
    return values


def _write_mask_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    text = f"""# Original SAM + Scene Graph vs GI SAM + Scene Graph

## Headline results

| Metric | Original | GI |
| --- | ---: | ---: |
| Processed frames/s of bag source time | {h['original_processed_fps']:.3f} | {h['gi_processed_fps']:.3f} |
| Grounding / HTTP p50 latency | {h['original_grounding_p50_ms']:.1f} ms | {h['gi_http_p50_ms']:.1f} ms |
| Frames with 3D detections | {h['original_detection_frames']} | {h['gi_detection_frames']} |
| Final graph nodes | {h['original_final_nodes']} | {h['gi_final_nodes']} |
| Mean GPU utilization | {h['original_gpu_util_mean']:.1f}% | {h['gi_gpu_util_mean']:.1f}% |
| Minimum unified memory available | {h['original_mem_available_min_gib']:.1f} GiB | {h['gi_mem_available_min_gib']:.1f} GiB |
| COCO-10 single-prompt mean latency | {h['original_coco_mean_ms']:.1f} ms | {h['gi_coco_mean_ms']:.1f} ms |
| COCO-10 best-mask mIoU | {h['original_coco_best_miou']:.4f} | {h['gi_coco_best_miou']:.4f} |

Bag teacher agreement (Original as reference): instance mIoU
`{h['bag_teacher_instance_miou']:.4f}`, recall@IoU 0.5
`{h['bag_teacher_recall_at_50']:.4f}`.

## Interpretation

The bag run uses 39 prompts and fixed camera source timestamps. GI's delivered
runtime is optimized for small prompt sets plus tracking; its 39-prompt path did
not outperform the original batched grounding path and produced fewer scene
detections. In contrast, the fixed COCO-10 single-prompt test shows that GI has
strong segmentation quality, so the primary integration bottleneck is the
multi-category Scene Graph workload rather than basic mask capability.

The `gi-threshold-0p8` sensitivity run produced no masks. The formal GI run uses
the delivery operating point, threshold 0.5. Scores are therefore not assumed to
be calibrated identically across the two models.

## Figures

- `hardware_timeseries.png`: GPU utilization, power, unified-memory headroom, temperature.
- `latency_boxplot.png`: grounding/HTTP and full-frame distributions.
- `mask_agreement_timeline.png`: mask agreement at fixed camera timestamps.
- `quality_vs_latency.png`: true COCO GT quality versus single-prompt latency.

Raw JSONL, full 3D points, graph JSON, model logs, resource samples, overlays,
and SHA256 files remain alongside this report. Bag mIoU is teacher agreement;
only the COCO-10 metrics use polygon ground truth.
"""
    path.write_text(text, encoding="utf-8")


def _stats(values: list[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None, "values": []}
    return {
        "count": len(numeric),
        "mean": mean(numeric),
        "p50": median(numeric),
        "p95": float(np.percentile(numeric, 95)),
        "min": min(numeric),
        "max": max(numeric),
        "values": numeric,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _edge_count(edges: Any) -> int:
    if isinstance(edges, list):
        return len(edges)
    if isinstance(edges, dict):
        return sum(len(value) if isinstance(value, list) else 1 for value in edges.values())
    return 0


def _node_categories(graph: dict[str, Any]) -> list[str]:
    nodes = graph.get("nodes", {})
    values = nodes.values() if isinstance(nodes, dict) else nodes
    return [
        str(node.get("category") or node.get("object_type") or node.get("type"))
        for node in values
        if node.get("category") or node.get("object_type") or node.get("type")
    ]


def _utc_ns(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1e9)


def _memory_gib(value: str) -> float:
    match = re.fullmatch(r"([0-9.]+)([KMG]iB)", value)
    if not match:
        return 0.0
    number = float(match.group(1))
    return number * {"KiB": 1 / 2**20, "MiB": 1 / 1024, "GiB": 1}[match.group(2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot the Thor Scene Graph A/B run.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
