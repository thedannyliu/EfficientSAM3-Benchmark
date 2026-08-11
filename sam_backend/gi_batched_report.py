from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .gi_tracking_report import _directed_mask_ious, _load_prediction
from .scene_graph_ab_report import _resource_summary, _stats


CONDITIONS = (
    "original",
    "gi_sequential",
    "gi_batched_tracker",
    "gi_batched_detector_only",
)


def main() -> None:
    args = parse_args()
    report = build_report(args.t01_root, args.t05_root, args.t06_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in report.items() if key != "profiles"}
    (args.output_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "per_frame.csv", report["frames"])
    _write_markdown(args.output_dir / "report.md", report)
    _make_plots(args.output_dir, report)
    print(json.dumps(report["headline"], indent=2))


def build_report(t01_root: Path, t05_root: Path, t06_root: Path) -> dict[str, Any]:
    paths = {
        "original": t01_root / "original-refresh1",
        "gi_sequential": t01_root / "gi-refresh1",
        "gi_batched_tracker": t05_root / "gi-batched-r1-formal",
        "gi_batched_detector_only": t06_root / "gi-batched-detector-only-r1",
    }
    profiles = {name: _read_jsonl(path / "profile.jsonl") for name, path in paths.items()}
    conditions = {
        name: _condition_summary(name, paths[name], profiles[name]) for name in CONDITIONS
    }
    manifest = _read_jsonl(t01_root / "input" / "manifest.jsonl")
    quality, frames = _quality_summary(manifest, paths, profiles)

    original_p50 = conditions["original"]["total_ms"]["p50"]
    sequential_p50 = conditions["gi_sequential"]["total_ms"]["p50"]
    detector_p50 = conditions["gi_batched_detector_only"]["total_ms"]["p50"]
    t05_quality = quality["original_to_gi_batched_tracker"]
    t06_quality = quality["original_to_gi_batched_detector_only"]
    detector_status = profiles["gi_batched_detector_only"]
    gates = {
        "detector_only_faster_than_original_p50": detector_p50 < original_p50,
        "detector_only_at_least_20_percent_faster_than_sequential": (
            detector_p50 <= 0.8 * sequential_p50
        ),
        "all_conditions_completed_100_frames": all(
            conditions[name]["frames"] == 100 for name in CONDITIONS
        ),
        "detector_only_retained_zero_tracker_states": all(
            int(row["status"].get("tracker_state_count", -1)) == 0
            for row in detector_status
        ),
        "teacher_recall_drop_vs_t05_at_most_0p02": (
            t05_quality["recall_at_50"] - t06_quality["recall_at_50"] <= 0.02
        ),
        "teacher_miou_drop_vs_t05_at_most_0p02": (
            t05_quality["instance_miou"] - t06_quality["instance_miou"] <= 0.02
        ),
        "temperature_below_80_c": max(
            conditions[name]["resources"]["gpu_temperature_c"]["max"]
            for name in CONDITIONS
        )
        < 80.0,
        "at_least_32_gib_available": min(
            conditions[name]["resources"]["mem_available_gib"]["min"]
            for name in CONDITIONS
        )
        >= 32.0,
    }
    gates["all_t06_gates_pass"] = all(gates.values())

    headline = {
        "original_p50_ms": original_p50,
        "gi_sequential_p50_ms": sequential_p50,
        "gi_batched_tracker_p50_ms": conditions["gi_batched_tracker"]["total_ms"]["p50"],
        "gi_batched_detector_only_p50_ms": detector_p50,
        "detector_only_speedup_vs_original": original_p50 / detector_p50,
        "detector_only_percent_faster_than_original": 100.0 * (original_p50 - detector_p50) / original_p50,
        "detector_only_speedup_vs_sequential_gi": sequential_p50 / detector_p50,
        "detector_only_teacher_miou": t06_quality["instance_miou"],
        "detector_only_teacher_recall_at_50": t06_quality["recall_at_50"],
        "all_t06_gates_pass": gates["all_t06_gates_pass"],
    }
    return {
        "headline": headline,
        "conditions": conditions,
        "quality": quality,
        "gates": gates,
        "frames": frames,
        "profiles": profiles,
        "interpretation_limits": [
            "Original SAM3.1 is a teacher reference, not ground truth.",
            "Client latency ends after the matching runtime status response and excludes benchmark NPZ artifact writing.",
            "Sequence wall time includes writing large prediction files to NAS and is not the detector throughput metric.",
            "Detector-only IDs are frame-local and are not temporal tracking identities.",
        ],
    }


def _condition_summary(name: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = min(int(row["started_wall_ns"]) for row in rows)
    ended = max(int(row["ended_wall_ns"]) for row in rows)
    total = [float(row["total_ms"]) for row in rows]
    if name == "original":
        runtime = [float(row["metadata"]["runtime_ms"]) for row in rows]
        process = runtime
        detect = runtime
    else:
        process = [float(row["status"]["process_ms"]) for row in rows]
        detect = [float(row["status"]["detect_ms"]) for row in rows]
        runtime = process
    summary = _read_json(path / "summary.json")
    return {
        "frames": len(rows),
        "sequence_wall_seconds": float(summary["sequence_wall_seconds"]),
        "total_ms": _stats(total),
        "runtime_process_ms": _stats(process),
        "runtime_detect_ms": _stats(detect),
        "client_boundary_ms": _stats([a - b for a, b in zip(total, runtime)]),
        "mask_observations": sum(int(row["mask_count"]) for row in rows),
        "nonempty_frames": sum(int(row["mask_count"]) > 0 for row in rows),
        "resources": _resource_summary(
            path / "resources.jsonl",
            {"started_wall_ns": started, "ended_wall_ns": ended},
        ),
    }


def _quality_summary(
    manifest: list[dict[str, Any]],
    paths: dict[str, Path],
    profiles: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = {
        "original_to_gi_sequential": ("original", "gi_sequential"),
        "original_to_gi_batched_tracker": ("original", "gi_batched_tracker"),
        "original_to_gi_batched_detector_only": ("original", "gi_batched_detector_only"),
        "gi_batched_tracker_to_detector_only": (
            "gi_batched_tracker",
            "gi_batched_detector_only",
        ),
    }
    collected = {name: [] for name in pairs}
    frame_rows = []
    label_counts = {name: Counter() for name in CONDITIONS}
    for item in manifest:
        index = int(item["frame_index"])
        predictions = {
            name: _load_prediction(_prediction_path(paths[name], name, item))
            for name in CONDITIONS
        }
        frame_row: dict[str, Any] = {
            "frame_index": index,
            "source_stamp_ns": int(item["source_stamp_ns"]),
        }
        for name, (masks, labels) in predictions.items():
            frame_row[f"{name}_masks"] = len(masks)
            label_counts[name].update(str(label) for label in labels)
        for comparison, (source, target) in pairs.items():
            values = _directed_mask_ious(*predictions[source], *predictions[target])
            collected[comparison].extend(values)
            frame_row[f"{comparison}_miou"] = mean(values) if values else None
        for name in CONDITIONS:
            frame_row[f"{name}_total_ms"] = float(profiles[name][index]["total_ms"])
        frame_rows.append(frame_row)
    result = {
        name: {
            "instances": len(values),
            "instance_miou": mean(values) if values else 0.0,
            "recall_at_50": sum(value >= 0.5 for value in values) / max(len(values), 1),
        }
        for name, values in collected.items()
    }
    result["label_observations"] = {
        name: dict(label_counts[name].most_common()) for name in CONDITIONS
    }
    return result, frame_rows


def _prediction_path(path: Path, name: str, item: dict[str, Any]) -> Path:
    filename = (
        f"{int(item['source_stamp_ns'])}.npz"
        if name == "original"
        else f"{int(item['frame_index']):05d}.npz"
    )
    return path / "predictions" / filename


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = report["profiles"]
    labels = ["Original", "GI sequential", "GI batched+tracker", "GI batched detector-only"]
    colors = ["#3366cc", "#dc3912", "#ff9900", "#109618"]

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(
        [[row["total_ms"] for row in profiles[name]] for name in CONDITIONS],
        labels=labels,
        showfliers=False,
    )
    axis.axhline(report["headline"]["original_p50_ms"], color="#3366cc", linestyle="--", alpha=0.7)
    axis.set_ylabel("client-visible latency (ms)")
    axis.set_title("Five-prompt per-frame detection latency")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_boxplot.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for name, label, color in zip(CONDITIONS, labels, colors):
        axis.plot(
            range(len(profiles[name])),
            [row["total_ms"] for row in profiles[name]],
            label=label,
            color=color,
            alpha=0.8,
        )
    axis.set_xlabel("fixed source frame index")
    axis.set_ylabel("client-visible latency (ms)")
    axis.set_title("Latency on identical source frames")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_timeline.png", dpi=160)
    plt.close(fig)

    frames = report["frames"]
    fig, axis = plt.subplots(figsize=(12, 5))
    for key, label, color in (
        ("original_to_gi_sequential_miou", "Original→GI sequential", "#dc3912"),
        ("original_to_gi_batched_tracker_miou", "Original→GI batched+tracker", "#ff9900"),
        ("original_to_gi_batched_detector_only_miou", "Original→GI detector-only", "#109618"),
    ):
        axis.plot(
            [row["frame_index"] for row in frames],
            [np.nan if row[key] is None else row[key] for row in frames],
            label=label,
            color=color,
            alpha=0.8,
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("fixed source frame index")
    axis.set_ylabel("directed instance IoU")
    axis.set_title("Agreement with Original SAM3.1 teacher")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "teacher_agreement_timeline.png", dpi=160)
    plt.close(fig)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    c = report["conditions"]
    q = report["quality"]
    g = report["gates"]
    rows = []
    for name, label in zip(CONDITIONS, ("Original SAM3.1", "GI sequential", "GI batched+tracker", "GI batched detector-only")):
        value = c[name]
        rows.append(
            f"| {label} | {value['total_ms']['mean']:.1f} | {value['total_ms']['p50']:.1f} | "
            f"{value['total_ms']['p95']:.1f} | {value['runtime_detect_ms']['p50']:.1f} | "
            f"{value['mask_observations']} |"
        )
    text = f"""# T05/T06 Batched GI Detection Report

## Outcome

GI batched detector-only p50 was {h['gi_batched_detector_only_p50_ms']:.1f} ms,
versus {h['original_p50_ms']:.1f} ms for Original SAM3.1. This is
{h['detector_only_percent_faster_than_original']:.1f}% lower latency
({h['detector_only_speedup_vs_original']:.3f}x throughput by p50 latency).

| Condition | Mean client ms | P50 client ms | P95 client ms | P50 detect ms | Mask observations |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Teacher agreement

| Comparison | Instance mIoU | Recall at IoU 0.5 |
| --- | ---: | ---: |
| Original to GI sequential | {q['original_to_gi_sequential']['instance_miou']:.4f} | {q['original_to_gi_sequential']['recall_at_50']:.4f} |
| Original to GI batched+tracker | {q['original_to_gi_batched_tracker']['instance_miou']:.4f} | {q['original_to_gi_batched_tracker']['recall_at_50']:.4f} |
| Original to GI batched detector-only | {q['original_to_gi_batched_detector_only']['instance_miou']:.4f} | {q['original_to_gi_batched_detector_only']['recall_at_50']:.4f} |
| GI batched+tracker to detector-only | {q['gi_batched_tracker_to_detector_only']['instance_miou']:.4f} | {q['gi_batched_tracker_to_detector_only']['recall_at_50']:.4f} |

These are teacher-agreement measurements, not ground-truth accuracy.

## Gates

{chr(10).join(f'- `{name}`: `{passed}`' for name, passed in g.items())}

## Interpretation

- Prompt batching removes repeated grounding-head calls.
- Detector-only mode additionally removes the 768 tracking backbone, temporal
  propagation, and tracker-state consolidation from an R1 image-detector use case.
- Client latency excludes benchmark NPZ writing. Sequence wall time includes NAS
  artifact writes and must not be presented as detector throughput.
- Detector-only IDs are frame-local. Stateful R30 remains the appropriate mode
  when temporal identity is required.

## Figures

- `latency_boxplot.png`
- `latency_timeline.png`
- `teacher_agreement_timeline.png`
"""
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report T05/T06 GI batched detection experiments.")
    parser.add_argument("--t01-root", type=Path, required=True)
    parser.add_argument("--t05-root", type=Path, required=True)
    parser.add_argument("--t06-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
