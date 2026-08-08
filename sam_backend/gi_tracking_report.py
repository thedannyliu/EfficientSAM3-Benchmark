from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .scene_graph_ab_report import _resource_summary, _stats, _utc_ns


CONDITIONS = ("original-refresh1", "gi-refresh1", "gi-refresh30")


def main() -> None:
    args = parse_args()
    report = build_report(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_frames(args.output_dir / "per_frame.csv", report["frames"])
    _write_markdown(args.output_dir / "report.md", report)
    _make_plots(args.root, args.output_dir, report)
    _make_contact_sheet(args.root, args.output_dir)
    print(json.dumps(report["headline"], indent=2))


def build_report(root: Path) -> dict[str, Any]:
    manifest = _read_jsonl(root / "input" / "manifest.jsonl")
    profiles = {
        name: _read_jsonl(root / name / "profile.jsonl") for name in CONDITIONS
    }
    conditions = {
        name: _condition_summary(root / name, profiles[name]) for name in CONDITIONS
    }
    comparisons, frames = _quality_summary(root, manifest)

    tracking = conditions["gi-refresh30"]["tracking_only"]
    repeated = conditions["gi-refresh1"]["all_frames"]
    teacher_r1 = comparisons["original_to_gi_refresh1"]
    teacher_r30 = comparisons["original_to_gi_refresh30"]
    self_buckets = comparisons["gi_refresh1_to_gi_refresh30"]["buckets"]
    bucket_values = [self_buckets[name]["instance_miou"] for name in ("1-9", "10-19", "20-29")]
    gates = {
        "tracking_p50_at_most_250_ms": tracking["total_ms"]["p50"] <= 250.0,
        "tracking_at_least_5x_faster_than_refresh1": (
            repeated["total_ms"]["p50"] / tracking["total_ms"]["p50"] >= 5.0
        ),
        "teacher_recall_drop_at_most_0p10": (
            teacher_r1["recall_at_50"] - teacher_r30["recall_at_50"] <= 0.10
        ),
        "no_monotonic_bucket_collapse": not (
            bucket_values[0] > bucket_values[1] > bucket_values[2]
        ),
        "completed_100_frames": all(
            conditions[name]["frames"] == 100 for name in CONDITIONS
        ),
        "temperature_below_80_c": max(
            conditions[name]["resources"]["gpu_temperature_c"]["max"]
            for name in CONDITIONS
        ) < 80.0,
        "at_least_32_gib_available": min(
            conditions[name]["resources"]["mem_available_gib"]["min"]
            for name in CONDITIONS
        ) >= 32.0,
    }
    gates["all_t01_gates_pass"] = all(gates.values())

    headline = {
        "original_effective_fps": conditions["original-refresh1"]["effective_fps"],
        "gi_refresh1_effective_fps": conditions["gi-refresh1"]["effective_fps"],
        "gi_refresh30_effective_fps": conditions["gi-refresh30"]["effective_fps"],
        "gi_tracking_only_p50_ms": tracking["total_ms"]["p50"],
        "gi_tracking_only_p95_ms": tracking["total_ms"]["p95"],
        "gi_refresh1_p50_ms": repeated["total_ms"]["p50"],
        "tracking_speedup_vs_gi_refresh1": (
            repeated["total_ms"]["p50"] / tracking["total_ms"]["p50"]
        ),
        "tracking_speedup_vs_original": (
            conditions["original-refresh1"]["all_frames"]["total_ms"]["p50"]
            / tracking["total_ms"]["p50"]
        ),
        "original_to_gi_refresh1_miou": teacher_r1["instance_miou"],
        "original_to_gi_refresh30_miou": teacher_r30["instance_miou"],
        "original_to_gi_refresh1_recall_at_50": teacher_r1["recall_at_50"],
        "original_to_gi_refresh30_recall_at_50": teacher_r30["recall_at_50"],
        "gi_refresh1_to_refresh30_miou": comparisons[
            "gi_refresh1_to_gi_refresh30"
        ]["instance_miou"],
        "all_t01_gates_pass": gates["all_t01_gates_pass"],
    }
    return {
        "headline": headline,
        "conditions": conditions,
        "comparisons": comparisons,
        "gates": gates,
        "frames": frames,
        "interpretation_limits": [
            "Original is a teacher reference for the bag sequence, not ground truth.",
            "The sequence is 100 frames from one 3.34-second camera interval.",
            "GI refresh1 retains tracker state while detecting each frame; it is not the prior stateless reset-per-frame condition.",
            "Runtime status backbone_ms and tracker_ms are exponential moving averages; process_ms and detect_ms are per-frame values.",
        ],
    }


def _condition_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _read_json(path / "summary.json")
    started = min(int(row["started_wall_ns"]) for row in rows)
    ended = max(int(row["ended_wall_ns"]) for row in rows)
    resources = _resource_summary(
        path / "resources.jsonl",
        {"started_wall_ns": started, "ended_wall_ns": ended},
    )
    result: dict[str, Any] = {
        "frames": len(rows),
        "sequence_wall_seconds": float(summary["sequence_wall_seconds"]),
        "effective_fps": len(rows) / float(summary["sequence_wall_seconds"]),
        "all_frames": _latency_summary(rows),
        "mask_observations": sum(int(row["mask_count"]) for row in rows),
        "nonempty_frames": sum(int(row["mask_count"]) > 0 for row in rows),
        "resources": resources,
    }
    if "model_init_ms" in summary:
        result["startup_seconds"] = float(summary["model_init_ms"]) / 1000.0
    else:
        result["startup_seconds"] = float(_read_json(path / "startup.json")["startup_seconds"])
    if path.name.startswith("gi-"):
        refresh = [row for row in rows if float(row["status"]["detect_ms"]) > 0]
        tracking = [row for row in rows if float(row["status"]["detect_ms"]) == 0]
        result["refresh_frames"] = _latency_summary(refresh)
        result["tracking_only"] = _latency_summary(tracking)
        result["refresh_indices"] = [int(row["frame_index"]) for row in refresh]
        result["lost_frames"] = sum(
            int(row["active_mask_count"]) < int(row["mask_count"]) for row in rows
        )
    return result


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "count": len(rows),
        "total_ms": _stats([float(row["total_ms"]) for row in rows]),
    }
    if rows and "status" in rows[0]:
        for key in ("process_ms", "detect_ms"):
            result[key] = _stats([float(row["status"][key]) for row in rows])
        for key in ("submit_ms", "wait_ms", "status_ms"):
            result[key] = _stats([float(row[key]) for row in rows])
    return result


def _quality_summary(
    root: Path, manifest: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = {
        "original_to_gi_refresh1": ("original-refresh1", "gi-refresh1"),
        "original_to_gi_refresh30": ("original-refresh1", "gi-refresh30"),
        "gi_refresh1_to_gi_refresh30": ("gi-refresh1", "gi-refresh30"),
    }
    values = {
        key: {"all": [], "buckets": {name: [] for name in ("refresh", "1-9", "10-19", "20-29")}}
        for key in pairs
    }
    frames = []
    for item in manifest:
        index = int(item["frame_index"])
        loaded = {
            name: _load_prediction(_prediction_path(root, name, item))
            for name in CONDITIONS
        }
        frame_row: dict[str, Any] = {
            "frame_index": index,
            "source_stamp_ns": int(item["source_stamp_ns"]),
        }
        for name in CONDITIONS:
            frame_row[f"{name}_masks"] = len(loaded[name][0])
        bucket = _phase_bucket(index)
        for key, (source, target) in pairs.items():
            source_masks, source_labels = loaded[source]
            target_masks, target_labels = loaded[target]
            ious = _directed_mask_ious(
                source_masks, source_labels, target_masks, target_labels
            )
            values[key]["all"].extend(ious)
            values[key]["buckets"][bucket].extend(ious)
            frame_row[f"{key}_miou"] = mean(ious) if ious else None
        frames.append(frame_row)

    comparisons = {}
    for key, collected in values.items():
        comparisons[key] = _agreement_summary(collected["all"])
        comparisons[key]["buckets"] = {
            name: _agreement_summary(bucket_values)
            for name, bucket_values in collected["buckets"].items()
        }
    return comparisons, frames


def _prediction_path(root: Path, condition: str, item: dict[str, Any]) -> Path:
    if condition == "original-refresh1":
        filename = f"{int(item['source_stamp_ns'])}.npz"
    else:
        filename = f"{int(item['frame_index']):05d}.npz"
    return root / condition / "predictions" / filename


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["mask_shape"])
        masks = np.unpackbits(
            archive["masks_packed"], bitorder=str(archive["bitorder"].item())
        )[: int(np.prod(shape))].astype(bool).reshape(shape)
        labels = archive["labels"].astype(str)
        lost = (
            archive["lost"].astype(bool)
            if "lost" in archive
            else np.zeros(len(masks), dtype=bool)
        )
        return masks[np.logical_not(lost)], labels[np.logical_not(lost)]


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


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def _phase_bucket(frame_index: int) -> str:
    phase = frame_index % 30
    if phase == 0:
        return "refresh"
    if phase <= 9:
        return "1-9"
    if phase <= 19:
        return "10-19"
    return "20-29"


def _agreement_summary(values: list[float]) -> dict[str, Any]:
    return {
        "instances": len(values),
        "instance_miou": mean(values) if values else 0.0,
        "recall_at_50": sum(value >= 0.5 for value in values) / max(len(values), 1),
    }


def _write_frames(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(root: Path, output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = {
        name: _read_jsonl(root / name / "profile.jsonl") for name in CONDITIONS
    }
    refresh30 = profiles["gi-refresh30"]
    latency_sets = [
        [row["total_ms"] for row in profiles["original-refresh1"]],
        [row["total_ms"] for row in profiles["gi-refresh1"]],
        [row["total_ms"] for row in refresh30 if row["status"]["detect_ms"] > 0],
        [row["total_ms"] for row in refresh30 if row["status"]["detect_ms"] == 0],
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(
        latency_sets,
        labels=["Original", "GI refresh=1", "GI refresh frames", "GI tracking-only"],
        showfliers=False,
    )
    axis.set_ylabel("client latency (ms)")
    axis.set_title("T01 latency distributions")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_boxplot.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for name, color in (
        ("original-refresh1", "#3366cc"),
        ("gi-refresh1", "#dc3912"),
        ("gi-refresh30", "#109618"),
    ):
        axis.plot(
            [row["frame_index"] if "frame_index" in row else index for index, row in enumerate(profiles[name])],
            [row["total_ms"] for row in profiles[name]],
            label=name,
            color=color,
            alpha=0.8,
        )
    for frame in (0, 30, 60, 90):
        axis.axvline(frame, color="black", alpha=0.12)
    axis.set_xlabel("frame index")
    axis.set_ylabel("client latency (ms)")
    axis.set_title("Latency over the fixed 100-frame sequence")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_timeline.png", dpi=160)
    plt.close(fig)

    frames = report["frames"]
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        [row["frame_index"] for row in frames],
        [np.nan if row["original_to_gi_refresh1_miou"] is None else row["original_to_gi_refresh1_miou"] for row in frames],
        label="Original→GI refresh=1",
        color="#dc3912",
    )
    axis.plot(
        [row["frame_index"] for row in frames],
        [np.nan if row["original_to_gi_refresh30_miou"] is None else row["original_to_gi_refresh30_miou"] for row in frames],
        label="Original→GI refresh=30",
        color="#109618",
    )
    axis.plot(
        [row["frame_index"] for row in frames],
        [np.nan if row["gi_refresh1_to_gi_refresh30_miou"] is None else row["gi_refresh1_to_gi_refresh30_miou"] for row in frames],
        label="GI refresh=1→refresh=30",
        color="#990099",
        alpha=0.75,
    )
    for frame in (0, 30, 60, 90):
        axis.axvline(frame, color="black", alpha=0.12)
    axis.set_ylim(0, 1)
    axis.set_xlabel("frame index")
    axis.set_ylabel("directed instance mIoU")
    axis.set_title("Teacher agreement and tracking stability")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "agreement_timeline.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = {
        "original-refresh1": "#3366cc",
        "gi-refresh1": "#dc3912",
        "gi-refresh30": "#109618",
    }
    for name in CONDITIONS:
        rows = _read_jsonl(root / name / "resources.jsonl")
        start_ns = min(int(row["started_wall_ns"]) for row in profiles[name])
        end_ns = max(int(row["ended_wall_ns"]) for row in profiles[name])
        rows = [row for row in rows if start_ns <= _utc_ns(row["timestamp_utc"]) <= end_ns]
        x = [(_utc_ns(row["timestamp_utc"]) - start_ns) / 1e9 for row in rows]
        axes[0, 0].plot(x, [row["nvidia_smi"]["gpu"]["utilization.gpu"] for row in rows], label=name, color=colors[name], alpha=0.8)
        axes[0, 1].plot(x, [row["nvidia_smi"]["gpu"]["power.draw"] for row in rows], label=name, color=colors[name], alpha=0.8)
        axes[1, 0].plot(x, [row["host_memory"]["memavailable_bytes"] / 2**30 for row in rows], label=name, color=colors[name], alpha=0.8)
        axes[1, 1].plot(x, [row["nvidia_smi"]["gpu"]["temperature.gpu"] for row in rows], label=name, color=colors[name], alpha=0.8)
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


def _make_contact_sheet(root: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    manifest = _read_jsonl(root / "input" / "manifest.jsonl")
    selected = (0, 29, 30, 59, 60, 89, 90, 99)
    colors = {
        "keyboard": np.asarray([51, 102, 204]),
        "table": np.asarray([220, 57, 18]),
        "book": np.asarray([16, 150, 24]),
        "computer desk": np.asarray([153, 0, 153]),
        "stool": np.asarray([255, 153, 0]),
    }
    fig, axes = plt.subplots(len(selected), len(CONDITIONS), figsize=(12, 24))
    for row_index, frame_index in enumerate(selected):
        item = manifest[frame_index]
        image = np.asarray(Image.open(root / "input" / "images" / item["image"]).convert("RGB"))
        for column, condition in enumerate(CONDITIONS):
            masks, labels = _load_prediction(_prediction_path(root, condition, item))
            overlay = image.copy()
            for mask, label in zip(masks, labels):
                color = colors.get(str(label), np.asarray([128, 128, 128]))
                overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
            axis = axes[row_index, column]
            axis.imshow(overlay)
            axis.axis("off")
            axis.set_title(f"frame {frame_index} | {condition} | {len(masks)} masks", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "overlay_contact_sheet.png", dpi=130)
    plt.close(fig)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    c = report["conditions"]
    q = report["comparisons"]
    g = report["gates"]
    text = f"""# T01: Five-Prompt GI Keyframe Detection and Tracking

## Outcome

T01 is a partial pass. GI tracking is useful and faster than Original, but the
pre-registered 5x speedup gate versus GI repeated detection was not reached.

| Metric | Original | GI refresh=1 | GI refresh=30 |
| --- | ---: | ---: | ---: |
| Effective sequence FPS | {h['original_effective_fps']:.3f} | {h['gi_refresh1_effective_fps']:.3f} | {h['gi_refresh30_effective_fps']:.3f} |
| All-frame p50 latency | {c['original-refresh1']['all_frames']['total_ms']['p50']:.1f} ms | {c['gi-refresh1']['all_frames']['total_ms']['p50']:.1f} ms | {c['gi-refresh30']['all_frames']['total_ms']['p50']:.1f} ms |
| Startup/model initialization | {c['original-refresh1']['startup_seconds']:.1f} s | {c['gi-refresh1']['startup_seconds']:.1f} s | {c['gi-refresh30']['startup_seconds']:.1f} s |

GI refresh=30 tracking-only p50 was {h['gi_tracking_only_p50_ms']:.1f} ms
and p95 was {h['gi_tracking_only_p95_ms']:.1f} ms. It was
{h['tracking_speedup_vs_gi_refresh1']:.2f}x faster than GI refresh=1 and
{h['tracking_speedup_vs_original']:.2f}x faster than Original by p50 latency.

## Quality

| Directed teacher comparison | Instance mIoU | Recall at IoU 0.5 |
| --- | ---: | ---: |
| Original to GI refresh=1 | {q['original_to_gi_refresh1']['instance_miou']:.4f} | {q['original_to_gi_refresh1']['recall_at_50']:.4f} |
| Original to GI refresh=30 | {q['original_to_gi_refresh30']['instance_miou']:.4f} | {q['original_to_gi_refresh30']['recall_at_50']:.4f} |
| GI refresh=1 to GI refresh=30 | {q['gi_refresh1_to_gi_refresh30']['instance_miou']:.4f} | {q['gi_refresh1_to_gi_refresh30']['recall_at_50']:.4f} |

The teacher-recall decrease was
{q['original_to_gi_refresh1']['recall_at_50'] - q['original_to_gi_refresh30']['recall_at_50']:.4f},
within the pre-registered 0.10 gate. This is teacher agreement, not ground-truth
accuracy.

## Screening gates

| Gate | Passed |
| --- | --- |
| Tracking-only p50 <= 250 ms | {g['tracking_p50_at_most_250_ms']} |
| Tracking at least 5x faster than GI refresh=1 | {g['tracking_at_least_5x_faster_than_refresh1']} |
| Teacher recall decrease <= 0.10 | {g['teacher_recall_drop_at_most_0p10']} |
| No monotonic agreement collapse across phase buckets | {g['no_monotonic_bucket_collapse']} |
| 100 frames completed in every condition | {g['completed_100_frames']} |
| Temperature below 80 C | {g['temperature_below_80_c']} |
| At least 32 GiB available memory | {g['at_least_32_gib_available']} |

Overall T01 gate: `{g['all_t01_gates_pass']}`.

## Figures

- `latency_boxplot.png`: condition and frame-type latency distributions.
- `latency_timeline.png`: per-frame latency and refresh positions.
- `agreement_timeline.png`: teacher agreement and GI tracking stability.
- `hardware_timeseries.png`: GPU, memory, power, and temperature.
- `overlay_contact_sheet.png`: frames around each refresh boundary.

Raw profiles, packed masks, runtime logs, resource JSONL, input hashes, and the
evaluation overlay remain beside this report. The full interpretation and next
attempt decision are tracked in `docs/gi_tracking_experiments.md`.
"""
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize T01 GI tracking evaluation.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
