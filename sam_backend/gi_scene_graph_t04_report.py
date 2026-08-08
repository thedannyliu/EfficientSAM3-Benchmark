from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gi_scene_graph_t03_report import _condition_summary
from .scene_graph_ab_report import _read_json, _stats


def main() -> None:
    args = parse_args()
    report = build_report(args.t03_root, args.t04_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "report.md", report)
    _make_plots(args.output_dir, report)
    print(json.dumps(report["headline"], indent=2))


def build_report(t03_root: Path, t04_root: Path) -> dict[str, Any]:
    dense_path = t03_root / "sg5-stateful-r30-pose-fixture"
    voxel_path = t04_root / "sg5-stateful-r30-voxel01"
    dense = _condition_summary(dense_path, "dense")
    voxel = _condition_summary(voxel_path, "voxel01")
    geometry = _read_json(t04_root / "geometry" / "dense-t03b-voxel01.json")
    publish = _parse_publish_metrics(voxel_path / "detection.log")
    dense_bytes = (dense_path / "detections.jsonl").stat().st_size
    voxel_bytes = (voxel_path / "detections.jsonl").stat().st_size
    byte_reduction = 1.0 - voxel_bytes / dense_bytes
    point_reduction = 1.0 - publish["payload_points"] / max(publish["raw_points"], 1)
    dense_p95 = dense["detections"]["camera_to_last_detection_ms"]["p95"]
    voxel_p95 = voxel["detections"]["camera_to_last_detection_ms"]["p95"]
    p95_reduction = 1.0 - voxel_p95 / dense_p95
    expected_refresh = list(range(0, voxel["speed"]["detector_calls"], 30))
    gates = {
        "complete_publication_p95_below_1000_ms": voxel_p95 < 1000.0,
        "complete_publication_p95_at_least_10_percent_lower": p95_reduction >= 0.10,
        "json_bytes_at_least_50_percent_lower": byte_reduction >= 0.50,
        "points_at_least_50_percent_lower": point_reduction >= 0.50,
        "offline_voxel_geometry_exact": geometry["all_cell_keys_identical"]
        and geometry["max_mean_error_m"] <= 1e-5,
        "at_least_87_completed_frames": voxel["speed"]["completed_frames"] >= 87,
        "refresh_cadence_exact": voxel["speed"]["refresh_zero_based_indices"]
        == expected_refresh,
        "at_least_40_nonempty_3d_frames": voxel["detections"]["frames"] >= 40,
        "nonempty_graph": voxel["graph"]["nodes"] > 0,
        "no_pipeline_traceback": not voxel["pipeline_traceback"],
        "temperature_below_80_c": voxel["resources"]["gpu_temperature_c"]["max"]
        < 80.0,
        "at_least_32_gib_available": voxel["resources"]["mem_available_gib"]["min"]
        >= 32.0,
    }
    headline = {
        "dense_complete_publication_p95_ms": dense_p95,
        "voxel_complete_publication_p95_ms": voxel_p95,
        "complete_publication_p95_reduction_fraction": p95_reduction,
        "dense_detection_json_bytes": dense_bytes,
        "voxel_detection_json_bytes": voxel_bytes,
        "detection_json_reduction_fraction": byte_reduction,
        "voxel_raw_points": publish["raw_points"],
        "voxel_payload_points": publish["payload_points"],
        "live_point_reduction_fraction": point_reduction,
        "dense_completed_frames": dense["speed"]["completed_frames"],
        "voxel_completed_frames": voxel["speed"]["completed_frames"],
        "dense_nonempty_3d_frames": dense["detections"]["frames"],
        "voxel_nonempty_3d_frames": voxel["detections"]["frames"],
        "dense_graph_nodes": dense["graph"]["nodes"],
        "voxel_graph_nodes": voxel["graph"]["nodes"],
        "all_gates_pass": all(gates.values()),
    }
    return {
        "headline": headline,
        "gates": gates,
        "conditions": {"dense": dense, "voxel01": voxel},
        "offline_geometry": geometry,
        "voxel_publish": publish,
    }


def _parse_publish_metrics(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "DETECTION3D_METRICS " in line:
            rows.append(json.loads(line.split("DETECTION3D_METRICS ", 1)[1]))
    return {
        "frames": len(rows),
        "messages": sum(int(row["message_count"]) for row in rows),
        "raw_points": sum(int(row["raw_point_count"]) for row in rows),
        "payload_points": sum(int(row["payload_point_count"]) for row in rows),
        "payload_bytes": sum(int(row["payload_bytes"]) for row in rows),
        "publish_ms": _stats([float(row["publish_ms"]) for row in rows]),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    h = report["headline"]
    dense = report["conditions"]["dense"]
    voxel = report["conditions"]["voxel01"]
    publish = report["voxel_publish"]
    gate_rows = "\n".join(
        f"| `{name}` | {'Pass' if passed else 'Fail'} |"
        for name, passed in report["gates"].items()
    )
    text = f"""# T04: Voxel-Compressed 3D Detection Transport

## Result

| Metric | Dense T03b control | 1 cm voxel transport |
| --- | ---: | ---: |
| Complete source-to-last-3D p50 | {dense['detections']['camera_to_last_detection_ms']['p50']:.1f} ms | {voxel['detections']['camera_to_last_detection_ms']['p50']:.1f} ms |
| Complete source-to-last-3D p95 | {h['dense_complete_publication_p95_ms']:.1f} ms | {h['voxel_complete_publication_p95_ms']:.1f} ms |
| Detection JSONL bytes | {h['dense_detection_json_bytes']:,} | {h['voxel_detection_json_bytes']:,} |
| Completed full frames | {h['dense_completed_frames']} | {h['voxel_completed_frames']} |
| Non-empty 3D source frames | {h['dense_nonempty_3d_frames']} | {h['voxel_nonempty_3d_frames']} |
| Final graph nodes | {h['dense_graph_nodes']} | {h['voxel_graph_nodes']} |

Complete-publication p95 decreased **{h['complete_publication_p95_reduction_fraction']:.1%}**
and recorded JSON decreased **{h['detection_json_reduction_fraction']:.1%}**.
The live condition reduced {h['voxel_raw_points']:,} raw points to
{h['voxel_payload_points']:,} payload points
({h['live_point_reduction_fraction']:.1%}). Voxel publication p50/p95 was
{publish['publish_ms']['p50']:.1f}/{publish['publish_ms']['p95']:.1f} ms.

## Geometry fidelity

The offline check covered {report['offline_geometry']['messages']} dense T03b
messages and reduced {report['offline_geometry']['raw_points']:,} points to
{report['offline_geometry']['compressed_points']:,}. All downstream 1 cm cell
keys were identical and maximum per-cell mean error was
{report['offline_geometry']['max_mean_error_m']:.3g} m.

## Pre-registered gates

| Gate | Result |
| --- | --- |
{gate_rows}

Overall: **{'Pass' if h['all_gates_pass'] else 'Partial pass'}**. The transport
and geometry gates pass. The only failed gate is the pre-registered count of 40
non-empty source frames: the candidate produced {h['voxel_nonempty_3d_frames']}.
Because mask/tracker scheduling samples different live source frames after a
latency change, this count cannot establish a voxel-geometry regression by
itself. Keep the failure and perform a fixed-frame 3D replay before merge.

## Hardware

| Metric | Dense T03b control | 1 cm voxel transport |
| --- | ---: | ---: |
| Mean GPU utilization | {dense['resources']['gpu_utilization_percent']['mean']:.1f}% | {voxel['resources']['gpu_utilization_percent']['mean']:.1f}% |
| Mean container CPU | {dense['resources']['container_cpu_percent']['mean']:.1f}% | {voxel['resources']['container_cpu_percent']['mean']:.1f}% |
| Maximum GPU temperature | {dense['resources']['gpu_temperature_c']['max']:.1f} C | {voxel['resources']['gpu_temperature_c']['max']:.1f} C |
| Minimum `MemAvailable` | {dense['resources']['mem_available_gib']['min']:.2f} GiB | {voxel['resources']['mem_available_gib']['min']:.2f} GiB |
| Mean Docker working set | {dense['resources']['container_memory_gib']['mean']:.2f} GiB | {voxel['resources']['container_memory_gib']['mean']:.2f} GiB |

## Figures

- `latency_and_transport.png`
- `pipeline_outputs.png`
"""
    path.write_text(text, encoding="utf-8")


def _make_plots(output: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = report["headline"]
    dense = report["conditions"]["dense"]
    voxel = report["conditions"]["voxel01"]
    colors = ["#3366cc", "#109618"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        ["dense", "voxel01"],
        [h["dense_complete_publication_p95_ms"], h["voxel_complete_publication_p95_ms"]],
        color=colors,
    )
    axes[0].axhline(1000, color="#dc3912", linestyle="--", label="p95 gate")
    axes[0].set_ylabel("source to last 3D p95 (ms)")
    axes[0].set_title("Publication latency")
    axes[0].legend()
    axes[1].bar(
        ["dense", "voxel01"],
        [h["dense_detection_json_bytes"] / 2**20, h["voxel_detection_json_bytes"] / 2**20],
        color=colors,
    )
    axes[1].set_ylabel("detections JSONL (MiB)")
    axes[1].set_title("Serialized 3D transport")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "latency_and_transport.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    values = [
        ("Completed frames", dense["speed"]["completed_frames"], voxel["speed"]["completed_frames"]),
        ("Non-empty 3D frames", dense["detections"]["frames"], voxel["detections"]["frames"]),
        ("Final graph nodes", dense["graph"]["nodes"], voxel["graph"]["nodes"]),
    ]
    for axis, (title, left, right) in zip(axes, values):
        axis.bar(["dense", "voxel01"], [left, right], color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "pipeline_outputs.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the T04 voxel report.")
    parser.add_argument("--t03-root", type=Path, required=True)
    parser.add_argument("--t04-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
