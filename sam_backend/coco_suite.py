from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


COMPONENT_FIELDS = [
    "image_encoder_ms",
    "text_encoder_ms",
    "prompt_encoder_ms",
    "mask_decoder_ms",
    "transformer_ms",
    "geometry_encoder_ms",
    "segmentation_head_ms",
    "grounding_ms",
    "detector_ms",
    "memory_attention_ms",
    "memory_encoder_ms",
]

PARAM_WEIGHT_FIELDS = [
    "params_total",
    "params_backbone",
    "params_image_encoder",
    "params_text_encoder",
    "params_transformer",
    "params_geometry_encoder",
    "params_segmentation_head",
    "params_prompt_encoder",
    "params_mask_decoder",
    "params_detector",
    "params_memory_encoder",
    "params_memory_attention",
    "weight_total_bytes",
    "weight_backbone_bytes",
    "weight_image_encoder_bytes",
    "weight_text_encoder_bytes",
    "weight_transformer_bytes",
    "weight_geometry_encoder_bytes",
    "weight_segmentation_head_bytes",
    "weight_prompt_encoder_bytes",
    "weight_mask_decoder_bytes",
    "weight_detector_bytes",
    "weight_memory_encoder_bytes",
    "weight_memory_attention_bytes",
    "checkpoint_file_bytes",
]


@dataclass(frozen=True)
class CocoRun:
    model_id: str
    backend: str
    prompt_mode: str
    checkpoint_path: str | None = None
    model_config: str | None = None
    external_repo: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_RUNS = [
    CocoRun("sam3", "sam3", "both", checkpoint_path="checkpoints/sam3/sam3.pt", external_repo="external/sam3"),
    CocoRun(
        "es3p1_weak_image_weak_text",
        "efficientsam3",
        "both",
        checkpoint_path="checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type",
            "efficientvit",
            "--model-name",
            "b0",
            "--text-encoder-type",
            "MobileCLIP-S0",
            "--text-encoder-context-length",
            "16",
            "--text-encoder-pos-embed-table-size",
            "16",
        ),
    ),
    CocoRun(
        "es3p1_strong_image_weak_text",
        "efficientsam3",
        "both",
        checkpoint_path="checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type",
            "efficientvit",
            "--model-name",
            "b2",
            "--text-encoder-type",
            "MobileCLIP-S0",
            "--text-encoder-context-length",
            "16",
            "--text-encoder-pos-embed-table-size",
            "16",
        ),
    ),
    CocoRun(
        "es3_weak_image_strong_available_text",
        "efficientsam3",
        "both",
        checkpoint_path="checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type",
            "efficientvit",
            "--model-name",
            "b0",
            "--text-encoder-type",
            "MobileCLIP-S1",
            "--text-encoder-context-length",
            "16",
            "--text-encoder-pos-embed-table-size",
            "77",
        ),
    ),
    CocoRun(
        "es3_strong_image_strong_available_text",
        "efficientsam3",
        "both",
        checkpoint_path="checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type",
            "efficientvit",
            "--model-name",
            "b2",
            "--text-encoder-type",
            "MobileCLIP-S1",
            "--text-encoder-context-length",
            "16",
            "--text-encoder-pos-embed-table-size",
            "77",
        ),
    ),
    CocoRun(
        "sam2p1_hiera_tiny",
        "sam2",
        "point",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_tiny.pt",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        external_repo="external/sam2",
    ),
    CocoRun(
        "sam2p1_hiera_small",
        "sam2",
        "point",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_small.pt",
        model_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        external_repo="external/sam2",
    ),
    CocoRun(
        "sam2p1_hiera_base_plus",
        "sam2",
        "point",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_base_plus.pt",
        model_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        external_repo="external/sam2",
    ),
    CocoRun(
        "sam2p1_hiera_large",
        "sam2",
        "point",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        external_repo="external/sam2",
    ),
    CocoRun(
        "efficient_sam2p1_hiera_tiny",
        "efficient-sam2",
        "point",
        checkpoint_path="checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        external_repo="external/Efficient-SAM2",
    ),
    CocoRun(
        "efficient_sam2p1_hiera_small",
        "efficient-sam2",
        "point",
        checkpoint_path="checkpoints/efficient-sam2/sam2.1_hiera_small.pt",
        model_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        external_repo="external/Efficient-SAM2",
    ),
    CocoRun(
        "efficient_sam2p1_hiera_base_plus",
        "efficient-sam2",
        "point",
        checkpoint_path="checkpoints/efficient-sam2/sam2.1_hiera_base_plus.pt",
        model_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        external_repo="external/Efficient-SAM2",
    ),
    CocoRun(
        "efficient_sam2p1_hiera_large",
        "efficient-sam2",
        "point",
        checkpoint_path="checkpoints/efficient-sam2/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        external_repo="external/Efficient-SAM2",
    ),
    CocoRun(
        "efficienttam_ti",
        "efficienttam",
        "point",
        checkpoint_path="checkpoints/efficienttam/efficienttam_ti.pt",
        model_config="configs/efficienttam/efficienttam_ti.yaml",
        external_repo="external/EfficientTAM",
    ),
    CocoRun(
        "efficienttam_s",
        "efficienttam",
        "point",
        checkpoint_path="checkpoints/efficienttam/efficienttam_s.pt",
        model_config="configs/efficienttam/efficienttam_s.yaml",
        external_repo="external/EfficientTAM",
    ),
    CocoRun(
        "mobilesam_vit_t",
        "mobilesam",
        "point",
        checkpoint_path="checkpoints/mobilesam/mobile_sam.pt",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_t"),
    ),
    CocoRun(
        "mobilesam_vit_b",
        "mobilesam",
        "point",
        checkpoint_path="checkpoints/mobilesam/sam_vit_b_01ec64.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_b"),
    ),
    CocoRun(
        "mobilesam_vit_l",
        "mobilesam",
        "point",
        checkpoint_path="checkpoints/mobilesam/sam_vit_l_0b3195.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_l"),
    ),
    CocoRun(
        "mobilesam_vit_h",
        "mobilesam",
        "point",
        checkpoint_path="checkpoints/mobilesam/sam_vit_h_4b8939.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_h"),
    ),
]


def main() -> None:
    args = parse_args()
    results = run_suite(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "coco_suite_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "backend", "prompt_mode", "status", "summary_json", "csv", "message"])
        writer.writeheader()
        writer.writerows(results)
    print(summary_path)
    component_summary_path = write_component_summary(args.output_dir)
    if component_summary_path:
        print(component_summary_path)


def run_suite(args: argparse.Namespace) -> list[dict[str, str]]:
    selected = {name for name in args.models} if args.models else None
    runs = [run for run in DEFAULT_RUNS if selected is None or run.model_id in selected]
    if selected:
        missing = selected - {run.model_id for run in DEFAULT_RUNS}
        if missing:
            raise ValueError(f"unknown model ids: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for run in runs:
        missing_path = _missing_required_path(run)
        run_dir = args.output_dir / run.model_id
        csv_path = run_dir / "profile.csv"
        summary_path = run_dir / "summary.json"
        cmd = _build_cmd(args, run, csv_path, summary_path)
        if missing_path and args.skip_missing:
            results.append(_result(run, "skipped", summary_path, csv_path, f"missing {missing_path}"))
            continue
        if args.dry_run:
            results.append(_result(run, "dry-run", summary_path, csv_path, " ".join(cmd)))
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            results.append(_result(run, "failed", summary_path, csv_path, f"exit {exc.returncode}: {' '.join(cmd)}"))
            continue
        results.append(_result(run, "ok", summary_path, csv_path, ""))
    return results


def _build_cmd(args: argparse.Namespace, run: CocoRun, csv_path: Path, summary_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sam_backend.profile_coco",
        "--model-id",
        run.model_id,
        "--backend",
        run.backend,
        "--device",
        args.device,
        "--manifest",
        str(args.manifest),
    ]
    limit = getattr(args, "limit", 0)
    if limit and limit > 0:
        cmd.extend(["--limit", str(limit)])
    cmd.extend(
        [
            "--prompt-mode",
            run.prompt_mode,
            "--eval-mode",
            getattr(args, "eval_mode", "both"),
            "--csv-output",
            str(csv_path),
            "--summary-output",
            str(summary_path),
        ]
    )
    if run.checkpoint_path:
        cmd.extend(["--checkpoint-path", run.checkpoint_path])
    if run.model_config:
        cmd.extend(["--model-config", run.model_config])
    if run.external_repo:
        cmd.extend(["--external-repo", run.external_repo])
    if args.overlay_dir:
        cmd.extend(["--overlay-dir", str(args.overlay_dir / run.model_id)])
    cmd.extend(run.extra_args)
    return cmd


def _missing_required_path(run: CocoRun) -> str | None:
    for value in (run.checkpoint_path, run.external_repo):
        if value and not Path(value).exists():
            return value
    return None


def _result(run: CocoRun, status: str, summary_path: Path, csv_path: Path, message: str) -> dict[str, str]:
    return {
        "model_id": run.model_id,
        "backend": run.backend,
        "prompt_mode": run.prompt_mode,
        "status": status,
        "summary_json": str(summary_path),
        "csv": str(csv_path),
        "message": message,
    }


def write_component_summary(output_dir: Path) -> Path | None:
    rows = []
    for profile_csv in sorted(output_dir.glob("*/profile.csv")):
        with profile_csv.open(newline="", encoding="utf-8") as f:
            profile_rows = list(csv.DictReader(f))
        by_prompt: dict[str, list[dict[str, str]]] = {}
        for row in profile_rows:
            by_prompt.setdefault(row.get("prompt_mode", ""), []).append(row)
        for prompt_mode, prompt_rows in sorted(by_prompt.items()):
            first = prompt_rows[0]
            total_ms = _mean(prompt_rows, "total_ms")
            row = {
                "model_id": first.get("model_id", profile_csv.parent.name),
                "backend": first.get("backend", ""),
                "prompt_mode": prompt_mode,
                "rows": len(prompt_rows),
                "samples": len({row.get("sample_id", "") for row in prompt_rows}),
                "mean_total_ms": total_ms,
                "effective_fps": 1000.0 / total_ms if isinstance(total_ms, float) and total_ms > 0 else "",
                "miou_best": _mean(prompt_rows, "best_iou"),
                "miou_merged": _mean(prompt_rows, "merged_iou"),
                "mean_cuda_peak_allocated_mb": _mean(prompt_rows, "cuda_peak_allocated_mb"),
                "mean_cuda_peak_reserved_mb": _mean(prompt_rows, "cuda_peak_reserved_mb"),
            }
            for field_name in COMPONENT_FIELDS:
                row[f"mean_{field_name}"] = _mean(prompt_rows, field_name)
            for field_name in PARAM_WEIGHT_FIELDS:
                row[field_name] = first.get(field_name, "")
            row.update(_readable_param_weight_fields(row))
            rows.append(row)

    if not rows:
        return None

    path = output_dir / "coco_suite_component_summary.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mean(rows: list[dict[str, str]], key: str) -> float | str:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return sum(values) / len(values) if values else ""


def _readable_param_weight_fields(row: dict[str, object]) -> dict[str, float | str]:
    fields: dict[str, float | str] = {}
    for key, value in list(row.items()):
        if key.startswith("params_"):
            fields[f"{key}_m"] = _numeric(value) / 1_000_000.0 if _numeric(value) is not None else ""
        elif key.startswith("weight_") and key.endswith("_bytes"):
            fields[f"{key[:-6]}_mb"] = _numeric(value) / (1024.0 * 1024.0) if _numeric(value) is not None else ""
        elif key == "checkpoint_file_bytes":
            fields["checkpoint_file_mb"] = _numeric(value) / (1024.0 * 1024.0) if _numeric(value) is not None else ""
    return fields


def _numeric(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed COCO profiling suite across the default model matrix.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Profile only the first N manifest rows; 0 means all rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="*", help="Optional subset of model ids to run.")
    parser.add_argument("--eval-mode", choices=["gt", "overlay", "both", "profile"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("results/coco_suite"))
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--skip-missing", action="store_true", help="Skip runs whose checkpoint or external repo is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Write the command matrix without running models.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
