from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SCRATCH_ROOT = Path("/storage/scratch1/9/eliu354/efficientsam3-benchmark")
DEFAULT_TINYVIT21_REPO = "/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/EfficientSam3-Distillation"
DEFAULT_TINYVIT21_CHECKPOINT = str(
    Path(DEFAULT_TINYVIT21_REPO) / "efficient_sam3_tinyvit21_stage1_e32_h200_full_sam3.pt"
)
EFFICIENTSAM3_TINYVIT21_REPO = os.environ.get("EFFICIENTSAM3_TINYVIT21_REPO", DEFAULT_TINYVIT21_REPO)
EFFICIENTSAM3_TINYVIT21_CHECKPOINT = os.environ.get(
    "EFFICIENTSAM3_TINYVIT21_CHECKPOINT",
    DEFAULT_TINYVIT21_CHECKPOINT,
)


@dataclass(frozen=True)
class SacoStreamRun:
    model_id: str
    backend: str
    stream_mode: str
    prompt_type: str = "auto"
    checkpoint_path: str | None = None
    model_config: str | None = None
    external_repo: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


VIDEO_RUNS = [
    SacoStreamRun(
        "mobilesam_vit_t_bbox_chain",
        "mobilesam",
        "bbox_chain",
        checkpoint_path="checkpoints/mobilesam/mobile_sam.pt",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_t"),
    ),
    SacoStreamRun(
        "sam1_vit_b_bbox_chain",
        "sam1",
        "bbox_chain",
        checkpoint_path="checkpoints/sam1/sam_vit_b_01ec64.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_b"),
    ),
    SacoStreamRun(
        "sam1_vit_l_bbox_chain",
        "sam1",
        "bbox_chain",
        checkpoint_path="checkpoints/sam1/sam_vit_l_0b3195.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_l"),
    ),
    SacoStreamRun(
        "sam1_vit_h_bbox_chain",
        "sam1",
        "bbox_chain",
        checkpoint_path="checkpoints/sam1/sam_vit_h_4b8939.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_h"),
    ),
    SacoStreamRun(
        "sam2p1_hiera_tiny_bbox_chain",
        "sam2",
        "bbox_chain",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_tiny.pt",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        external_repo="external/sam2",
    ),
    SacoStreamRun(
        "sam2p1_hiera_tiny_native",
        "sam2",
        "native_video",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_tiny.pt",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        external_repo="external/sam2",
    ),
    SacoStreamRun(
        "sam2p1_hiera_large_bbox_chain",
        "sam2",
        "bbox_chain",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        external_repo="external/sam2",
    ),
    SacoStreamRun(
        "sam2p1_hiera_large_native",
        "sam2",
        "native_video",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        external_repo="external/sam2",
    ),
    SacoStreamRun(
        "sam3_ref_text_bbox_chain",
        "sam3",
        "text_bbox_chain",
        prompt_type="text",
        checkpoint_path="checkpoints/sam3/sam3.pt",
        external_repo="external/sam3",
    ),
    SacoStreamRun(
        "sam3_ref_native",
        "sam3",
        "native_video",
        prompt_type="text",
        checkpoint_path="checkpoints/sam3/sam3.pt",
        external_repo="external/sam3",
    ),
    SacoStreamRun(
        "sam3p1_ref_native",
        "sam3p1",
        "native_video",
        prompt_type="text",
        checkpoint_path="checkpoints/sam3p1/sam3.1_multiplex.pt",
        external_repo="external/sam3",
    ),
    SacoStreamRun(
        "efficientsam3_ev_m_text_bbox_chain",
        "efficientsam3",
        "text_bbox_chain",
        prompt_type="text",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "efficientvit",
            "--model-name", "b1",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_rv_m_text_bbox_chain",
        "efficientsam3",
        "text_bbox_chain",
        prompt_type="text",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_repvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "repvit",
            "--model-name", "m1.1",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_tv_m_text_bbox_chain",
        "efficientsam3",
        "text_bbox_chain",
        prompt_type="text",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "tinyvit",
            "--model-name", "11m",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
]


IMAGE_PER_FRAME_RUNS = [
    SacoStreamRun(
        "mobilesam_vit_t_image_per_frame",
        "mobilesam",
        "image_per_frame",
        checkpoint_path="checkpoints/mobilesam/mobile_sam.pt",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_t"),
    ),
    SacoStreamRun(
        "sam1_vit_b_image_per_frame",
        "sam1",
        "image_per_frame",
        checkpoint_path="checkpoints/sam1/sam_vit_b_01ec64.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_b"),
    ),
    SacoStreamRun(
        "sam1_vit_l_image_per_frame",
        "sam1",
        "image_per_frame",
        checkpoint_path="checkpoints/sam1/sam_vit_l_0b3195.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_l"),
    ),
    SacoStreamRun(
        "sam1_vit_h_image_per_frame",
        "sam1",
        "image_per_frame",
        checkpoint_path="checkpoints/sam1/sam_vit_h_4b8939.pth",
        external_repo="external/MobileSAM",
        extra_args=("--mobile-sam-model-type", "vit_h"),
    ),
    SacoStreamRun(
        "sam2p1_hiera_tiny_image_per_frame",
        "sam2",
        "image_per_frame",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_tiny.pt",
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        external_repo="external/sam2",
    ),
    SacoStreamRun(
        "sam2p1_hiera_large_image_per_frame",
        "sam2",
        "image_per_frame",
        checkpoint_path="checkpoints/sam2/sam2.1_hiera_large.pt",
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        external_repo="external/sam2",
    ),
    SacoStreamRun(
        "sam3_ref_image_per_frame",
        "sam3",
        "image_per_frame",
        prompt_type="text",
        checkpoint_path="checkpoints/sam3/sam3.pt",
        external_repo="external/sam3",
    ),
    SacoStreamRun(
        "efficientsam3_ev_m_image_per_frame",
        "efficientsam3",
        "image_per_frame",
        prompt_type="text",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "efficientvit",
            "--model-name", "b1",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_rv_m_image_per_frame",
        "efficientsam3",
        "image_per_frame",
        prompt_type="text",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_repvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "repvit",
            "--model-name", "m1.1",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_tv_m_image_per_frame",
        "efficientsam3",
        "image_per_frame",
        prompt_type="text",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "tinyvit",
            "--model-name", "11m",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_tv_m_image_per_frame_point",
        "efficientsam3",
        "image_per_frame",
        prompt_type="point",
        checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt",
        external_repo="external/efficientsam3",
        extra_args=(
            "--backbone-type", "tinyvit",
            "--model-name", "11m",
            "--text-encoder-type", "MobileCLIP-S0",
            "--text-encoder-context-length", "16",
            "--text-encoder-pos-embed-table-size", "16",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_tinyvit21_image_per_frame_point",
        "efficientsam3",
        "image_per_frame",
        prompt_type="point",
        checkpoint_path=EFFICIENTSAM3_TINYVIT21_CHECKPOINT,
        external_repo=EFFICIENTSAM3_TINYVIT21_REPO,
        extra_args=(
            "--backbone-type", "tinyvit",
            "--model-name", "21m",
        ),
    ),
    SacoStreamRun(
        "efficientsam3_tinyvit21_image_per_frame_text",
        "efficientsam3",
        "image_per_frame",
        prompt_type="text",
        checkpoint_path=EFFICIENTSAM3_TINYVIT21_CHECKPOINT,
        external_repo=EFFICIENTSAM3_TINYVIT21_REPO,
        extra_args=(
            "--backbone-type", "tinyvit",
            "--model-name", "21m",
        ),
    ),
]

DEFAULT_RUNS = VIDEO_RUNS


def main() -> None:
    args = parse_args()
    rows = run_suite(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "saco_stream_suite_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_id", "backend", "stream_mode", "status", "summary_json", "csv", "pred_json", "official_eval_json", "overlay_dir", "message"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def run_suite(args: argparse.Namespace) -> list[dict[str, str]]:
    selected = {name for name in args.models} if args.models else None
    available_runs = _runs_for_mode_set(getattr(args, "mode_set", "video"))
    runs = [run for run in available_runs if selected is None or run.model_id in selected]
    if selected:
        missing = selected - {run.model_id for run in available_runs}
        if missing:
            raise ValueError(f"unknown model ids: {', '.join(sorted(missing))}")
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_dir = args.output_dir / run.model_id
        csv_path = run_dir / "frames.csv"
        summary_path = run_dir / "summary.json"
        pred_json = run_dir / "saco_veval_preds.json"
        eval_json = run_dir / "saco_veval_eval_res.json"
        overlay_dir = args.overlay_dir / run.model_id if args.overlay_dir else run_dir / "overlays"
        cmd = _build_cmd(args, run, csv_path, summary_path, pred_json, eval_json, overlay_dir)
        missing_path = _missing_required_path(args, run)
        if missing_path and args.skip_missing:
            results.append(_result(run, "skipped", summary_path, csv_path, pred_json, eval_json, overlay_dir, f"missing {missing_path}"))
            continue
        if args.dry_run:
            results.append(_result(run, "dry-run", summary_path, csv_path, pred_json, eval_json, overlay_dir, " ".join(cmd)))
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            results.append(_result(run, "failed", summary_path, csv_path, pred_json, eval_json, overlay_dir, f"exit {exc.returncode}: {' '.join(cmd)}"))
            continue
        results.append(_result(run, "ok", summary_path, csv_path, pred_json, eval_json, overlay_dir, ""))
    return results


def _runs_for_mode_set(mode_set: str) -> list[SacoStreamRun]:
    if mode_set == "video":
        return VIDEO_RUNS
    if mode_set == "image_per_frame":
        return IMAGE_PER_FRAME_RUNS
    if mode_set == "all":
        return VIDEO_RUNS + IMAGE_PER_FRAME_RUNS
    raise ValueError(f"unknown mode set: {mode_set}")


def _build_cmd(
    args: argparse.Namespace,
    run: SacoStreamRun,
    csv_path: Path,
    summary_path: Path,
    pred_json: Path,
    eval_json: Path,
    overlay_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sam_backend.profile_saco_stream",
        "--manifest",
        str(args.manifest),
        "--model-id",
        run.model_id,
        "--backend",
        run.backend,
        "--stream-mode",
        run.stream_mode,
        "--prompt-type",
        run.prompt_type,
        "--device",
        args.device,
        "--max-frames",
        str(args.max_frames),
        "--input-fps",
        str(args.input_fps),
        "--csv-output",
        str(csv_path),
        "--summary-output",
        str(summary_path),
        "--pred-json",
        str(pred_json),
        "--overlay-root",
        str(overlay_dir),
    ]
    if args.gt_annotation_file:
        cmd.extend(["--gt-annotation-file", str(args.gt_annotation_file), "--official-eval-json", str(eval_json)])
    checkpoint = _resolve_path(args, run.checkpoint_path)
    external = _resolve_path(args, run.external_repo)
    if checkpoint:
        cmd.extend(["--checkpoint-path", str(checkpoint)])
    if run.model_config:
        cmd.extend(["--model-config", run.model_config])
    if external:
        cmd.extend(["--external-repo", str(external)])
    cmd.extend(run.extra_args)
    return cmd


def _missing_required_path(args: argparse.Namespace, run: SacoStreamRun) -> str | None:
    for value in (run.checkpoint_path, run.external_repo):
        resolved = _resolve_path(args, value)
        if value and (resolved is None or not resolved.exists()):
            return str(args.scratch_root / value)
    return None


def _resolve_path(args: argparse.Namespace, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    scratch_path = args.scratch_root / path
    if scratch_path.exists():
        return scratch_path
    if path.exists():
        return path
    return scratch_path


def _result(
    run: SacoStreamRun,
    status: str,
    summary_path: Path,
    csv_path: Path,
    pred_json: Path,
    eval_json: Path,
    overlay_dir: Path,
    message: str,
) -> dict[str, str]:
    return {
        "model_id": run.model_id,
        "backend": run.backend,
        "stream_mode": run.stream_mode,
        "status": status,
        "summary_json": str(summary_path),
        "csv": str(csv_path),
        "pred_json": str(pred_json),
        "official_eval_json": str(eval_json),
        "overlay_dir": str(overlay_dir),
        "message": message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SA-Co/VEval-SAV stream benchmark suite.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-annotation-file", type=Path)
    parser.add_argument("--models", nargs="*", help="Optional subset of model IDs.")
    parser.add_argument("--mode-set", choices=["video", "image_per_frame", "all"], default="video")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/saco_stream"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("overlays/saco_stream"))
    parser.add_argument("--scratch-root", type=Path, default=Path(os.environ.get("SAM_BENCH_SCRATCH", DEFAULT_SCRATCH_ROOT)))
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
