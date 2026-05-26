from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


EFFICIENTSAM3_REPO = "Simon7108528/EfficientSAM3"


@dataclass(frozen=True)
class Variant:
    model_id: str
    checkpoint: str
    backbone_type: str
    model_name: str
    text_encoder_type: str
    text_context: int = 16
    text_pos_embed_table_size: int = 16
    interpolate_pos_embed: bool = False


DEFAULT_VARIANTS = [
    Variant(
        "es3p1_weak_image_weak_text",
        "stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt",
        "efficientvit",
        "b0",
        "MobileCLIP-S0",
    ),
    Variant(
        "es3p1_strong_image_weak_text",
        "stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
        "efficientvit",
        "b2",
        "MobileCLIP-S0",
    ),
    Variant(
        "es3_weak_image_strong_available_text",
        "stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
        "efficientvit",
        "b0",
        "MobileCLIP-S1",
        text_pos_embed_table_size=77,
    ),
    Variant(
        "es3_strong_image_strong_available_text",
        "stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
        "efficientvit",
        "b2",
        "MobileCLIP-S1",
        text_pos_embed_table_size=77,
    ),
]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in DEFAULT_VARIANTS:
        checkpoint_path = hf_hub_download(
            repo_id=EFFICIENTSAM3_REPO,
            filename=variant.checkpoint,
            local_dir=args.checkpoint_dir,
        )
        for video in args.videos:
            csv_path = args.output_dir / f"{variant.model_id}-{video.stem}.csv"
            overlay_path = args.overlay_dir / f"{variant.model_id}-{video.stem}.mp4"
            cmd = [
                sys.executable,
                "-m",
                "sam_backend.profile_video",
                "--model-id",
                variant.model_id,
                "--backend",
                "efficientsam3",
                "--checkpoint-path",
                checkpoint_path,
                "--device",
                args.device,
                "--backbone-type",
                variant.backbone_type,
                "--model-name",
                variant.model_name,
                "--text-encoder-type",
                variant.text_encoder_type,
                "--text-encoder-context-length",
                str(variant.text_context),
                "--text-encoder-pos-embed-table-size",
                str(variant.text_pos_embed_table_size),
                "--prompt",
                args.prompt,
                "--video",
                str(video),
                "--max-frames",
                str(args.max_frames),
                "--frame-stride",
                str(args.frame_stride),
                "--csv-output",
                str(csv_path),
                "--overlay-output",
                str(overlay_path),
            ]
            if variant.interpolate_pos_embed:
                cmd.append("--interpolate-pos-embed")
            subprocess.run(cmd, check=True)
            rows.append(summarize_csv(csv_path, variant, video, overlay_path))

    summary_path = args.output_dir / "efficientsam3_variant_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(summary_path)


def summarize_csv(csv_path: Path, variant: Variant, video: Path, overlay_path: Path) -> dict[str, str | float | int]:
    totals = []
    image = []
    text = []
    grounding = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            totals.append(float(row["total_ms"]))
            image.append(float(row["image_encoder_ms"]))
            text.append(float(row["text_encoder_ms"]))
            grounding.append(float(row["grounding_ms"]))
            params_total = int(row["params_total"])
            params_image = int(row["params_image_encoder"])
            params_text = int(row["params_text_encoder"])
    return {
        "model_id": variant.model_id,
        "video": str(video),
        "frames": len(totals),
        "mean_total_ms": sum(totals) / len(totals) if totals else 0.0,
        "mean_image_encoder_ms": sum(image) / len(image) if image else 0.0,
        "mean_text_encoder_ms": sum(text) / len(text) if text else 0.0,
        "mean_grounding_ms": sum(grounding) / len(grounding) if grounding else 0.0,
        "params_total": params_total if totals else 0,
        "params_image_encoder": params_image if totals else 0,
        "params_text_encoder": params_text if totals else 0,
        "csv": str(csv_path),
        "overlay": str(overlay_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run default EfficientSAM3 2x2 variants.")
    parser.add_argument("--videos", type=Path, nargs="+", default=[Path("videos/test1.mov"), Path("videos/test2.mov")])
    parser.add_argument("--prompt", default="monitor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/efficientsam3_variants"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("overlays/efficientsam3_variants"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
