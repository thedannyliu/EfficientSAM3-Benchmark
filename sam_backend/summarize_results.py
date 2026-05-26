from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


SAM3_IMAGE_ENCODER_M = 461.84
SAM3_TEXT_ENCODER_M = 353.72
SAM3_IMAGE_TEXT_M = SAM3_IMAGE_ENCODER_M + SAM3_TEXT_ENCODER_M


MODEL_CATALOG = [
    {
        "readme_model_name": "SAM3",
        "backbone": "ViT-H + SAM3 text encoder",
        "image_params_m": SAM3_IMAGE_ENCODER_M,
        "text_params_m": SAM3_TEXT_ENCODER_M,
        "notes": "Original SAM3 reference from EfficientSAM3 README parameter statements.",
    },
    {
        "readme_model_name": "SAM3-LiteText-S0-16",
        "backbone": "SAM3 ViT-H + MobileCLIP-S0",
        "image_params_m": SAM3_IMAGE_ENCODER_M,
        "text_params_m": 42.54,
        "notes": "SAM3-LiteText table.",
    },
    {
        "readme_model_name": "SAM3-LiteText-S1-16",
        "backbone": "SAM3 ViT-H + MobileCLIP-S1",
        "image_params_m": SAM3_IMAGE_ENCODER_M,
        "text_params_m": 63.53,
        "notes": "SAM3-LiteText table.",
    },
    {
        "readme_model_name": "SAM3-LiteText-L-16",
        "backbone": "SAM3 ViT-H + MobileCLIP2-L",
        "image_params_m": SAM3_IMAGE_ENCODER_M,
        "text_params_m": 123.80,
        "notes": "SAM3-LiteText table.",
    },
    {
        "readme_model_name": "ES-EV-S",
        "backbone": "EfficientViT-B0",
        "image_params_m": 0.68,
        "text_params_m": "",
        "notes": "EfficientSAM3 image encoder table.",
    },
    {
        "readme_model_name": "ES-EV-M",
        "backbone": "EfficientViT-B1",
        "image_params_m": 4.64,
        "text_params_m": "",
        "notes": "EfficientSAM3 image encoder table.",
    },
    {
        "readme_model_name": "ES-EV-L",
        "backbone": "EfficientViT-B2",
        "image_params_m": 14.98,
        "text_params_m": "",
        "notes": "EfficientSAM3 image encoder table.",
    },
    {
        "readme_model_name": "ES-EV-S-MC-S1",
        "backbone": "EfficientViT-B0 + MobileCLIP-S1",
        "image_params_m": 0.68,
        "text_params_m": 63.56,
        "notes": "EfficientSAM3 text encoder + image encoder table.",
    },
    {
        "readme_model_name": "ES-EV-M-MC-S1",
        "backbone": "EfficientViT-B1 + MobileCLIP-S1",
        "image_params_m": 4.64,
        "text_params_m": 63.56,
        "notes": "EfficientSAM3 text encoder + image encoder table.",
    },
    {
        "readme_model_name": "ES-EV-L-MC-S1",
        "backbone": "EfficientViT-B2 + MobileCLIP-S1",
        "image_params_m": 14.98,
        "text_params_m": 63.56,
        "notes": "EfficientSAM3 text encoder + image encoder table.",
    },
]


def main() -> None:
    args = parse_args()
    csv_paths = list(discover_csvs(args.inputs))
    summaries = [summarize_csv(path) for path in csv_paths]
    summaries = [row for row in summaries if row is not None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, summaries, SUMMARY_FIELDS)
    write_rows(args.catalog_output, build_catalog_rows(), CATALOG_FIELDS)

    print(args.output)
    print(args.catalog_output)
    for row in summaries:
        print(
            f"{row['model_id']}: frames={row['frames']} "
            f"mean={_fmt(row['mean_total_ms'])}ms "
            f"mean_fps={_fmt(row['mean_total_fps'])} "
            f"p50={_fmt(row['p50_total_ms'])}ms "
            f"p95={_fmt(row['p95_total_ms'])}ms "
            f"csv={row['csv']}"
        )


SUMMARY_FIELDS = [
    "model_id",
    "backend",
    "video",
    "csv",
    "overlay",
    "frames",
    "prompt",
    "mean_total_ms",
    "mean_total_fps",
    "p50_total_ms",
    "p50_total_fps",
    "p95_total_ms",
    "p95_total_fps",
    "min_total_ms",
    "max_total_fps",
    "max_total_ms",
    "min_total_fps",
    "mean_image_encoder_ms",
    "mean_image_encoder_fps",
    "mean_text_encoder_ms",
    "mean_text_encoder_fps",
    "mean_grounding_ms",
    "mean_grounding_fps",
    "mean_other_ms",
    "mean_other_fps",
    "mean_mask_count",
    "mean_score_max",
    "params_total",
    "params_total_m",
    "params_total_pct_of_sam3_image_text",
    "params_image_encoder",
    "params_image_encoder_m",
    "params_image_encoder_pct_of_sam3_image",
    "params_text_encoder",
    "params_text_encoder_m",
    "params_text_encoder_pct_of_sam3_text",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
]


CATALOG_FIELDS = [
    "readme_model_name",
    "backbone",
    "image_params_m",
    "image_params_pct_of_sam3_image",
    "text_params_m",
    "text_params_pct_of_sam3_text",
    "image_text_params_m",
    "image_text_params_pct_of_sam3_image_text",
    "notes",
]


def discover_csvs(inputs: list[Path]) -> Iterable[Path]:
    paths = inputs or [Path("results")]
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.csv") if _looks_like_result_csv(p))
        elif path.suffix.lower() == ".csv" and _looks_like_result_csv(path):
            yield path


def _looks_like_result_csv(path: Path) -> bool:
    if path.name in {"benchmark_summary.csv", "image_check_summary.csv", "model_catalog.csv"}:
        return False
    return path.is_file()


def summarize_csv(path: Path) -> dict[str, object] | None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "total_ms" not in rows[0]:
        return None

    totals = sorted(_float(row.get("total_ms")) for row in rows if _float(row.get("total_ms")) is not None)
    if not totals:
        return None

    first = rows[0]
    params_total = _float(first.get("params_total"))
    params_image = _float(first.get("params_image_encoder"))
    params_text = _float(first.get("params_text_encoder"))
    mean_total_ms = _mean(totals)
    p50_total_ms = _percentile(totals, 0.50)
    p95_total_ms = _percentile(totals, 0.95)
    min_total_ms = min(totals)
    max_total_ms = max(totals)
    mean_image_encoder_ms = _mean_field(rows, "image_encoder_ms")
    mean_text_encoder_ms = _mean_field(rows, "text_encoder_ms")
    mean_grounding_ms = _mean_field(rows, "grounding_ms")
    mean_other_ms = _mean_field(rows, "other_ms")

    return {
        "model_id": first.get("model_id") or path.stem,
        "backend": first.get("backend", ""),
        "video": first.get("video", ""),
        "csv": str(path),
        "overlay": _overlay_for(path),
        "frames": len(rows),
        "prompt": first.get("prompt", ""),
        "mean_total_ms": mean_total_ms,
        "mean_total_fps": _fps(mean_total_ms),
        "p50_total_ms": p50_total_ms,
        "p50_total_fps": _fps(p50_total_ms),
        "p95_total_ms": p95_total_ms,
        "p95_total_fps": _fps(p95_total_ms),
        "min_total_ms": min_total_ms,
        "max_total_fps": _fps(min_total_ms),
        "max_total_ms": max_total_ms,
        "min_total_fps": _fps(max_total_ms),
        "mean_image_encoder_ms": mean_image_encoder_ms,
        "mean_image_encoder_fps": _fps(mean_image_encoder_ms),
        "mean_text_encoder_ms": mean_text_encoder_ms,
        "mean_text_encoder_fps": _fps(mean_text_encoder_ms),
        "mean_grounding_ms": mean_grounding_ms,
        "mean_grounding_fps": _fps(mean_grounding_ms),
        "mean_other_ms": mean_other_ms,
        "mean_other_fps": _fps(mean_other_ms),
        "mean_mask_count": _mean_field(rows, "mask_count"),
        "mean_score_max": _mean_field(rows, "score_max"),
        "params_total": _int_or_empty(params_total),
        "params_total_m": _params_m(params_total),
        "params_total_pct_of_sam3_image_text": _pct(_params_m(params_total), SAM3_IMAGE_TEXT_M),
        "params_image_encoder": _int_or_empty(params_image),
        "params_image_encoder_m": _params_m(params_image),
        "params_image_encoder_pct_of_sam3_image": _pct(_params_m(params_image), SAM3_IMAGE_ENCODER_M),
        "params_text_encoder": _int_or_empty(params_text),
        "params_text_encoder_m": _params_m(params_text),
        "params_text_encoder_pct_of_sam3_text": _pct(_params_m(params_text), SAM3_TEXT_ENCODER_M),
        "cuda_peak_allocated_mb": max(_float(row.get("cuda_peak_allocated_mb")) or 0.0 for row in rows),
        "cuda_peak_reserved_mb": max(_float(row.get("cuda_peak_reserved_mb")) or 0.0 for row in rows),
    }


def build_catalog_rows() -> list[dict[str, object]]:
    rows = []
    for item in MODEL_CATALOG:
        image_params = _float(item["image_params_m"])
        text_params = _float(item["text_params_m"])
        combined = (image_params or 0.0) + (text_params or 0.0) if text_params is not None else image_params
        rows.append(
            {
                **item,
                "image_params_pct_of_sam3_image": _pct(image_params, SAM3_IMAGE_ENCODER_M),
                "text_params_pct_of_sam3_text": _pct(text_params, SAM3_TEXT_ENCODER_M),
                "image_text_params_m": combined,
                "image_text_params_pct_of_sam3_image_text": _pct(combined, SAM3_IMAGE_TEXT_M),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _overlay_for(csv_path: Path) -> str:
    parts = csv_path.parts
    if "results" in parts:
        idx = parts.index("results")
        relative = Path(*parts[idx + 1 :]).with_suffix(".mp4")
        overlay = Path("overlays") / relative
        if overlay.exists():
            return str(overlay)
    return ""


def _float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    return _mean([value for value in values if value is not None])


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    return sorted_values[int((len(sorted_values) - 1) * q)]


def _params_m(params: float | None) -> float | None:
    return params / 1_000_000.0 if params is not None else None


def _pct(value: float | None, reference: float) -> float | str:
    if value is None:
        return ""
    return value / reference * 100.0


def _fps(latency_ms: float | None) -> float | str:
    if latency_ms is None or latency_ms <= 0:
        return ""
    return 1000.0 / latency_ms


def _int_or_empty(value: float | None) -> int | str:
    return int(value) if value is not None else ""


def _fmt(value: object) -> str:
    number = _float(value)
    return "NA" if number is None else f"{number:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SAM3/EfficientSAM3 benchmark CSV files.")
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV files or directories containing CSV files. Defaults to results/.")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_summary.csv"))
    parser.add_argument("--catalog-output", type=Path, default=Path("results/model_catalog.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
