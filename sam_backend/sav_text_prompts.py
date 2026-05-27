from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    if args.command == "init":
        payload = init_prompt_file(args.manifest, args.output, args.review_dir)
        print(json.dumps(payload, indent=2))
    elif args.command == "apply":
        summary = apply_prompt_file(args.manifest, args.prompts, args.output, allow_empty=args.allow_empty)
        print(json.dumps(summary, indent=2))
    else:
        raise ValueError(f"unknown command: {args.command}")


def init_prompt_file(manifest: Path, output: Path, review_dir: Path | None = None) -> dict[str, Any]:
    rows = _read_manifest(manifest)
    payload = {
        "source_manifest": str(manifest),
        "prompt_protocol": {
            "scope": "manual SA-V text labels for selected official object IDs",
            "rule": "Fill text_prompt with a short noun phrase for the class/appearance, and instance_hint with how to identify the selected GT object among similar instances.",
            "verification": "Text-only top-1 localization may select a different same-class instance. Use instance_hint for human review, and use GT-assisted matching only as a diagnostic metric.",
            "examples": ["person", "white car", "red sign", "overhead wire"],
        },
        "samples": [],
    }
    for row in rows:
        sample = {
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "object_id": row["object_id"],
            "prompt_frame_index": row["prompt_frame_index"],
            "point": row["point"],
            "initial_mask_area_ratio": row.get("initial_mask_area_ratio"),
            "initial_bbox_xyxy": row.get("initial_bbox_xyxy"),
            "text_prompt": "",
            "instance_hint": "",
            "review_note": "",
        }
        if review_dir is not None:
            sample["review_overlay"] = str(review_dir / f"{row['sample_id']}.png")
        payload["samples"].append(sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def apply_prompt_file(manifest: Path, prompts: Path, output: Path, allow_empty: bool = False) -> dict[str, Any]:
    rows = _read_manifest(manifest)
    prompt_payload = json.loads(prompts.read_text(encoding="utf-8"))
    by_sample = {sample["sample_id"]: sample for sample in prompt_payload.get("samples", [])}
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in by_sample]
    if missing:
        raise ValueError(f"missing text prompt samples: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    empty = []
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            sample = by_sample[row["sample_id"]]
            text_prompt = str(sample.get("text_prompt", "")).strip()
            if not text_prompt:
                empty.append(row["sample_id"])
            if text_prompt or allow_empty:
                row = {
                    **row,
                    "text_prompt": text_prompt,
                    "text_prompt_instance_hint": str(sample.get("instance_hint", "")).strip(),
                    "text_prompt_source": str(prompts),
                    "text_prompt_review_note": str(sample.get("review_note", "")).strip(),
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1

    if empty and not allow_empty:
        output.unlink(missing_ok=True)
        raise ValueError(f"empty text_prompt values: {', '.join(empty)}")

    return {
        "source_manifest": str(manifest),
        "prompts": str(prompts),
        "output_manifest": str(output),
        "rows": written,
        "empty_text_prompts": empty,
    }


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or apply manual text prompts for fixed SA-V object IDs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a JSON file to fill with manual text prompts.")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--review-dir", type=Path)

    apply = subparsers.add_parser("apply", help="Merge filled manual text prompts back into a JSONL manifest.")
    apply.add_argument("--manifest", type=Path, required=True)
    apply.add_argument("--prompts", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
