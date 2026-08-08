from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .backends import BackendConfig, Prompt, create_backend


def main() -> None:
    args = parse_args()
    samples = _read_jsonl(args.samples)
    prompts = _load_prompts(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    init_started = time.perf_counter()
    if args.backend == "original-node":
        predictor = _OriginalNodePredictor(args.scene_graph_source, prompts)
    else:
        predictor = create_backend(
            BackendConfig(
                backend="instinctsam-http",
                runtime_url=args.runtime_url,
                runtime_timeout=args.runtime_timeout,
            )
        )
    model_init_ms = (time.perf_counter() - init_started) * 1000.0

    profile_path = args.output_dir / "profile.jsonl"
    sequence_started = time.perf_counter()
    with profile_path.open("w", encoding="utf-8") as profile_file:
        for sample in samples:
            image_path = _image_path(sample, args.image_dir)
            image = Image.open(image_path).convert("RGB")
            started_wall_ns = time.time_ns()
            started = time.perf_counter()
            if args.backend == "original-node":
                masks, labels, scores, metadata = predictor.predict(image)
            else:
                prediction = predictor.predict(np.asarray(image), Prompt(texts=prompts))
                masks = np.asarray(prediction.masks, dtype=bool)
                labels = np.asarray(prediction.metadata.get("labels", []), dtype=np.str_)
                scores = np.asarray(prediction.scores, dtype=np.float32)
                metadata = prediction.metadata
            total_ms = (time.perf_counter() - started) * 1000.0
            source_stamp_ns = int(sample["source_stamp_ns"])
            output_path = predictions_dir / f"{source_stamp_ns}.npz"
            _save_masks(output_path, masks, labels, scores)
            row = {
                "source_stamp_ns": source_stamp_ns,
                "image": str(image_path),
                "prediction": str(output_path),
                "prompt_count": len(prompts),
                "mask_count": int(len(masks)),
                "total_ms": total_ms,
                "started_wall_ns": started_wall_ns,
                "ended_wall_ns": time.time_ns(),
                "metadata": metadata,
            }
            profile_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            profile_file.flush()

    close = getattr(predictor, "close", None)
    if close is not None:
        close()
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "frames": len(samples),
                "prompts": prompts,
                "model_init_ms": model_init_ms,
                "sequence_wall_seconds": time.perf_counter() - sequence_started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class _OriginalNodePredictor:
    def __init__(self, scene_graph_source: Path, prompts: list[str]) -> None:
        import rclpy
        import torch

        sys.path.insert(0, str(scene_graph_source / "src"))
        from detection_ros_node import Detection

        rclpy.init()
        self.rclpy = rclpy
        self.torch = torch
        self.node = Detection()
        self.prompts = prompts

    def predict(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.bfloat16
        ):
            state = self.node.processor.set_image(image)
            results = self.node._grounding_batched(state, self.prompts)
        self.torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - started) * 1000.0

        masks = []
        labels = []
        scores = []
        for prompt, result in zip(self.prompts, results):
            for index, mask in enumerate(result.get("masks") or []):
                masks.append(np.asarray(mask, dtype=bool))
                labels.append(prompt)
                values = result.get("scores") or []
                scores.append(float(values[index]) if index < len(values) else 0.0)
        shape = (0, image.height, image.width) if not masks else None
        mask_array = np.empty(shape, dtype=bool) if shape else np.stack(masks)
        return (
            mask_array,
            np.asarray(labels, dtype=np.str_),
            np.asarray(scores, dtype=np.float32),
            {"runtime_ms": runtime_ms},
        )

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def _save_masks(
    path: Path, masks: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> None:
    masks = np.asarray(masks, dtype=bool)
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        mask_shape=np.asarray(masks.shape, dtype=np.int64),
        masks_packed=np.packbits(masks.reshape(-1), bitorder="little"),
        bitorder=np.asarray("little"),
        labels=np.asarray(labels, dtype=np.str_),
        scores=np.asarray(scores, dtype=np.float32),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_prompts(path: Path) -> list[str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    objects = list(
        dict.fromkeys(
            config["movable_types"]
            + config["container_types"]
            + config["surface_types"]
        )
    )
    return [value.replace("_", " ") for value in objects]


def _image_path(sample: dict[str, Any], image_dir: Path | None) -> Path:
    source = Path(sample["image"])
    return image_dir / source.name if image_dir is not None else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump source-aligned masks for Scene Graph detector comparison."
    )
    parser.add_argument("--backend", choices=["original-node", "instinctsam-http"], required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-graph-source", type=Path, default=Path("/workspace/src/scene_graph"))
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8767")
    parser.add_argument("--runtime-timeout", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
