"""Compare PyTorch and TensorRT TinyViT encoders through the same SAM2-L decoder."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np


OUTPUT_NAMES = ("high_res_s0", "high_res_s1", "image_embedding")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--sam2-root", required=True)
    parser.add_argument("--distill-root", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.999)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _torch_dtype(torch, trt, dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
    }[dtype]


class TensorRtEncoder:
    def __init__(self, engine_path: str, torch, trt):
        self.torch = torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_dtype = _torch_dtype(torch, trt, self.engine.get_tensor_dtype("image"))
        self.outputs = {}
        if not self.context.set_input_shape("image", (1, 3, 1024, 1024)):
            raise RuntimeError("TensorRT rejected the SAM2 image shape")
        for name in OUTPUT_NAMES:
            shape = tuple(self.context.get_tensor_shape(name))
            self.outputs[name] = torch.empty(
                shape,
                dtype=_torch_dtype(torch, trt, self.engine.get_tensor_dtype(name)),
                device="cuda",
            )
            if not self.context.set_tensor_address(name, self.outputs[name].data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT output {name}")

    def __call__(self, image):
        image = image.to(dtype=self.input_dtype).contiguous()
        if not self.context.set_tensor_address("image", image.data_ptr()):
            raise RuntimeError("failed to bind TensorRT image")
        stream = self.torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT encoder enqueue failed")
        return self.outputs


def _build_model(args, torch):
    from sam2.build_sam import build_sam2

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        args.sam2_checkpoint,
        device="cuda",
        mode="eval",
        apply_postprocessing=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _patch_pytorch_student(model, args, torch):
    from sam_backend.sam2_stage1 import patch_stage1_forward_image

    patch_stage1_forward_image(
        model,
        torch,
        sam2_distill_root=args.distill_root,
        student_checkpoint_path=args.student_checkpoint,
        sam2_checkpoint_path=args.sam2_checkpoint,
        device="cuda",
        requested_family="auto",
        requested_model_name="",
        requested_backbone_checkpoint="",
        legacy_tinyvit_checkpoint="",
        requested_adapter_mode="auto",
        fallback_model_name="",
    )


def _patch_tensorrt_student(model, engine):
    position_encoding = model.image_encoder.neck.position_encoding

    def forward_image(self, image):
        outputs = engine(image)
        backbone_fpn = [
            outputs["high_res_s0"].float(),
            outputs["high_res_s1"].float(),
            outputs["image_embedding"].float(),
        ]
        return {
            "vision_features": backbone_fpn[-1],
            "vision_pos_enc": [position_encoding(feature).float() for feature in backbone_fpn],
            "backbone_fpn": backbone_fpn,
        }

    model.forward_image = types.MethodType(forward_image, model)


def _read_frames(path: str, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), count)
    indices = np.linspace(0, total - 1, count, dtype=np.int64)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to decode video frame {index}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def _predict(predictor, frame: np.ndarray) -> dict[str, np.ndarray]:
    height, width = frame.shape[:2]
    predictor.set_image(frame)
    prompts = {
        "point": {
            "point_coords": np.asarray([[width * 0.5, height * 0.5]], dtype=np.float32),
            "point_labels": np.asarray([1], dtype=np.int32),
        },
        "box": {
            "box": np.asarray(
                [width * 0.2, height * 0.2, width * 0.8, height * 0.8], dtype=np.float32
            ),
        },
    }
    results = {}
    for name, prompt in prompts.items():
        masks, _, _ = predictor.predict(
            **prompt,
            multimask_output=False,
            return_logits=True,
        )
        results[name] = masks[0].astype(np.float32, copy=False)
    return results


def _binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = left > 0.0
    right_mask = right > 0.0
    union = np.logical_or(left_mask, right_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left_mask, right_mask).sum() / union)


def main() -> int:
    args = _arguments()
    sys.path.insert(0, str(Path(args.sam2_root).resolve()))
    sys.path.insert(0, str(Path(args.distill_root).resolve()))

    import tensorrt as trt
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    frames = _read_frames(args.video, args.frames)
    pytorch_model = _build_model(args, torch)
    _patch_pytorch_student(pytorch_model, args, torch)
    pytorch_predictor = SAM2ImagePredictor(pytorch_model)
    references = [_predict(pytorch_predictor, frame) for frame in frames]
    del pytorch_predictor, pytorch_model
    torch.cuda.empty_cache()

    tensorrt_model = _build_model(args, torch)
    engine = TensorRtEncoder(args.engine, torch, trt)
    _patch_tensorrt_student(tensorrt_model, engine)
    tensorrt_predictor = SAM2ImagePredictor(tensorrt_model)

    rows = []
    for frame_index, (frame, reference) in enumerate(zip(frames, references, strict=True)):
        candidate = _predict(tensorrt_predictor, frame)
        for prompt in ("point", "box"):
            difference = candidate[prompt] - reference[prompt]
            reference_mask = reference[prompt] > 0.0
            candidate_mask = candidate[prompt] > 0.0
            rows.append(
                {
                    "frame": frame_index,
                    "prompt": prompt,
                    "binary_iou": _binary_iou(reference[prompt], candidate[prompt]),
                    "reference_foreground_pixels": int(reference_mask.sum()),
                    "candidate_foreground_pixels": int(candidate_mask.sum()),
                    "disagreement_pixels": int(
                        np.logical_xor(reference_mask, candidate_mask).sum()
                    ),
                    "mean_abs_logit": float(np.abs(difference).mean()),
                    "max_abs_logit": float(np.abs(difference).max()),
                }
            )

    minimum_iou = min(row["binary_iou"] for row in rows)
    report = {
        "passed": minimum_iou >= args.minimum_mask_iou,
        "minimum_mask_iou_required": args.minimum_mask_iou,
        "minimum_mask_iou": minimum_iou,
        "mean_mask_iou": float(np.mean([row["binary_iou"] for row in rows])),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "engine": str(Path(args.engine).resolve()),
        "sam2_checkpoint": str(Path(args.sam2_checkpoint).resolve()),
        "video": str(Path(args.video).resolve()),
        "frames": args.frames,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
