"""Compare PyTorch and TensorRT TinyViT encoders through the same SAM2-L decoder."""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import sys
import threading
import time
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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sequential-frames", action="store_true")
    parser.add_argument("--decode-warmup", type=int, default=0)
    parser.add_argument("--stream-throughput-frames", type=int, default=0)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.999)
    parser.add_argument("--minimum-mean-mask-iou", type=float, default=0.0)
    parser.add_argument(
        "--trt-position-mode",
        choices=("full", "shape-only"),
        default="full",
        help="shape-only skips image positional values that SAM2ImagePredictor discards",
    )
    parser.add_argument("--trt-native-outputs", action="store_true")
    parser.add_argument("--trt-gpu-preprocess", action="store_true")
    parser.add_argument("--trt-gpu-preprocess-fp16", action="store_true")
    parser.add_argument("--decoder-autocast-fp16", action="store_true")
    parser.add_argument("--cache-prompts", action="store_true")
    parser.add_argument(
        "--compile-components",
        choices=("none", "reduce-overhead", "max-autotune"),
        default="none",
    )
    parser.add_argument(
        "--mask-transfer",
        choices=("all-logits", "binary-only"),
        default="all-logits",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.trt_native_outputs and not args.decoder_autocast_fp16:
        parser.error("--trt-native-outputs requires --decoder-autocast-fp16")
    if args.trt_gpu_preprocess_fp16 and not args.trt_gpu_preprocess:
        parser.error("--trt-gpu-preprocess-fp16 requires --trt-gpu-preprocess")
    return args


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


def _patch_tensorrt_student(model, engine, native_outputs: bool, position_mode: str):
    position_encoding = model.image_encoder.neck.position_encoding

    def forward_image(self, image):
        outputs = engine(image)
        backbone_fpn = [outputs[name] for name in OUTPUT_NAMES]
        if not native_outputs:
            backbone_fpn = [feature.float() for feature in backbone_fpn]
        if position_mode == "shape-only":
            vision_pos_enc = backbone_fpn
        else:
            vision_pos_enc = [
                position_encoding(feature).float() for feature in backbone_fpn
            ]
        return {
            "vision_features": backbone_fpn[-1],
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
        }

    model.forward_image = types.MethodType(forward_image, model)


def _read_frames(
    path: str, count: int, sequential: bool, decode_warmup: int
) -> tuple[list[np.ndarray], list[float]]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), count)
    indices = (
        np.arange(count, dtype=np.int64)
        if sequential
        else np.linspace(0, total - 1, count, dtype=np.int64)
    )
    frames = []
    decode_ms = []
    if sequential:
        for _ in range(decode_warmup):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("failed to decode warmup frame")
    for index in indices:
        if not sequential:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to decode video frame {index}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        decode_ms.append((time.perf_counter() - started) * 1000.0)
    capture.release()
    return frames, decode_ms


def _set_image(
    predictor,
    frame: np.ndarray,
    torch,
    gpu_preprocess: bool,
    gpu_preprocess_fp16: bool,
) -> None:
    if not gpu_preprocess:
        predictor.set_image(frame)
        return

    import torch.nn.functional as F

    predictor.reset_predictor()
    predictor._orig_hw = [frame.shape[:2]]
    input_image = torch.from_numpy(frame).to(device=predictor.device)
    input_dtype = torch.float16 if gpu_preprocess_fp16 else torch.float32
    input_image = (
        input_image.permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=input_dtype)
        .div_(255.0)
    )
    input_image = F.interpolate(
        input_image,
        size=(predictor.model.image_size, predictor.model.image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if not hasattr(predictor, "_gpu_normalize"):
        predictor._gpu_normalize = (
            input_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            input_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )
    mean, std = predictor._gpu_normalize
    input_image.sub_(mean).div_(std)

    backbone_out = predictor.model.forward_image(input_image)
    _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
    if predictor.model.directly_add_no_mem_embed:
        no_mem_embed = predictor.model.no_mem_embed.to(dtype=vision_feats[-1].dtype)
        vision_feats[-1] = vision_feats[-1] + no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(
            vision_feats[::-1], predictor._bb_feat_sizes[::-1], strict=True
        )
    ][::-1]
    predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    predictor._is_image_set = True


def _predict_mask(
    predictor,
    prompt: dict,
    torch,
    decoder_autocast_fp16: bool,
    mask_transfer: str,
    cache_prompts: bool,
) -> np.ndarray:
    if getattr(predictor.model, "_tinyvit_components_compiled", False):
        torch.compiler.cudagraph_mark_step_begin()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if decoder_autocast_fp16
        else contextlib.nullcontext()
    )
    with autocast:
        if mask_transfer == "all-logits":
            masks, _, _ = predictor.predict(
                **prompt,
                multimask_output=False,
                return_logits=True,
            )
            return masks[0].astype(np.float32, copy=False)

        prompt_values = (
            prompt.get("point_coords"),
            prompt.get("point_labels"),
            prompt.get("box"),
        )
        prepared = None
        if cache_prompts:
            prompt_key = tuple(
                None
                if value is None
                else (value.shape, value.dtype.str, value.tobytes())
                for value in prompt_values
            ) + tuple(predictor._orig_hw[0])
            prompt_cache = getattr(predictor, "_tinyvit_prompt_cache", {})
            prepared = prompt_cache.get(prompt_key)
        if prepared is None:
            prepared = predictor._prep_prompts(*prompt_values, None, True)
            if cache_prompts:
                prompt_cache[prompt_key] = prepared
                predictor._tinyvit_prompt_cache = prompt_cache
        mask_input, coords, labels, box = prepared
        masks, _, _ = predictor._predict(
            coords,
            labels,
            box,
            mask_input,
            multimask_output=False,
            return_logits=False,
        )
    return masks[0, 0].detach().cpu().numpy()


def _predict(
    predictor,
    frame: np.ndarray,
    torch,
    *,
    gpu_preprocess: bool = False,
    gpu_preprocess_fp16: bool = False,
    decoder_autocast_fp16: bool = False,
    mask_transfer: str = "all-logits",
    cache_prompts: bool = False,
    prompt_names: tuple[str, ...] = ("point", "box"),
) -> tuple[dict[str, np.ndarray], dict]:
    height, width = frame.shape[:2]
    torch.cuda.synchronize()
    started = time.perf_counter()
    _set_image(
        predictor,
        frame,
        torch,
        gpu_preprocess,
        gpu_preprocess_fp16,
    )
    torch.cuda.synchronize()
    set_image_ms = (time.perf_counter() - started) * 1000.0
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
    prompt_ms = {}
    for name in prompt_names:
        prompt = prompts[name]
        torch.cuda.synchronize()
        started = time.perf_counter()
        results[name] = _predict_mask(
            predictor,
            prompt,
            torch,
            decoder_autocast_fp16,
            mask_transfer,
            cache_prompts,
        )
        torch.cuda.synchronize()
        prompt_ms[name] = (time.perf_counter() - started) * 1000.0
    return results, {"set_image_ms": set_image_ms, "prompt_ms": prompt_ms}


def _stream_once(
    predictor,
    video: str,
    frame_count: int,
    decode_warmup: int,
    torch,
    *,
    threaded_decode: bool,
    gpu_preprocess: bool,
    gpu_preprocess_fp16: bool,
    decoder_autocast_fp16: bool,
    mask_transfer: str,
    cache_prompts: bool,
) -> dict[str, float]:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    for _ in range(decode_warmup):
        ok, _ = capture.read()
        if not ok:
            raise RuntimeError("failed to decode stream warmup frame")

    def consume(frame):
        _predict(
            predictor,
            frame,
            torch,
            gpu_preprocess=gpu_preprocess,
            gpu_preprocess_fp16=gpu_preprocess_fp16,
            decoder_autocast_fp16=decoder_autocast_fp16,
            mask_transfer=mask_transfer,
            cache_prompts=cache_prompts,
            prompt_names=("point",),
        )

    started = time.perf_counter()
    if not threaded_decode:
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to decode sequential stream frame")
            consume(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        frame_queue: queue.Queue = queue.Queue(maxsize=2)
        errors = []

        def produce() -> None:
            try:
                for _ in range(frame_count):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("failed to decode threaded stream frame")
                    frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as error:
                errors.append(error)
            finally:
                frame_queue.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            consume(frame)
        producer.join()
        if errors:
            raise errors[0]
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    capture.release()
    return {
        "frames": frame_count,
        "elapsed_ms": elapsed_ms,
        "fps": 1000.0 * frame_count / elapsed_ms,
    }


def _stream_throughput(predictor, args, torch) -> dict | None:
    if args.stream_throughput_frames <= 0:
        return None
    settings = {
        "gpu_preprocess": args.trt_gpu_preprocess,
        "gpu_preprocess_fp16": args.trt_gpu_preprocess_fp16,
        "decoder_autocast_fp16": args.decoder_autocast_fp16,
        "mask_transfer": args.mask_transfer,
        "cache_prompts": args.cache_prompts,
    }
    sequential = _stream_once(
        predictor,
        args.video,
        args.stream_throughput_frames,
        args.decode_warmup,
        torch,
        threaded_decode=False,
        **settings,
    )
    threaded = _stream_once(
        predictor,
        args.video,
        args.stream_throughput_frames,
        args.decode_warmup,
        torch,
        threaded_decode=True,
        **settings,
    )
    return {
        "sequential": sequential,
        "threaded_decode": threaded,
        "threaded_speedup": threaded["fps"] / sequential["fps"],
    }


def _compile_components(model, torch, mode: str) -> None:
    if mode == "none":
        return
    model.sam_prompt_encoder.forward = torch.compile(
        model.sam_prompt_encoder.forward,
        mode=mode,
        fullgraph=True,
        dynamic=False,
    )
    model.sam_mask_decoder.forward = torch.compile(
        model.sam_mask_decoder.forward,
        mode=mode,
        fullgraph=True,
        dynamic=False,
    )
    model._tinyvit_components_compiled = True


def _timing_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "fps": float(1000.0 / array.mean()),
    }


def _backend_timing(rows: list[dict], decode_ms: list[float]) -> dict:
    set_image = [row["set_image_ms"] for row in rows]
    point = [row["prompt_ms"]["point"] for row in rows]
    box = [row["prompt_ms"]["box"] for row in rows]
    point_model = [left + right for left, right in zip(set_image, point, strict=True)]
    point_e2e = [
        decode + model for decode, model in zip(decode_ms, point_model, strict=True)
    ]
    two_prompt_e2e = [
        decode + image + point_ms + box_ms
        for decode, image, point_ms, box_ms in zip(
            decode_ms, set_image, point, box, strict=True
        )
    ]
    return {
        "set_image": _timing_summary(set_image),
        "point_prompt": _timing_summary(point),
        "box_prompt": _timing_summary(box),
        "point_model_pipeline": _timing_summary(point_model),
        "point_end_to_end": _timing_summary(point_e2e),
        "two_prompt_end_to_end": _timing_summary(two_prompt_e2e),
    }


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

    frames, decode_ms = _read_frames(
        args.video, args.frames, args.sequential_frames, args.decode_warmup
    )
    pytorch_model = _build_model(args, torch)
    _patch_pytorch_student(pytorch_model, args, torch)
    pytorch_predictor = SAM2ImagePredictor(pytorch_model)
    for _ in range(args.warmup):
        _predict(pytorch_predictor, frames[0], torch)
    reference_rows = [_predict(pytorch_predictor, frame, torch) for frame in frames]
    references = [row[0] for row in reference_rows]
    pytorch_timing_rows = [row[1] for row in reference_rows]
    del pytorch_predictor, pytorch_model
    torch.cuda.empty_cache()

    tensorrt_model = _build_model(args, torch)
    engine = TensorRtEncoder(args.engine, torch, trt)
    _patch_tensorrt_student(
        tensorrt_model,
        engine,
        native_outputs=args.trt_native_outputs,
        position_mode=args.trt_position_mode,
    )
    _compile_components(tensorrt_model, torch, args.compile_components)
    tensorrt_predictor = SAM2ImagePredictor(tensorrt_model)
    for _ in range(args.warmup):
        _predict(
            tensorrt_predictor,
            frames[0],
            torch,
            gpu_preprocess=args.trt_gpu_preprocess,
            gpu_preprocess_fp16=args.trt_gpu_preprocess_fp16,
            decoder_autocast_fp16=args.decoder_autocast_fp16,
            mask_transfer=args.mask_transfer,
            cache_prompts=args.cache_prompts,
        )

    rows = []
    tensorrt_timing_rows = []
    for frame_index, (frame, reference) in enumerate(zip(frames, references, strict=True)):
        candidate, timing_row = _predict(
            tensorrt_predictor,
            frame,
            torch,
            gpu_preprocess=args.trt_gpu_preprocess,
            gpu_preprocess_fp16=args.trt_gpu_preprocess_fp16,
            decoder_autocast_fp16=args.decoder_autocast_fp16,
            mask_transfer=args.mask_transfer,
            cache_prompts=args.cache_prompts,
        )
        tensorrt_timing_rows.append(timing_row)
        for prompt in ("point", "box"):
            difference = (
                candidate[prompt] - reference[prompt]
                if args.mask_transfer == "all-logits"
                else None
            )
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
                    "mean_abs_logit": (
                        float(np.abs(difference).mean())
                        if difference is not None
                        else None
                    ),
                    "max_abs_logit": (
                        float(np.abs(difference).max())
                        if difference is not None
                        else None
                    ),
                }
            )

    minimum_iou = min(row["binary_iou"] for row in rows)
    mean_iou = float(np.mean([row["binary_iou"] for row in rows]))
    pytorch_timing = _backend_timing(pytorch_timing_rows, decode_ms)
    tensorrt_timing = _backend_timing(tensorrt_timing_rows, decode_ms)
    stream_throughput = _stream_throughput(tensorrt_predictor, args, torch)
    report = {
        "passed": (
            minimum_iou >= args.minimum_mask_iou
            and mean_iou >= args.minimum_mean_mask_iou
        ),
        "minimum_mask_iou_required": args.minimum_mask_iou,
        "minimum_mean_mask_iou_required": args.minimum_mean_mask_iou,
        "minimum_mask_iou": minimum_iou,
        "mean_mask_iou": mean_iou,
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "engine": str(Path(args.engine).resolve()),
        "sam2_checkpoint": str(Path(args.sam2_checkpoint).resolve()),
        "video": str(Path(args.video).resolve()),
        "frames": args.frames,
        "sequential_frames": args.sequential_frames,
        "warmup": args.warmup,
        "decode_warmup": args.decode_warmup,
        "optimization": {
            "trt_position_mode": args.trt_position_mode,
            "trt_native_outputs": args.trt_native_outputs,
            "trt_gpu_preprocess": args.trt_gpu_preprocess,
            "trt_gpu_preprocess_fp16": args.trt_gpu_preprocess_fp16,
            "decoder_autocast_fp16": args.decoder_autocast_fp16,
            "compile_components": args.compile_components,
            "mask_transfer": args.mask_transfer,
            "cache_prompts": args.cache_prompts,
        },
        "decode": _timing_summary(decode_ms),
        "stream_throughput": stream_throughput,
        "timing": {
            "pytorch": pytorch_timing,
            "tensorrt": tensorrt_timing,
            "point_model_speedup": (
                pytorch_timing["point_model_pipeline"]["mean_ms"]
                / tensorrt_timing["point_model_pipeline"]["mean_ms"]
            ),
            "point_end_to_end_speedup": (
                pytorch_timing["point_end_to_end"]["mean_ms"]
                / tensorrt_timing["point_end_to_end"]["mean_ms"]
            ),
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
