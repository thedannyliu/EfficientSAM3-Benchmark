"""Export and validate a distilled SAM2 TinyViT image encoder with TensorRT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


OUTPUT_NAMES = ("high_res_s0", "high_res_s1", "image_embedding")
EXPECTED_SHAPES = ((1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--distill-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--exporter", choices=("dynamo", "legacy"), default="dynamo")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--builder-optimization-level", type=int, choices=range(0, 6), default=3)
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


def _network_flags(trt) -> int:
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return flags


def _torch_dtype(torch, trt, dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
    }[dtype]


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: float,
    allow_tf32: bool,
    optimization_level: int,
) -> dict[str, object]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
    if not allow_tf32 and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = optimization_level
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the encoder engine")
    engine_path.write_bytes(serialized)
    return {
        "build_seconds": build_seconds,
        "engine_bytes": engine_path.stat().st_size,
        "inputs": {
            network.get_input(index).name: list(network.get_input(index).shape)
            for index in range(network.num_inputs)
        },
        "outputs": {
            network.get_output(index).name: list(network.get_output(index).shape)
            for index in range(network.num_outputs)
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_float_initializers_to_bfloat16(onnx, model) -> int:
    """Repair Dynamo's FP32 Conv+BN-folded weights in an otherwise BF16 graph."""
    import numpy as np

    converted = 0
    for initializer in model.graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        values = onnx.numpy_helper.to_array(initializer).astype(np.float32, copy=False)
        bits = values.view(np.uint32)
        rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
        bfloat16 = ((bits + rounding) >> np.uint32(16)).astype(np.uint16)
        initializer.ClearField("float_data")
        initializer.raw_data = bfloat16.tobytes()
        initializer.data_type = onnx.TensorProto.BFLOAT16
        converted += 1
    return converted


def _benchmark_pytorch(model, image, warmup: int, runs: int) -> dict[str, float | int]:
    import numpy as np
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
    torch.cuda.synchronize()
    timings = []
    with torch.inference_mode():
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(image)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    values = np.asarray(timings, dtype=np.float64)
    return {
        "warmup": warmup,
        "runs": runs,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "fps": float(1000.0 / values.mean()),
    }


def _run_engine(engine_path: Path, image, warmup: int, runs: int):
    import numpy as np
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the encoder engine")
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    tensors = {"image": image.contiguous()}
    if not context.set_input_shape("image", tuple(image.shape)):
        raise RuntimeError(f"TensorRT rejected image shape {tuple(image.shape)}")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"unresolved TensorRT shape for {name}: {shape}")
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            tensors[name] = torch.empty(
                shape,
                dtype=_torch_dtype(torch, trt, engine.get_tensor_dtype(name)),
                device="cuda",
            )
        if not context.set_tensor_address(name, tensors[name].data_ptr()):
            raise RuntimeError(f"failed to bind TensorRT tensor {name}")

    for _ in range(warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT warmup enqueue failed")
    stream.synchronize()
    timings = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT benchmark enqueue failed")
        end.record(stream)
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    values = np.asarray(timings, dtype=np.float64)
    return tuple(tensors[name] for name in OUTPUT_NAMES), {
        "warmup": warmup,
        "runs": runs,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "fps": float(1000.0 / values.mean()),
    }


def main() -> int:
    args = _arguments()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(Path(args.distill_root).resolve()))
    import onnx
    import tensorrt as trt
    import torch
    from sam2_distill.models.stage1_checkpoint import (
        extract_state_dict,
        infer_adapter_mode,
        infer_stage1_model_name,
        infer_student_family,
    )
    from sam2_distill.models.stage1_student import build_stage1_student

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"loading {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(payload)
    model_name = infer_stage1_model_name(payload, state, fallback="")
    family = infer_student_family(payload, model_name)
    adapter_mode = infer_adapter_mode(payload, state)
    if not model_name:
        raise RuntimeError("checkpoint does not identify a TinyViT model")
    model = build_stage1_student(family, model_name, None, adapter_mode)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint is not an exact model match: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    class EncoderOutputs(torch.nn.Module):
        def __init__(self, student):
            super().__init__()
            self.student = student

        def forward(self, image):
            outputs = self.student(image)
            return outputs["high_res_s0"], outputs["high_res_s1"], outputs["image_embed"]

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    model = EncoderOutputs(model).to(device="cuda", dtype=dtype).eval()
    torch.manual_seed(20260722)
    image = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=dtype)
    with torch.inference_mode():
        reference = model(image)
    torch.cuda.synchronize()
    pytorch_timing = _benchmark_pytorch(model, image, args.warmup, args.runs)
    shapes = tuple(tuple(tensor.shape) for tensor in reference)
    if shapes != EXPECTED_SHAPES:
        raise RuntimeError(f"SAM2 encoder contract mismatch: expected={EXPECTED_SHAPES}, got={shapes}")

    onnx_path = output_dir / f"encoder.{args.precision}.onnx"
    engine_path = output_dir / f"encoder.{args.precision}.engine"
    print(f"exporting {onnx_path}", flush=True)
    export_options = {
        "input_names": ["image"],
        "output_names": list(OUTPUT_NAMES),
        "opset_version": 18,
        "dynamo": args.exporter == "dynamo",
        "external_data": False,
    }
    if args.exporter == "dynamo":
        export_options["verify"] = False
    torch.onnx.export(model, (image,), str(onnx_path), **export_options)
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    bfloat16_initializers_converted = 0
    if args.precision == "bf16":
        bfloat16_initializers_converted = _convert_float_initializers_to_bfloat16(
            onnx, onnx_model
        )
        onnx.save(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"building {engine_path}", flush=True)
    build = _build_engine(
        onnx_path,
        engine_path,
        args.workspace_gib,
        args.allow_tf32,
        args.builder_optimization_level,
    )
    torch.cuda.synchronize()
    actual, timing = _run_engine(engine_path, image, args.warmup, args.runs)

    parity = {}
    passed = True
    for name, expected, observed in zip(OUTPUT_NAMES, reference, actual, strict=True):
        expected_f32 = expected.float()
        observed_f32 = observed.float()
        difference = observed_f32 - expected_f32
        relative_l2 = float(difference.norm() / expected_f32.norm().clamp_min(1.0e-12))
        cosine = float(torch.nn.functional.cosine_similarity(
            expected_f32.flatten(), observed_f32.flatten(), dim=0
        ))
        finite = bool(torch.isfinite(observed_f32).all())
        parity[name] = {
            "shape": list(observed.shape),
            "finite": finite,
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
            "relative_l2": relative_l2,
            "cosine": cosine,
        }
        passed &= finite and tuple(observed.shape) == tuple(expected.shape)
        cosine_limit = {"fp32": 0.99999, "fp16": 0.9999, "bf16": 0.999}[args.precision]
        relative_l2_limit = {"fp32": 2.0e-3, "fp16": 1.0e-2, "bf16": 5.0e-2}[
            args.precision
        ]
        passed &= cosine >= cosine_limit
        passed &= relative_l2 <= relative_l2_limit

    report = {
        "passed": passed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_name": model_name,
        "student_family": family,
        "adapter_mode": adapter_mode,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": args.precision,
        "exporter": args.exporter,
        "allow_tf32": args.allow_tf32,
        "pytorch_oracle_tf32": False,
        "builder_optimization_level": args.builder_optimization_level,
        "seed": 20260722,
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "tensorrt": trt.__version__,
        "onnx_bytes": onnx_path.stat().st_size,
        "bfloat16_initializers_converted": bfloat16_initializers_converted,
        "build": build,
        "pytorch_timing": pytorch_timing,
        "timing": timing,
        "speedup_over_pytorch": pytorch_timing["mean_ms"] / timing["mean_ms"],
        "parity": parity,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
