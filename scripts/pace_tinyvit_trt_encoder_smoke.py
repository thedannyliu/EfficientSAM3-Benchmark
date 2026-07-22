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
    parser.add_argument(
        "--mixed-precision-profile",
        choices=(
            "none",
            "fp16",
            "softmax_fp32",
            "norm_fp32",
            "softmax_norm_fp32",
            "attention_core_fp32",
            "matmul_softmax_fp32",
            "projection_fp32",
            "matmul_fp32",
            "conv_fp32",
            "matmul_projection_fp32",
            "conv_matmul_fp32",
        ),
        default="none",
    )
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


def _mixed_node_selected(node, profile: str, graph_outputs: set[str]) -> bool:
    is_softmax = node.op_type == "Softmax"
    is_norm = node.op_type in ("LayerNormalization", "InstanceNormalization")
    is_matmul = node.op_type in ("MatMul", "Gemm")
    is_conv = node.op_type == "Conv"
    is_projection = any(output in graph_outputs for output in node.output)
    return (
        (profile == "softmax_fp32" and is_softmax)
        or (profile == "norm_fp32" and is_norm)
        or (profile == "softmax_norm_fp32" and (is_softmax or is_norm))
        or (
            profile == "attention_core_fp32"
            and (is_softmax or "scaled_dot_product_attention" in node.name.lower())
        )
        or (profile == "matmul_softmax_fp32" and (is_matmul or is_softmax))
        or (profile == "projection_fp32" and is_projection)
        or (profile == "matmul_fp32" and is_matmul)
        or (profile == "conv_fp32" and is_conv)
        or (profile == "matmul_projection_fp32" and (is_matmul or is_projection))
        or (profile == "conv_matmul_fp32" and (is_conv or is_matmul))
    )


def _restore_selected_fp32_initializers(onnx, fp16_model, fp32_model, profile: str) -> dict:
    if profile in ("none", "fp16"):
        return {"fp32_initializer_count": 0, "fp32_initializer_bytes": 0}

    graph_outputs = {value.name for value in fp16_model.graph.output}
    fp32_initializers = {initializer.name: initializer for initializer in fp32_model.graph.initializer}
    existing_names = {initializer.name for initializer in fp16_model.graph.initializer}
    restored = {}
    restored_bytes = 0
    for node in fp16_model.graph.node:
        if not _mixed_node_selected(node, profile, graph_outputs):
            continue
        for input_index, name in enumerate(node.input):
            fp32_initializer = fp32_initializers.get(name)
            if fp32_initializer is None:
                continue
            restored_name = restored.get(name)
            if restored_name is None:
                restored_name = f"{name}__mixed_fp32_initializer"
                if restored_name in existing_names:
                    raise RuntimeError(f"duplicate mixed initializer name: {restored_name}")
                restored_initializer = onnx.TensorProto()
                restored_initializer.CopyFrom(fp32_initializer)
                restored_initializer.name = restored_name
                fp16_model.graph.initializer.append(restored_initializer)
                existing_names.add(restored_name)
                restored[name] = restored_name
                restored_bytes += len(restored_initializer.raw_data)
            node.input[input_index] = restored_name
    return {
        "fp32_initializer_count": len(restored),
        "fp32_initializer_bytes": restored_bytes,
    }


def _insert_fp32_islands(onnx, model, profile: str) -> dict[str, object]:
    if profile in ("none", "fp16"):
        return {"profile": profile, "fp32_node_count": 0, "fp32_nodes": [], "casts": 0}

    inferred = onnx.shape_inference.infer_shapes(model)
    element_types = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type"):
            element_types[value.name] = tensor_type.elem_type
    for initializer in inferred.graph.initializer:
        element_types[initializer.name] = initializer.data_type

    nodes = list(model.graph.node)
    graph_outputs = {value.name for value in model.graph.output}
    selected = [_mixed_node_selected(node, profile, graph_outputs) for node in nodes]
    if not any(selected):
        raise RuntimeError(f"mixed precision profile selected no ONNX nodes: {profile}")
    consumers = {}
    for node_index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(node_index)

    new_nodes = []
    fp32_outputs = {}
    casts = 0
    fp32_nodes = []
    for node_index, node in enumerate(nodes):
        if not selected[node_index]:
            new_nodes.append(node)
            continue

        fp32_nodes.append(node.name or f"{node.op_type}_{node_index}")
        rewritten_inputs = []
        for input_index, name in enumerate(node.input):
            if not name:
                rewritten_inputs.append(name)
            elif name in fp32_outputs:
                rewritten_inputs.append(fp32_outputs[name])
            elif element_types.get(name) == onnx.TensorProto.FLOAT16:
                cast_output = f"{name}__mixed_to_fp32_{node_index}_{input_index}"
                new_nodes.append(
                    onnx.helper.make_node(
                        "Cast",
                        [name],
                        [cast_output],
                        name=f"mixed_to_fp32_{node_index}_{input_index}",
                        to=onnx.TensorProto.FLOAT,
                    )
                )
                casts += 1
                rewritten_inputs.append(cast_output)
            else:
                rewritten_inputs.append(name)
        node.input[:] = rewritten_inputs

        original_outputs = list(node.output)
        rewritten_outputs = [f"{name}__mixed_fp32" if name else name for name in original_outputs]
        node.output[:] = rewritten_outputs
        new_nodes.append(node)
        for output_index, (original, rewritten) in enumerate(
            zip(original_outputs, rewritten_outputs, strict=True)
        ):
            if not original:
                continue
            fp32_outputs[original] = rewritten
            needs_fp16 = original in graph_outputs or any(
                not selected[consumer] for consumer in consumers.get(original, [])
            )
            if needs_fp16:
                new_nodes.append(
                    onnx.helper.make_node(
                        "Cast",
                        [rewritten],
                        [original],
                        name=f"mixed_to_fp16_{node_index}_{output_index}",
                        to=onnx.TensorProto.FLOAT16,
                    )
                )
                casts += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return {
        "profile": profile,
        "fp32_node_count": len(fp32_nodes),
        "fp32_nodes": fp32_nodes,
        "casts": casts,
    }


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
    if args.mixed_precision_profile != "none" and args.precision != "fp32":
        raise ValueError("mixed precision requires an FP32 ONNX export and FP32 oracle")
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

    oracle_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    model = EncoderOutputs(model).to(device="cuda", dtype=oracle_dtype).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.manual_seed(20260722)
    image = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=oracle_dtype)
    with torch.inference_mode():
        reference = model(image)
    torch.cuda.synchronize()
    pytorch_timing = _benchmark_pytorch(model, image, args.warmup, args.runs)
    shapes = tuple(tuple(tensor.shape) for tensor in reference)
    if shapes != EXPECTED_SHAPES:
        raise RuntimeError(f"SAM2 encoder contract mismatch: expected={EXPECTED_SHAPES}, got={shapes}")

    export_options = {
        "input_names": ["image"],
        "output_names": list(OUTPUT_NAMES),
        "opset_version": 18,
        "dynamo": args.exporter == "dynamo",
        "external_data": False,
    }
    if args.exporter == "dynamo":
        export_options["verify"] = False
    fp32_onnx_path = None
    if args.mixed_precision_profile not in ("none", "fp16"):
        fp32_onnx_path = output_dir / "encoder.oracle_fp32.onnx"
        print(f"exporting {fp32_onnx_path}", flush=True)
        torch.onnx.export(model, (image,), str(fp32_onnx_path), **export_options)

    engine_precision = "fp16" if args.mixed_precision_profile != "none" else args.precision
    export_model = model
    export_image = image
    if args.mixed_precision_profile != "none":
        del model
        torch.cuda.empty_cache()
        export_student = build_stage1_student(family, model_name, None, adapter_mode)
        incompatible = export_student.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("checkpoint changed while rebuilding the FP16 export model")
        export_model = EncoderOutputs(export_student).to(
            device="cuda", dtype=torch.float16
        ).eval()
        export_image = image.to(dtype=torch.float16)
    onnx_path = output_dir / f"encoder.{engine_precision}.onnx"
    engine_path = output_dir / f"encoder.{engine_precision}.engine"
    print(f"exporting {onnx_path}", flush=True)
    torch.onnx.export(export_model, (export_image,), str(onnx_path), **export_options)
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    restored_initializers = {"fp32_initializer_count": 0, "fp32_initializer_bytes": 0}
    if fp32_onnx_path is not None:
        fp32_onnx_model = onnx.load(fp32_onnx_path, load_external_data=False)
        restored_initializers = _restore_selected_fp32_initializers(
            onnx, onnx_model, fp32_onnx_model, args.mixed_precision_profile
        )
    mixed_precision = _insert_fp32_islands(
        onnx, onnx_model, args.mixed_precision_profile
    )
    mixed_precision.update(restored_initializers)
    bfloat16_initializers_converted = 0
    if engine_precision == "bf16":
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
    actual, timing = _run_engine(engine_path, export_image, args.warmup, args.runs)

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
        parity_precision = engine_precision
        cosine_limit = {"fp32": 0.99999, "fp16": 0.9999, "bf16": 0.999}[
            parity_precision
        ]
        relative_l2_limit = {"fp32": 2.0e-3, "fp16": 1.0e-2, "bf16": 5.0e-2}[
            parity_precision
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
        "parameters": parameter_count,
        "precision": args.precision,
        "oracle_precision": args.precision,
        "engine_precision": engine_precision,
        "exporter": args.exporter,
        "allow_tf32": args.allow_tf32,
        "mixed_precision_profile": args.mixed_precision_profile,
        "mixed_precision": mixed_precision,
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
