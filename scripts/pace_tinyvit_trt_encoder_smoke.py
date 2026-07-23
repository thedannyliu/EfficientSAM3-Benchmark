"""Export and validate a distilled SAM2 TinyViT image encoder with TensorRT."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path


OUTPUT_NAMES = ("high_res_s0", "high_res_s1", "image_embedding")
EXPECTED_SHAPES = ((1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64))
FLOAT_ONNX_TYPES = frozenset((1, 10, 16))  # FLOAT, FLOAT16, BFLOAT16


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
    parser.add_argument(
        "--quantization-mode", choices=("none", "fp8", "int8", "int4"), default="none"
    )
    parser.add_argument(
        "--quantization-op-set",
        choices=(
            "conv_matmul",
            "matmul",
            "conv",
            "attention_matmul",
            "linear_matmul",
            "backbone_conv",
            "neck_conv",
        ),
        default="conv_matmul",
    )
    parser.add_argument(
        "--quantization-scope-regex",
        action="append",
        default=[],
        help="Quantize Conv/MatMul nodes whose exported module scope matches this regex",
    )
    parser.add_argument(
        "--calibration-method",
        choices=("max", "entropy", "awq_clip", "awq_lite", "rtn_dq"),
        default="max",
    )
    parser.add_argument("--calibration-video")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--builder-optimization-level", type=int, choices=range(0, 6), default=3)
    parser.add_argument("--max-aux-streams", type=int, choices=range(0, 9))
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


def _calibration_frames(video_path: str, count: int):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open calibration video: {video_path}")
    frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), count)
    indices = np.linspace(0, frame_count - 1, count, dtype=np.int64)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to decode calibration frame {index}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - mean) / std
        frames.append(np.transpose(normalized, (2, 0, 1)).astype(np.float16))
    capture.release()
    return frames


class _CalibrationReader:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def get_next(self):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return {"image": frame[None, ...]}

    def get_first(self):
        return {"image": self.frames[0][None, ...]}

    def rewind(self):
        self.index = 0

    def __iter__(self):
        for frame in self.frames:
            yield {"image": frame[None, ...]}


def _semantic_scope(node) -> str:
    metadata = {item.key: item.value for item in getattr(node, "metadata_props", [])}
    encoded_scopes = metadata.get("pkg.torch.onnx.name_scopes")
    if encoded_scopes:
        scopes = ast.literal_eval(encoded_scopes)
        module_scopes = [scope for scope in scopes if scope.startswith("student.")]
        if module_scopes:
            return module_scopes[-1]
    scope = node.name.strip("/").replace("/", ".")
    scope = scope.replace(".blocks.blocks.", ".blocks.")
    scope = re.sub(r"\.(?:Conv|MatMul(?:_\d+)?)$", "", scope)
    for output_name in ("high_res_s0", "high_res_s1", "image_embed"):
        scope = scope.replace(
            f"student.{output_name}", f"student.projections.{output_name}"
        )
    return scope


def _quantization_selection(
    model, profile: str, scope_patterns: list[str] | None = None
) -> tuple[list[str], list[str] | None]:
    if scope_patterns:
        expressions = [re.compile(pattern) for pattern in scope_patterns]
        nodes = [
            node.name
            for node in model.graph.node
            if node.op_type in ("Conv", "MatMul")
            and any(expression.search(_semantic_scope(node)) for expression in expressions)
        ]
        if not nodes:
            raise RuntimeError(
                f"quantization scopes selected no Conv/MatMul nodes: {scope_patterns}"
            )
        return ["Conv", "MatMul"], nodes
    if profile == "conv_matmul":
        return ["Conv", "MatMul"], None
    if profile in ("matmul", "conv"):
        return [{"matmul": "MatMul", "conv": "Conv"}[profile]], None

    nodes = []
    for node in model.graph.node:
        is_attention = (
            node.op_type == "MatMul" and "scaled_dot_product_attention" in node.name
        )
        is_neck_conv = node.op_type == "Conv" and node.name.startswith("node_conv2d_")
        selected = (
            (profile == "attention_matmul" and is_attention)
            or (profile == "linear_matmul" and node.op_type == "MatMul" and not is_attention)
            or (profile == "backbone_conv" and node.op_type == "Conv" and not is_neck_conv)
            or (profile == "neck_conv" and is_neck_conv)
        )
        if selected:
            if not node.name:
                raise RuntimeError(f"quantization profile {profile} selected an unnamed node")
            nodes.append(node.name)
    if not nodes:
        raise RuntimeError(f"quantization profile selected no ONNX nodes: {profile}")
    op_type = "MatMul" if profile.endswith("matmul") else "Conv"
    return [op_type], nodes


def _network_flags(trt) -> int:
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return flags


def _mixed_node_selected(
    node, profile: str, graph_outputs: set[str], element_types: dict[str, int] | None = None
) -> bool:
    is_softmax = node.op_type == "Softmax"
    is_norm = node.op_type in ("LayerNormalization", "InstanceNormalization")
    known_input_types = [
        element_types[name] for name in node.input if element_types and name in element_types
    ]
    is_matmul = node.op_type in ("MatMul", "Gemm") and (
        element_types is None
        or (
            any(data_type in FLOAT_ONNX_TYPES for data_type in known_input_types)
            and all(data_type in FLOAT_ONNX_TYPES for data_type in known_input_types)
        )
    )
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
    inferred = onnx.shape_inference.infer_shapes(fp16_model)
    element_types = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type"):
            element_types[value.name] = tensor_type.elem_type
    for initializer in inferred.graph.initializer:
        element_types[initializer.name] = initializer.data_type
    fp16_initializers = {
        initializer.name: initializer for initializer in fp16_model.graph.initializer
    }
    fp32_initializers = {
        initializer.name: initializer for initializer in fp32_model.graph.initializer
    }
    fp32_nodes = {node.name: node for node in fp32_model.graph.node if node.name}
    existing_names = {initializer.name for initializer in fp16_model.graph.initializer}
    restored = {}
    restored_bytes = 0
    for node_index, node in enumerate(fp16_model.graph.node):
        if not _mixed_node_selected(node, profile, graph_outputs, element_types):
            continue
        fp32_node = fp32_nodes.get(node.name)
        if fp32_node is None or fp32_node.op_type != node.op_type:
            continue
        for input_index, name in enumerate(node.input):
            if input_index >= len(fp32_node.input):
                continue
            fp16_initializer = fp16_initializers.get(name)
            fp32_initializer = fp32_initializers.get(fp32_node.input[input_index])
            if (
                fp16_initializer is None
                or fp32_initializer is None
                or fp16_initializer.data_type != onnx.TensorProto.FLOAT16
                or fp32_initializer.data_type != onnx.TensorProto.FLOAT
                or tuple(fp16_initializer.dims) != tuple(fp32_initializer.dims)
            ):
                continue
            initializer_pair = (name, fp32_initializer.name)
            restored_name = restored.get(initializer_pair)
            if restored_name is None:
                restored_name = f"mixed_fp32_initializer_{node_index}_{input_index}"
                if restored_name in existing_names:
                    raise RuntimeError(f"duplicate mixed initializer name: {restored_name}")
                restored_initializer = onnx.TensorProto()
                restored_initializer.CopyFrom(fp32_initializer)
                restored_initializer.name = restored_name
                fp16_model.graph.initializer.append(restored_initializer)
                existing_names.add(restored_name)
                restored[initializer_pair] = restored_name
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
    selected = [
        _mixed_node_selected(node, profile, graph_outputs, element_types) for node in nodes
    ]
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
    max_aux_streams: int | None,
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
    if max_aux_streams is not None:
        config.max_aux_streams = max_aux_streams
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
    if args.quantization_mode != "none" and args.mixed_precision_profile != "none":
        raise ValueError("quantization and FP32-island search are separate experiments")
    if args.quantization_mode != "none" and args.precision != "fp32":
        raise ValueError("quantization requires an FP32 PyTorch oracle")
    if args.quantization_mode != "none" and not args.calibration_video:
        raise ValueError("quantization requires --calibration-video")
    if args.quantization_mode == "int4" and args.calibration_method not in (
        "awq_clip",
        "awq_lite",
        "rtn_dq",
    ):
        raise ValueError("INT4 requires an AWQ or RTN calibration method")
    if args.quantization_mode in ("fp8", "int8") and args.calibration_method not in (
        "max",
        "entropy",
    ):
        raise ValueError("FP8/INT8 requires max or entropy calibration")
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

    low_precision_export = (
        args.mixed_precision_profile != "none" or args.quantization_mode != "none"
    )
    engine_precision = "fp16" if low_precision_export else args.precision
    export_model = model
    export_image = image
    if low_precision_export:
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
    quantization = {
        "mode": args.quantization_mode,
        "op_set": args.quantization_op_set,
        "calibration_method": args.calibration_method,
        "calibration_video": args.calibration_video,
        "calibration_samples": 0,
        "selected_nodes": [],
        "selected_scopes": {},
        "quantize_linear_nodes": 0,
        "dequantize_linear_nodes": 0,
    }
    if args.quantization_mode != "none":
        from modelopt.onnx.quantization import quantize

        frames = _calibration_frames(args.calibration_video, args.calibration_samples)
        quantized_path = output_dir / f"encoder.{args.quantization_mode}.onnx"
        print(f"quantizing {quantized_path}", flush=True)
        quantize_op_types, nodes_to_quantize = _quantization_selection(
            onnx_model, args.quantization_op_set, args.quantization_scope_regex
        )
        quantize(
            str(onnx_path),
            quantize_mode=args.quantization_mode,
            calibration_data_reader=_CalibrationReader(frames),
            calibration_method=args.calibration_method,
            calibration_eps=["cuda:0", "cpu"],
            op_types_to_quantize=quantize_op_types,
            nodes_to_quantize=nodes_to_quantize,
            high_precision_dtype="fp16",
            output_path=str(quantized_path),
        )
        onnx_path = quantized_path
        engine_path = output_dir / f"encoder.{args.quantization_mode}.engine"
        quantized_model = onnx.load(onnx_path, load_external_data=False)
        onnx.checker.check_model(quantized_model)
        quantize_linear_nodes = sum(
            node.op_type == "QuantizeLinear" for node in quantized_model.graph.node
        )
        dequantize_linear_nodes = sum(
            node.op_type == "DequantizeLinear" for node in quantized_model.graph.node
        )
        if (
            args.quantization_mode in ("fp8", "int8")
            and nodes_to_quantize
            and quantize_linear_nodes == 0
        ):
            raise RuntimeError(
                "selected nodes produced no Q/DQ nodes; the requested layers are not "
                "supported by this quantizer"
            )
        quantization.update(
            {
                "calibration_samples": len(frames),
                "selected_nodes": nodes_to_quantize or [],
                "selected_scopes": {
                    node.name: _semantic_scope(node)
                    for node in onnx_model.graph.node
                    if nodes_to_quantize and node.name in nodes_to_quantize
                },
                "quantize_linear_nodes": quantize_linear_nodes,
                "dequantize_linear_nodes": dequantize_linear_nodes,
            }
        )
    print(f"building {engine_path}", flush=True)
    build = _build_engine(
        onnx_path,
        engine_path,
        args.workspace_gib,
        args.allow_tf32,
        args.builder_optimization_level,
        args.max_aux_streams,
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
        "quantization": quantization,
        "pytorch_oracle_tf32": False,
        "builder_optimization_level": args.builder_optimization_level,
        "max_aux_streams": args.max_aux_streams,
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
