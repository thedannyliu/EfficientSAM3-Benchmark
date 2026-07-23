"""Alternately benchmark TensorRT engines on one GPU to reduce clock-state bias."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _torch_dtype(torch, trt, dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
    }[dtype]


class EngineRunner:
    def __init__(self, path: Path, image, torch, trt):
        self.torch = torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self.context = self.engine.create_execution_context()
        if not self.context.set_input_shape("image", tuple(image.shape)):
            raise RuntimeError(f"engine rejected input shape: {path}")
        self.tensors = {"image": image}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.tensors[name] = torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    dtype=_torch_dtype(torch, trt, self.engine.get_tensor_dtype(name)),
                    device="cuda",
                )
            if not self.context.set_tensor_address(name, self.tensors[name].data_ptr()):
                raise RuntimeError(f"failed to bind {name}: {path}")

    def run(self, stream):
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")


def _summary(values, np):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p99_ms": float(np.percentile(array, 99)),
        "fps": float(1000.0 / array.mean()),
    }


def main() -> int:
    args = _arguments()
    import numpy as np
    import tensorrt as trt
    import torch

    paths = {}
    for value in args.engine:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        paths[label] = Path(path).resolve()
    image = torch.randn(1, 3, 1024, 1024, dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    runners = {label: EngineRunner(path, image, torch, trt) for label, path in paths.items()}
    labels = list(runners)
    for _ in range(args.warmup):
        for runner in runners.values():
            runner.run(stream)
    stream.synchronize()

    gpu_ms = {label: [] for label in labels}
    wall_ms = {label: [] for label in labels}
    for round_index in range(args.rounds):
        order = labels if round_index % 2 == 0 else list(reversed(labels))
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record(stream)
            runners[label].run(stream)
            end.record(stream)
            end.synchronize()
            wall_ms[label].append((time.perf_counter() - wall_start) * 1000.0)
            gpu_ms[label].append(float(start.elapsed_time(end)))

    baseline = labels[0]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "tensorrt": trt.__version__,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "engines": {},
    }
    baseline_mean = _summary(gpu_ms[baseline], np)["mean_ms"]
    for label in labels:
        gpu = _summary(gpu_ms[label], np)
        report["engines"][label] = {
            "path": str(paths[label]),
            "gpu_timing": gpu,
            "wall_timing": _summary(wall_ms[label], np),
            "speedup_over_first": baseline_mean / gpu["mean_ms"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
