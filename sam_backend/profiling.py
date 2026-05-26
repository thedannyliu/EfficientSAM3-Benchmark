from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterator


@dataclass(slots=True)
class ComponentProfile:
    image_encoder_ms: float = 0.0
    text_encoder_ms: float = 0.0
    grounding_ms: float = 0.0


def parameter_counts(model: Any) -> dict[str, int]:
    def count(module: Any) -> int:
        if module is None or not hasattr(module, "parameters"):
            return 0
        return sum(param.numel() for param in module.parameters())

    backbone = getattr(model, "backbone", None)
    return {
        "params_total": count(model),
        "params_backbone": count(backbone),
        "params_image_encoder": count(getattr(backbone, "vision_backbone", None)),
        "params_text_encoder": count(getattr(backbone, "language_backbone", None)),
        "params_transformer": count(getattr(model, "transformer", None)),
        "params_geometry_encoder": count(getattr(model, "geometry_encoder", None)),
        "params_segmentation_head": count(getattr(model, "segmentation_head", None)),
    }


@contextmanager
def component_timer(model: Any, torch_module: Any) -> Iterator[dict[str, float]]:
    timings = asdict(ComponentProfile())
    patches: list[tuple[Any, str, Any]] = []

    def synchronize() -> None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()

    def patch(obj: Any, attr: str, field: str) -> None:
        if obj is None or not hasattr(obj, attr):
            return
        original = getattr(obj, attr)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            synchronize()
            start = perf_counter()
            out = original(*args, **kwargs)
            synchronize()
            timings[field] += (perf_counter() - start) * 1000.0
            return out

        patches.append((obj, attr, original))
        setattr(obj, attr, wrapped)

    backbone = getattr(model, "backbone", None)
    patch(backbone, "forward_image", "image_encoder_ms")
    patch(backbone, "forward_text", "text_encoder_ms")
    patch(model, "forward_grounding", "grounding_ms")

    try:
        yield timings
    finally:
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)


def cuda_memory_mb(torch_module: Any) -> dict[str, float | None]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {
            "cuda_allocated_mb": None,
            "cuda_reserved_mb": None,
            "cuda_peak_allocated_mb": None,
            "cuda_peak_reserved_mb": None,
        }
    scale = 1024.0 * 1024.0
    return {
        "cuda_allocated_mb": cuda.memory_allocated() / scale,
        "cuda_reserved_mb": cuda.memory_reserved() / scale,
        "cuda_peak_allocated_mb": cuda.max_memory_allocated() / scale,
        "cuda_peak_reserved_mb": cuda.max_memory_reserved() / scale,
    }
