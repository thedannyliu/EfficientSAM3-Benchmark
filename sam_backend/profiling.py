from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterator


@dataclass(slots=True)
class ComponentProfile:
    image_encoder_ms: float = 0.0
    text_encoder_ms: float = 0.0
    prompt_encoder_ms: float = 0.0
    mask_decoder_ms: float = 0.0
    transformer_ms: float = 0.0
    geometry_encoder_ms: float = 0.0
    segmentation_head_ms: float = 0.0
    grounding_ms: float = 0.0
    detector_ms: float = 0.0
    memory_attention_ms: float = 0.0
    memory_encoder_ms: float = 0.0


def parameter_counts(model: Any) -> dict[str, int]:
    def count(module: Any) -> int:
        if module is None or not hasattr(module, "parameters"):
            return 0
        return sum(param.numel() for param in module.parameters())

    def weight_bytes(module: Any) -> int:
        if module is None or not hasattr(module, "parameters"):
            return 0
        return sum(param.numel() * param.element_size() for param in module.parameters())

    backbone = _first_module(model, ["backbone", "detector.backbone", "tracker.backbone"])
    image_encoder = _first_module(
        model,
        [
            "backbone.vision_backbone",
            "image_encoder",
            "detector.image_encoder",
            "tracker.image_encoder",
        ],
    )
    text_encoder = _first_module(model, ["backbone.language_backbone", "detector.backbone.language_backbone"])
    transformer = _first_module(model, ["transformer", "detector.transformer"])
    geometry_encoder = _first_module(model, ["geometry_encoder", "detector.geometry_encoder"])
    segmentation_head = _first_module(model, ["segmentation_head", "detector.segmentation_head"])
    prompt_encoder = _first_module(
        model,
        [
            "sam_prompt_encoder",
            "inst_interactive_predictor.model.sam_prompt_encoder",
            "tracker.sam_prompt_encoder",
            "tracker.interactive_sam_prompt_encoder",
        ],
    )
    mask_decoder = _first_module(
        model,
        [
            "sam_mask_decoder",
            "inst_interactive_predictor.model.sam_mask_decoder",
            "tracker.sam_mask_decoder",
            "tracker.interactive_sam_mask_decoder",
        ],
    )
    detector = _first_module(model, ["detector", "detection_head"])
    memory_encoder = _first_module(model, ["memory_encoder", "tracker.memory_encoder"])
    memory_attention = _first_module(model, ["memory_attention", "tracker.memory_attention"])
    return {
        "params_total": count(model),
        "params_backbone": count(backbone),
        "params_image_encoder": count(image_encoder),
        "params_text_encoder": count(text_encoder),
        "params_transformer": count(transformer),
        "params_geometry_encoder": count(geometry_encoder),
        "params_segmentation_head": count(segmentation_head),
        "params_prompt_encoder": count(prompt_encoder),
        "params_mask_decoder": count(mask_decoder),
        "params_detector": count(detector),
        "params_memory_encoder": count(memory_encoder),
        "params_memory_attention": count(memory_attention),
        "weight_total_bytes": weight_bytes(model),
        "weight_backbone_bytes": weight_bytes(backbone),
        "weight_image_encoder_bytes": weight_bytes(image_encoder),
        "weight_text_encoder_bytes": weight_bytes(text_encoder),
        "weight_transformer_bytes": weight_bytes(transformer),
        "weight_geometry_encoder_bytes": weight_bytes(geometry_encoder),
        "weight_segmentation_head_bytes": weight_bytes(segmentation_head),
        "weight_prompt_encoder_bytes": weight_bytes(prompt_encoder),
        "weight_mask_decoder_bytes": weight_bytes(mask_decoder),
        "weight_detector_bytes": weight_bytes(detector),
        "weight_memory_encoder_bytes": weight_bytes(memory_encoder),
        "weight_memory_attention_bytes": weight_bytes(memory_attention),
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
    patch(_first_module(model, ["image_encoder", "detector.image_encoder", "tracker.image_encoder"]), "forward", "image_encoder_ms")
    patch(_first_module(model, ["transformer", "detector.transformer"]), "forward", "transformer_ms")
    patch(_first_module(model, ["geometry_encoder", "detector.geometry_encoder"]), "forward", "geometry_encoder_ms")
    patch(_first_module(model, ["segmentation_head", "detector.segmentation_head"]), "forward", "segmentation_head_ms")
    patch(_first_module(model, ["detector", "detection_head"]), "forward", "detector_ms")
    patch(
        _first_module(
            model,
            [
                "sam_prompt_encoder",
                "inst_interactive_predictor.model.sam_prompt_encoder",
                "tracker.sam_prompt_encoder",
                "tracker.interactive_sam_prompt_encoder",
            ],
        ),
        "forward",
        "prompt_encoder_ms",
    )
    patch(
        _first_module(
            model,
            [
                "sam_mask_decoder",
                "inst_interactive_predictor.model.sam_mask_decoder",
                "tracker.sam_mask_decoder",
                "tracker.interactive_sam_mask_decoder",
            ],
        ),
        "forward",
        "mask_decoder_ms",
    )
    patch(_first_module(model, ["memory_attention", "tracker.memory_attention"]), "forward", "memory_attention_ms")
    patch(_first_module(model, ["memory_encoder", "tracker.memory_encoder"]), "forward", "memory_encoder_ms")

    try:
        yield timings
    finally:
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)


def _first_module(root: Any, paths: list[str]) -> Any:
    for path in paths:
        obj = root
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


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
