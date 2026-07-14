from __future__ import annotations

from typing import Any


def configure_sam3_nms(torch_module: Any, backend: str = "auto") -> str:
    """Select Thor-safe SAM3 perflib implementations without editing the external repo."""
    normalized = backend.strip().lower()
    if normalized not in {"auto", "native", "torch"}:
        raise ValueError("nms_backend must be one of: auto, native, torch")

    resolved = normalized
    if normalized == "auto":
        resolved = "native"
        if torch_module.cuda.is_available():
            major, _ = torch_module.cuda.get_device_capability()
            if major >= 10:
                resolved = "torch"

    if resolved == "torch":
        from sam3.perflib import connected_components as cc_module
        from sam3.perflib import nms as nms_module

        nms_module.generic_nms = generic_nms_torch
        cc_module.connected_components = connected_components_cpu_safe
    return resolved


def generic_nms_torch(ious: Any, scores: Any, iou_threshold: float = 0.5) -> Any:
    """Sequential generic NMS using regular PyTorch operations on the input device."""
    if ious.dim() != 2 or ious.size(0) != ious.size(1):
        raise ValueError("ious must be a square matrix")
    if scores.dim() != 1 or scores.size(0) != ious.size(0):
        raise ValueError("scores must match the IoU matrix")
    order = scores.argsort(descending=True, stable=True)
    if order.numel() == 0:
        return order
    kept = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[ious[current, remaining] <= iou_threshold]
    import torch

    return torch.stack(kept).to(dtype=torch.int64)


def connected_components_cpu_safe(input_tensor: Any) -> tuple[Any, Any]:
    """Call SAM3's CPU fallback while preserving valid empty-batch behavior."""
    import torch

    if input_tensor.numel() == 0:
        empty = torch.zeros_like(input_tensor, dtype=torch.int64)
        return empty, empty.clone()

    from sam3.perflib.connected_components import connected_components_cpu

    return connected_components_cpu(input_tensor)
