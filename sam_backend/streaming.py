from __future__ import annotations

import re
from typing import Any

import numpy as np

from .overlay import merge_masks


def masks_to_mono8(masks: Any, frame_hw: tuple[int, int]) -> np.ndarray:
    merged = merge_masks(masks, frame_hw)
    if merged is None:
        return np.zeros(frame_hw, dtype=np.uint8)
    return (merged.astype(np.uint8) * 255).astype(np.uint8)


def masks_to_bbox_xyxy(masks: Any, frame_hw: tuple[int, int], min_area: int = 1) -> tuple[float, float, float, float] | None:
    merged = merge_masks(masks, frame_hw)
    if merged is None or int(merged.sum()) < min_area:
        return None
    ys, xs = np.nonzero(merged)
    if xs.size == 0 or ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def left_panel_click_to_image_point(
    x: int,
    y: int,
    frame_hw: tuple[int, int],
) -> tuple[float, float] | None:
    height, width = frame_hw
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    return (float(x), float(y))


def parse_tegrastats_gr3d(line: str) -> float | None:
    match = re.search(r"\bGR3D_FREQ\s+([0-9]+(?:\.[0-9]+)?)%", line)
    if match is None:
        return None
    return float(match.group(1))
