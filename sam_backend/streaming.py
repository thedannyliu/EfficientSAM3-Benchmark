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


def masks_to_bbox_xyxy(
    masks: Any,
    frame_hw: tuple[int, int],
    min_area: int = 1,
    scale: float = 1.0,
) -> tuple[float, float, float, float] | None:
    merged = merge_masks(masks, frame_hw)
    if merged is None or int(merged.sum()) < min_area:
        return None
    ys, xs = np.nonzero(merged)
    if xs.size == 0 or ys.size == 0:
        return None
    height, width = frame_hw
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    bbox_scale = max(1.0, float(scale))
    if bbox_scale > 1.0:
        box_width = x2 - x1 + 1.0
        box_height = y2 - y1 + 1.0
        x_pad = (bbox_scale - 1.0) * box_width / 2.0
        y_pad = (bbox_scale - 1.0) * box_height / 2.0
        x1 -= x_pad
        y1 -= y_pad
        x2 += x_pad
        y2 += y_pad
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(width - 1), x2)
    y2 = min(float(height - 1), y2)
    return (x1, y1, x2, y2)


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


def parse_text_prompts(prompt: str, prompts: str = "") -> list[str]:
    raw = prompts.strip()
    if not raw:
        return [prompt.strip()] if prompt.strip() else []
    if re.search(r"[,;\n]", raw):
        values = re.split(r"[,;\n]+", raw)
    else:
        values = raw.split()
    return [value.strip() for value in values if value.strip()]
