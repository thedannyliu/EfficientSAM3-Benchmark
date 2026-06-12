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


def parse_tegrastats_gr3d(line: str) -> float | None:
    match = re.search(r"\bGR3D_FREQ\s+([0-9]+(?:\.[0-9]+)?)%", line)
    if match is None:
        return None
    return float(match.group(1))
