from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def overlay_prediction(
    frame_rgb: np.ndarray,
    masks: Any,
    boxes: Any = None,
    scores: Any = None,
    alpha: float = 0.45,
) -> np.ndarray:
    frame = frame_rgb.copy()
    merged = merge_masks(masks, frame.shape[:2])
    if merged is not None:
        color = np.zeros_like(frame)
        color[:, :] = (30, 220, 80)
        frame = np.where(merged[..., None], (frame * (1.0 - alpha) + color * alpha).astype(np.uint8), frame)

    for idx, box in enumerate(to_numpy(boxes)):
        if len(box) < 4:
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in box[:4]]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 80, 30), 2)
        score_values = to_numpy(scores)
        if idx < len(score_values):
            cv2.putText(
                frame,
                f"{float(score_values[idx]):.2f}",
                (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 80, 30),
                2,
                cv2.LINE_AA,
            )
    return frame


def merge_masks(masks: Any, frame_hw: tuple[int, int]) -> np.ndarray | None:
    values = to_numpy(masks)
    if values.size == 0:
        return None
    if values.ndim == 4:
        values = values[:, 0]
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        return None
    merged = values.astype(bool).any(axis=0).astype(np.uint8)
    h, w = frame_hw
    if merged.shape != (h, w):
        merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_NEAREST)
    return merged.astype(bool)


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    dtype = str(getattr(value, "dtype", ""))
    if dtype.endswith("bfloat16") and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)
