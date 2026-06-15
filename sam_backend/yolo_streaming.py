from __future__ import annotations

from typing import Any

import numpy as np


def detections_to_arrays(detections: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    masks = [det["mask"] for det in detections if det.get("mask") is not None]
    boxes = [det["box"] for det in detections if det.get("box") is not None]
    scores = [float(det["score"]) for det in detections if det.get("score") != ""]
    return masks, boxes, scores


def max_score(scores: list[float]) -> float | None:
    return max(scores) if scores else None
