from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from .backends import BackendConfig, Prediction, Prompt, _as_pil_image


class InstinctSamHttpBackend:
    """Stateless image interface to the evaluation-only InstinctSAM HTTP overlay."""

    def __init__(self, config: BackendConfig) -> None:
        self.base_url = config.runtime_url.rstrip("/")
        self.timeout = config.runtime_timeout
        self.model = None
        self.torch = None

    def predict(self, image: Any, prompt: Prompt) -> Prediction:
        prompts = prompt.texts or ([prompt.text] if prompt.text else [])
        if not prompts:
            raise ValueError("instinctsam-http requires at least one text prompt")
        if prompt.points or prompt.boxes:
            raise ValueError("instinctsam-http supports text prompts only")

        pil_image = _as_pil_image(image).convert("RGB")
        frame_rgb = np.asarray(pil_image)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode InstinctSAM input")

        started = time.perf_counter()
        self._request("/reset", b"", "application/octet-stream")
        self._request(
            "/prompt",
            json.dumps({"text": ",".join(prompts)}, separators=(",", ":")).encode(),
            "application/json",
        )
        frame_response = json.loads(
            self._request("/frame.jpg", encoded.tobytes(), "image/jpeg")
        )
        input_sequence = int(frame_response["input_sequence"])
        masks, labels, scores, lost, frame_number = self._wait_for_masks(input_sequence)
        status = json.loads(self._request("/status.json"))
        latency_ms = (time.perf_counter() - started) * 1000.0

        keep = np.logical_not(lost)
        masks = masks[keep]
        labels = labels[keep]
        scores = scores[keep]
        boxes = [_mask_box(mask) for mask in masks]
        return Prediction(
            masks=masks,
            boxes=boxes,
            scores=scores,
            latency_ms=latency_ms,
            metadata={
                "backend": "instinctsam-http",
                "input_sequence": input_sequence,
                "runtime_frame": frame_number,
                "labels": labels.tolist(),
                "status": status,
            },
        )

    def _wait_for_masks(
        self, input_sequence: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                payload = self._request("/masks.npz")
            except urllib.error.HTTPError as error:
                if error.code != 503:
                    raise
                time.sleep(0.01)
                continue
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if int(archive["schema_version"]) < 2:
                    raise RuntimeError("InstinctSAM mask API schema 2 or newer is required")
                if int(archive["input_sequence"]) != input_sequence:
                    time.sleep(0.01)
                    continue
                shape = tuple(int(value) for value in archive["mask_shape"])
                bitorder = str(archive["bitorder"].item())
                values = np.unpackbits(
                    archive["masks_packed"], bitorder=bitorder
                )[: int(np.prod(shape))]
                return (
                    values.astype(bool).reshape(shape),
                    archive["labels"].astype(str),
                    archive["scores"].astype(np.float32),
                    archive["lost"].astype(bool),
                    int(archive["frame"]),
                )
        raise TimeoutError(
            f"timed out waiting for InstinctSAM input_sequence={input_sequence}"
        )

    def _request(
        self, path: str, data: bytes | None = None, content_type: str | None = None
    ) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()


def _mask_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
