from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(slots=True)
class Prompt:
    text: str | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Prediction:
    masks: Any
    boxes: Any = None
    scores: Any = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendConfig:
    backend: str = "null"
    checkpoint_path: str | None = None
    device: str | None = None
    backbone_type: str = "efficientvit"
    model_name: str = "b0"
    text_encoder_type: str | None = None
    text_encoder_context_length: int = 77
    text_encoder_pos_embed_table_size: int | None = None
    interpolate_pos_embed: bool = False


class SegmentationBackend(Protocol):
    def predict(self, image: Any, prompt: Prompt) -> Prediction:
        ...


class NullBackend:
    """Dependency-light backend for smoke-testing benchmark and ROS plumbing."""

    def predict(self, image: Any, prompt: Prompt) -> Prediction:
        pil_image = _as_pil_image(image)
        start = perf_counter()
        mask = np.zeros((pil_image.height, pil_image.width), dtype=np.uint8)
        latency_ms = (perf_counter() - start) * 1000.0
        return Prediction(
            masks=[mask],
            boxes=[],
            scores=[],
            latency_ms=latency_ms,
            metadata={"backend": "null", "prompt": prompt.text},
        )


class Sam3ImageBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.torch = _import_required("torch")
        builder = _import_required("sam3.model_builder")
        processor_mod = _import_required("sam3.model.sam3_image_processor")

        if config.backend == "sam3":
            self.model = builder.build_sam3_image_model(
                checkpoint_path=config.checkpoint_path,
                device=config.device,
                text_encoder_type=config.text_encoder_type,
                text_encoder_context_length=config.text_encoder_context_length,
                text_encoder_pos_embed_table_size=config.text_encoder_pos_embed_table_size,
                interpolate_pos_embed=config.interpolate_pos_embed,
                enable_inst_interactivity=False,
            )
        elif config.backend == "efficientsam3":
            if not config.checkpoint_path:
                raise ValueError("--checkpoint-path is required for EfficientSAM3")
            self.model = builder.build_efficientsam3_image_model(
                checkpoint_path=config.checkpoint_path,
                device=config.device,
                backbone_type=config.backbone_type,
                model_name=config.model_name,
                text_encoder_type=config.text_encoder_type,
                text_encoder_context_length=config.text_encoder_context_length,
                text_encoder_pos_embed_table_size=config.text_encoder_pos_embed_table_size,
                interpolate_pos_embed=config.interpolate_pos_embed,
                enable_inst_interactivity=False,
            )
        else:
            raise ValueError(f"unsupported SAM backend: {config.backend}")

        if config.device and hasattr(self.model, "to"):
            self.model = self.model.to(config.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.processor = processor_mod.Sam3Processor(self.model, device=config.device or "cuda")

    def predict(self, image: Any, prompt: Prompt) -> Prediction:
        pil_image = _as_pil_image(image)
        self._synchronize()
        start = perf_counter()
        with self.torch.inference_mode():
            state = self.processor.set_image(pil_image)
            output = self._run_prompt(state, prompt)
        self._synchronize()
        latency_ms = (perf_counter() - start) * 1000.0
        return Prediction(
            masks=output.get("masks"),
            boxes=output.get("boxes"),
            scores=output.get("scores"),
            latency_ms=latency_ms,
            metadata={"backend": self.config.backend},
        )

    def _run_prompt(self, state: Any, prompt: Prompt) -> dict[str, Any]:
        if prompt.text:
            return self.processor.set_text_prompt(state=state, prompt=prompt.text)
        if prompt.points:
            masks, scores, logits = self.model.predict_inst(
                state,
                point_coords=prompt.points,
                point_labels=prompt.labels or [1] * len(prompt.points),
            )
            return {"masks": masks, "scores": scores, "logits": logits}
        raise ValueError("prompt must include text or points")

    def _synchronize(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()


def create_backend(config: BackendConfig) -> SegmentationBackend:
    if config.backend == "null":
        return NullBackend()
    if config.backend in {"sam3", "efficientsam3"}:
        return Sam3ImageBackend(config)
    raise ValueError(f"unknown backend: {config.backend}")


def _as_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3:
            return Image.fromarray(image[..., :3]).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _import_required(module_name: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install the selected SAM backend "
            "in this Python environment before running real inference."
        ) from exc
