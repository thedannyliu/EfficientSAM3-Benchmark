from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(slots=True)
class Prompt:
    text: str | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


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
    enable_inst_interactivity: bool = False
    model_config: str | None = None
    external_repo: str | None = None
    mobile_sam_model_type: str = "vit_t"


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
        self.config = resolve_backend_config(config)
        _prepend_repo_path(config.external_repo)
        self.torch = _import_required("torch")
        builder = _import_required("sam3.model_builder")
        processor_mod = _import_required("sam3.model.sam3_image_processor")

        if self.config.backend == "sam3":
            self.model = builder.build_sam3_image_model(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
                enable_inst_interactivity=self.config.enable_inst_interactivity,
            )
        elif self.config.backend == "efficientsam3":
            if not self.config.checkpoint_path:
                raise ValueError("--checkpoint-path is required for EfficientSAM3")
            self.model = builder.build_efficientsam3_image_model(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
                backbone_type=self.config.backbone_type,
                model_name=self.config.model_name,
                text_encoder_type=self.config.text_encoder_type,
                text_encoder_context_length=self.config.text_encoder_context_length,
                text_encoder_pos_embed_table_size=self.config.text_encoder_pos_embed_table_size,
                interpolate_pos_embed=self.config.interpolate_pos_embed,
                enable_inst_interactivity=self.config.enable_inst_interactivity,
            )
        else:
            raise ValueError(f"unsupported SAM backend: {self.config.backend}")

        if self.config.device and hasattr(self.model, "to"):
            self.model = self.model.to(self.config.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.processor = processor_mod.Sam3Processor(self.model, device=self.config.device or "cuda")

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
        if prompt.boxes:
            masks, scores, logits = self.model.predict_inst(
                state,
                point_coords=None,
                point_labels=None,
                box=np.asarray(prompt.boxes, dtype=np.float32),
                multimask_output=False,
            )
            return {"masks": masks, "scores": scores, "logits": logits}
        raise ValueError("prompt must include text, points, or boxes")

    def _synchronize(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()


class Sam2PointImageBackend:
    def __init__(self, config: BackendConfig, package: str) -> None:
        self.config = config
        self.package = package
        _prepend_repo_path(config.external_repo or _default_external_repo(config.backend))
        self.torch = _import_required("torch")

        if package == "efficient_track_anything":
            builder = _import_required("efficient_track_anything.build_efficienttam")
            predictor_mod = _import_required("efficient_track_anything.efficienttam_image_predictor")
            if not config.model_config:
                raise ValueError("--model-config is required for EfficientTAM")
            self.model = builder.build_efficienttam(
                config.model_config,
                config.checkpoint_path,
                device=config.device or "cuda",
            )
            self.processor = predictor_mod.EfficientTAMImagePredictor(self.model)
        else:
            builder = _import_required("sam2.build_sam")
            predictor_mod = _import_required("sam2.sam2_image_predictor")
            if not config.model_config:
                raise ValueError("--model-config is required for SAM2-style backends")
            self.model = builder.build_sam2(
                config.model_config,
                config.checkpoint_path,
                device=config.device or "cuda",
            )
            self.processor = predictor_mod.SAM2ImagePredictor(self.model)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict(self, image: Any, prompt: Prompt) -> Prediction:
        if prompt.text:
            raise ValueError(f"{self.config.backend} supports point prompts in this benchmark, not text prompts")
        if not prompt.points and not prompt.boxes:
            raise ValueError(f"{self.config.backend} requires at least one point or box prompt")

        pil_image = _as_pil_image(image)
        image_np = np.asarray(pil_image)
        point_coords = np.asarray(prompt.points, dtype=np.float32) if prompt.points else None
        point_labels = np.asarray(prompt.labels or [1] * len(prompt.points), dtype=np.int32) if prompt.points else None
        box = np.asarray(prompt.boxes[0], dtype=np.float32) if prompt.boxes else None

        self._synchronize()
        start = perf_counter()
        with self.torch.inference_mode():
            self.processor.set_image(image_np)
            masks, scores, logits = self.processor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )
        self._synchronize()
        latency_ms = (perf_counter() - start) * 1000.0
        return Prediction(
            masks=masks,
            scores=scores,
            latency_ms=latency_ms,
            metadata={"backend": self.config.backend, "prompt_type": "point"},
        )

    def _synchronize(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()


class MobileSamPointImageBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        _prepend_repo_path(config.external_repo or _default_external_repo(config.backend))
        self.torch = _import_required("torch")
        mobile_sam = _import_required("mobile_sam")
        if not config.checkpoint_path:
            raise ValueError("--checkpoint-path is required for MobileSAM")

        self.model = mobile_sam.sam_model_registry[config.mobile_sam_model_type](checkpoint=config.checkpoint_path)
        if config.device and hasattr(self.model, "to"):
            self.model.to(device=config.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.processor = mobile_sam.SamPredictor(self.model)

    def predict(self, image: Any, prompt: Prompt) -> Prediction:
        if prompt.text:
            raise ValueError("mobilesam supports point prompts in this benchmark, not text prompts")
        if not prompt.points and not prompt.boxes:
            raise ValueError("mobilesam requires at least one point or box prompt")

        pil_image = _as_pil_image(image)
        image_np = np.asarray(pil_image)
        point_coords = np.asarray(prompt.points, dtype=np.float32) if prompt.points else None
        point_labels = np.asarray(prompt.labels or [1] * len(prompt.points), dtype=np.int32) if prompt.points else None
        box = np.asarray(prompt.boxes[0], dtype=np.float32) if prompt.boxes else None

        self._synchronize()
        start = perf_counter()
        with self.torch.inference_mode():
            self.processor.set_image(image_np)
            masks, scores, logits = self.processor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )
        self._synchronize()
        latency_ms = (perf_counter() - start) * 1000.0
        return Prediction(
            masks=masks,
            scores=scores,
            latency_ms=latency_ms,
            metadata={"backend": self.config.backend, "prompt_type": "point"},
        )

    def _synchronize(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()


def create_backend(config: BackendConfig) -> SegmentationBackend:
    config = resolve_backend_config(config)
    if config.backend == "null":
        return NullBackend()
    if config.backend in {"sam3", "efficientsam3"}:
        return Sam3ImageBackend(config)
    if config.backend in {"sam2", "efficient-sam2"}:
        return Sam2PointImageBackend(config, "sam2")
    if config.backend == "efficienttam":
        return Sam2PointImageBackend(config, "efficient_track_anything")
    if config.backend == "mobilesam":
        return MobileSamPointImageBackend(config)
    raise ValueError(f"unknown backend: {config.backend}")


def resolve_backend_config(config: BackendConfig) -> BackendConfig:
    if config.backend != "efficientsam3" or not config.checkpoint_path:
        return config
    resolved = _infer_efficientsam3_image_config(config.checkpoint_path)
    if resolved is None:
        return config
    backbone_type, model_name = resolved
    return BackendConfig(
        backend=config.backend,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        backbone_type=backbone_type,
        model_name=model_name,
        text_encoder_type=config.text_encoder_type,
        text_encoder_context_length=config.text_encoder_context_length,
        text_encoder_pos_embed_table_size=config.text_encoder_pos_embed_table_size,
        interpolate_pos_embed=config.interpolate_pos_embed,
        enable_inst_interactivity=config.enable_inst_interactivity,
        model_config=config.model_config,
        external_repo=config.external_repo,
        mobile_sam_model_type=config.mobile_sam_model_type,
    )


def _infer_efficientsam3_image_config(checkpoint_path: str) -> tuple[str, str] | None:
    stem = Path(checkpoint_path).stem
    mapping = {
        "efficient_sam3_repvit_s": ("repvit", "m0.9"),
        "efficient_sam3_repvit_m": ("repvit", "m1.1"),
        "efficient_sam3_repvit_l": ("repvit", "m2.3"),
        "efficient_sam3_tinyvit_s": ("tinyvit", "5m"),
        "efficient_sam3_tinyvit_m": ("tinyvit", "11m"),
        "efficient_sam3_tinyvit_l": ("tinyvit", "21m"),
        "efficient_sam3_efficientvit_s": ("efficientvit", "b0"),
        "efficient_sam3_efficientvit_m": ("efficientvit", "b1"),
        "efficient_sam3_efficientvit_l": ("efficientvit", "b2"),
    }
    return mapping.get(stem)


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


def _prepend_repo_path(repo_path: str | None) -> None:
    if not repo_path:
        return
    path = str(Path(repo_path).resolve())
    if Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)


def _default_external_repo(backend: str) -> str | None:
    defaults = {
        "sam2": "external/sam2",
        "efficient-sam2": "external/Efficient-SAM2",
        "efficienttam": "external/EfficientTAM",
        "mobilesam": "external/MobileSAM",
    }
    path = defaults.get(backend)
    if path and Path(path).exists():
        return path
    return None
