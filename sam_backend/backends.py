from __future__ import annotations

from contextlib import nullcontext
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
    texts: list[str] = field(default_factory=list)
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
    autocast_dtype: str = "bfloat16"
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
                load_from_HF=False,
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
        with self.torch.inference_mode(), _autocast_context(
            self.torch,
            self.config.device,
            self.config.autocast_dtype,
        ):
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
        if prompt.texts:
            outputs = [self.processor.set_text_prompt(state=state, prompt=text) for text in prompt.texts]
            return {
                "masks": _concat_prediction_values(output.get("masks") for output in outputs),
                "boxes": _concat_prediction_values(output.get("boxes") for output in outputs),
                "scores": _concat_prediction_values(output.get("scores") for output in outputs),
            }
        if prompt.text:
            return self.processor.set_text_prompt(state=state, prompt=prompt.text)
        if prompt.points:
            point_coords = np.asarray(prompt.points, dtype=np.float32)
            point_labels = np.asarray(prompt.labels or [1] * len(prompt.points), dtype=np.int32)
            masks, scores, logits = self.model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
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
    if config.backend in {"mobilesam", "sam1"}:
        return MobileSamPointImageBackend(config)
    raise ValueError(f"unknown backend: {config.backend}")


def resolve_backend_config(config: BackendConfig) -> BackendConfig:
    if config.backend != "efficientsam3" or not config.checkpoint_path:
        return config
    resolved = _infer_efficientsam3_image_config(config.checkpoint_path)
    if resolved is None:
        return config
    backbone_type, model_name, text_encoder_type, context_length, pos_embed_table_size = resolved
    return BackendConfig(
        backend=config.backend,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        backbone_type=backbone_type,
        model_name=model_name,
        text_encoder_type=config.text_encoder_type or text_encoder_type,
        text_encoder_context_length=(
            context_length
            if context_length is not None and config.text_encoder_context_length == 77
            else config.text_encoder_context_length
        ),
        text_encoder_pos_embed_table_size=config.text_encoder_pos_embed_table_size or pos_embed_table_size,
        interpolate_pos_embed=config.interpolate_pos_embed,
        enable_inst_interactivity=config.enable_inst_interactivity,
        autocast_dtype=config.autocast_dtype,
        model_config=config.model_config,
        external_repo=config.external_repo,
        mobile_sam_model_type=config.mobile_sam_model_type,
    )


def _infer_efficientsam3_image_config(checkpoint_path: str) -> tuple[str, str, str | None, int | None, int | None] | None:
    stem = Path(checkpoint_path).stem
    mapping = {
        "efficient_sam3_repvit_s": ("repvit", "m0.9", None, None, None),
        "efficient_sam3_repvit_m": ("repvit", "m1.1", None, None, None),
        "efficient_sam3_repvit_l": ("repvit", "m2.3", None, None, None),
        "efficient_sam3_tinyvit_s": ("tinyvit", "5m", None, None, None),
        "efficient_sam3_tinyvit_m": ("tinyvit", "11m", None, None, None),
        "efficient_sam3_tinyvit_l": ("tinyvit", "21m", None, None, None),
        "efficient_sam3_tinyvit21_stage1_e32_h200_full_sam3": ("tinyvit", "21m", None, None, None),
        "efficient_sam3_efficientvit_s": ("efficientvit", "b0", None, None, None),
        "efficient_sam3_efficientvit_m": ("efficientvit", "b1", None, None, None),
        "efficient_sam3_efficientvit_l": ("efficientvit", "b2", None, None, None),
        "efficientsam3_efficientvit": ("efficientvit", "b1", "MobileCLIP-S0", 16, 16),
        "efficientsam3_repvit": ("repvit", "m1.1", "MobileCLIP-S0", 16, 16),
        "efficientsam3_tinyvit": ("tinyvit", "11m", "MobileCLIP-S0", 16, 16),
    }
    return mapping.get(stem)


def _autocast_context(torch_module: Any, device: str | None, dtype_name: str) -> Any:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return nullcontext()
    if device is not None and not str(device).startswith("cuda"):
        return nullcontext()
    dtype = _resolve_autocast_dtype(torch_module, dtype_name)
    if dtype is None:
        return nullcontext()
    return torch_module.autocast("cuda", dtype=dtype)


def _resolve_autocast_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized = dtype_name.strip().lower()
    if normalized in {"", "none", "false", "off", "float32", "fp32"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch_module.float16
    raise ValueError("autocast_dtype must be one of: bfloat16, float16, float32, none")


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


def _concat_prediction_values(values: Any) -> np.ndarray:
    arrays = []
    for value in values:
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach()
        dtype = str(getattr(value, "dtype", ""))
        if dtype.endswith("bfloat16") and hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "cpu"):
            value = value.cpu()
        array = np.asarray(value)
        if array.size == 0:
            continue
        if array.ndim == 0:
            array = array.reshape(1)
        arrays.append(array)
    if not arrays:
        return np.asarray([])
    return np.concatenate(arrays, axis=0)


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
    root = Path(repo_path).resolve()
    candidates = [root]
    nested_sam3_parent = root / "sam3"
    if (nested_sam3_parent / "sam3" / "model_builder.py").exists():
        candidates.insert(0, nested_sam3_parent)
    for candidate in reversed(candidates):
        path = str(candidate)
        if candidate.exists() and path not in sys.path:
            sys.path.insert(0, path)


def _default_external_repo(backend: str) -> str | None:
    defaults = {
        "sam2": "external/sam2",
        "efficient-sam2": "external/Efficient-SAM2",
        "efficienttam": "external/EfficientTAM",
        "mobilesam": "external/MobileSAM",
        "sam1": "external/MobileSAM",
    }
    path = defaults.get(backend)
    if path and Path(path).exists():
        return path
    return None
