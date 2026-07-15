from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


MODEL_BY_IMAGE_PROJ_CHANNELS = {
    384: "tiny_vit_21m_512.dist_in22k_ft_in1k",
    256: "tiny_vit_11m_224.dist_in22k_ft_in1k",
    160: "tiny_vit_5m_224.dist_in22k_ft_in1k",
}

CHECKPOINT_NAME_BY_MODEL = {
    "tiny_vit_21m_512.dist_in22k_ft_in1k": "tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors",
    "tiny_vit_11m_224.dist_in22k_ft_in1k": "tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors",
    "tiny_vit_5m_224.dist_in22k_ft_in1k": "tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors",
    "repvit_m0_9.dist_450e_in1k": "repvit_m0_9.dist_450e_in1k.safetensors",
    "repvit_m2_3.dist_450e_in1k": "repvit_m2_3.dist_450e_in1k.safetensors",
}


def extract_stage1_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            if any(name.startswith("module.") for name in value):
                return {
                    name.removeprefix("module."): tensor
                    for name, tensor in value.items()
                }
            return value
    raise KeyError("checkpoint must contain one of: model, model_state, state_dict")


def resolve_stage1_spec(
    checkpoint: dict[str, Any],
    state_dict: dict[str, Any],
    *,
    requested_family: str,
    requested_model_name: str,
    fallback_model_name: str,
    requested_adapter_mode: str,
) -> tuple[str, str, str]:
    args = checkpoint.get("args")
    args = args if isinstance(args, dict) else {}

    metadata_model_name = args.get("model_name")
    if requested_model_name:
        if isinstance(metadata_model_name, str) and metadata_model_name != requested_model_name:
            raise ValueError(
                f"student_model_name={requested_model_name} conflicts with "
                f"checkpoint model_name={metadata_model_name}"
            )
        model_name = requested_model_name
    elif isinstance(metadata_model_name, str):
        model_name = metadata_model_name
    else:
        model_name = _infer_tinyvit_model_name(state_dict, fallback_model_name)

    inferred_family = args.get("student_family")
    if inferred_family not in {"tinyvit", "repvit"}:
        inferred_family = "repvit" if model_name.startswith("repvit_") else "tinyvit"
    family = _resolve_requested_value(
        "student_family",
        requested_family,
        inferred_family,
        {"tinyvit", "repvit"},
    )

    inferred_adapter_mode = args.get("adapter_mode")
    if inferred_adapter_mode not in {"projection", "residual_dwconv"}:
        inferred_adapter_mode = (
            "residual_dwconv"
            if any(key.startswith("adapters.") for key in state_dict)
            else "projection"
        )
    adapter_mode = _resolve_requested_value(
        "student_adapter_mode",
        requested_adapter_mode,
        inferred_adapter_mode,
        {"projection", "residual_dwconv"},
    )
    if family == "repvit" and adapter_mode != "projection":
        raise ValueError("RepViT Stage1 currently supports projection mode only")
    return family, model_name, adapter_mode


def resolve_stage1_backbone_checkpoint(
    model_name: str, requested_checkpoint: str
) -> Path | None:
    if not requested_checkpoint:
        return None
    requested = Path(requested_checkpoint)
    expected_name = CHECKPOINT_NAME_BY_MODEL.get(model_name)
    if expected_name is not None:
        candidate = requested.parent / expected_name
        if candidate.exists():
            return candidate
    return requested


def build_stage1_student_compat(
    *,
    student_family: str,
    model_name: str,
    checkpoint_path: str | None,
    adapter_mode: str,
) -> Any:
    try:
        from sam2_distill.models.stage1_student import build_stage1_student
    except ImportError:
        build_stage1_student = None
    if build_stage1_student is not None:
        try:
            return build_stage1_student(
                student_family=student_family,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                adapter_mode=adapter_mode,
            )
        except ImportError:
            pass

    from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter

    if student_family == "tinyvit":
        return _build_tinyvit_student(
            TinyViTSAM2Adapter,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            adapter_mode=adapter_mode,
        )
    if student_family != "repvit":
        raise ValueError(f"Unsupported Stage1 student family: {student_family}")
    if adapter_mode != "projection":
        raise ValueError("RepViT Stage1 currently supports projection mode only")

    try:
        from sam2_distill.models.repvit_adapter import RepViTSAM2Adapter
    except ImportError:
        RepViTSAM2Adapter = _fallback_repvit_adapter(TinyViTSAM2Adapter)
    return RepViTSAM2Adapter(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )


def _build_tinyvit_student(
    adapter_class: Any,
    *,
    model_name: str,
    checkpoint_path: str | None,
    adapter_mode: str,
) -> Any:
    parameters = inspect.signature(adapter_class.__init__).parameters
    kwargs = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
    }
    if "adapter_mode" in parameters:
        kwargs["adapter_mode"] = adapter_mode
    elif adapter_mode != "projection":
        raise RuntimeError(
            "This SAM2-Distillation-Pipeline version does not support "
            f"adapter_mode={adapter_mode}; update the external pipeline"
        )
    return adapter_class(**kwargs)


def _fallback_repvit_adapter(tinyvit_adapter_class: Any) -> Any:
    from torch import nn

    class RepViTSAM2Adapter(tinyvit_adapter_class):
        def __init__(self, model_name: str, checkpoint_path: str | None) -> None:
            if not model_name.startswith("repvit_"):
                raise ValueError(f"Expected a RepViT model name, got {model_name}")
            parameters = inspect.signature(tinyvit_adapter_class.__init__).parameters
            kwargs = {
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
            }
            if "adapter_mode" in parameters:
                kwargs["adapter_mode"] = "projection"
            super().__init__(**kwargs)
            channels = list(self.backbone.body.feature_info.channels())
            final_feature_idx = len(channels) - 1
            self.target_to_feature_idx["image_embed"] = final_feature_idx
            self.projections["image_embed"] = nn.Conv2d(
                channels[final_feature_idx], 256, kernel_size=1
            )

    return RepViTSAM2Adapter


def _infer_tinyvit_model_name(
    state_dict: dict[str, Any], fallback: str
) -> str:
    weight = state_dict.get("projections.image_embed.weight")
    if hasattr(weight, "ndim") and weight.ndim == 4:
        inferred = MODEL_BY_IMAGE_PROJ_CHANNELS.get(int(weight.shape[1]))
        if inferred is not None:
            return inferred
    return fallback


def _resolve_requested_value(
    field_name: str,
    requested: str,
    inferred: str,
    allowed: set[str],
) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", *allowed}:
        choices = ", ".join(["auto", *sorted(allowed)])
        raise ValueError(f"{field_name} must be one of: {choices}")
    if normalized != "auto" and normalized != inferred:
        raise ValueError(
            f"{field_name}={normalized} conflicts with checkpoint value={inferred}"
        )
    return inferred
