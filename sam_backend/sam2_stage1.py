from __future__ import annotations

import inspect
import sys
import types
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


def patch_stage1_forward_image(
    predictor: Any,
    torch_module: Any,
    *,
    sam2_distill_root: str | Path,
    student_checkpoint_path: str,
    sam2_checkpoint_path: str,
    device: str,
    requested_family: str,
    requested_model_name: str,
    requested_backbone_checkpoint: str,
    legacy_tinyvit_checkpoint: str,
    requested_adapter_mode: str,
    fallback_model_name: str,
) -> dict[str, Any]:
    distill_root = Path(sam2_distill_root)
    if distill_root.exists():
        sys.path.insert(0, str(distill_root))
    try:
        from sam2_distill.edgetam.compat import patch_edgetam_perceiver_view
    except ImportError as exc:
        raise RuntimeError(
            "Stage1 student loading requires SAM2-Distillation-Pipeline on "
            "sam2_distill_root/PYTHONPATH"
        ) from exc

    patch_edgetam_perceiver_view()
    checkpoint = torch_module.load(
        student_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_stage1_state_dict(checkpoint)
    student_family, student_model_name, student_adapter_mode = resolve_stage1_spec(
        checkpoint,
        state_dict,
        requested_family=requested_family,
        requested_model_name=requested_model_name,
        fallback_model_name=fallback_model_name,
        requested_adapter_mode=requested_adapter_mode,
    )
    if not requested_backbone_checkpoint and student_family == "tinyvit":
        requested_backbone_checkpoint = legacy_tinyvit_checkpoint
    backbone_checkpoint = resolve_stage1_backbone_checkpoint(
        student_model_name,
        requested_backbone_checkpoint,
    )
    student = build_stage1_student_compat(
        student_family=student_family,
        model_name=student_model_name,
        checkpoint_path=(
            str(backbone_checkpoint) if backbone_checkpoint is not None else None
        ),
        adapter_mode=student_adapter_mode,
    ).to(device)
    incompatible = student.load_state_dict(state_dict, strict=False)
    if student_family == "repvit":
        missing_non_backbone = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("backbone.")
        ]
        missing_backbone_without_source = (
            backbone_checkpoint is None
            and any(key.startswith("backbone.") for key in incompatible.missing_keys)
        )
        if (
            missing_non_backbone
            or missing_backbone_without_source
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "RepViT Stage1 checkpoint is incomplete or incompatible: "
                f"missing={incompatible.missing_keys[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)

    position_encoding = predictor.image_encoder.neck.position_encoding

    @torch_module.inference_mode()
    def forward_image(self: Any, img_batch: Any) -> dict[str, Any]:
        features = student(img_batch)
        backbone_fpn = [
            features["high_res_s0"].float(),
            features["high_res_s1"].float(),
            features["image_embed"].float(),
        ]
        vision_pos_enc = [position_encoding(feat).float() for feat in backbone_fpn]
        return {
            "vision_features": backbone_fpn[-1],
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
        }

    predictor.forward_image = types.MethodType(forward_image, predictor)
    predictor._stage1_student = student
    return {
        "student_checkpoint_path": student_checkpoint_path,
        "student_family": student_family,
        "student_model_name": student_model_name,
        "student_backbone_checkpoint_path": (
            str(backbone_checkpoint) if backbone_checkpoint is not None else ""
        ),
        "student_adapter_mode": student_adapter_mode,
        "tinyvit_checkpoint_path": (
            str(backbone_checkpoint)
            if student_family == "tinyvit" and backbone_checkpoint is not None
            else ""
        ),
        "tinyvit_model_name": (
            student_model_name if student_family == "tinyvit" else ""
        ),
        "stage1_checkpoint_step": checkpoint.get("step", ""),
        "stage1_checkpoint_epoch": checkpoint.get("epoch", ""),
        "stage1_missing_keys": len(incompatible.missing_keys),
        "stage1_unexpected_keys": len(incompatible.unexpected_keys),
        "sam2_checkpoint_path": sam2_checkpoint_path,
    }


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
