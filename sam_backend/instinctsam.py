from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


def install_instinctsam_components(
    model: Any,
    builder: Any,
    *,
    text_checkpoint: str | None,
    vision_checkpoint: str | None,
    device: str,
) -> None:
    """Graft official InstinctSAM component checkpoints into a SAM3 image model."""
    if text_checkpoint:
        model.backbone.language_backbone = build_gitext(
            checkpoint=Path(text_checkpoint),
            bpe_path=_builder_bpe_path(builder),
            device=device,
        )
    if vision_checkpoint:
        patch_efficientsam3_hiera(builder)
        trunk = builder._create_student_vision_backbone("hiera", "large").trunk
        load_hiera_trunk_checkpoint(trunk, Path(vision_checkpoint), builder.torch)
        model.backbone.vision_backbone.trunk = trunk.to(device).eval()


def build_gitext(
    *,
    checkpoint: Path,
    bpe_path: str,
    device: str,
    context_length: int = 16,
) -> Any:
    import torch
    import torch.nn as nn
    from sam3.model.tokenizer_ve import SimpleTokenizer

    class CleanTextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = 512
            self.context_length = context_length
            self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
            self.token_embedding = nn.Embedding(49408, dim)
            self.pos_embedding = nn.Parameter(torch.empty(77, dim))
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=12)
            self.ln_final = nn.LayerNorm(dim)
            self.projector = nn.Linear(dim, 256)

        def set_context_length(self, value: int) -> None:
            self.context_length = value

        def forward(self, text: Any, input_boxes: Any = None, device: Any = None) -> tuple[Any, Any, Any]:
            del input_boxes
            tokens = self.tokenizer(text, context_length=self.context_length).to(device)
            embeddings = self.token_embedding(tokens) * math.sqrt(512)
            encoded = embeddings + self.pos_embedding[: tokens.shape[1]].unsqueeze(0)
            padding = tokens == 0
            encoded = self.transformer(encoded, src_key_padding_mask=padding)
            memory = self.projector(self.ln_final(encoded))
            return padding, memory.transpose(0, 1), embeddings.transpose(0, 1)

    encoder = CleanTextEncoder()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    result = encoder.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"GIText checkpoint is incompatible: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    return encoder.to(device).eval()


def patch_efficientsam3_hiera(builder: Any) -> None:
    """Add InstinctSAM's SAM2 Hiera-L student trunk to EfficientSAM3."""
    if getattr(builder, "_sam_bench_instinctsam_hiera", False):
        return
    if not hasattr(builder, "_create_student_vision_backbone"):
        raise RuntimeError(
            "InstinctSAM Hiera-L requires an EfficientSAM3 model_builder; "
            "use external/efficientsam3"
        )
    original = builder._create_student_vision_backbone

    def create_student_vision_backbone(
        backbone_type: str,
        model_name: str,
        compile_mode: str | None = None,
        enable_inst_interactivity: bool = True,
    ) -> Any:
        if backbone_type != "hiera":
            return original(
                backbone_type,
                model_name,
                compile_mode=compile_mode,
                enable_inst_interactivity=enable_inst_interactivity,
            )
        if model_name != "large":
            raise ValueError("InstinctSAM Hiera supports model_name='large' only")

        import timm

        nn = builder.nn
        position_encoding = builder._create_position_encoding(precompute_resolution=1008)
        backbone = timm.create_model(
            "sam2_hiera_large",
            pretrained=False,
            features_only=True,
            out_indices=(2,),
        )

        class HieraTrunkWrapper(nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.model = wrapped
                self.channel_list = [wrapped.feature_info.channels()[-1]]

            def forward(self, x: Any) -> Any:
                x = x[0] if isinstance(x, list) else x
                if x.shape[-2:] != (1152, 1152):
                    x = builder.F.interpolate(
                        x,
                        size=(1152, 1152),
                        mode="bilinear",
                        align_corners=False,
                    )
                return self.model(x)[-1]

        wrapped_backbone = HieraTrunkWrapper(backbone)
        student_encoder = builder.ImageStudentEncoder(
            backbone=wrapped_backbone,
            in_channels=wrapped_backbone.channel_list[0],
            embed_dim=1024,
            embed_size=72,
            img_size=1008,
        )
        student_encoder.channel_list = [1024]

        class ListWrapper(nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.model = wrapped
                self.channel_list = wrapped.channel_list

            def forward(self, x: Any) -> list[Any]:
                return [self.model(x)]

        trunk = ListWrapper(student_encoder)
        if compile_mode:
            trunk = builder.torch.compile(trunk, mode=compile_mode)
        return builder._create_vit_neck(
            position_encoding,
            trunk,
            enable_inst_interactivity=enable_inst_interactivity,
        )

    builder._create_student_vision_backbone = create_student_vision_backbone
    builder._sam_bench_instinctsam_hiera = True


def load_hiera_trunk_checkpoint(trunk: Any, checkpoint: Path, torch_module: Any) -> tuple[int, int]:
    payload = torch_module.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("trunk", payload) if isinstance(payload, dict) else payload
    prefix = "backbone.vision_backbone.trunk."
    state = {(key[len(prefix):] if key.startswith(prefix) else key): value for key, value in state.items()}
    target = trunk.state_dict()
    compatible = {key: value for key, value in state.items() if key in target and value.shape == target[key].shape}
    if len(compatible) < int(len(target) * 0.9):
        raise RuntimeError(
            f"Hiera-L checkpoint matched only {len(compatible)}/{len(target)} trunk tensors"
        )
    result = trunk.load_state_dict(compatible, strict=False)
    return len(compatible), len(result.missing_keys)


def _builder_bpe_path(builder: Any) -> str:
    return str(Path(builder.__file__).resolve().parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def patch_efficientsam3_vit_base(builder: Any) -> None:
    """Add InstinctSAM's ViT-B/16 student trunk to an EfficientSAM3 builder."""
    if getattr(builder, "_sam_bench_instinctsam_vit_base", False):
        return
    if not hasattr(builder, "_create_student_vision_backbone"):
        raise RuntimeError(
            "InstinctSAM ViT-B patch requires an EfficientSAM3 model_builder with "
            "_create_student_vision_backbone; use external/efficientsam3 for instinctsam_vitb"
        )

    original_create_student_vision_backbone = builder._create_student_vision_backbone

    def create_student_vision_backbone(
        backbone_type: str,
        model_name: str,
        compile_mode: str | None = None,
        enable_inst_interactivity: bool = True,
    ) -> Any:
        if backbone_type != "vit_base":
            return original_create_student_vision_backbone(
                backbone_type,
                model_name,
                compile_mode=compile_mode,
                enable_inst_interactivity=enable_inst_interactivity,
            )
        if model_name != "base":
            raise ValueError("InstinctSAM ViT-B supports model_name='base' only")

        torch = builder.torch
        nn = builder.nn
        timm = __import__("timm")

        position_encoding = builder._create_position_encoding(precompute_resolution=1008)
        backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=1008,
            global_pool="",
        )

        class ViTBaseTrunkWrapper(nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model
                self.channel_list = [model.embed_dim]

            def forward(self, x: Any) -> Any:
                x = x[0] if isinstance(x, list) else x
                tokens = self.model.forward_features(x)
                if isinstance(tokens, dict):
                    if "x_norm_patchtokens" in tokens:
                        tokens = tokens["x_norm_patchtokens"]
                    elif "x" in tokens:
                        tokens = tokens["x"]
                    else:
                        tokens = next(iter(tokens.values()))
                if tokens.ndim == 4:
                    return tokens
                if tokens.ndim != 3:
                    raise ValueError(f"unexpected ViT-B feature shape: {tuple(tokens.shape)}")

                patch_count = self.model.patch_embed.num_patches
                if tokens.shape[1] > patch_count:
                    tokens = tokens[:, -patch_count:, :]
                side = int(patch_count**0.5)
                batch, _, channels = tokens.shape
                return tokens.reshape(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()

        wrapped_backbone = ViTBaseTrunkWrapper(backbone)
        student_encoder = builder.ImageStudentEncoder(
            backbone=wrapped_backbone,
            in_channels=wrapped_backbone.channel_list[0],
            embed_dim=1024,
            embed_size=72,
            img_size=1008,
        )
        student_encoder.channel_list = [1024]

        class ListWrapper(nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model
                self.channel_list = model.channel_list

            def forward(self, x: Any) -> list[Any]:
                return [self.model(x)]

        final_trunk = ListWrapper(student_encoder)
        if compile_mode:
            final_trunk = torch.compile(final_trunk, mode=compile_mode)

        return builder._create_vit_neck(
            position_encoding,
            final_trunk,
            enable_inst_interactivity=enable_inst_interactivity,
        )

    builder._create_student_vision_backbone = create_student_vision_backbone
    builder._sam_bench_instinctsam_vit_base = True


def build_merged_vitb_checkpoint(
    *,
    teacher_checkpoint: Path,
    trunk_checkpoint: Path,
    text_checkpoint: Path,
    output_checkpoint: Path,
    external_repo: str | None,
    device: str,
) -> None:
    import sys

    from .backends import _prepend_repo_path

    _prepend_repo_path(external_repo)
    builder = __import__("sam3.model_builder", fromlist=["*"])
    patch_efficientsam3_vit_base(builder)
    torch = builder.torch

    model = builder.build_efficientsam3_image_model(
        checkpoint_path=str(teacher_checkpoint),
        load_from_HF=False,
        device=device,
        backbone_type="vit_base",
        model_name="base",
        text_encoder_type="MobileCLIP-S1",
        text_encoder_context_length=77,
        text_encoder_pos_embed_table_size=77,
        enable_inst_interactivity=True,
    )

    trunk_state = torch.load(trunk_checkpoint, map_location="cpu", weights_only=False)
    trunk_state = trunk_state.get("trunk", trunk_state)
    trunk_result = model.backbone.vision_backbone.trunk.load_state_dict(trunk_state, strict=False)
    print(
        "trunk:",
        f"missing={len(trunk_result.missing_keys)}",
        f"unexpected={len(trunk_result.unexpected_keys)}",
        file=sys.stderr,
    )

    text_state = torch.load(text_checkpoint, map_location="cpu", weights_only=False)
    text_state = text_state.get("model", text_state)
    language_backbone = model.backbone.language_backbone
    target_state = language_backbone.state_dict()
    remapped = {}
    for target_key, target_value in target_state.items():
        candidates = [
            key
            for key in text_state
            if key == target_key
            or key.endswith("." + target_key)
            or key.endswith("language_backbone." + target_key)
        ]
        if not candidates:
            continue
        value = text_state[candidates[0]]
        if value.shape != target_value.shape and "pos_embed" in target_key:
            value = value[..., : target_value.shape[-2], :].contiguous()
        if value.shape == target_value.shape:
            remapped[target_key] = value
    text_result = language_backbone.load_state_dict(remapped, strict=False)
    print(
        "text:",
        f"matched={len(remapped)}",
        f"missing={len(text_result.missing_keys)}",
        f"unexpected={len(text_result.unexpected_keys)}",
        file=sys.stderr,
    )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    print(output_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the merged InstinctSAM ViT-B checkpoint for benchmarking.")
    parser.add_argument("--teacher", type=Path, default=Path("checkpoints/sam3/sam3.pt"))
    parser.add_argument("--trunk", type=Path, required=True)
    parser.add_argument("--text", type=Path, default=Path("checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/instinctsam/instinctsam_vitb_concept.pt"))
    parser.add_argument("--external-repo", default="external/efficientsam3")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    build_merged_vitb_checkpoint(
        teacher_checkpoint=args.teacher,
        trunk_checkpoint=args.trunk,
        text_checkpoint=args.text,
        output_checkpoint=args.out,
        external_repo=args.external_repo,
        device=args.device,
    )


if __name__ == "__main__":
    main()
