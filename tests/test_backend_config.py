from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from sam_backend import BackendConfig, resolve_backend_config
from sam_backend.backends import _prepend_repo_path, _resolve_autocast_dtype


class BackendConfigTest(unittest.TestCase):
    def test_resolves_repvit_s_checkpoint_filename(self) -> None:
        config = resolve_backend_config(
            BackendConfig(
                backend="efficientsam3",
                checkpoint_path="checkpoints/efficient_sam3_repvit_s.pt",
            )
        )

        self.assertEqual(config.backbone_type, "repvit")
        self.assertEqual(config.model_name, "m0.9")

    def test_resolves_full_efficientsam3_checkpoint_text_encoder(self) -> None:
        config = resolve_backend_config(
            BackendConfig(
                backend="efficientsam3",
                checkpoint_path="checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt",
            )
        )

        self.assertEqual(config.backbone_type, "efficientvit")
        self.assertEqual(config.model_name, "b1")
        self.assertEqual(config.text_encoder_type, "MobileCLIP-S0")
        self.assertEqual(config.text_encoder_context_length, 16)
        self.assertEqual(config.text_encoder_pos_embed_table_size, 16)

    def test_resolves_distilled_tinyvit21_checkpoint_filename(self) -> None:
        config = resolve_backend_config(
            BackendConfig(
                backend="efficientsam3",
                checkpoint_path="/tmp/efficient_sam3_tinyvit21_stage1_e32_h200_full_sam3.pt",
            )
        )

        self.assertEqual(config.backbone_type, "tinyvit")
        self.assertEqual(config.model_name, "21m")

    def test_resolves_instinctsam_vitb_checkpoint_filename(self) -> None:
        config = resolve_backend_config(
            BackendConfig(
                backend="efficientsam3",
                checkpoint_path="checkpoints/instinctsam/instinctsam_vitb_concept.pt",
            )
        )

        self.assertEqual(config.backbone_type, "vit_base")
        self.assertEqual(config.model_name, "base")
        self.assertEqual(config.text_encoder_type, "MobileCLIP-S1")
        self.assertEqual(config.text_encoder_context_length, 16)
        self.assertEqual(config.text_encoder_pos_embed_table_size, 77)

    def test_leaves_unknown_checkpoint_filename_unchanged(self) -> None:
        config = resolve_backend_config(
            BackendConfig(
                backend="efficientsam3",
                checkpoint_path="checkpoints/custom.pt",
                backbone_type="tinyvit",
                model_name="11m",
            )
        )

        self.assertEqual(config.backbone_type, "tinyvit")
        self.assertEqual(config.model_name, "11m")

    def test_resolves_autocast_dtype_aliases(self) -> None:
        class TorchModule:
            bfloat16 = "bf16"
            float16 = "fp16"

        self.assertEqual(_resolve_autocast_dtype(TorchModule, "bfloat16"), "bf16")
        self.assertEqual(_resolve_autocast_dtype(TorchModule, "fp16"), "fp16")
        self.assertIsNone(_resolve_autocast_dtype(TorchModule, "none"))

    def test_prepend_repo_path_handles_nested_efficientsam3_layout(self) -> None:
        original_path = list(sys.path)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sam3" / "sam3").mkdir(parents=True)
            (root / "sam3" / "sam3" / "model_builder.py").write_text("", encoding="utf-8")

            try:
                _prepend_repo_path(str(root))

                self.assertEqual(sys.path[0], str(root / "sam3"))
                self.assertEqual(sys.path[1], str(root))
            finally:
                sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
