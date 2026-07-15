from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_backend.sam2_stage1 import (
    extract_stage1_state_dict,
    resolve_stage1_backbone_checkpoint,
    resolve_stage1_spec,
)


class Sam2Stage1CompatibilityTest(unittest.TestCase):
    def test_extracts_model_state_and_strips_module_prefix(self) -> None:
        state = {"module.projections.image_embed.weight": np.zeros((1, 1, 1, 1))}
        extracted = extract_stage1_state_dict({"model_state": state})
        self.assertEqual(list(extracted), ["projections.image_embed.weight"])

    def test_resolves_tinyvit_metadata_from_current_checkpoint(self) -> None:
        checkpoint = {
            "args": {
                "student_family": "tinyvit",
                "model_name": "tiny_vit_5m_224.dist_in22k_ft_in1k",
                "adapter_mode": "projection",
            }
        }
        result = resolve_stage1_spec(
            checkpoint,
            {},
            requested_family="tinyvit",
            requested_model_name="tiny_vit_5m_224.dist_in22k_ft_in1k",
            fallback_model_name="tiny_vit_21m_512.dist_in22k_ft_in1k",
            requested_adapter_mode="projection",
        )
        self.assertEqual(
            result,
            (
                "tinyvit",
                "tiny_vit_5m_224.dist_in22k_ft_in1k",
                "projection",
            ),
        )

    def test_explicit_repvit_name_disambiguates_384_channel_projection(self) -> None:
        state = {"projections.image_embed.weight": np.zeros((256, 384, 1, 1))}
        result = resolve_stage1_spec(
            {},
            state,
            requested_family="repvit",
            requested_model_name="repvit_m0_9.dist_450e_in1k",
            fallback_model_name="tiny_vit_21m_512.dist_in22k_ft_in1k",
            requested_adapter_mode="projection",
        )
        self.assertEqual(
            result,
            ("repvit", "repvit_m0_9.dist_450e_in1k", "projection"),
        )

    def test_rejects_family_that_conflicts_with_checkpoint(self) -> None:
        checkpoint = {
            "args": {
                "student_family": "tinyvit",
                "model_name": "tiny_vit_5m_224.dist_in22k_ft_in1k",
            }
        }
        with self.assertRaisesRegex(ValueError, "student_family=repvit conflicts"):
            resolve_stage1_spec(
                checkpoint,
                {},
                requested_family="repvit",
                requested_model_name="",
                fallback_model_name="tiny_vit_5m_224.dist_in22k_ft_in1k",
                requested_adapter_mode="auto",
            )

    def test_resolves_expected_backbone_name_next_to_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "repvit_m0_9.dist_450e_in1k.safetensors"
            expected.touch()
            resolved = resolve_stage1_backbone_checkpoint(
                "repvit_m0_9.dist_450e_in1k",
                str(root / "model.safetensors"),
            )
        self.assertEqual(resolved, expected)


if __name__ == "__main__":
    unittest.main()
