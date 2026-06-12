from __future__ import annotations

import unittest

from sam_backend import BackendConfig, resolve_backend_config


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


if __name__ == "__main__":
    unittest.main()
