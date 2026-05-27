from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.overlay import to_numpy
from sam_backend.profile_image import profile_image


class NullProfileImageTest(unittest.TestCase):
    def test_to_numpy_handles_torch_bfloat16(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        values = to_numpy(torch.ones(2, dtype=torch.bfloat16))

        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.tolist(), [1.0, 1.0])

    def test_profile_image_writes_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "cats.png"
            overlay_path = tmp / "overlay.png"
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[12:36, 20:44] = (255, 255, 255)
            self.assertTrue(cv2.imwrite(str(image_path), frame))

            args = argparse.Namespace(
                model_id="null-cats",
                backend="null",
                checkpoint_path=None,
                device="cpu",
                model_config=None,
                external_repo=None,
                backbone_type="efficientvit",
                model_name="b0",
                text_encoder_type=None,
                text_encoder_context_length=77,
                text_encoder_pos_embed_table_size=None,
                interpolate_pos_embed=False,
                prompt="cats",
                point=None,
                point_label=None,
                point_normalized=False,
                image=image_path,
                json_output=None,
                overlay_output=overlay_path,
            )
            summary = profile_image(args)

            self.assertEqual(summary["backend"], "null")
            self.assertEqual(summary["prompt"], "cats")
            self.assertEqual(summary["mask_count"], 1)
            self.assertTrue(overlay_path.exists())

    def test_profile_image_accepts_normalized_point_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "cat1.jpeg"
            frame = np.zeros((50, 100, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), frame))

            args = argparse.Namespace(
                model_id="null-cat1-point",
                backend="null",
                checkpoint_path=None,
                device="cpu",
                model_config=None,
                external_repo=None,
                backbone_type="efficientvit",
                model_name="b0",
                text_encoder_type=None,
                text_encoder_context_length=77,
                text_encoder_pos_embed_table_size=None,
                interpolate_pos_embed=False,
                prompt=None,
                point=["0.5,0.5"],
                point_label=[1],
                point_normalized=True,
                image=image_path,
                json_output=None,
                overlay_output=None,
            )
            summary = profile_image(args)

            self.assertEqual(summary["points"], [(50.0, 25.0)])
            self.assertEqual(summary["labels"], [1])


if __name__ == "__main__":
    unittest.main()
