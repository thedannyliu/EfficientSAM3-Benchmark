from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.profile_sav_frames import profile_sav_frames


class SAVFrameProfileTest(unittest.TestCase):
    def test_null_backend_profiles_point_and_text_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames_dir = tmp / "JPEGImages_24fps" / "video_a"
            mask_dir = tmp / "Annotations_6fps" / "video_a" / "001"
            frames_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for frame_idx in [0, 4]:
                frame = np.zeros((20, 30, 3), dtype=np.uint8)
                frame[5:15, 10:20] = 255
                self.assertTrue(cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.jpg"), frame))
                mask = np.zeros((20, 30), dtype=np.uint8)
                mask[5:15, 10:20] = 255
                self.assertTrue(cv2.imwrite(str(mask_dir / f"{frame_idx:05d}.png"), mask))

            manifest = tmp / "sav_text.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sav_video_a_001",
                        "video_id": "video_a",
                        "frames_dir": str(frames_dir),
                        "annotations_dir": str(tmp / "Annotations_6fps" / "video_a"),
                        "object_id": "001",
                        "prompt_frame_index": 0,
                        "point": [14.5, 9.5],
                        "point_label": 1,
                        "text_prompt": "white square",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                manifest=manifest,
                limit=0,
                max_frames=2,
                frame_stride=1,
                model_id="null_sav_frames",
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
                mobile_sam_model_type="vit_t",
                prompt_mode="both",
                csv_output=tmp / "frames.csv",
                summary_output=tmp / "summary.json",
                overlay_dir=None,
            )

            summary = profile_sav_frames(args)

            self.assertEqual(summary["videos"], 1)
            self.assertEqual(summary["rows"], 4)
            self.assertEqual(summary["prompt_modes"], ["point", "text"])

            with args.csv_output.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["prompt_mode"] for row in rows}, {"point", "text"})
            point_rows = [row for row in rows if row["prompt_mode"] == "point"]
            self.assertEqual(point_rows[0]["point_x"], "14.5")
            self.assertEqual(point_rows[0]["point_y"], "9.5")

    def test_text_prompt_requires_text_enabled_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            frames_dir = tmp / "JPEGImages_24fps" / "video_a"
            mask_dir = tmp / "Annotations_6fps" / "video_a" / "001"
            frames_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(frames_dir / "00000.jpg"), np.zeros((10, 10, 3), dtype=np.uint8)))
            mask = np.ones((10, 10), dtype=np.uint8) * 255
            self.assertTrue(cv2.imwrite(str(mask_dir / "00000.png"), mask))
            manifest = tmp / "sav.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sav_video_a_001",
                        "video_id": "video_a",
                        "frames_dir": str(frames_dir),
                        "annotations_dir": str(tmp / "Annotations_6fps" / "video_a"),
                        "object_id": "001",
                        "prompt_frame_index": 0,
                        "point": [5.0, 5.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                manifest=manifest,
                limit=0,
                max_frames=1,
                frame_stride=1,
                model_id="null_sav_frames",
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
                mobile_sam_model_type="vit_t",
                prompt_mode="text",
                csv_output=tmp / "frames.csv",
                summary_output=None,
                overlay_dir=None,
            )

            with self.assertRaisesRegex(ValueError, "text prompt missing"):
                profile_sav_frames(args)


if __name__ == "__main__":
    unittest.main()
