from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.coco_manifest import build_coco_manifest
from sam_backend.profile_coco import _build_prompt, profile_coco


class CocoProfileNullTest(unittest.TestCase):
    def test_manifest_and_null_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_dir = tmp / "images"
            image_dir.mkdir()
            image_path = image_dir / "000000000001.jpg"
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            frame[5:15, 10:20] = (255, 255, 255)
            self.assertTrue(cv2.imwrite(str(image_path), frame))

            annotations_path = tmp / "instances.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": image_path.name, "width": 30, "height": 20}],
                        "categories": [{"id": 1, "name": "square"}],
                        "annotations": [
                            {
                                "id": 7,
                                "image_id": 1,
                                "category_id": 1,
                                "iscrowd": 0,
                                "area": 100,
                                "bbox": [10, 5, 10, 10],
                                "segmentation": [[10, 5, 19, 5, 19, 14, 10, 14]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest_rows = build_coco_manifest(
                annotations=annotations_path,
                image_dir=image_dir,
                count=1,
                seed=123,
                min_area=1,
            )
            self.assertEqual(manifest_rows[0]["text_prompt"], "square")
            self.assertEqual(manifest_rows[0]["annotation_id"], 7)

            manifest_path = tmp / "manifest.jsonl"
            manifest_path.write_text(json.dumps(manifest_rows[0]) + "\n", encoding="utf-8")
            csv_path = tmp / "profile.csv"
            summary_path = tmp / "summary.json"
            args = argparse.Namespace(
                manifest=manifest_path,
                model_id="null-coco",
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
                prompt_mode="both",
                csv_output=csv_path,
                summary_output=summary_path,
                overlay_dir=None,
            )
            summary = profile_coco(args)

            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["prompt_modes"]["text"]["miou_best"], 0.0)
            self.assertEqual(summary["prompt_modes"]["point"]["miou_best"], 0.0)

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["prompt_mode"] for row in rows], ["text", "point"])

            box_prompt = _build_prompt(manifest_rows[0], "box")
            self.assertEqual(box_prompt.boxes[0], (10.0, 5.0, 20.0, 15.0))

            args.prompt_mode = "all"
            csv_path_all = tmp / "profile_all.csv"
            args.csv_output = csv_path_all
            summary = profile_coco(args)
            self.assertEqual(summary["rows"], 3)

            with csv_path_all.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["prompt_mode"] for row in rows], ["text", "point", "box"])
            self.assertEqual(rows[-1]["box_x1"], "10.0")
            self.assertEqual(rows[-1]["box_y1"], "5.0")
            self.assertEqual(rows[-1]["box_x2"], "20.0")
            self.assertEqual(rows[-1]["box_y2"], "15.0")


if __name__ == "__main__":
    unittest.main()
