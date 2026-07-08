from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_backend.manifest_mask_layout import write_mask_layout


class ManifestMaskLayoutTest(unittest.TestCase):
    def test_write_one_frame_mask_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "image.jpg"
            Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(image_path)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sa1b test",
                        "image_path": str(image_path),
                        "width": 12,
                        "height": 10,
                        "segmentation": [[2, 3, 6, 3, 6, 7, 2, 7]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output = write_mask_layout(manifest, tmp / "layout", copy_images=True)

            self.assertTrue((output / "JPEGImages_24fps" / "sa1b_test" / "00000.jpg").exists())
            mask_path = output / "Annotations_6fps" / "sa1b_test" / "1" / "00000.png"
            self.assertTrue(mask_path.exists())
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(int(mask.sum()), 0)
            self.assertEqual((output / "sav_train_benchmark.txt").read_text(encoding="utf-8"), "sa1b_test\n")


if __name__ == "__main__":
    unittest.main()
