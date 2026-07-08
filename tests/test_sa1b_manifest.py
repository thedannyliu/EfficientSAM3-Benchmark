from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam_backend.sa1b_manifest import build_sa1b_manifest


class Sa1bManifestTest(unittest.TestCase):
    def test_build_manifest_from_extracted_json_and_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.zeros((12, 14, 3), dtype=np.uint8)
            image[2:8, 3:9] = 255
            Image.fromarray(image).save(root / "sa_1.jpg")
            (root / "sa_1.json").write_text(
                json.dumps(
                    {
                        "image": {"file_name": "sa_1.jpg", "width": 14, "height": 12, "id": "sa_1"},
                        "annotations": [
                            {
                                "id": 5,
                                "area": 36,
                                "bbox": [3, 2, 6, 6],
                                "segmentation": [[3, 2, 8, 2, 8, 7, 3, 7]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = build_sa1b_manifest(root, root, count=1, seed=123, min_area=1)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "sa1b")
            self.assertEqual(rows[0]["image_id"], "sa_1")
            self.assertEqual(rows[0]["annotation_id"], 5)
            self.assertEqual(rows[0]["category_name"], "object")
            self.assertEqual(rows[0]["bbox_xywh"], [3, 2, 6, 6])
            self.assertEqual(rows[0]["width"], 14)
            self.assertEqual(rows[0]["height"], 12)


if __name__ == "__main__":
    unittest.main()
