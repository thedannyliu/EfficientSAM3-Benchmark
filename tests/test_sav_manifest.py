from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.sav_manifest import build_sav_manifest


class SavManifestTest(unittest.TestCase):
    def test_build_sav_manifest_selects_largest_first_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_root = root / "JPEGImages_24fps"
            ann_root = root / "Annotations_6fps"
            for video_id in ["video_a", "video_b", "video_c"]:
                (frames_root / video_id).mkdir(parents=True)
                (ann_root / video_id / "000").mkdir(parents=True)
                (ann_root / video_id / "001").mkdir(parents=True)
                frame = np.zeros((20, 30, 3), dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(frames_root / video_id / "00000.jpg"), frame))
                small = np.zeros((20, 30), dtype=np.uint8)
                small[1:4, 1:4] = 255
                large = np.zeros((20, 30), dtype=np.uint8)
                large[5:15, 10:20] = 255
                self.assertTrue(cv2.imwrite(str(ann_root / video_id / "000" / "00000.png"), small))
                self.assertTrue(cv2.imwrite(str(ann_root / video_id / "001" / "00000.png"), large))

            rows = build_sav_manifest(root, count=3, seed=1)

            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["object_id"] == "001" for row in rows))
            self.assertTrue(all(row["prompt_frame_index"] == 0 for row in rows))
            self.assertTrue(all(row["initial_mask_area"] == 100 for row in rows))
            self.assertTrue(all(row["selection"] == "random_video_seeded_largest_first_mask" for row in rows))

    def test_salient_policy_filters_tiny_and_thin_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_root = root / "JPEGImages_24fps"
            ann_root = root / "Annotations_6fps"
            video_id = "video_a"
            (frames_root / video_id).mkdir(parents=True)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(frames_root / video_id / "00000.jpg"), frame))
            for object_id in ["000", "001", "002"]:
                (ann_root / video_id / object_id).mkdir(parents=True)

            tiny = np.zeros((100, 100), dtype=np.uint8)
            tiny[1:6, 1:6] = 255
            thin = np.zeros((100, 100), dtype=np.uint8)
            thin[10:12, 0:90] = 255
            salient = np.zeros((100, 100), dtype=np.uint8)
            salient[40:70, 40:70] = 255
            self.assertTrue(cv2.imwrite(str(ann_root / video_id / "000" / "00000.png"), tiny))
            self.assertTrue(cv2.imwrite(str(ann_root / video_id / "001" / "00000.png"), thin))
            self.assertTrue(cv2.imwrite(str(ann_root / video_id / "002" / "00000.png"), salient))

            rows = build_sav_manifest(
                root,
                count=1,
                seed=1,
                selection_policy="salient_first_mask",
                min_area_ratio=0.01,
                max_aspect_ratio=4.0,
            )

            self.assertEqual(rows[0]["object_id"], "002")
            self.assertEqual(rows[0]["selection"], "random_video_seeded_salient_first_mask")
            self.assertAlmostEqual(rows[0]["initial_mask_area_ratio"], 0.09)
            self.assertLessEqual(rows[0]["initial_bbox_aspect_ratio"], 4.0)


if __name__ == "__main__":
    unittest.main()
