from __future__ import annotations

import unittest

import numpy as np

from sam_backend.streaming import left_panel_click_to_image_point, masks_to_bbox_xyxy, masks_to_mono8, parse_tegrastats_gr3d


class StreamingHelpersTest(unittest.TestCase):
    def test_masks_to_mono8_merges_and_scales_masks(self) -> None:
        masks = np.zeros((2, 4, 5), dtype=np.uint8)
        masks[0, 1:3, 1:3] = 1
        masks[1, 2:4, 3:5] = 1

        mask = masks_to_mono8(masks, (4, 5))

        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(int(mask[1, 1]), 255)
        self.assertEqual(int(mask[3, 4]), 255)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_masks_to_mono8_returns_blank_mask_without_predictions(self) -> None:
        mask = masks_to_mono8([], (3, 4))

        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (3, 4))
        self.assertEqual(int(mask.max()), 0)

    def test_masks_to_bbox_xyxy(self) -> None:
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1

        self.assertEqual(masks_to_bbox_xyxy(mask, (5, 6)), (2.0, 1.0, 4.0, 3.0))
        self.assertIsNone(masks_to_bbox_xyxy(mask, (5, 6), min_area=100))
        self.assertIsNone(masks_to_bbox_xyxy([], (5, 6)))

    def test_left_panel_click_to_image_point(self) -> None:
        self.assertEqual(left_panel_click_to_image_point(10, 5, (20, 30)), (10.0, 5.0))
        self.assertIsNone(left_panel_click_to_image_point(30, 5, (20, 30)))
        self.assertIsNone(left_panel_click_to_image_point(10, 20, (20, 30)))

    def test_parse_tegrastats_gr3d(self) -> None:
        line = "RAM 4388/62801MB CPU [1%@729] GR3D_FREQ 42%@306 EMC_FREQ 3%@2133"

        self.assertEqual(parse_tegrastats_gr3d(line), 42.0)
        self.assertIsNone(parse_tegrastats_gr3d("RAM 4388/62801MB CPU [1%@729]"))


if __name__ == "__main__":
    unittest.main()
