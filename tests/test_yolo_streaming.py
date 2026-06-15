from __future__ import annotations

import unittest

import numpy as np

from sam_backend.yolo_streaming import detections_to_arrays, max_score


class YoloStreamingHelpersTest(unittest.TestCase):
    def test_detections_to_arrays_keeps_present_masks_boxes_and_scores(self) -> None:
        mask = np.ones((3, 4), dtype=bool)
        box = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        masks, boxes, scores = detections_to_arrays(
            [
                {"mask": mask, "box": box, "score": 0.7},
                {"mask": None, "box": None, "score": ""},
            ]
        )

        self.assertEqual(len(masks), 1)
        self.assertTrue(np.array_equal(masks[0], mask))
        self.assertEqual(len(boxes), 1)
        self.assertTrue(np.array_equal(boxes[0], box))
        self.assertEqual(scores, [0.7])

    def test_max_score(self) -> None:
        self.assertEqual(max_score([0.1, 0.9, 0.3]), 0.9)
        self.assertIsNone(max_score([]))


if __name__ == "__main__":
    unittest.main()
