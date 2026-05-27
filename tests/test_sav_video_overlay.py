from __future__ import annotations

import unittest

import numpy as np

from sam_backend.profile_sav_video import _overlay_video_frame


class SAVVideoOverlayTest(unittest.TestCase):
    def test_overlay_video_frame_draws_prediction_and_gt(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        pred = np.zeros((20, 30), dtype=bool)
        pred[5:15, 8:18] = True
        gt = np.zeros((40, 60), dtype=bool)
        gt[10:30, 16:36] = True
        item = {"video_id": "sav_test", "object_id": "000"}

        overlay = _overlay_video_frame(frame, pred, gt, "model", item, 0, 0.5)

        self.assertEqual(overlay.shape, frame.shape)
        self.assertGreater(int(overlay.sum()), 0)
        self.assertGreater(int(overlay[20, 25, 1]), int(frame[20, 25, 1]))


if __name__ == "__main__":
    unittest.main()
