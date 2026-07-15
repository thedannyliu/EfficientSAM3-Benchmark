from __future__ import annotations

import unittest

import numpy as np

from sam_backend.sam2_video_demo import (
    display_scale,
    overlay_masks,
    realtime_repeat_count,
    resize_masks,
    scale_prompts,
)


class Sam2VideoDemoTest(unittest.TestCase):
    def test_scales_point_and_box_prompts_to_inference_frames(self) -> None:
        prompts = [
            {"prompt_mode": "point", "point": [960.0, 540.0], "label": 1},
            {"prompt_mode": "box", "box": [100.0, 200.0, 500.0, 600.0]},
        ]

        scaled = scale_prompts(
            prompts,
            source_size=(1920, 1080),
            inference_size=(1024, 576),
        )

        self.assertEqual(scaled[0]["point"], [512.0, 288.0])
        self.assertEqual(
            scaled[1]["box"],
            [100.0 * 1024 / 1920, 200.0 * 576 / 1080, 500.0 * 1024 / 1920, 320.0],
        )

    def test_realtime_repeat_count_quantizes_latency_at_source_fps(self) -> None:
        self.assertEqual(realtime_repeat_count(0.0, 30.0), 1)
        self.assertEqual(realtime_repeat_count(33.0, 30.0), 1)
        self.assertEqual(realtime_repeat_count(34.0, 30.0), 2)
        self.assertEqual(realtime_repeat_count(100.0, 30.0), 3)

    def test_resizes_binary_masks_back_to_source_resolution(self) -> None:
        logits = np.full((2, 1, 2, 3), -1.0, dtype=np.float32)
        logits[0, 0, 0, 0] = 1.0
        logits[1, 0, 1, 2] = 1.0

        masks = resize_masks(logits, target_size=(6, 4))

        self.assertEqual(masks.shape, (2, 4, 6))
        self.assertTrue(masks.dtype == np.bool_)
        self.assertEqual(int(masks[0].sum()), 4)
        self.assertEqual(int(masks[1].sum()), 4)

    def test_overlay_keeps_original_frame_dimensions(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        masks = np.zeros((3, 40, 60), dtype=bool)
        masks[0, 10:20, 10:20] = True
        masks[1, 20:30, 20:30] = True
        masks[2, 5:10, 40:50] = True

        overlay = overlay_masks(
            frame,
            masks,
            [1, 2, 3],
            model_id="sam2p1_l",
            frame_index=1,
            latency_ms=10.0,
        )

        self.assertEqual(overlay.shape, frame.shape)
        self.assertGreater(int(overlay.sum()), 0)

    def test_display_scale_bounds_portrait_video(self) -> None:
        self.assertAlmostEqual(display_scale((3840, 2160), 1600, 900), 900 / 3840)
        self.assertEqual(display_scale((480, 640), 1600, 900), 1.0)


if __name__ == "__main__":
    unittest.main()
