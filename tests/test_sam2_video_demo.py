from __future__ import annotations

import unittest

import numpy as np

from sam_backend.sam2_video_demo import (
    display_scale,
    ffmpeg_video_args,
    masks_to_label_map,
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

    def test_compacts_three_masks_into_object_id_label_map(self) -> None:
        masks = np.zeros((3, 4, 5), dtype=bool)
        masks[0, 0:2, 0:2] = True
        masks[1, 2:4, 2:4] = True
        masks[2, 1:3, 4] = True

        label_map = masks_to_label_map(masks, [1, 2, 3])

        self.assertEqual(label_map.dtype, np.uint8)
        self.assertEqual(set(np.unique(label_map)), {0, 1, 2, 3})
        self.assertEqual(int(label_map[0, 0]), 1)
        self.assertEqual(int(label_map[3, 3]), 2)
        self.assertEqual(int(label_map[2, 4]), 3)

    def test_display_scale_bounds_portrait_video(self) -> None:
        self.assertAlmostEqual(display_scale((3840, 2160), 1600, 900), 900 / 3840)
        self.assertEqual(display_scale((480, 640), 1600, 900), 1.0)

    def test_ffmpeg_software_and_hardware_codec_arguments_differ(self) -> None:
        self.assertEqual(
            ffmpeg_video_args(
                "libx264",
                preset="medium",
                crf=18,
                width=1920,
                height=1080,
                fps=30.0,
            ),
            ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
        )
        self.assertEqual(
            ffmpeg_video_args(
                "h264_v4l2m2m",
                preset="medium",
                crf=18,
                width=3840,
                height=2160,
                fps=60.0,
            ),
            ["-c:v", "h264_v4l2m2m", "-b:v", "80M"],
        )


if __name__ == "__main__":
    unittest.main()
