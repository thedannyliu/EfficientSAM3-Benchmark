from __future__ import annotations

import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import cv2
import numpy as np

from sam_backend.sam2_video_demo import (
    FFmpegVideoWriter,
    ModelArtifacts,
    PromptSelector,
    VideoInfo,
    build_parser,
    display_scale,
    ffmpeg_video_args,
    load_prompts,
    masks_from_label_map,
    masks_to_label_map,
    model_save_speed_overrides,
    model_specs_from_args,
    overlay_masks,
    realtime_repeat_count,
    resize_masks,
    resolve_model_save_speeds,
    rolling_fps,
    save_model_outputs,
    select_audio_stream_index,
    smooth_transition_frames,
    scale_prompts,
)


class Sam2VideoDemoTest(unittest.TestCase):
    def test_builds_three_model_comparison_matrix(self) -> None:
        args = build_parser().parse_args(["--video-path", "demo.mov"])

        specs = model_specs_from_args(args)

        self.assertEqual(
            [spec.model_id for spec in specs],
            ["sam2p1_l", "tv21m_mse_cos", "tv5m_projection"],
        )
        self.assertEqual(
            specs[1].student_model_name,
            "tiny_vit_21m_512.dist_in22k_ft_in1k",
        )
        self.assertEqual(specs[1].student_adapter_mode, "auto")
        self.assertEqual(specs[2].student_adapter_mode, "projection")

    def test_legacy_tinyvit_arguments_still_target_tv5(self) -> None:
        args = build_parser().parse_args(
            [
                "--video-path",
                "demo.mov",
                "--tinyvit-stage1-checkpoint",
                "legacy-stage1.pt",
                "--tinyvit-backbone-checkpoint",
                "legacy-backbone.safetensors",
            ]
        )

        self.assertEqual(args.tv5_stage1_checkpoint, "legacy-stage1.pt")
        self.assertEqual(args.tv5_backbone_checkpoint, "legacy-backbone.safetensors")

    def test_selects_only_tv5_model(self) -> None:
        args = build_parser().parse_args(
            ["--video-path", "demo.mov", "--models", "tv5m_projection"]
        )

        specs = model_specs_from_args(args)

        self.assertEqual([spec.model_id for spec in specs], ["tv5m_projection"])

    def test_selects_tv21_and_tv5_models_for_backfill(self) -> None:
        args = build_parser().parse_args(
            [
                "--video-path",
                "demo.mov",
                "--models",
                "tv21m_mse_cos",
                "tv5m_projection",
                "--prompts-json",
                "previous/prompts.json",
            ]
        )

        self.assertEqual(
            [spec.model_id for spec in model_specs_from_args(args)],
            ["tv21m_mse_cos", "tv5m_projection"],
        )
        self.assertEqual(args.prompts_json, "previous/prompts.json")

    def test_resolves_per_model_save_speeds(self) -> None:
        args = build_parser().parse_args(
            [
                "--video-path",
                "demo.mov",
                "--sam2-save-speed",
                "20",
                "--tv21-save-speed",
                "10",
                "--tv5-save-speed",
                "5",
            ]
        )
        specs = model_specs_from_args(args)
        overrides = model_save_speed_overrides(args)

        speeds = resolve_model_save_speeds(
            specs,
            timing_mode="both",
            default_speed=args.save_speed,
            overrides=overrides,
        )

        self.assertEqual(
            speeds,
            {"sam2p1_l": 20.0, "tv21m_mse_cos": 10.0, "tv5m_projection": 5.0},
        )

    def test_source_fps_ignores_per_model_save_speeds(self) -> None:
        args = build_parser().parse_args(["--video-path", "demo.mov"])

        speeds = resolve_model_save_speeds(
            model_specs_from_args(args),
            timing_mode="source_fps",
            default_speed=2.0,
            overrides={"sam2p1_l": 20.0},
        )

        self.assertEqual(set(speeds.values()), {1.0})

    def test_loads_saved_point_and_box_prompts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                '[{"prompt_mode":"point","point":[10,20],"label":1},'
                '{"prompt_mode":"box","box":[1,2,30,40]}]',
                encoding="utf-8",
            )

            prompts = load_prompts(path)

        self.assertEqual(prompts[0]["point"], [10.0, 20.0])
        self.assertEqual(prompts[1]["box"], [1.0, 2.0, 30.0, 40.0])

    def test_rejects_empty_saved_prompts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "between 1 and 255"):
                load_prompts(path)

    def test_prompt_selector_accepts_more_than_three_objects(self) -> None:
        selector = PromptSelector(np.zeros((40, 60, 3), dtype=np.uint8), 1600, 900)

        with patch("builtins.print"):
            for x in (10, 20, 30, 40):
                selector._on_mouse(cv2.EVENT_LBUTTONDOWN, x, 10, 0, None)
                selector._on_mouse(cv2.EVENT_LBUTTONUP, x, 10, 0, None)

        self.assertEqual(len(selector.prompts), 4)

    def test_prompt_selector_starts_with_one_object(self) -> None:
        selector = PromptSelector(np.zeros((40, 60, 3), dtype=np.uint8), 1600, 900)
        selector.prompts.append(
            {"prompt_mode": "point", "point": [10.0, 10.0], "label": 1}
        )

        with (
            patch("sam_backend.sam2_video_demo.cv2.namedWindow"),
            patch("sam_backend.sam2_video_demo.cv2.setMouseCallback"),
            patch("sam_backend.sam2_video_demo.cv2.imshow"),
            patch("sam_backend.sam2_video_demo.cv2.waitKey", return_value=13),
            patch("sam_backend.sam2_video_demo.cv2.destroyWindow"),
            patch("builtins.print"),
        ):
            prompts = selector.run()

        self.assertEqual(len(prompts), 1)

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

    def test_realtime_speed_reduces_output_frames_without_changing_fps(self) -> None:
        self.assertEqual(realtime_repeat_count(1000.0, 30.0, 1.0), 30)
        self.assertEqual(realtime_repeat_count(1000.0, 30.0, 2.0), 15)
        self.assertEqual(realtime_repeat_count(1000.0, 30.0, 4.0), 8)

    def test_smooth_transition_replaces_repeated_latency_frames(self) -> None:
        previous = np.zeros((2, 2, 3), dtype=np.uint8)
        current = np.full((2, 2, 3), 120, dtype=np.uint8)

        frames = smooth_transition_frames(previous, current, 4)

        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [30, 60, 90, 120])

    def test_save_speed_applies_only_to_realtime_output(self) -> None:
        artifacts = ModelArtifacts(
            result={"model_id": "demo", "object_count": 1},
            mask_dir=Path("masks"),
            latencies_ms=[100.0],
            last_overlay=np.zeros((2, 2, 3), dtype=np.uint8),
        )
        with (
            TemporaryDirectory() as output_dir,
            patch("sam_backend.sam2_video_demo.shutil.which", return_value="ffmpeg"),
            patch(
                "sam_backend.sam2_video_demo.resolve_ffmpeg_codec",
                return_value="libx264",
            ),
            patch("sam_backend.sam2_video_demo._encode_model_output") as encode,
        ):
            paths = save_model_outputs(
                artifacts,
                video_path=Path("source.mov"),
                video_info=VideoInfo(width=2, height=2, fps=30.0, frame_count=1),
                output_dir=Path(output_dir),
                timing_mode="both",
                speed=2.0,
                codec="libx264",
                preset="medium",
                crf=18,
            )

        self.assertTrue(paths["source_fps"].endswith("demo_source_fps.mp4"))
        self.assertTrue(paths["realtime"].endswith("demo_realtime_2x.mp4"))
        self.assertEqual(
            [call.kwargs["speed"] for call in encode.call_args_list],
            [1.0, 2.0],
        )

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

    def test_restores_the_selected_number_of_masks_from_label_map(self) -> None:
        label_map = np.zeros((4, 5), dtype=np.uint8)
        label_map[0:2, 0:2] = 1
        label_map[2:4, 3:5] = 4

        masks = masks_from_label_map(label_map, 4)

        self.assertEqual(masks.shape, (4, 4, 5))
        self.assertEqual(int(masks[0].sum()), 4)
        self.assertEqual(int(masks[1].sum()), 0)
        self.assertEqual(int(masks[3].sum()), 4)

    def test_rejects_zero_objects_when_restoring_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 255"):
            masks_from_label_map(np.zeros((2, 2), dtype=np.uint8), 0)

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

    def test_ffmpeg_pads_short_audio_before_using_shortest(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch("sam_backend.sam2_video_demo.shutil.which", return_value="ffmpeg"),
            patch(
                "sam_backend.sam2_video_demo._ffprobe_audio_stream_index",
                return_value=1,
            ),
            patch(
                "sam_backend.sam2_video_demo.subprocess.Popen",
                return_value=Mock(),
            ) as popen,
        ):
            writer = FFmpegVideoWriter(
                Path(directory) / "output.mp4",
                width=64,
                height=48,
                fps=30.0,
                source_path=Path("source.mov"),
                codec="libx264",
                preset="medium",
                crf=18,
            )
            command = popen.call_args.args[0]
            writer.log_handle.close()

        self.assertEqual(
            command[command.index("-af") : command.index("-af") + 2],
            ["-af", "apad"],
        )
        self.assertIn("-shortest", command)

    def test_rolling_display_fps_uses_recent_completion_times(self) -> None:
        self.assertIsNone(rolling_fps(deque([1.0])))
        self.assertEqual(rolling_fps(deque([1.0, 1.5, 2.0])), 2.0)

    def test_selects_first_decodable_audio_stream_from_iphone_mov(self) -> None:
        payload = {
            "streams": [
                {"index": 2, "codec_name": "none"},
                {"index": 1, "codec_name": "aac"},
                {"index": 3, "codec_name": "unknown"},
            ]
        }

        self.assertEqual(select_audio_stream_index(payload), 1)

    def test_ignores_mov_audio_streams_without_a_decoder(self) -> None:
        payload = {"streams": [{"index": 2, "codec_name": "none"}]}

        self.assertIsNone(select_audio_stream_index(payload))


if __name__ == "__main__":
    unittest.main()
