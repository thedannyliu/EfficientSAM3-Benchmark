from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.thor_pipeline_smoke import _build_prompt, _capture_source, run_smoke


class ThorPipelineSmokeTest(unittest.TestCase):
    def test_capture_source_accepts_camera_index_or_path(self) -> None:
        self.assertEqual(_capture_source("0"), 0)
        self.assertEqual(_capture_source("videos/test1.mov"), "videos/test1.mov")

    def test_build_prompt_accepts_normalized_point(self) -> None:
        args = argparse.Namespace(point=["0.25,0.5"], point_label=[1], point_normalized=True, prompt="monitor")

        prompt = _build_prompt(args, width=200, height=100)

        self.assertEqual(prompt.points, [(50.0, 50.0)])
        self.assertEqual(prompt.labels, [1])
        self.assertIsNone(prompt.text)

    def test_null_smoke_writes_sample_and_overlay_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = tmp / "input.mp4"
            self._write_video(video_path)
            args = argparse.Namespace(
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
                prompt="monitor",
                point=None,
                point_label=None,
                point_normalized=False,
                video=str(video_path),
                frame_id="camera",
                max_frames=1,
                output_jsonl=tmp / "result.jsonl",
                overlay_output=tmp / "overlay.mp4",
                sample_frame_dir=tmp / "sampled_frames",
                overlay_frame_dir=tmp / "overlay_frames",
            )

            run_smoke(args)

            rows = [json.loads(line) for line in args.output_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["backend"], "null")
            self.assertEqual(rows[0]["frame_id"], "camera")
            self.assertTrue((args.sample_frame_dir / "frame_000000.png").exists())
            self.assertTrue((args.overlay_frame_dir / "frame_000000.png").exists())
            self.assertTrue(args.overlay_output.exists())

    def _write_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[4:20, 8:24] = (255, 255, 255)
            writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
