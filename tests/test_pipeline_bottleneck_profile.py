from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.pipeline_bottleneck_profile import _bottleneck_hint, profile_pipeline


class PipelineBottleneckProfileTest(unittest.TestCase):
    def test_null_sam_profile_writes_timing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "sample.jpg"
            image = np.zeros((16, 24, 3), dtype=np.uint8)
            image[4:12, 6:18] = 255
            self.assertTrue(cv2.imwrite(str(image_path), image))
            manifest_path = tmp / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "image_path": str(image_path),
                        "image_id": 1,
                        "annotation_id": 2,
                        "category_name": "square",
                        "text_prompt": "square",
                        "width": 24,
                        "height": 16,
                        "point": [12, 8],
                        "point_label": 1,
                        "bbox_xywh": [6, 4, 12, 8],
                        "segmentation": [[6, 4, 17, 4, 17, 11, 6, 11]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                manifest=manifest_path,
                limit=1,
                suite="sam",
                model_id="null_pipeline",
                device="cpu",
                warmup=1,
                repeat=2,
                input_mode="read-each-time",
                with_gt=True,
                with_torch_profiler=False,
                csv_output=tmp / "pipeline.csv",
                summary_output=tmp / "summary.json",
                backend="null",
                checkpoint_path=None,
                model_config=None,
                external_repo=None,
                backbone_type="efficientvit",
                model_name="b0",
                text_encoder_type=None,
                text_encoder_context_length=77,
                text_encoder_pos_embed_table_size=None,
                interpolate_pos_embed=False,
                mobile_sam_model_type="vit_t",
                prompt_mode="point",
                family="yolo-seg",
                weights="",
                imgsz=640,
                conf=0.25,
                iou=0.7,
                max_det=100,
                max_det_for_iou=100,
                agnostic_nms=None,
            )

            summary = profile_pipeline(args)

            self.assertEqual(summary["suite"], "sam")
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["bottleneck_hint"], "cpu_or_no_cuda_timing")
            self.assertGreaterEqual(summary["mean_total_pipeline_ms"], 0.0)

            with args.csv_output.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["prompt_mode"], "point")
            self.assertIn("predict_wall_ms", rows[0])
            self.assertIn("predict_cuda_window_ms", rows[0])
            self.assertIn("gt_decode_ms", rows[0])

    def test_bottleneck_hint_prefers_cpu_gap_when_cuda_time_is_small(self) -> None:
        summary = {
            "cuda_available": True,
            "mean_predict_cuda_window_ms": 5.0,
            "mean_predict_torch_cuda_kernel_ms": "",
            "gpu_time_fraction_of_pipeline": 0.05,
            "mean_predict_cpu_gap_ms": 90.0,
            "mean_total_pipeline_ms": 100.0,
            "mean_non_predict_pipeline_ms": 5.0,
            "mean_postprocess_ms": 1.0,
        }

        self.assertEqual(_bottleneck_hint(summary), "cpu_wrapper_sync_or_copy_bound")


if __name__ == "__main__":
    unittest.main()
