from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.sav_video_report import write_sav_video_report


class SavVideoReportTest(unittest.TestCase):
    def test_write_sav_video_report_summarizes_model_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "sam2_tiny"
            model_dir.mkdir()
            with (model_dir / "frames_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_id",
                        "backend",
                        "frames_tracked",
                        "gt_frames_evaluated",
                        "mean_iou",
                        "effective_fps",
                        "image_encoder_ms",
                        "params_total",
                        "weight_total_bytes",
                        "overlay_video",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model_id": "sam2_tiny",
                        "backend": "sam2",
                        "frames_tracked": "5",
                        "gt_frames_evaluated": "2",
                        "mean_iou": "0.5",
                        "effective_fps": "20",
                        "image_encoder_ms": "7",
                        "params_total": "11",
                        "weight_total_bytes": "44",
                        "overlay_video": "overlay.mp4",
                    }
                )

            report = write_sav_video_report(root)

            with report.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["model_id"], "sam2_tiny")
            self.assertEqual(rows[0]["frames_tracked"], "5")
            self.assertEqual(rows[0]["mean_image_encoder_ms"], "7.0")
            self.assertEqual(rows[0]["overlay_videos"], "1")
            self.assertEqual(rows[0]["params_total_m"], "1.1e-05")
            self.assertEqual(rows[0]["weight_total_mb"], str(44 / (1024.0 * 1024.0)))


if __name__ == "__main__":
    unittest.main()
