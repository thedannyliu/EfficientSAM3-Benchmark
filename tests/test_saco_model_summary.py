from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.saco_model_summary import collect_offline_rows, collect_ros_rows


class SacoModelSummaryTest(unittest.TestCase):
    def test_collect_offline_rows_reads_frames_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "sam3_ref_native"
            model_dir.mkdir()
            _write_csv(
                model_dir / "frames_summary.csv",
                [
                    {
                        "model_id": "sam3_ref_native",
                        "backend": "sam3",
                        "stream_mode": "native_video",
                        "frames": "10",
                        "mean_iou": "0.5",
                        "mean_latency_ms": "20",
                        "mean_end_to_end_ms": "25",
                        "effective_fps": "40",
                        "params_total": "1000000",
                        "weight_total_bytes": "1048576",
                    }
                ],
            )

            rows = collect_offline_rows(root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["layer"], "offline")
            self.assertEqual(rows[0]["model_id"], "sam3_ref_native")
            self.assertEqual(rows[0]["mean_model_latency_ms"], "20")
            self.assertEqual(rows[0]["params_total_m"], 1.0)
            self.assertEqual(rows[0]["weight_total_mb"], 1.0)

    def test_collect_ros_rows_reads_recorder_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "mobilesam_vit_t_bbox_chain"
            model_dir.mkdir()
            _write_csv(
                model_dir / "summary.csv",
                [
                    {
                        "frames": "5",
                        "mean_latency_ms": "11",
                        "mean_end_to_end_ms": "22",
                        "mean_end_to_end_fps": "45.45",
                        "mean_mask_count": "1",
                    }
                ],
            )
            _write_csv(
                root / "ros_saco_stream_summary.csv",
                [
                    {
                        "model_id": "mobilesam_vit_t_bbox_chain",
                        "status": "ok",
                        "overlay_video": "overlay.mp4",
                    }
                ],
            )

            rows = collect_ros_rows(root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["layer"], "ros_video_stream")
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["end_to_end_fps"], "45.45")
            self.assertEqual(rows[0]["overlay_count"], 1)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
