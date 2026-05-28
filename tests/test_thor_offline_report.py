from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.thor_offline_report import write_thor_offline_reports


class ThorOfflineReportTest(unittest.TestCase):
    def test_write_thor_offline_reports_groups_tasks_and_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = tmp / "results" / "thor" / "offline"
            output = root / "reports"

            coco_summary = root / "coco" / "run_coco" / "coco_suite_component_summary.csv"
            coco_summary.parent.mkdir(parents=True)
            _write_csv(
                coco_summary,
                [
                    {
                        "model_id": "sam3",
                        "backend": "sam3",
                        "prompt_mode": "both",
                        "rows": "2",
                        "mean_total_ms": "10",
                        "params_total": "12",
                        "weight_total_bytes": "48",
                    }
                ],
            )

            sav_summary = root / "sav" / "run_sav" / "sam2p1_hiera_tiny" / "frames_summary.csv"
            sav_summary.parent.mkdir(parents=True)
            _write_csv(
                sav_summary,
                [
                    {
                        "model_id": "sam2p1_hiera_tiny",
                        "backend": "sam2",
                        "frames_tracked": "5",
                        "effective_fps": "20",
                        "params_total": "22",
                        "weight_total_bytes": "88",
                        "overlay_video": "overlay.mp4",
                    }
                ],
            )

            yoloe_summary = root / "yoloe_edgetam" / "run_yoloe" / "frames_summary.csv"
            yoloe_summary.parent.mkdir(parents=True)
            _write_csv(
                yoloe_summary,
                [
                    {
                        "source_id": "test1",
                        "status": "ok",
                        "frames_tracked": "7",
                        "effective_tracking_fps": "11",
                        "yoloe_params_total": "30",
                        "edgetam_params_total": "40",
                        "yoloe_weight_total_bytes": "120",
                        "edgetam_weight_total_bytes": "160",
                        "overlay_video": "overlay.mp4",
                    }
                ],
            )

            _write_file(tmp / "checkpoints" / "sam3" / "sam3.pt", 13)
            _write_file(tmp / "checkpoints" / "sam3" / "config.json", 3)
            _write_file(tmp / "checkpoints" / "sam2" / "sam2.1_hiera_tiny.pt", 17)
            _write_file(tmp / "checkpoints" / "yoloe" / "yoloe-26m-seg.pt", 19)
            _write_file(tmp / "checkpoints" / "edgetam" / "edgetam.pt", 23)

            paths = write_thor_offline_reports(root, output, tmp)

            self.assertIn(output / "thor_offline_coco_summary.csv", paths)
            self.assertIn(output / "thor_offline_sav_summary.csv", paths)
            self.assertIn(output / "thor_offline_yoloe_edgetam_summary.csv", paths)
            self.assertIn(output / "thor_offline_all_summary.csv", paths)
            self.assertIn(output / "thor_offline_model_storage_components.csv", paths)

            coco_rows = _read_csv(output / "thor_offline_coco_summary.csv")
            self.assertEqual(coco_rows[0]["task"], "coco")
            self.assertEqual(coco_rows[0]["run_id"], "run_coco")
            self.assertEqual(coco_rows[0]["storage_total_bytes"], "16")

            sav_rows = _read_csv(output / "thor_offline_sav_summary.csv")
            self.assertEqual(sav_rows[0]["frames_tracked"], "5")
            self.assertEqual(sav_rows[0]["overlay_videos"], "1")
            self.assertEqual(sav_rows[0]["storage_total_bytes"], "17")

            yoloe_rows = _read_csv(output / "thor_offline_yoloe_edgetam_summary.csv")
            self.assertEqual(yoloe_rows[0]["model_id"], "yoloe_26m_seg_edgetam")
            self.assertEqual(yoloe_rows[0]["params_total"], "70")
            self.assertEqual(yoloe_rows[0]["weight_total_bytes"], "280")
            self.assertEqual(yoloe_rows[0]["storage_total_bytes"], "42")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)


if __name__ == "__main__":
    unittest.main()
