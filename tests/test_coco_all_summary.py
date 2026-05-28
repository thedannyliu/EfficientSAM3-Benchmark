from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.coco_all_summary import write_coco_all_summary


class CocoAllSummaryTest(unittest.TestCase):
    def test_merges_sam_and_yolo_model_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sam_dir = tmp / "sam"
            yolo_dir = tmp / "yolo"
            sam_dir.mkdir()
            yolo_dir.mkdir()
            _write_csv(
                sam_dir / "coco_suite_component_summary.csv",
                [
                    {
                        "model_id": "sam2p1_hiera_tiny",
                        "backend": "sam2",
                        "prompt_mode": "point",
                        "samples": "2",
                        "rows": "2",
                        "effective_fps": "5",
                        "miou_best": "0.2",
                        "params_total_m": "38.9",
                        "params_detector_m": "",
                        "checkpoint_file_mb": "148.8",
                    }
                ],
            )
            _write_csv(
                yolo_dir / "yolo_coco_model_summary.csv",
                [
                    {
                        "model_id": "yoloe_26n_seg",
                        "family": "yoloe-seg",
                        "samples": "2",
                        "rows": "2",
                        "effective_fps": "30",
                        "miou_best": "0.4",
                        "params_total_m": "12.3",
                        "params_detector_m": "12.3",
                        "checkpoint_file_mb": "25.0",
                    }
                ],
            )

            output = write_coco_all_summary(sam_dir, yolo_dir, tmp / "all.csv")

            rows = _read_csv(output)
            self.assertEqual([row["model_id"] for row in rows], ["sam2p1_hiera_tiny", "yoloe_26n_seg"])
            self.assertEqual(rows[0]["suite"], "sam_coco")
            self.assertEqual(rows[0]["prompt_mode"], "point")
            self.assertEqual(rows[1]["suite"], "yolo_coco")
            self.assertEqual(rows[1]["prompt_mode"], "text")
            self.assertEqual(rows[1]["effective_fps"], "30")
            self.assertEqual(rows[1]["params_detector_m"], "12.3")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
