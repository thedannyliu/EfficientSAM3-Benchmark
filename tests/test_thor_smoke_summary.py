from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.thor_smoke_summary import write_smoke_summary


class ThorSmokeSummaryTest(unittest.TestCase):
    def test_merge_smoke_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sam2d = root / "sav_sam2d" / "sam2_stage1" / "benchmark_summary.csv"
            _write_csv(
                sam2d,
                [
                    {
                        "mode": "video_tracking",
                        "prompt": "point",
                        "model": "sam2p1_l",
                        "status": "pass",
                        "J&F": "0.5",
                        "J": "0.4",
                        "F": "0.6",
                        "elapsed_sec": "10",
                        "sec_per_video": "10",
                    }
                ],
            )
            coco = root / "sa1b_sam_family" / "coco_suite_model_summary.csv"
            _write_csv(
                coco,
                [
                    {
                        "model_id": "sam3",
                        "backend": "sam3",
                        "prompt_mode": "point",
                        "samples": "1",
                        "rows": "1",
                        "miou_best": "0.7",
                        "AP": "0.8",
                        "AP50": "1.0",
                        "AP75": "0.0",
                        "mean_total_ms": "20",
                        "effective_fps": "50",
                    }
                ],
            )

            output = write_smoke_summary(root, root / "summary.csv")

            rows = _read_csv(output)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["task"], "video_tracking")
            self.assertEqual(rows[0]["model_id"], "sam2p1_l")
            self.assertEqual(rows[0]["J&F"], "0.5")
            self.assertEqual(rows[1]["task"], "image_segmentation")
            self.assertEqual(rows[1]["model_id"], "sam3")
            self.assertEqual(rows[1]["AP50"], "1.0")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
