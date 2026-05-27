from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.coco_suite import run_suite, write_component_summary


class CocoSuiteTest(unittest.TestCase):
    def test_dry_run_builds_selected_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = argparse.Namespace(
                manifest=tmp / "manifest.jsonl",
                limit=1,
                device="cpu",
                models=["sam3"],
                output_dir=tmp / "results",
                overlay_dir=None,
                skip_missing=False,
                dry_run=True,
            )

            rows = run_suite(args)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_id"], "sam3")
            self.assertEqual(rows[0]["status"], "dry-run")
            self.assertIn("sam_backend.profile_coco", rows[0]["message"])
            self.assertIn("--external-repo external/sam3", rows[0]["message"])
            self.assertIn("--limit 1", rows[0]["message"])

    def test_unknown_model_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = argparse.Namespace(
                manifest=tmp / "manifest.jsonl",
                limit=0,
                device="cpu",
                models=["missing_model"],
                output_dir=tmp / "results",
                overlay_dir=None,
                skip_missing=False,
                dry_run=True,
            )

            with self.assertRaises(ValueError):
                run_suite(args)

    def test_component_summary_groups_by_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / "model_a"
            run_dir.mkdir()
            with (run_dir / "profile.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_id",
                        "backend",
                        "sample_id",
                        "prompt_mode",
                        "total_ms",
                        "best_iou",
                        "merged_iou",
                        "image_encoder_ms",
                        "params_total",
                        "weight_total_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model_id": "model_a",
                        "backend": "null",
                        "sample_id": "s1",
                        "prompt_mode": "point",
                        "total_ms": "10",
                        "best_iou": "0.2",
                        "merged_iou": "0.3",
                        "image_encoder_ms": "4",
                        "params_total": "12",
                        "weight_total_bytes": "48",
                    }
                )

            summary_path = write_component_summary(tmp)

            self.assertIsNotNone(summary_path)
            with summary_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["model_id"], "model_a")
            self.assertEqual(rows[0]["mean_total_ms"], "10.0")
            self.assertEqual(rows[0]["mean_image_encoder_ms"], "4.0")
            self.assertEqual(rows[0]["params_total_m"], "1.2e-05")
            self.assertEqual(rows[0]["weight_total_mb"], str(48 / (1024.0 * 1024.0)))


if __name__ == "__main__":
    unittest.main()
