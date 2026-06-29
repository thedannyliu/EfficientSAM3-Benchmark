from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.coco_suite import (
    run_suite,
    write_component_summary,
    write_model_summary,
)


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

    def test_dry_run_builds_instinctsam_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = argparse.Namespace(
                manifest=tmp / "manifest.jsonl",
                limit=1,
                device="cpu",
                models=["instinctsam_vitb"],
                output_dir=tmp / "results",
                overlay_dir=None,
                skip_missing=False,
                dry_run=True,
            )

            rows = run_suite(args)

            self.assertEqual(rows[0]["model_id"], "instinctsam_vitb")
            self.assertEqual(rows[0]["status"], "dry-run")
            self.assertIn("--checkpoint-path checkpoints/instinctsam/instinctsam_vitb_concept.pt", rows[0]["message"])
            self.assertIn("--backbone-type vit_base", rows[0]["message"])
            self.assertIn("--text-encoder-type MobileCLIP-S1", rows[0]["message"])

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

    def test_model_summary_groups_by_prompt(self) -> None:
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
                        "prompt_mode": "text",
                        "total_ms": "10",
                        "best_iou": "0.2",
                        "merged_iou": "0.3",
                        "params_total": "2000000",
                        "weight_total_bytes": "8000000",
                    }
                )
                writer.writerow(
                    {
                        "model_id": "model_a",
                        "backend": "null",
                        "sample_id": "s1",
                        "prompt_mode": "point",
                        "total_ms": "30",
                        "best_iou": "0.4",
                        "merged_iou": "0.5",
                        "params_total": "2000000",
                        "weight_total_bytes": "8000000",
                    }
                )

            summary_path = write_model_summary(tmp)

            self.assertIsNotNone(summary_path)
            with summary_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["model_id"], "model_a")
            self.assertEqual(rows[0]["prompt_mode"], "point")
            self.assertEqual(rows[0]["samples"], "1")
            self.assertEqual(rows[0]["rows"], "1")
            self.assertEqual(rows[0]["effective_fps"], str(1000.0 / 30.0))
            self.assertAlmostEqual(float(rows[0]["miou_best"]), 0.4)
            self.assertEqual(rows[0]["params_total_m"], "2.0")
            self.assertEqual(rows[1]["prompt_mode"], "text")
            self.assertEqual(rows[1]["effective_fps"], "100.0")
            self.assertAlmostEqual(float(rows[1]["miou_best"]), 0.2)


if __name__ == "__main__":
    unittest.main()
