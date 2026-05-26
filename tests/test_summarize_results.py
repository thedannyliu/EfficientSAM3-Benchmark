from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from sam_backend.summarize_results import build_catalog_rows, discover_csvs, summarize_csv


class SummarizeResultsTest(unittest.TestCase):
    def test_summarize_profile_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_id",
                        "backend",
                        "video",
                        "prompt",
                        "total_ms",
                        "image_encoder_ms",
                        "text_encoder_ms",
                        "grounding_ms",
                        "other_ms",
                        "mask_count",
                        "score_max",
                        "params_total",
                        "params_image_encoder",
                        "params_text_encoder",
                        "cuda_peak_allocated_mb",
                        "cuda_peak_reserved_mb",
                    ],
                )
                writer.writeheader()
                for total in [10.0, 20.0, 40.0]:
                    writer.writerow(
                        {
                            "model_id": "sam3-litetext-l-test",
                            "backend": "sam3",
                            "video": "videos/test.mov",
                            "prompt": "monitor",
                            "total_ms": total,
                            "image_encoder_ms": 2.0,
                            "text_encoder_ms": 3.0,
                            "grounding_ms": 4.0,
                            "other_ms": 1.0,
                            "mask_count": 1,
                            "score_max": 0.5,
                            "params_total": 815560000,
                            "params_image_encoder": 461840000,
                            "params_text_encoder": 353720000,
                            "cuda_peak_allocated_mb": 100,
                            "cuda_peak_reserved_mb": 200,
                        }
                    )

            summary = summarize_csv(path)

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["mean_total_ms"], 70.0 / 3.0)
            self.assertAlmostEqual(float(summary["mean_total_fps"]), 1000.0 / (70.0 / 3.0))
            self.assertEqual(summary["p50_total_ms"], 20.0)
            self.assertEqual(summary["p50_total_fps"], 50.0)
            self.assertEqual(summary["p95_total_ms"], 20.0)
            self.assertEqual(summary["mean_image_encoder_fps"], 500.0)
            self.assertAlmostEqual(float(summary["mean_text_encoder_fps"]), 1000.0 / 3.0)
            self.assertEqual(summary["mean_grounding_fps"], 250.0)
            self.assertAlmostEqual(float(summary["params_total_pct_of_sam3_image_text"]), 100.0)

    def test_catalog_contains_readme_models(self) -> None:
        rows = build_catalog_rows()
        names = {row["readme_model_name"] for row in rows}

        self.assertIn("SAM3", names)
        self.assertIn("SAM3-LiteText-L-16", names)
        self.assertIn("ES-EV-L-MC-S1", names)

    def test_discover_skips_generated_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep = root / "run.csv"
            skip = root / "benchmark_summary.csv"
            keep.write_text("total_ms\n1\n", encoding="utf-8")
            skip.write_text("total_ms\n1\n", encoding="utf-8")

            self.assertEqual(list(discover_csvs([root])), [keep])


if __name__ == "__main__":
    unittest.main()
