from __future__ import annotations

import unittest
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from sam_backend.profile_yoloe_edgetam import (
    _area_reground_reason,
    _load_sources,
    _localization_diagnostics,
    _mask_iou,
    _overlay_frame,
    _resolve_edgetam_model_config,
)


class YoloeEdgeTamProfileHelpersTest(unittest.TestCase):
    def test_area_jump_rule(self) -> None:
        self.assertEqual(_area_reground_reason(10.0, 30.0, 2.5), "mask_area_jump")
        self.assertEqual(_area_reground_reason(10.0, 20.0, 2.5), "")

    def test_resolve_edgetam_model_config_for_hydra_package_path(self) -> None:
        self.assertEqual(
            _resolve_edgetam_model_config("configs/edgetam.yaml", "external/EdgeTAM"),
            "configs/edgetam.yaml",
        )
        self.assertEqual(
            _resolve_edgetam_model_config("external/EdgeTAM/sam2/configs/edgetam.yaml", "external/EdgeTAM"),
            "configs/edgetam.yaml",
        )

    def test_resolve_edgetam_model_config_for_absolute_external_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "external" / "EdgeTAM"
            config = repo / "sam2" / "configs" / "edgetam.yaml"
            self.assertEqual(_resolve_edgetam_model_config(str(config), str(repo)), "configs/edgetam.yaml")

    def test_mask_iou(self) -> None:
        left = np.zeros((10, 10), dtype=bool)
        right = np.zeros((10, 10), dtype=bool)
        left[:5, :5] = True
        right[:5, :5] = True
        right[5:, 5:] = True
        self.assertAlmostEqual(_mask_iou(left, right), 0.5)

    def test_overlay_frame_draws_mask_and_label(self) -> None:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        mask = np.zeros((32, 48), dtype=bool)
        mask[8:20, 12:28] = True
        overlay = _overlay_frame(frame, mask, "person", "video", 3, 1, "")

        self.assertEqual(overlay.shape, frame.shape)
        self.assertGreater(int(overlay.sum()), 0)
        self.assertGreater(int(overlay[12, 16, 1]), 0)

    def test_localization_diagnostics_reports_top1_and_best_instance(self) -> None:
        gt = np.zeros((10, 10), dtype=bool)
        gt[:5, :5] = True
        wrong = np.zeros((10, 10), dtype=bool)
        wrong[5:, 5:] = True
        right = gt.copy()
        diagnostics = _localization_diagnostics(
            [
                {"mask": wrong, "confidence": 0.9},
                {"mask": right, "confidence": 0.7},
            ],
            gt,
        )

        self.assertEqual(diagnostics["yoloe_initial_detection_count"], 2)
        self.assertEqual(diagnostics["yoloe_initial_best_rank"], 2)
        self.assertEqual(diagnostics["yoloe_initial_localization_note"], "same_prompt_different_instance")
        self.assertAlmostEqual(diagnostics["yoloe_initial_top1_gt_iou"], 0.0)
        self.assertAlmostEqual(diagnostics["yoloe_initial_best_gt_iou"], 1.0)

    def test_load_sources_preserves_manifest_gt_and_instance_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "video_id": "video",
                        "frames_dir": "frames",
                        "annotations_dir": "anns",
                        "object_id": "003",
                        "text_prompt": "person",
                        "text_prompt_instance_hint": "rightmost person",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sources = _load_sources(argparse.Namespace(manifest=manifest, limit=0, frames_dir=None, video_path=None, source_id=None))

        self.assertEqual(sources[0]["source_id"], "video")
        self.assertEqual(sources[0]["annotations_dir"], "anns")
        self.assertEqual(sources[0]["object_id"], "003")
        self.assertEqual(sources[0]["instance_hint"], "rightmost person")


if __name__ == "__main__":
    unittest.main()
