from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.backends import Prompt
from sam_backend.profile_saco_stream import (
    BBoxChainState,
    _decode_uncompressed_rle,
    _image_per_frame_prompt,
    _initial_prompt_frame_index,
    _official_eval_script,
    profile_saco_stream,
)
from sam_backend.saco_manifest import build_saco_veval_manifest
from sam_backend.saco_stream_suite import run_suite


class SacoStreamTests(unittest.TestCase):
    def test_build_manifest_selects_positive_video_np_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            annot = tmp / "annotation.json"
            annot.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "id": 1,
                                "video_name": "sav_a",
                                "file_names": ["sav_a/00000.jpg"],
                                "height": 4,
                                "width": 5,
                                "length": 1,
                            },
                            {
                                "id": 2,
                                "video_name": "sav_b",
                                "file_names": ["sav_b/00000.jpg"],
                                "height": 4,
                                "width": 5,
                                "length": 1,
                            },
                        ],
                        "annotations": [
                            {
                                "id": 10,
                                "video_id": 1,
                                "category_id": 7,
                                "segmentations": [{"size": [4, 5], "counts": [6, 2, 12]}],
                            }
                        ],
                        "video_np_pairs": [
                            {"id": 1, "video_id": 1, "category_id": 7, "noun_phrase": "red box", "num_masklets": 1},
                            {"id": 2, "video_id": 2, "category_id": 8, "noun_phrase": "blue box", "num_masklets": 0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = build_saco_veval_manifest(
                annotation_path=annot,
                media_root=tmp / "media",
                output_path=tmp / "manifest.jsonl",
                count=20,
                seed=1,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text_prompt"], "red box")
            self.assertTrue(rows[0]["is_positive"])

    def test_bbox_chain_uses_initial_prompt_then_box(self) -> None:
        state = BBoxChainState(initial_prompt=Prompt(points=[(2.0, 3.0)], labels=[1]), bbox_min_area=1)

        prompt, mode = state.next_prompt(0)
        self.assertEqual(mode, "point")
        self.assertEqual(prompt.points, [(2.0, 3.0)])

        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1
        state.update([mask], mask.shape)
        prompt, mode = state.next_prompt(1)

        self.assertEqual(mode, "box")
        self.assertEqual(prompt.boxes, [(2.0, 1.0, 4.0, 3.0)])

    def test_bbox_chain_can_delay_initial_prompt_until_visible_gt(self) -> None:
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1
        state = BBoxChainState(
            initial_prompt=Prompt(points=[(3.0, 2.0)], labels=[1]),
            bbox_min_area=1,
            initial_prompt_frame_index=2,
        )

        prompt, mode = state.next_prompt(0)
        self.assertIsNone(prompt)
        self.assertEqual(mode, "pre_prompt")

        prompt, mode = state.next_prompt(2)
        self.assertEqual(mode, "point")
        self.assertEqual(prompt.points, [(3.0, 2.0)])

        self.assertEqual(_initial_prompt_frame_index("point", {2: mask}, 4, {"source_id": "sample"}), 2)

    def test_uncompressed_rle_decode(self) -> None:
        decoded = _decode_uncompressed_rle([2, 3, 7], (3, 4))
        self.assertEqual(decoded.shape, (3, 4))
        self.assertEqual(int(decoded.sum()), 3)

    def test_null_stream_profile_writes_overlay_and_pred_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            media = tmp / "media" / "sav_a"
            media.mkdir(parents=True)
            for idx in range(2):
                frame = np.zeros((16, 20, 3), dtype=np.uint8)
                frame[:, :, 1] = 80 + idx
                self.assertTrue(cv2.imwrite(str(media / f"{idx:05d}.jpg"), frame))

            manifest = tmp / "manifest.jsonl"
            row = {
                "dataset": "saco-veval-sav",
                "source_id": "sav_a_7",
                "video_id": 1,
                "video_name": "sav_a",
                "category_id": 7,
                "noun_phrase": "green square",
                "text_prompt": "green square",
                "is_positive": True,
                "media_root": str(tmp / "media"),
                "file_names": ["sav_a/00000.jpg", "sav_a/00001.jpg"],
                "height": 16,
                "width": 20,
                "length": 2,
                "annotations": [
                    {
                        "id": 10,
                        "video_id": 1,
                        "category_id": 7,
                        "segmentations": [
                            {"size": [16, 20], "counts": [85, 20, 215]},
                            {"size": [16, 20], "counts": [85, 20, 215]},
                        ],
                    }
                ],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            args = argparse.Namespace(
                manifest=manifest,
                limit=0,
                max_frames=2,
                model_id="null_text",
                backend="null",
                stream_mode="text_bbox_chain",
                prompt_type="text",
                prompt="",
                checkpoint_path=None,
                device="cpu",
                model_config=None,
                external_repo=None,
                backbone_type="efficientvit",
                model_name="b0",
                text_encoder_type=None,
                text_encoder_context_length=77,
                text_encoder_pos_embed_table_size=None,
                interpolate_pos_embed=False,
                mobile_sam_model_type="vit_t",
                bbox_min_area=1,
                bbox_scale=1.0,
                input_fps=30.0,
                csv_output=tmp / "frames.csv",
                summary_output=tmp / "summary.json",
                pred_json=tmp / "pred.json",
                gt_annotation_file=None,
                official_eval_json=None,
                overlay_root=tmp / "overlay",
                overlay_fps=30.0,
            )

            summary = profile_saco_stream(args)

            self.assertEqual(summary["frames"], 2)
            self.assertTrue((tmp / "frames.csv").exists())
            self.assertTrue((tmp / "pred.json").exists())
            self.assertTrue((tmp / "overlay" / "sav_a_7" / "overlay.mp4").exists())
            with (tmp / "frames.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["prompt_mode"], "text")

    def test_point_stream_profile_starts_at_first_visible_gt_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            media = tmp / "media" / "sav_a"
            media.mkdir(parents=True)
            for idx in range(2):
                frame = np.zeros((16, 20, 3), dtype=np.uint8)
                frame[:, :, 1] = 80 + idx
                self.assertTrue(cv2.imwrite(str(media / f"{idx:05d}.jpg"), frame))

            manifest = tmp / "manifest.jsonl"
            row = {
                "dataset": "saco-veval-sav",
                "source_id": "sav_a_7",
                "video_id": 1,
                "video_name": "sav_a",
                "category_id": 7,
                "noun_phrase": "green square",
                "text_prompt": "green square",
                "is_positive": True,
                "media_root": str(tmp / "media"),
                "file_names": ["sav_a/00000.jpg", "sav_a/00001.jpg"],
                "height": 16,
                "width": 20,
                "length": 2,
                "annotations": [
                    {
                        "id": 10,
                        "video_id": 1,
                        "category_id": 7,
                        "segmentations": [
                            {"size": [16, 20], "counts": [320]},
                            {"size": [16, 20], "counts": [85, 20, 215]},
                        ],
                    }
                ],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            args = argparse.Namespace(
                manifest=manifest,
                limit=0,
                max_frames=2,
                model_id="null_point",
                backend="null",
                stream_mode="bbox_chain",
                prompt_type="point",
                prompt="",
                checkpoint_path=None,
                device="cpu",
                model_config=None,
                external_repo=None,
                backbone_type="efficientvit",
                model_name="b0",
                text_encoder_type=None,
                text_encoder_context_length=77,
                text_encoder_pos_embed_table_size=None,
                interpolate_pos_embed=False,
                mobile_sam_model_type="vit_t",
                bbox_min_area=1,
                bbox_scale=1.0,
                input_fps=30.0,
                csv_output=tmp / "frames.csv",
                summary_output=tmp / "summary.json",
                pred_json=tmp / "pred.json",
                gt_annotation_file=None,
                official_eval_json=None,
                overlay_root=None,
                overlay_fps=30.0,
            )

            summary = profile_saco_stream(args)

            self.assertEqual(summary["eval_start_frame"], 1)
            self.assertEqual(summary["frames"], 1)
            with (tmp / "frames.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["frame_index"], "1")
            self.assertEqual(rows[0]["prompt_mode"], "point")

    def test_image_per_frame_point_skips_source_without_gt_in_profiled_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            media = tmp / "media" / "sav_a"
            media.mkdir(parents=True)
            frame = np.zeros((16, 20, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(media / "00000.jpg"), frame))

            manifest = tmp / "manifest.jsonl"
            row = {
                "dataset": "saco-veval-sav",
                "source_id": "sav_a_7",
                "video_id": 1,
                "video_name": "sav_a",
                "category_id": 7,
                "noun_phrase": "green square",
                "text_prompt": "green square",
                "is_positive": True,
                "media_root": str(tmp / "media"),
                "file_names": ["sav_a/00000.jpg"],
                "height": 16,
                "width": 20,
                "length": 1,
                "annotations": [
                    {
                        "id": 10,
                        "video_id": 1,
                        "category_id": 7,
                        "segmentations": [{"size": [16, 20], "counts": [320]}],
                    }
                ],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            args = argparse.Namespace(
                manifest=manifest,
                limit=0,
                max_frames=1,
                model_id="null_point",
                backend="null",
                stream_mode="image_per_frame",
                prompt_type="point",
                prompt="",
                checkpoint_path=None,
                device="cpu",
                model_config=None,
                external_repo=None,
                backbone_type="efficientvit",
                model_name="b0",
                text_encoder_type=None,
                text_encoder_context_length=77,
                text_encoder_pos_embed_table_size=None,
                interpolate_pos_embed=False,
                mobile_sam_model_type="vit_t",
                bbox_min_area=1,
                bbox_scale=1.0,
                input_fps=30.0,
                csv_output=tmp / "frames.csv",
                summary_output=tmp / "summary.json",
                pred_json=tmp / "pred.json",
                gt_annotation_file=None,
                official_eval_json=None,
                overlay_root=None,
                overlay_fps=30.0,
            )

            summary = profile_saco_stream(args)

            self.assertEqual(summary["frames"], 0)
            self.assertEqual(summary["sources"], 0)

    def test_image_per_frame_uses_current_frame_point_prompt(self) -> None:
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1
        args = argparse.Namespace(prompt="")
        item = {"source_id": "sample", "text_prompt": "green square"}

        prompt, mode = _image_per_frame_prompt(args, item, mask, "point")
        self.assertEqual(mode, "point")
        self.assertIsNotNone(prompt)
        self.assertAlmostEqual(prompt.points[0][0], 3.0)
        self.assertAlmostEqual(prompt.points[0][1], 2.0)

        prompt, mode = _image_per_frame_prompt(args, item, np.zeros_like(mask), "point")
        self.assertIsNone(prompt)
        self.assertEqual(mode, "no_prompt")

    def test_suite_skip_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                manifest=manifest,
                gt_annotation_file=None,
                models=["sam3p1_ref_native"],
                device="cpu",
                max_frames=1,
                input_fps=30.0,
                output_dir=tmp / "out",
                overlay_dir=tmp / "overlay",
                scratch_root=tmp / "scratch",
                skip_missing=True,
                dry_run=False,
            )

            rows = run_suite(args)

            self.assertEqual(rows[0]["status"], "skipped")
            self.assertIn("sam3.1_multiplex.pt", rows[0]["message"])

    def test_suite_efficientsam3_uses_text_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                manifest=manifest,
                gt_annotation_file=None,
                models=["efficientsam3_ev_m_text_bbox_chain"],
                mode_set="video",
                device="cpu",
                max_frames=1,
                input_fps=30.0,
                output_dir=tmp / "out",
                overlay_dir=tmp / "overlay",
                scratch_root=tmp / "scratch",
                skip_missing=False,
                dry_run=True,
            )

            rows = run_suite(args)

            self.assertEqual(rows[0]["status"], "dry-run")
            self.assertIn("--prompt-type text", rows[0]["message"])
            self.assertIn("--text-encoder-type MobileCLIP-S0", rows[0]["message"])

    def test_suite_can_dry_run_image_per_frame_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                manifest=manifest,
                gt_annotation_file=None,
                models=["sam3_ref_image_per_frame"],
                mode_set="image_per_frame",
                device="cpu",
                max_frames=1,
                input_fps=30.0,
                output_dir=tmp / "out",
                overlay_dir=tmp / "overlay",
                scratch_root=tmp / "scratch",
                skip_missing=False,
                dry_run=True,
            )

            rows = run_suite(args)

            self.assertEqual(rows[0]["status"], "dry-run")
            self.assertIn("--stream-mode image_per_frame", rows[0]["message"])

    def test_suite_can_dry_run_efficientsam3_image_per_frame_point_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                manifest=manifest,
                gt_annotation_file=None,
                models=[
                    "efficientsam3_tv_m_image_per_frame_point",
                    "efficientsam3_tinyvit21_image_per_frame_point",
                    "efficientsam3_tinyvit21_image_per_frame_text",
                ],
                mode_set="image_per_frame",
                device="cpu",
                max_frames=1,
                input_fps=30.0,
                output_dir=tmp / "out",
                overlay_dir=tmp / "overlay",
                scratch_root=tmp / "scratch",
                skip_missing=False,
                dry_run=True,
            )

            rows = run_suite(args)
            messages = {row["model_id"]: row["message"] for row in rows}

            self.assertEqual(len(rows), 3)
            self.assertIn("--prompt-type point", messages["efficientsam3_tv_m_image_per_frame_point"])
            self.assertIn("--model-name 11m", messages["efficientsam3_tv_m_image_per_frame_point"])
            self.assertIn("--prompt-type point", messages["efficientsam3_tinyvit21_image_per_frame_point"])
            self.assertIn("--prompt-type text", messages["efficientsam3_tinyvit21_image_per_frame_text"])
            self.assertIn("--backbone-type tinyvit", messages["efficientsam3_tinyvit21_image_per_frame_point"])
            self.assertIn("--model-name 21m", messages["efficientsam3_tinyvit21_image_per_frame_point"])
            self.assertIn("external/efficientsam3", messages["efficientsam3_tinyvit21_image_per_frame_point"])
            self.assertNotIn("EfficientSam3-Distillation", messages["efficientsam3_tinyvit21_image_per_frame_point"])
            self.assertIn("efficient_sam3_tinyvit21_stage1_e32_h200_full_sam3.pt", messages["efficientsam3_tinyvit21_image_per_frame_text"])

    def test_suite_can_dry_run_instinctsam_image_per_frame_point_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                manifest=manifest,
                gt_annotation_file=None,
                models=[
                    "instinctsam_vitb_image_per_frame_point",
                    "instinctsam_vitb_image_per_frame_text",
                ],
                mode_set="image_per_frame",
                device="cpu",
                max_frames=1,
                input_fps=30.0,
                output_dir=tmp / "out",
                overlay_dir=tmp / "overlay",
                scratch_root=tmp / "scratch",
                skip_missing=False,
                dry_run=True,
            )

            rows = run_suite(args)
            messages = {row["model_id"]: row["message"] for row in rows}

            self.assertEqual(len(rows), 2)
            self.assertIn("--prompt-type point", messages["instinctsam_vitb_image_per_frame_point"])
            self.assertIn("--prompt-type text", messages["instinctsam_vitb_image_per_frame_text"])
            self.assertIn("checkpoints/instinctsam/instinctsam_vitb_concept.pt", messages["instinctsam_vitb_image_per_frame_point"])
            self.assertIn("--backbone-type vit_base", messages["instinctsam_vitb_image_per_frame_point"])
            self.assertIn("--model-name base", messages["instinctsam_vitb_image_per_frame_point"])
            self.assertIn("--text-encoder-type MobileCLIP-S1", messages["instinctsam_vitb_image_per_frame_text"])
            self.assertIn("--text-encoder-pos-embed-table-size 77", messages["instinctsam_vitb_image_per_frame_text"])

    def test_official_eval_script_handles_nested_sam3_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "sam3" / "sam3" / "eval" / "saco_veval_eval.py"
            script.parent.mkdir(parents=True)
            script.write_text("", encoding="utf-8")

            self.assertEqual(_official_eval_script(str(root)), script)


if __name__ == "__main__":
    unittest.main()
