from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_backend.profile_yolo_coco import (
    _best_box_iou,
    _extract_detections,
    _filter_detections_by_class,
    _mask_ious,
    _predict_kwargs,
    _split_yolo_layers,
    _yolo_parameter_counts,
)
from sam_backend.yolo_coco_suite import (
    run_suite,
    weight_names_for_preset,
    write_component_summary,
    write_model_summary,
)


class TensorLike:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "TensorLike":
        return self

    def cpu(self) -> np.ndarray:
        return self.value


class Boxes:
    xyxy = TensorLike([[1, 1, 5, 5], [8, 8, 12, 12]])
    conf = TensorLike([0.9, 0.7])
    cls = TensorLike([0, 1])


class Masks:
    data = TensorLike(np.stack([np.eye(8, dtype=np.float32), np.ones((8, 8), dtype=np.float32)]))


class Result:
    boxes = Boxes()
    masks = Masks()
    names = {0: "cow", 1: "person"}


class FakeParam:
    def __init__(self, n: int) -> None:
        self.n = n

    def numel(self) -> int:
        return self.n

    def element_size(self) -> int:
        return 4


class FakeLayer:
    def __init__(self, type_name: str, params: int) -> None:
        self.type = type_name
        self.param = FakeParam(params)

    def parameters(self) -> list[FakeParam]:
        return [self.param]


class FakeYoloModel:
    def __init__(self, layers: list[FakeLayer]) -> None:
        self.model = layers

    def parameters(self) -> list[FakeParam]:
        return [param for layer in self.model for param in layer.parameters()]


class YoloCocoProfileTest(unittest.TestCase):
    def test_extract_and_filter_detections(self) -> None:
        detections = _extract_detections(Result(), (16, 16))

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "cow")
        self.assertEqual(detections[0]["mask"].shape, (16, 16))
        self.assertEqual(len(_filter_detections_by_class(detections, "person")), 1)

    def test_mask_and_box_iou(self) -> None:
        gt = np.zeros((10, 10), dtype=bool)
        gt[:5, :5] = True
        pred = gt.copy()
        wrong = np.zeros((10, 10), dtype=bool)
        wrong[5:, 5:] = True

        self.assertEqual(_mask_ious([wrong, pred], gt), (1.0, 0.5))
        box_iou = _best_box_iou([np.asarray([0, 0, 5, 5])], {"bbox_xywh": [0, 0, 5, 5]})
        self.assertAlmostEqual(box_iou, 1.0)

    def test_predict_kwargs_keeps_optional_nms(self) -> None:
        args = argparse.Namespace(imgsz=640, conf=0.2, iou=0.7, max_det=20, agnostic_nms=False, device="cpu")

        kwargs = _predict_kwargs(args)

        self.assertEqual(kwargs["max_det"], 20)
        self.assertFalse(kwargs["agnostic_nms"])

    def test_yolo_suite_dry_run_omits_world_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = argparse.Namespace(
                manifest=tmp / "manifest.jsonl",
                limit=1,
                device="cpu",
                preset="quick",
                models=None,
                imgsz=640,
                conf=0.25,
                iou=0.7,
                max_det=100,
                agnostic_nms=None,
                eval_mode="both",
                output_dir=tmp / "results",
                overlay_dir=tmp / "overlays",
                dry_run=True,
            )

            rows = run_suite(args)

            self.assertEqual([row["model_id"] for row in rows], ["yoloe_26n_seg", "yolo11n_seg"])
            self.assertTrue(all("yolo-world" not in row["message"] for row in rows))

    def test_weight_names_for_preset_matches_quick_models(self) -> None:
        self.assertEqual(weight_names_for_preset("quick"), ["yoloe-26n-seg.pt", "yolo11n-seg.pt"])

    def test_all_preset_contains_all_requested_yoloe_variants(self) -> None:
        weights = set(weight_names_for_preset("all"))
        self.assertTrue(
            {
                "yoloe-11s-seg.pt",
                "yoloe-11m-seg.pt",
                "yoloe-11l-seg.pt",
                "yoloe-v8s-seg.pt",
                "yoloe-v8m-seg.pt",
                "yoloe-v8l-seg.pt",
                "yoloe-26n-seg.pt",
                "yoloe-26s-seg.pt",
                "checkpoints/yoloe/yoloe-26m-seg.pt",
                "yoloe-26l-seg.pt",
                "yoloe-26x-seg.pt",
            }.issubset(weights)
        )

    def test_yolo_parameter_counts_split_backbone_neck_head(self) -> None:
        layers = [
            FakeLayer("Conv", 1),
            FakeLayer("SPPF", 2),
            FakeLayer("Upsample", 3),
            FakeLayer("Concat", 4),
            FakeLayer("Segment", 5),
        ]

        counts = _yolo_parameter_counts(FakeYoloModel(layers))

        self.assertEqual(counts["params_yolo_backbone"], 3)
        self.assertEqual(counts["params_yolo_neck"], 7)
        self.assertEqual(counts["params_yolo_head"], 5)
        self.assertEqual(counts["yolo_backbone_layers"], "0-1")
        self.assertEqual(counts["yolo_neck_layers"], "2-3")
        self.assertEqual(counts["yolo_head_layers"], "4")

    def test_yolo_split_falls_back_to_after_spp(self) -> None:
        layers = [FakeLayer("Conv", 1), FakeLayer("SPPF", 2), FakeLayer("Segment", 3)]

        split = _split_yolo_layers(layers)

        self.assertEqual(len(split["backbone"]), 2)
        self.assertEqual(len(split["neck"]), 0)
        self.assertEqual(len(split["head"]), 1)

    def test_yolo_component_summary_contains_miou_and_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / "yolo11n_seg"
            run_dir.mkdir()
            with (run_dir / "profile.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_id",
                        "family",
                        "weights",
                        "sample_id",
                        "total_ms",
                        "best_iou",
                        "merged_iou",
                        "best_box_iou",
                        "target_detection_count",
                        "predict_ms",
                        "params_total",
                        "weight_total_bytes",
                        "checkpoint_file_bytes",
                        "params_yolo_backbone",
                        "params_yolo_neck",
                        "params_yolo_head",
                        "weight_yolo_backbone_bytes",
                        "weight_yolo_neck_bytes",
                        "weight_yolo_head_bytes",
                        "yolo_backbone_layers",
                        "yolo_neck_layers",
                        "yolo_head_layers",
                        "component_note",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model_id": "yolo11n_seg",
                        "family": "yolo-seg",
                        "weights": "yolo11n-seg.pt",
                        "sample_id": "s1",
                        "total_ms": "20",
                        "best_iou": "0.4",
                        "merged_iou": "0.5",
                        "best_box_iou": "0.6",
                        "target_detection_count": "1",
                        "predict_ms": "18",
                        "params_total": "1000000",
                        "weight_total_bytes": "4000000",
                        "checkpoint_file_bytes": "2000000",
                        "params_yolo_backbone": "300000",
                        "params_yolo_neck": "400000",
                        "params_yolo_head": "300000",
                        "weight_yolo_backbone_bytes": "1200000",
                        "weight_yolo_neck_bytes": "1600000",
                        "weight_yolo_head_bytes": "1200000",
                        "yolo_backbone_layers": "0-9",
                        "yolo_neck_layers": "10-21",
                        "yolo_head_layers": "22",
                        "component_note": "note",
                    }
                )

            summary_path = write_component_summary(tmp)

            self.assertIsNotNone(summary_path)
            with summary_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["miou_best"], "0.4")
            self.assertEqual(rows[0]["effective_fps"], "50.0")
            self.assertEqual(rows[0]["params_yolo_neck_m"], "0.4")
            self.assertEqual(rows[0]["yolo_head_layers"], "22")
            self.assertEqual(rows[0]["checkpoint_file_mb"], str(2000000 / (1024.0 * 1024.0)))

    def test_yolo_model_summary_is_concise_one_row_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / "yoloe_26n_seg"
            run_dir.mkdir()
            with (run_dir / "profile.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_id",
                        "family",
                        "weights",
                        "sample_id",
                        "total_ms",
                        "best_iou",
                        "merged_iou",
                        "predict_ms",
                        "params_total",
                        "weight_total_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model_id": "yoloe_26n_seg",
                        "family": "yoloe-seg",
                        "weights": "yoloe-26n-seg.pt",
                        "sample_id": "s1",
                        "total_ms": "25",
                        "best_iou": "0.25",
                        "merged_iou": "0.30",
                        "predict_ms": "20",
                        "params_total": "2000000",
                        "weight_total_bytes": "8000000",
                    }
                )

            summary_path = write_model_summary(tmp)

            self.assertIsNotNone(summary_path)
            with summary_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["model_id"], "yoloe_26n_seg")
            self.assertEqual(rows[0]["effective_fps"], "40.0")
            self.assertEqual(rows[0]["miou_best"], "0.25")
            self.assertEqual(rows[0]["params_total_m"], "2.0")


if __name__ == "__main__":
    unittest.main()
