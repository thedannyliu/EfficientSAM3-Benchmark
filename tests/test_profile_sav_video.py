from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam_backend.profile_sav_video import _add_initial_prompt, _efficienttam_hydra_overrides, _prepare_efficient_sam2_predictor


class DummyEfficientSam2Predictor:
    def __init__(self) -> None:
        self.init_memory_info_called = False

    def init_memory_info(self, enable_MeP_info: bool = False) -> None:
        self.init_memory_info_called = True
        self.enable_MeP_info = enable_MeP_info


class DummyPromptPredictor:
    def __init__(self) -> None:
        self.calls = []

    def add_new_mask(self, **kwargs) -> None:
        self.calls.append(("mask", kwargs))

    def add_new_points_or_box(self, **kwargs) -> None:
        self.calls.append(("point", kwargs))


class ProfileSavVideoTest(unittest.TestCase):
    def test_prepare_efficient_sam2_predictor_adds_time_log(self) -> None:
        predictor = DummyEfficientSam2Predictor()

        _prepare_efficient_sam2_predictor(predictor)

        self.assertTrue(predictor.init_memory_info_called)
        self.assertFalse(predictor.enable_MeP_info)
        self.assertEqual(predictor.time_log, {})
        self.assertFalse(predictor.Mem_Frame_Prune)

    def test_efficienttam_overrides_disable_image_encoder_compile(self) -> None:
        self.assertIn("++model.compile_image_encoder=False", _efficienttam_hydra_overrides())

    def test_add_initial_prompt_uses_gt_mask_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask_dir = root / "000"
            mask_dir.mkdir(parents=True)
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[2:6, 3:7] = 255
            self.assertTrue(cv2.imwrite(str(mask_dir / "00000.png"), mask))
            item = {"video_id": "video_a", "object_id": "000", "annotations_dir": str(root)}
            predictor = DummyPromptPredictor()

            _add_initial_prompt(
                predictor=predictor,
                state="state",
                item=item,
                frame_idx=0,
                obj_id=1,
                points=np.asarray([[5.0, 4.0]], dtype=np.float32),
                labels=np.asarray([1], dtype=np.int32),
                init_prompt="mask",
            )

            self.assertEqual(predictor.calls[0][0], "mask")
            call = predictor.calls[0][1]
            self.assertEqual(call["inference_state"], "state")
            self.assertEqual(call["frame_idx"], 0)
            self.assertEqual(call["obj_id"], 1)
            self.assertEqual(call["mask"].shape, (8, 10))
            self.assertTrue(call["mask"][2, 3])

    def test_add_initial_prompt_uses_point_by_default(self) -> None:
        predictor = DummyPromptPredictor()
        points = np.asarray([[5.0, 4.0]], dtype=np.float32)
        labels = np.asarray([1], dtype=np.int32)

        _add_initial_prompt(
            predictor=predictor,
            state="state",
            item={"video_id": "video_a", "object_id": "000"},
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            init_prompt="point",
        )

        self.assertEqual(predictor.calls[0][0], "point")
        call = predictor.calls[0][1]
        self.assertIs(call["points"], points)
        self.assertIs(call["labels"], labels)


if __name__ == "__main__":
    unittest.main()
