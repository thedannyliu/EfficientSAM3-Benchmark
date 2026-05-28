from __future__ import annotations

import unittest

from sam_backend.profile_sav_video import _prepare_efficient_sam2_predictor


class DummyEfficientSam2Predictor:
    def __init__(self) -> None:
        self.init_memory_info_called = False

    def init_memory_info(self, enable_MeP_info: bool = False) -> None:
        self.init_memory_info_called = True
        self.enable_MeP_info = enable_MeP_info


class ProfileSavVideoTest(unittest.TestCase):
    def test_prepare_efficient_sam2_predictor_adds_time_log(self) -> None:
        predictor = DummyEfficientSam2Predictor()

        _prepare_efficient_sam2_predictor(predictor)

        self.assertTrue(predictor.init_memory_info_called)
        self.assertFalse(predictor.enable_MeP_info)
        self.assertEqual(predictor.time_log, {})


if __name__ == "__main__":
    unittest.main()
