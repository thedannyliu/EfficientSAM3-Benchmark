import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "pace_tinyvit_trt_mask_parity.py"
SPEC = importlib.util.spec_from_file_location("tinyvit_trt_mask_parity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EndToEndTimingTest(unittest.TestCase):
    def test_backend_timing_includes_decode_and_one_prompt(self) -> None:
        rows = [
            {"set_image_ms": 4.0, "prompt_ms": {"point": 2.0, "box": 3.0}},
            {"set_image_ms": 6.0, "prompt_ms": {"point": 2.0, "box": 3.0}},
        ]
        timing = MODULE._backend_timing(rows, [1.0, 1.0])
        self.assertEqual(timing["set_image"]["mean_ms"], 5.0)
        self.assertEqual(timing["point_model_pipeline"]["mean_ms"], 7.0)
        self.assertEqual(timing["point_end_to_end"]["mean_ms"], 8.0)
        self.assertEqual(timing["two_prompt_end_to_end"]["mean_ms"], 11.0)


if __name__ == "__main__":
    unittest.main()
