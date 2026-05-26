from __future__ import annotations

import argparse
import unittest

from sam_backend.benchmark import run_benchmark


class NullBenchmarkTest(unittest.TestCase):
    def test_synthetic_null_benchmark(self) -> None:
        args = argparse.Namespace(
            backend="null",
            checkpoint_path=None,
            device=None,
            backbone_type="efficientvit",
            model_name="b0",
            prompt="person",
            image=None,
            video=None,
            synthetic_frames=3,
            width=64,
            height=48,
            max_frames=64,
            warmup=1,
            runs=2,
            output=None,
        )
        result = run_benchmark(args)
        self.assertEqual(result["backend"], "null")
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["frames_loaded"], 3)
        self.assertIsNotNone(result["latency_ms"]["mean"])


if __name__ == "__main__":
    unittest.main()
