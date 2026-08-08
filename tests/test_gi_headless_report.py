from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_backend.gi_headless_report import _parity_summary


class GiHeadlessReportTest(unittest.TestCase):
    def test_parity_summary_requires_exact_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            values = {
                "mask_shape": np.asarray([1, 2, 2]),
                "masks_packed": np.asarray([128], dtype=np.uint8),
                "labels": np.asarray(["table"]),
                "ids": np.asarray([1]),
                "lost": np.asarray([False]),
                "scores": np.asarray([0.8], dtype=np.float32),
            }
            np.savez(left / "00000.npz", **values)
            np.savez(right / "00000.npz", **values)
            rows = [{"frame_index": 0, "total_ms": 1.0}]

            summary, frames = _parity_summary(left, right, rows, rows)

            self.assertEqual(summary["exact_frames"], 1)
            self.assertTrue(frames[0]["exact"])


if __name__ == "__main__":
    unittest.main()
