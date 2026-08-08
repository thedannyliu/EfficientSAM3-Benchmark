from __future__ import annotations

import unittest

import numpy as np

from sam_backend.gi_tracking_report import (
    _agreement_summary,
    _directed_mask_ious,
    _phase_bucket,
)


class GiTrackingReportTest(unittest.TestCase):
    def test_phase_buckets_match_refresh_period(self) -> None:
        self.assertEqual(_phase_bucket(0), "refresh")
        self.assertEqual(_phase_bucket(9), "1-9")
        self.assertEqual(_phase_bucket(10), "10-19")
        self.assertEqual(_phase_bucket(29), "20-29")
        self.assertEqual(_phase_bucket(30), "refresh")

    def test_directed_iou_requires_matching_label(self) -> None:
        left = np.asarray([[[True, True], [False, False]]])
        right = np.asarray([[[True, False], [False, False]]])
        self.assertEqual(
            _directed_mask_ious(left, np.asarray(["table"]), right, np.asarray(["table"])),
            [0.5],
        )
        self.assertEqual(
            _directed_mask_ious(left, np.asarray(["table"]), right, np.asarray(["book"])),
            [0.0],
        )

    def test_agreement_summary(self) -> None:
        summary = _agreement_summary([0.25, 0.5, 0.75])
        self.assertEqual(summary["instance_miou"], 0.5)
        self.assertEqual(summary["recall_at_50"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
