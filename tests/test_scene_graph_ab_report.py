from __future__ import annotations

import unittest

import numpy as np

from sam_backend.scene_graph_ab_report import _directed_mask_ious, _memory_gib, _stats


class SceneGraphAbReportTest(unittest.TestCase):
    def test_memory_gib_converts_docker_units(self) -> None:
        self.assertEqual(_memory_gib("2GiB"), 2.0)
        self.assertEqual(_memory_gib("512MiB"), 0.5)
        self.assertEqual(_memory_gib("1048576KiB"), 1.0)

    def test_directed_mask_iou_matches_only_the_same_label(self) -> None:
        cup = np.asarray([[True, True], [False, False]])
        book = np.asarray([[False, False], [True, True]])
        target_cup = np.asarray([[True, False], [False, False]])

        values = _directed_mask_ious(
            np.stack([cup, book]),
            np.asarray(["cup", "book"]),
            np.stack([target_cup]),
            np.asarray(["cup"]),
        )

        self.assertEqual(values, [0.5, 0.0])

    def test_stats_reports_empty_and_percentile_values(self) -> None:
        self.assertIsNone(_stats([])["mean"])
        values = _stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(values["count"], 4)
        self.assertEqual(values["p50"], 2.5)
        self.assertAlmostEqual(values["p95"], 3.85)


if __name__ == "__main__":
    unittest.main()
