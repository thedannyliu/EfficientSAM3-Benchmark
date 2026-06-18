from __future__ import annotations

import unittest

from sam_backend.profile_multi_prompt_image import _grid_points, _parse_counts, summarize_rows


class MultiPromptImageHelpersTest(unittest.TestCase):
    def test_parse_counts_accepts_commas_and_spaces(self) -> None:
        self.assertEqual(_parse_counts("1,2 3,5"), [1, 2, 3, 5])

    def test_grid_points_returns_requested_count_inside_frame(self) -> None:
        points = _grid_points(width=100, height=50, count=15)

        self.assertEqual(len(points), 15)
        for x, y in points:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 100)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 50)

    def test_summary_groups_by_suite_model_and_target_count(self) -> None:
        rows = [
            {"suite": "mobilesam_points", "model_id": "mobilesam_vit_t", "target_count": 2, "model_ms": 10, "mask_count": 2},
            {"suite": "mobilesam_points", "model_id": "mobilesam_vit_t", "target_count": 2, "model_ms": 20, "mask_count": 2},
            {"suite": "mobilesam_points", "model_id": "mobilesam_vit_t", "target_count": 5, "model_ms": 50, "mask_count": 5},
        ]

        summary = summarize_rows(rows)

        by_count = {row["target_count"]: row for row in summary}
        self.assertEqual(by_count["2"]["images"], 2)
        self.assertEqual(by_count["2"]["mean_model_ms"], 15)
        self.assertEqual(by_count["5"]["mean_mask_count"], 5)


if __name__ == "__main__":
    unittest.main()
