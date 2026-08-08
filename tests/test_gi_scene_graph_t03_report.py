import json
import tempfile
import unittest
from pathlib import Path

from sam_backend.gi_scene_graph_t03_report import (
    _detection_summary,
    _linear_slope,
    _parse_detection_log,
)


class GiSceneGraphT03ReportTest(unittest.TestCase):
    def test_detector_log_separates_refresh_and_tracking(self):
        lines = [
            '[INFO] [10.000] [detection]: === Processing frame 1 ===',
            '[INFO] [10.300] [detection]: DETECTOR_METRICS '
            + json.dumps(
                {
                    "input_sequence": 1,
                    "detect_ms": 200.0,
                    "http_total_ms": 250.0,
                    "process_ms": 220.0,
                    "mask_count": 1,
                    "initialized_this_frame": True,
                }
            ),
            '[INFO] [10.400] [detection]: === Completed frame 1 ===',
            '[INFO] [10.500] [detection]: SYNC #1',
            '[INFO] [10.501] [detection]: Skipping frame 1 — still processing previous frame',
            '[INFO] [11.000] [detection]: === Processing frame 2 ===',
            '[INFO] [11.100] [detection]: DETECTOR_METRICS '
            + json.dumps(
                {
                    "input_sequence": 2,
                    "detect_ms": 0.0,
                    "http_total_ms": 90.0,
                    "process_ms": 75.0,
                    "mask_count": 2,
                    "initialized_this_frame": False,
                }
            ),
            '[INFO] [11.200] [detection]: === Completed frame 2 ===',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detection.log"
            path.write_text("\n".join(lines), encoding="utf-8")
            result = _parse_detection_log(path)
        self.assertEqual(result["detector_calls"], 2)
        self.assertEqual(result["completed_frames"], 2)
        self.assertEqual(result["refresh_zero_based_indices"], [0])
        self.assertEqual(result["tracking_http_total_ms"]["p50"], 90.0)
        self.assertEqual(result["busy_skips"], 1)

    def test_detection_summary_uses_first_and_last_message_per_stamp(self):
        rows = [
            {
                "source_stamp_ns": 1_000_000_000,
                "camera_to_detection_ms": 100.0,
                "detection": {"category": "book"},
            },
            {
                "source_stamp_ns": 1_000_000_000,
                "camera_to_detection_ms": 140.0,
                "detection": {"category": "table"},
            },
            {
                "source_stamp_ns": 2_000_000_000,
                "camera_to_detection_ms": 120.0,
                "detection": {"category": "book"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = _detection_summary(path)
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["camera_to_first_detection_ms"]["values"], [100.0, 120.0])
        self.assertEqual(result["camera_to_last_detection_ms"]["values"], [140.0, 120.0])
        self.assertEqual(result["category_counts"], {"book": 2, "table": 1})

    def test_linear_slope(self):
        self.assertAlmostEqual(_linear_slope([0.0, 1.0, 2.0], [10.0, 13.0, 16.0]), 3.0)
        self.assertEqual(_linear_slope([0.0], [10.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
