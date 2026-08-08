import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_backend.gi_voxel_geometry_check import check_points, run_check
from sam_backend.gi_scene_graph_t04_report import _parse_publish_metrics


class GiVoxelGeometryCheckTest(unittest.TestCase):
    def test_voxel_means_preserve_downstream_cells(self):
        points = np.array(
            [
                [0.001, 0.002, 1.001],
                [0.009, 0.008, 1.009],
                [0.011, 0.002, 1.001],
                [-0.001, -0.002, 1.001],
                [-0.009, -0.008, 1.009],
            ]
        )
        result = check_points(points, 0.01)
        self.assertTrue(result["cell_keys_identical"])
        self.assertLessEqual(result["max_mean_error_m"], 1e-12)
        self.assertLess(result["compressed_points"], result["raw_points"])

    def test_streaming_check_reads_detection_payload(self):
        rows = [
            {"detection": {"points": [[0.001, 0.001, 1.001], [0.002, 0.002, 1.002]]}},
            {"detection": {"points": [[0.011, 0.001, 1.001]]}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = run_check(path, 0.01)
        self.assertEqual(result["messages"], 2)
        self.assertEqual(result["raw_points"], 3)
        self.assertEqual(result["compressed_points"], 2)
        self.assertTrue(result["all_cell_keys_identical"])

    def test_publish_metrics_are_aggregated(self):
        rows = [
            {
                "message_count": 2,
                "raw_point_count": 100,
                "payload_point_count": 10,
                "payload_bytes": 1000,
                "publish_ms": 20.0,
            },
            {
                "message_count": 1,
                "raw_point_count": 50,
                "payload_point_count": 5,
                "payload_bytes": 500,
                "publish_ms": 10.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detection.log"
            path.write_text(
                "".join(
                    "[INFO] [1.0] [detection]: DETECTION3D_METRICS "
                    + json.dumps(row)
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            result = _parse_publish_metrics(path)
        self.assertEqual(result["messages"], 3)
        self.assertEqual(result["raw_points"], 150)
        self.assertEqual(result["payload_points"], 15)
        self.assertEqual(result["payload_bytes"], 1500)
        self.assertEqual(result["publish_ms"]["p50"], 15.0)


if __name__ == "__main__":
    unittest.main()
