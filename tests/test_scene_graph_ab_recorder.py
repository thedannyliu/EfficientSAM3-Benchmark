from __future__ import annotations

import unittest

from sam_backend.scene_graph_ab_recorder import _edge_count, _stamp_ns


class SceneGraphAbRecorderTest(unittest.TestCase):
    def test_stamp_ns_preserves_integer_nanoseconds(self) -> None:
        stamp = type("Stamp", (), {"sec": 12, "nanosec": 345})()

        self.assertEqual(_stamp_ns(stamp), 12_000_000_345)

    def test_edge_count_supports_serialized_graph_layouts(self) -> None:
        self.assertEqual(_edge_count([{"relation": "near"}]), 1)
        self.assertEqual(_edge_count({"a": [{}, {}], "b": [{}]}), 3)
        self.assertEqual(_edge_count({"a": {"relation": "on"}}), 1)


if __name__ == "__main__":
    unittest.main()
