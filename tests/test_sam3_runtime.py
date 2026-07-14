from __future__ import annotations

import unittest


class Sam3RuntimeTest(unittest.TestCase):
    def test_torch_nms_suppresses_overlapping_lower_score(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from sam_backend.sam3_runtime import generic_nms_torch

        ious = torch.tensor(
            [
                [1.0, 0.8, 0.1],
                [0.8, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        scores = torch.tensor([0.9, 0.7, 0.8])

        kept = generic_nms_torch(ious, scores, 0.5)

        self.assertEqual(kept.tolist(), [0, 2])

    def test_torch_nms_rejects_invalid_shapes(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from sam_backend.sam3_runtime import generic_nms_torch

        with self.assertRaisesRegex(ValueError, "square matrix"):
            generic_nms_torch(torch.zeros(2, 3), torch.zeros(2))

    def test_connected_components_preserves_empty_batch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from sam_backend.sam3_runtime import connected_components_cpu_safe

        values = torch.zeros((0, 1, 32, 32), dtype=torch.uint8)
        labels, counts = connected_components_cpu_safe(values)

        self.assertEqual(labels.shape, values.shape)
        self.assertEqual(counts.shape, values.shape)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(counts.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
