from __future__ import annotations

import unittest

import numpy as np

from sam_backend.sam2_mask_postprocess import (
    overlay_label_mask,
    prediction_layer_alpha,
    resolve_lead_frame_offsets,
    resolve_lead_frames,
)


class Sam2MaskPostprocessTest(unittest.TestCase):
    def test_resolves_seconds_to_nearest_frame(self) -> None:
        self.assertEqual(
            resolve_lead_frames(30.0, lead_seconds=0.1, lead_frames=None),
            3,
        )
        self.assertEqual(
            resolve_lead_frames(24.0, lead_seconds=0.1, lead_frames=None),
            2,
        )

    def test_explicit_frame_lead_takes_effect(self) -> None:
        self.assertEqual(
            resolve_lead_frames(30.0, lead_seconds=None, lead_frames=2),
            2,
        )

    def test_resolves_prediction_stack_to_frame_offsets(self) -> None:
        offsets = resolve_lead_frame_offsets(
            30.0,
            lead_seconds=None,
            lead_frames=None,
            lead_seconds_list=[0.05, 0.1, 0.15, 0.2, 0.25],
        )

        self.assertEqual(offsets, [2, 3, 5, 6, 8])

    def test_prediction_stack_strengthens_with_horizon(self) -> None:
        alphas = [prediction_layer_alpha(index, 5, 0.2) for index in range(5)]

        for actual, expected in zip(
            alphas,
            [0.04, 0.08, 0.12, 0.16, 0.2],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_transparent_mask_changes_only_masked_pixels(self) -> None:
        frame = np.full((2, 3, 3), 100, dtype=np.uint8)
        label_map = np.zeros((2, 3), dtype=np.uint8)
        label_map[0, 1] = 1

        output = overlay_label_mask(frame, label_map, alpha=0.25)

        self.assertTrue(np.array_equal(output[0, 0], frame[0, 0]))
        self.assertFalse(np.array_equal(output[0, 1], frame[0, 1]))
        self.assertTrue(np.array_equal(output[1, 2], frame[1, 2]))

    def test_rejects_mask_with_wrong_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            overlay_label_mask(
                np.zeros((4, 5, 3), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.uint8),
                alpha=0.2,
            )


if __name__ == "__main__":
    unittest.main()
