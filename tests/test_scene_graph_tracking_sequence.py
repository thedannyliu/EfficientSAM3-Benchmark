from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_backend.scene_graph_tracking_sequence import _decode_masks, _save_snapshot


class SceneGraphTrackingSequenceTest(unittest.TestCase):
    def test_mask_snapshot_round_trip(self) -> None:
        masks = np.zeros((2, 3, 4), dtype=bool)
        masks[0, 1, 1:3] = True
        masks[1, 2, 3] = True
        payload = io.BytesIO()
        np.savez(
            payload,
            schema_version=np.asarray(2, dtype=np.int64),
            frame=np.asarray(7, dtype=np.int64),
            input_sequence=np.asarray(9, dtype=np.int64),
            mask_shape=np.asarray(masks.shape, dtype=np.int64),
            masks_packed=np.packbits(masks.reshape(-1), bitorder="little"),
            bitorder=np.asarray("little"),
            ids=np.asarray([3, 4]),
            scores=np.asarray([0.8, 0.7], dtype=np.float32),
            lost=np.asarray([False, True]),
            labels=np.asarray(["table", "book"]),
        )

        snapshot = _decode_masks(payload.getvalue())

        np.testing.assert_array_equal(snapshot["masks"], masks)
        self.assertEqual(snapshot["input_sequence"], 9)
        self.assertEqual(snapshot["labels"].tolist(), ["table", "book"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prediction.npz"
            _save_snapshot(path, snapshot)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["ids"].tolist(), [3, 4])
                self.assertEqual(archive["lost"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
