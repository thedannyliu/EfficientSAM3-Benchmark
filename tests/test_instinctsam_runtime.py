from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from sam_backend import BackendConfig, Prompt, create_backend


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class InstinctSamRuntimeTest(unittest.TestCase):
    def test_predict_decodes_sequence_aligned_masks(self) -> None:
        masks = np.zeros((2, 6, 8), dtype=bool)
        masks[0, 1:4, 2:6] = True
        masks[1, 0, 0] = True
        payload = io.BytesIO()
        np.savez(
            payload,
            schema_version=np.asarray(2),
            frame=np.asarray(17),
            input_sequence=np.asarray(3),
            mask_shape=np.asarray(masks.shape),
            masks_packed=np.packbits(masks.reshape(-1), bitorder="little"),
            bitorder=np.asarray("little"),
            ids=np.asarray([4, 5]),
            scores=np.asarray([0.8, 0.2], dtype=np.float32),
            lost=np.asarray([False, True]),
            labels=np.asarray(["cow", "cow"]),
        )
        responses = [
            _Response(b'{"ok":true}'),
            _Response(b'{"ok":true}'),
            _Response(b'{"ok":true,"input_sequence":3}'),
            _Response(payload.getvalue()),
            _Response(json.dumps({"detect_ms": 12.5}).encode()),
        ]

        with patch("urllib.request.urlopen", side_effect=responses):
            backend = create_backend(
                BackendConfig(backend="instinctsam-http", runtime_url="http://runtime")
            )
            prediction = backend.predict(
                np.zeros((6, 8, 3), dtype=np.uint8), Prompt(text="cow")
            )

        self.assertEqual(prediction.masks.shape, (1, 6, 8))
        self.assertEqual(int(prediction.masks[0].sum()), 12)
        self.assertEqual(prediction.boxes, [[2, 1, 5, 3]])
        self.assertAlmostEqual(float(prediction.scores[0]), 0.8)
        self.assertEqual(prediction.metadata["input_sequence"], 3)
        self.assertEqual(prediction.metadata["labels"], ["cow"])


if __name__ == "__main__":
    unittest.main()
