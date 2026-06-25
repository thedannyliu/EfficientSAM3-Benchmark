from __future__ import annotations

import unittest

import numpy as np

from sam_backend.backends import Prompt, Sam3ImageBackend


class Sam3ImageBackendTest(unittest.TestCase):
    def test_point_prompt_uses_numpy_arrays_for_predict_inst(self) -> None:
        backend = Sam3ImageBackend.__new__(Sam3ImageBackend)
        backend.model = _FakeSam3Model()

        output = backend._run_prompt(
            {"state": "fake"},
            Prompt(points=[(4.0, 5.0)], labels=[1]),
        )

        self.assertEqual(output["scores"].tolist(), [0.9])


class _FakeSam3Model:
    def predict_inst(self, inference_state, **kwargs):
        self.inference_state = inference_state
        point_coords = kwargs["point_coords"]
        point_labels = kwargs["point_labels"]
        assert isinstance(point_coords, np.ndarray)
        assert isinstance(point_labels, np.ndarray)
        assert point_coords.dtype == np.float32
        assert point_labels.dtype == np.int32
        return np.zeros((1, 2, 3), dtype=np.uint8), np.array([0.9]), np.zeros((1, 2, 3), dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
