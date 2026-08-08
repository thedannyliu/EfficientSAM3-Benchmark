from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam_backend.scene_graph_mask_dump import _load_prompts, _save_masks


class SceneGraphMaskDumpTest(unittest.TestCase):
    def test_saves_packbit_masks_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "masks.npz"
            masks = np.zeros((2, 3, 5), dtype=bool)
            masks[0, 1, 2] = True
            _save_masks(
                path,
                masks,
                np.asarray(["cup", "table"]),
                np.asarray([0.9, 0.8]),
            )

            with np.load(path, allow_pickle=False) as archive:
                shape = tuple(int(value) for value in archive["mask_shape"])
                decoded = np.unpackbits(
                    archive["masks_packed"], bitorder=str(archive["bitorder"].item())
                )[: int(np.prod(shape))].reshape(shape)

            np.testing.assert_array_equal(decoded.astype(bool), masks)

    def test_load_prompts_deduplicates_and_normalizes_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "movable_types": ["light_switch", "cup"],
                        "container_types": ["cup", "trash_can"],
                        "surface_types": ["table"],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _load_prompts(path), ["light switch", "cup", "trash can", "table"]
            )


if __name__ == "__main__":
    unittest.main()
