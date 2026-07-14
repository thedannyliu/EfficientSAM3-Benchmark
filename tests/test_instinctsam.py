from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sam_backend.instinctsam import install_instinctsam_video_components


class InstinctSamVideoTest(unittest.TestCase):
    def test_video_components_are_installed_on_detector(self) -> None:
        detector = object()
        video_model = SimpleNamespace(detector=detector)
        builder = object()

        with patch("sam_backend.instinctsam.install_instinctsam_components") as install:
            install_instinctsam_video_components(
                video_model,
                builder,
                text_checkpoint="text.pt",
                vision_checkpoint="vision.pt",
                device="cuda:0",
            )

        install.assert_called_once_with(
            detector,
            builder,
            text_checkpoint="text.pt",
            vision_checkpoint="vision.pt",
            device="cuda:0",
        )

    def test_video_model_requires_detector(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not expose detector"):
            install_instinctsam_video_components(
                object(),
                object(),
                text_checkpoint="text.pt",
                vision_checkpoint=None,
                device="cuda:0",
            )


if __name__ == "__main__":
    unittest.main()
