from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in dependency-light envs
    torch = None

from sam_backend.profiling import parameter_counts


@unittest.skipIf(torch is None, "torch is required for module parameter tests")
class ProfilingParameterCountsTest(unittest.TestCase):
    def test_sam3_style_components_are_counted(self) -> None:
        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vision_backbone = torch.nn.Linear(2, 3, bias=False)
                self.language_backbone = torch.nn.Linear(3, 5, bias=False)

        class Interactive(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sam_prompt_encoder = torch.nn.Linear(5, 7, bias=False)
                self.sam_mask_decoder = torch.nn.Linear(7, 11, bias=False)

        class InteractiveWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Interactive()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = Backbone()
                self.transformer = torch.nn.Linear(11, 13, bias=False)
                self.geometry_encoder = torch.nn.Linear(13, 17, bias=False)
                self.segmentation_head = torch.nn.Linear(17, 19, bias=False)
                self.inst_interactive_predictor = InteractiveWrapper()

        counts = parameter_counts(Model())

        self.assertEqual(counts["params_image_encoder"], 6)
        self.assertEqual(counts["params_text_encoder"], 15)
        self.assertEqual(counts["params_transformer"], 143)
        self.assertEqual(counts["params_geometry_encoder"], 221)
        self.assertEqual(counts["params_segmentation_head"], 323)
        self.assertEqual(counts["params_prompt_encoder"], 35)
        self.assertEqual(counts["params_mask_decoder"], 77)
        self.assertGreater(counts["params_backbone"], 0)

    def test_sam2_style_memory_and_mobile_sam_names_are_counted(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_encoder = torch.nn.Linear(2, 3, bias=False)
                self.prompt_encoder = torch.nn.Linear(3, 5, bias=False)
                self.mask_decoder = torch.nn.Linear(5, 7, bias=False)
                self.memory_attention = torch.nn.Linear(7, 11, bias=False)
                self.memory_encoder = torch.nn.Linear(11, 13, bias=False)

        counts = parameter_counts(Model())

        self.assertEqual(counts["params_backbone"], 6)
        self.assertEqual(counts["params_image_encoder"], 6)
        self.assertEqual(counts["params_prompt_encoder"], 15)
        self.assertEqual(counts["params_mask_decoder"], 35)
        self.assertEqual(counts["params_memory_attention"], 77)
        self.assertEqual(counts["params_memory_encoder"], 143)

    def test_ultralytics_like_model_counts_total_as_detector(self) -> None:
        class UltralyticsLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False))

        UltralyticsLike.__module__ = "ultralytics.nn.tasks"
        counts = parameter_counts(UltralyticsLike())

        self.assertEqual(counts["params_total"], 6)
        self.assertEqual(counts["params_detector"], 6)
        self.assertEqual(counts["weight_detector_bytes"], counts["weight_total_bytes"])


if __name__ == "__main__":
    unittest.main()
