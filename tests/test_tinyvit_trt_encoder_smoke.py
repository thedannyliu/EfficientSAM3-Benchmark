import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "pace_tinyvit_trt_encoder_smoke.py"
SPEC = importlib.util.spec_from_file_location("tinyvit_trt_encoder_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QuantizationSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        nodes = [
            types.SimpleNamespace(op_type="Conv", name="node_Conv_1"),
            types.SimpleNamespace(op_type="Conv", name="node_conv2d_27"),
            types.SimpleNamespace(op_type="MatMul", name="node_MatMul_1"),
            types.SimpleNamespace(
                op_type="MatMul", name="node_scaled_dot_product_attention"
            ),
        ]
        self.model = types.SimpleNamespace(graph=types.SimpleNamespace(node=nodes))

    def test_architecture_profiles_select_disjoint_nodes(self) -> None:
        expected = {
            "backbone_conv": ["node_Conv_1"],
            "neck_conv": ["node_conv2d_27"],
            "linear_matmul": ["node_MatMul_1"],
            "attention_matmul": ["node_scaled_dot_product_attention"],
        }
        for profile, node_names in expected.items():
            with self.subTest(profile=profile):
                _, selected = MODULE._quantization_selection(self.model, profile)
                self.assertEqual(selected, node_names)

    def test_whole_operator_profiles_preserve_onnx_spelling(self) -> None:
        self.assertEqual(
            MODULE._quantization_selection(self.model, "matmul"), (["MatMul"], None)
        )
        self.assertEqual(
            MODULE._quantization_selection(self.model, "conv"), (["Conv"], None)
        )

    def test_semantic_scope_selects_one_layer(self) -> None:
        metadata = types.SimpleNamespace(
            key="pkg.torch.onnx.name_scopes",
            value=str(["", "student.backbone.body.stages_2.blocks.4.mlp.fc1", "linear"]),
        )
        node = types.SimpleNamespace(
            op_type="MatMul", name="node_MatMul_1", metadata_props=[metadata]
        )
        model = types.SimpleNamespace(graph=types.SimpleNamespace(node=[node]))
        op_types, selected = MODULE._quantization_selection(
            model, "conv_matmul", [r"stages_2\.blocks\.4\.mlp\.fc1$"]
        )
        self.assertEqual(op_types, ["Conv", "MatMul"])
        self.assertEqual(selected, ["node_MatMul_1"])
        self.assertEqual(
            MODULE._semantic_scope(node),
            "student.backbone.body.stages_2.blocks.4.mlp.fc1",
        )

    def test_legacy_node_name_is_normalized_to_same_scope(self) -> None:
        node = types.SimpleNamespace(
            name="/student/backbone/body/stages_2/blocks/blocks.4/mlp/fc1/MatMul",
            metadata_props=[],
        )
        self.assertEqual(
            MODULE._semantic_scope(node),
            "student.backbone.body.stages_2.blocks.4.mlp.fc1",
        )

    def test_calibration_reader_supports_awq_iteration(self) -> None:
        import numpy as np

        frames = [np.zeros((3, 2, 2), dtype=np.float16) for _ in range(2)]
        rows = list(MODULE._CalibrationReader(frames))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["image"].shape, (1, 3, 2, 2))

    def test_calibration_sampling_does_not_seek_past_short_video(self) -> None:
        import numpy as np

        class Capture:
            def __init__(self) -> None:
                self.position = 0
                self.positions = []

            def isOpened(self) -> bool:
                return True

            def get(self, _property: int) -> int:
                return 3

            def set(self, _property: int, position: int) -> None:
                self.position = position
                self.positions.append(position)

            def read(self):
                return self.position < 3, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        capture = Capture()
        fake_cv2 = types.SimpleNamespace(
            CAP_PROP_FRAME_COUNT=1,
            CAP_PROP_POS_FRAMES=2,
            COLOR_BGR2RGB=3,
            INTER_LINEAR=4,
            VideoCapture=lambda _path: capture,
            cvtColor=lambda frame, _conversion: frame,
            resize=lambda frame, _size, interpolation: frame,
        )
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            frames = MODULE._calibration_frames("short.mov", 5)
        self.assertEqual(len(frames), 5)
        self.assertLess(max(capture.positions), 3)


if __name__ == "__main__":
    unittest.main()
