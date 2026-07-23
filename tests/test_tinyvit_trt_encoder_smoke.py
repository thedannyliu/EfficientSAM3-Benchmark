import importlib.util
import types
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
