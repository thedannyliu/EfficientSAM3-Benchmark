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


if __name__ == "__main__":
    unittest.main()
