from __future__ import annotations

from pathlib import Path
import unittest

from nostrain.model import ModelState, TensorState
from nostrain.pytorch import (
    export_pytorch_state_dict,
    import_pytorch_state_dict,
    infer_training_runtime_from_pytorch_state_dict,
)
from nostrain.runtime import LINEAR_REGRESSION_RUNTIME, MLP_REGRESSION_RUNTIME
from tests.helpers import assert_model_state_almost_equal


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class PyTorchAdapterTests(unittest.TestCase):
    def test_linear_state_exports_to_direct_linear_module_keys(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")

        exported = export_pytorch_state_dict(state)
        restored = import_pytorch_state_dict(exported)

        self.assertEqual(
            tuple(parameter.name for parameter in exported.parameters),
            ("bias", "weight"),
        )
        self.assertEqual(
            infer_training_runtime_from_pytorch_state_dict(exported),
            LINEAR_REGRESSION_RUNTIME,
        )
        assert_model_state_almost_equal(self, restored, state, places=12)

    def test_mlp_state_import_accepts_nested_module_prefixes(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")
        exported = export_pytorch_state_dict(state, runtime_name=MLP_REGRESSION_RUNTIME)
        prefixed = ModelState(
            parameters=tuple(
                TensorState(
                    name=f"module.training_stack.{parameter.name}",
                    shape=parameter.shape,
                    values=parameter.values,
                )
                for parameter in exported.parameters
            )
        )

        restored = import_pytorch_state_dict(prefixed)

        self.assertEqual(
            infer_training_runtime_from_pytorch_state_dict(prefixed),
            MLP_REGRESSION_RUNTIME,
        )
        assert_model_state_almost_equal(self, restored, state, places=12)

    def test_import_rejects_mixed_or_unsupported_pytorch_parameter_layouts(self) -> None:
        invalid_state = ModelState(
            parameters=(
                TensorState(name="weight", shape=(1, 2), values=(0.1, -0.2)),
                TensorState(name="output.bias", shape=(1,), values=(0.0,)),
            )
        )

        with self.assertRaises(ValueError):
            import_pytorch_state_dict(invalid_state)


if __name__ == "__main__":
    unittest.main()
