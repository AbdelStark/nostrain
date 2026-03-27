from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nostrain.model import ModelState
from nostrain.pytorch import model_state_to_torch_module
from nostrain.stateio import (
    load_model_state,
    load_model_state_document,
    write_model_state,
)
from tests.helpers import fake_torch_imports

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StateIoTests(unittest.TestCase):
    def test_numpy_npz_roundtrip_preserves_state_and_runtime_metadata(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "state.npz"
            write_model_state(
                archive_path,
                state,
                state_format="numpy-npz",
                runtime_name="mlp-regression",
            )

            restored = load_model_state_document(archive_path)

        self.assertEqual(restored.runtime_name, "mlp-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())

    def test_load_model_state_auto_detects_numpy_npz_extension(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "state.npz"
            write_model_state(archive_path, state, state_format="numpy-npz")

            restored = load_model_state(archive_path)

        self.assertEqual(restored.to_json_obj(), state.to_json_obj())

    def test_pytorch_state_dict_roundtrip_preserves_state_and_runtime_metadata(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "state.pt.npz"
            write_model_state(
                archive_path,
                state,
                state_format="pytorch-state-dict",
                runtime_name="mlp-regression",
            )

            restored = load_model_state_document(archive_path)

        self.assertEqual(restored.runtime_name, "mlp-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())

    def test_load_model_state_auto_detects_pytorch_state_dict_extension(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "state.pt.npz"
            write_model_state(
                archive_path,
                state,
                state_format="pytorch-state-dict",
                runtime_name="linear-regression",
            )

            restored = load_model_state(archive_path)

        self.assertEqual(restored.to_json_obj(), state.to_json_obj())

    def test_pytorch_state_dict_loading_normalizes_nested_module_prefixes(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "prefixed-state.pt.npz"
            with archive_path.open("wb") as handle:
                np.savez(
                    handle,
                    __nostrain_schema__=np.array("nostrain-pytorch-state-dict"),
                    __nostrain_version__=np.array(1, dtype=np.int64),
                    __nostrain_runtime__=np.array("linear-regression"),
                    **{
                        "module.model.weight": np.asarray([[0.0, 0.0]], dtype=np.float64),
                        "module.model.bias": np.asarray([0.0], dtype=np.float64),
                    },
                )

            restored = load_model_state_document(archive_path)

        self.assertEqual(restored.runtime_name, "linear-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())

    def test_native_torch_checkpoint_roundtrip_preserves_state(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory, fake_torch_imports():
            checkpoint_path = Path(temporary_directory) / "state.pt"
            write_model_state(
                checkpoint_path,
                state,
                state_format="pytorch-state-dict",
                runtime_name="mlp-regression",
            )

            restored = load_model_state_document(checkpoint_path)

        self.assertEqual(restored.runtime_name, "mlp-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())

    def test_native_torch_module_checkpoint_roundtrip_preserves_state(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory, fake_torch_imports():
            checkpoint_path = Path(temporary_directory) / "state.pt"
            write_model_state(
                checkpoint_path,
                state,
                state_format="pytorch-state-dict",
                runtime_name="mlp-regression",
                torch_checkpoint_payload_kind="module",
            )

            restored = load_model_state_document(checkpoint_path)

        self.assertEqual(restored.runtime_name, "mlp-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())

    def test_native_torch_checkpoint_loading_accepts_wrapped_state_dicts(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory, fake_torch_imports():
            import torch

            checkpoint_path = Path(temporary_directory) / "wrapped-state.pth"
            torch.save(
                {
                    "epoch": 7,
                    "runtime": "linear-regression",
                    "state_dict": {
                        "module.training_stack.weight": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
                        "module.training_stack.bias": torch.tensor([0.0], dtype=torch.float64),
                    },
                },
                checkpoint_path,
            )

            restored = load_model_state_document(checkpoint_path)

        self.assertEqual(restored.runtime_name, "linear-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())

    def test_native_torch_checkpoint_loading_accepts_nested_module_bundles(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")

        with tempfile.TemporaryDirectory() as temporary_directory, fake_torch_imports():
            import torch

            checkpoint_path = Path(temporary_directory) / "wrapped-module.pth"
            module = model_state_to_torch_module(state, runtime_name="linear-regression")
            torch.save(
                {
                    "runtime": "linear-regression",
                    "checkpoint": {
                        "model": module,
                    },
                },
                checkpoint_path,
            )

            restored = load_model_state_document(checkpoint_path)

        self.assertEqual(restored.runtime_name, "linear-regression")
        self.assertEqual(restored.state.to_json_obj(), state.to_json_obj())


if __name__ == "__main__":
    unittest.main()
