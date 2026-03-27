from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from nostrain.model import ModelState
from nostrain.stateio import load_model_state, load_model_state_document, write_model_state


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


if __name__ == "__main__":
    unittest.main()
