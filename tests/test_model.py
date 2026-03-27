from __future__ import annotations

import unittest
from pathlib import Path

from nostrain.model import ModelState, apply_delta, compute_delta, state_digest
from tests.helpers import assert_model_state_almost_equal

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ModelStateTests(unittest.TestCase):
    def test_delta_roundtrip_recovers_current_state(self) -> None:
        initial = ModelState.from_path(FIXTURES / "initial_state.json")
        current = ModelState.from_path(FIXTURES / "current_state.json")

        delta = compute_delta(initial, current)
        reconstructed = apply_delta(initial, delta)

        assert_model_state_almost_equal(self, reconstructed, current, places=12)

    def test_state_digest_is_deterministic(self) -> None:
        first = ModelState.from_path(FIXTURES / "initial_state.json")
        second = ModelState.from_path(FIXTURES / "initial_state.json")

        self.assertEqual(state_digest(first), state_digest(second))


if __name__ == "__main__":
    unittest.main()
