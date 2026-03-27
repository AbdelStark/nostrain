from __future__ import annotations

import unittest
from pathlib import Path

from nostrain.compression import (
    CompressionCodec,
    compress_delta,
    decompress_payload,
    inspect_payload,
)
from nostrain.model import ModelState, compute_delta
from tests.helpers import assert_model_state_almost_equal

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CompressionTests(unittest.TestCase):
    def setUp(self) -> None:
        initial = ModelState.from_path(FIXTURES / "initial_state.json")
        current = ModelState.from_path(FIXTURES / "current_state.json")
        self.delta = compute_delta(initial, current)

    def test_full_density_roundtrip_preserves_shape_and_values(self) -> None:
        payload = compress_delta(self.delta, topk_ratio=1.0, codec=CompressionCodec.ZLIB)
        restored = decompress_payload(payload)
        assert_model_state_almost_equal(self, restored, self.delta, places=2)

    def test_payload_inspection_reports_sparse_metadata(self) -> None:
        payload = compress_delta(self.delta, topk_ratio=0.25, codec="zlib")
        summary = inspect_payload(payload.payload)

        self.assertEqual(summary.codec.value, "zlib")
        self.assertEqual(summary.parameter_count, self.delta.parameter_count)
        self.assertEqual(summary.total_values, self.delta.total_values)
        self.assertEqual(summary.selected_values, 3)
        self.assertLess(summary.density, 0.5)


if __name__ == "__main__":
    unittest.main()
