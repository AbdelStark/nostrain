from __future__ import annotations

from pathlib import Path
import unittest

from nostrain.compression import compress_delta
from nostrain.model import ModelState, compute_delta, state_digest
from nostrain.protocol import GradientEventMetadata, build_gradient_event, parse_gradient_event


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ProtocolTests(unittest.TestCase):
    def test_event_roundtrip_validates_embedded_payload(self) -> None:
        initial = ModelState.from_path(FIXTURES / "initial_state.json")
        current = ModelState.from_path(FIXTURES / "current_state.json")
        delta = compute_delta(initial, current)
        payload = compress_delta(delta, topk_ratio=0.5)

        metadata = GradientEventMetadata(
            run_name="demo-run",
            round_index=4,
            worker_id="worker-pubkey",
            model_hash=state_digest(initial),
            inner_steps=250,
            created_at=1_700_000_000,
        )
        event = build_gradient_event(metadata, payload)
        parsed = parse_gradient_event(event.to_json_obj())

        self.assertEqual(parsed.metadata.run_name, metadata.run_name)
        self.assertEqual(parsed.metadata.round_index, metadata.round_index)
        self.assertEqual(parsed.metadata.worker_id, metadata.worker_id)
        self.assertEqual(parsed.metadata.model_hash, metadata.model_hash)
        self.assertEqual(parsed.payload.selected_values, payload.selected_values)
        self.assertEqual(parsed.event.tag_map()["compression"], payload.compression_label)


if __name__ == "__main__":
    unittest.main()
