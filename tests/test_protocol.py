from __future__ import annotations

from pathlib import Path
import unittest

from nostrain.compression import compress_delta
from nostrain.crypto import secret_key_to_public_key
from nostrain.model import ModelState, compute_delta, state_digest
from nostrain.protocol import GradientEventMetadata, build_gradient_event, parse_gradient_event


FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_SECRET_KEY = "0000000000000000000000000000000000000000000000000000000000000003"


class ProtocolTests(unittest.TestCase):
    def test_signed_event_roundtrip_validates_embedded_payload(self) -> None:
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
        event = build_gradient_event(
            metadata,
            payload,
            secret_key_hex=TEST_SECRET_KEY,
        )
        parsed = parse_gradient_event(event.to_json_obj())

        self.assertTrue(parsed.event.is_signed)
        self.assertEqual(parsed.event.pubkey, secret_key_to_public_key(TEST_SECRET_KEY))
        self.assertEqual(parsed.metadata.run_name, metadata.run_name)
        self.assertEqual(parsed.metadata.round_index, metadata.round_index)
        self.assertEqual(parsed.metadata.worker_id, metadata.worker_id)
        self.assertEqual(parsed.metadata.model_hash, metadata.model_hash)
        self.assertEqual(parsed.payload.selected_values, payload.selected_values)
        self.assertEqual(parsed.event.tag_map()["compression"], payload.compression_label)
        self.assertEqual(parsed.event.event_id, parsed.event.fingerprint())

    def test_parse_rejects_tampered_signed_event_content(self) -> None:
        initial = ModelState.from_path(FIXTURES / "initial_state.json")
        current = ModelState.from_path(FIXTURES / "current_state.json")
        delta = compute_delta(initial, current)
        payload = compress_delta(delta, topk_ratio=0.5)
        event = build_gradient_event(
            GradientEventMetadata(
                run_name="demo-run",
                round_index=4,
                worker_id="worker-pubkey",
                model_hash=state_digest(initial),
                created_at=1_700_000_000,
            ),
            payload,
            secret_key_hex=TEST_SECRET_KEY,
        ).to_json_obj()
        event["content"] = payload.payload[::-1]

        with self.assertRaises(ValueError):
            parse_gradient_event(event)


if __name__ == "__main__":
    unittest.main()
