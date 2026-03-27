from __future__ import annotations

import unittest
from pathlib import Path

from nostrain.compression import compress_delta
from nostrain.crypto import secret_key_to_public_key
from nostrain.model import ModelState, compute_delta, state_digest
from nostrain.protocol import (
    CheckpointEventMetadata,
    GradientEventMetadata,
    HeartbeatEventMetadata,
    build_checkpoint_event,
    build_gradient_event,
    build_heartbeat_event,
    parse_checkpoint_event,
    parse_gradient_event,
    parse_heartbeat_event,
)
from nostrain.training import TrainingCheckpoint, TrainingRoundSummary

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
            example_count=128,
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
        self.assertEqual(parsed.metadata.example_count, 128)
        self.assertEqual(parsed.payload.selected_values, payload.selected_values)
        self.assertEqual(parsed.event.tag_map()["compression"], payload.compression_label)
        self.assertEqual(parsed.event.tag_map()["examples"], "128")
        self.assertEqual(parsed.event.event_id, parsed.event.fingerprint())

    def test_signed_heartbeat_roundtrip_preserves_worker_metadata(self) -> None:
        event = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name="demo-run",
                worker_id="worker-pubkey",
                current_round=7,
                heartbeat_interval=45,
                example_count=64,
                capabilities=("gradient-event", "signed-events"),
                advertised_relays=("ws://127.0.0.1:8765",),
                created_at=1_700_000_100,
            ),
            secret_key_hex=TEST_SECRET_KEY,
        )
        parsed = parse_heartbeat_event(event.to_json_obj())

        self.assertTrue(parsed.event.is_signed)
        self.assertEqual(parsed.event.pubkey, secret_key_to_public_key(TEST_SECRET_KEY))
        self.assertEqual(parsed.metadata.run_name, "demo-run")
        self.assertEqual(parsed.metadata.worker_id, "worker-pubkey")
        self.assertEqual(parsed.metadata.current_round, 7)
        self.assertEqual(parsed.metadata.heartbeat_interval, 45)
        self.assertEqual(parsed.metadata.example_count, 64)
        self.assertEqual(
            parsed.metadata.capabilities,
            ("gradient-event", "signed-events"),
        )
        self.assertEqual(
            parsed.metadata.advertised_relays,
            ("ws://127.0.0.1:8765",),
        )
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

    def test_parse_rejects_heartbeat_with_non_empty_content(self) -> None:
        event = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name="demo-run",
                worker_id="worker-pubkey",
                current_round=7,
                created_at=1_700_000_100,
            ),
            secret_key_hex=TEST_SECRET_KEY,
        ).to_json_obj()
        event["content"] = "{}"

        with self.assertRaises(ValueError):
            parse_heartbeat_event(event)

    def test_checkpoint_event_roundtrip_preserves_embedded_checkpoint(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")
        checkpoint = TrainingCheckpoint(
            run_name="demo-run",
            worker_id="worker-pubkey",
            relay_urls=("ws://127.0.0.1:8765",),
            next_round=1,
            current_state=state,
            momentum_state=None,
            rounds=(
                TrainingRoundSummary(
                    round_index=0,
                    model_hash_before=state_digest(state),
                    model_hash_after=state_digest(state),
                    local_loss_before=1.0,
                    local_loss_after_inner=0.5,
                    local_loss_after_outer=0.5,
                    collected_event_count=1,
                    known_workers=("worker-pubkey",),
                    collected_workers=("worker-pubkey",),
                    completion_reason="timeout",
                    published_gradient_event_id="a" * 64,
                    published_heartbeat_event_id="b" * 64,
                    published_checkpoint_event_id="",
                    configured_relays=("ws://127.0.0.1:8765",),
                    published_heartbeat_relays=("ws://127.0.0.1:8765",),
                    published_gradient_relays=("ws://127.0.0.1:8765",),
                    published_checkpoint_relays=(),
                    collected_from_relays=("ws://127.0.0.1:8765",),
                    failed_relays=(),
                ),
            ),
            late_gradients=(),
            late_reconciliations=(),
            updated_at=1_700_000_200,
        )
        event = build_checkpoint_event(
            CheckpointEventMetadata(
                run_name="demo-run",
                worker_id="worker-pubkey",
                round_index=0,
                next_round=1,
                model_hash=state_digest(state),
                rounds_completed=1,
                history_slot=2,
                created_at=1_700_000_250,
            ),
            checkpoint.to_json_obj(),
            secret_key_hex=TEST_SECRET_KEY,
        )
        parsed = parse_checkpoint_event(event.to_json_obj())

        self.assertTrue(parsed.event.is_signed)
        self.assertEqual(parsed.metadata.run_name, "demo-run")
        self.assertEqual(parsed.metadata.worker_id, "worker-pubkey")
        self.assertEqual(parsed.metadata.round_index, 0)
        self.assertEqual(parsed.metadata.history_slot, 2)
        self.assertEqual(parsed.metadata.next_round, 1)
        self.assertEqual(parsed.metadata.model_hash, state_digest(state))
        self.assertEqual(parsed.metadata.rounds_completed, 1)
        self.assertEqual(parsed.checkpoint_data["next_round"], 1)
        self.assertEqual(parsed.checkpoint_data["updated_at"], 1_700_000_200)
        self.assertEqual(parsed.event.event_id, parsed.event.fingerprint())


if __name__ == "__main__":
    unittest.main()
