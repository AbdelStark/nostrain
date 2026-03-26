from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import unittest

import websockets

from nostrain.compression import compress_delta
from nostrain.model import ModelState, compute_delta, state_digest
from nostrain.protocol import (
    GradientEventMetadata,
    HeartbeatEventMetadata,
    NostrainEvent,
    build_gradient_event,
    build_heartbeat_event,
    parse_nostrain_event,
    parse_gradient_event,
)
from nostrain.relay import (
    collect_gradient_events,
    collect_gradient_events_across_relays,
    collect_heartbeat_events,
)
from tests.helpers import assert_model_state_almost_equal, assert_state_json_almost_equal


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
WORKER_SECRET_KEYS = {
    "worker-a": "0000000000000000000000000000000000000000000000000000000000000003",
    "worker-b": "0000000000000000000000000000000000000000000000000000000000000004",
    "worker-c": "0000000000000000000000000000000000000000000000000000000000000005",
}


def _frame(payload: list[Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _build_event(
    *,
    worker_id: str,
    current_fixture: str,
    created_at: int,
    run_name: str = "demo-run",
    round_index: int = 7,
) -> dict[str, Any]:
    initial = ModelState.from_path(FIXTURES / "initial_state.json")
    current = ModelState.from_path(FIXTURES / current_fixture)
    delta = compute_delta(initial, current)
    payload = compress_delta(delta, topk_ratio=1.0)
    event = build_gradient_event(
        GradientEventMetadata(
            run_name=run_name,
            round_index=round_index,
            worker_id=worker_id,
            model_hash=state_digest(initial),
            inner_steps=500,
            created_at=created_at,
        ),
        payload,
        secret_key_hex=WORKER_SECRET_KEYS[worker_id],
    )
    return event.to_json_obj()


def _build_heartbeat(
    *,
    worker_id: str,
    created_at: int,
    current_round: int = 7,
    run_name: str = "demo-run",
    heartbeat_interval: int = 60,
    capabilities: tuple[str, ...] = ("gradient-event",),
    advertised_relays: tuple[str, ...] = (),
) -> dict[str, Any]:
    event = build_heartbeat_event(
        HeartbeatEventMetadata(
            run_name=run_name,
            worker_id=worker_id,
            current_round=current_round,
            heartbeat_interval=heartbeat_interval,
            capabilities=capabilities,
            advertised_relays=advertised_relays,
            created_at=created_at,
        ),
        secret_key_hex=WORKER_SECRET_KEYS[worker_id],
    )
    return event.to_json_obj()


class MockRelay:
    def __init__(self, seeded_events: list[dict[str, Any]] | None = None) -> None:
        self._seeded_events = list(seeded_events or [])
        self._events: list[dict[str, Any]] = []
        self._subscriptions: list[dict[str, Any]] = []
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self.url = ""

    def __enter__(self) -> "MockRelay":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("mock relay failed to start")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._loop is None or self._stop_event is None or self._thread is None:
            return
        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())
        loop.close()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        self._events = list(self._seeded_events)
        async with websockets.serve(self._handle_connection, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            self.url = f"ws://127.0.0.1:{port}"
            self._ready.set()
            await self._stop_event.wait()

    async def _handle_connection(self, websocket) -> None:
        try:
            async for raw_message in websocket:
                frame = json.loads(raw_message)
                message_type = frame[0]
                if message_type == "EVENT":
                    await self._handle_publish(websocket, frame)
                    continue
                if message_type == "REQ":
                    await self._handle_subscription(websocket, frame)
                    continue
                if message_type == "CLOSE":
                    await self._handle_close(websocket, frame)
        finally:
            self._subscriptions = [
                subscription
                for subscription in self._subscriptions
                if subscription["websocket"] is not websocket
            ]

    async def _handle_publish(self, websocket, frame: list[Any]) -> None:
        accepted = True
        message = "stored"
        parsed_event_json: dict[str, Any] | None = None
        event_id = ""
        try:
            parsed = parse_nostrain_event(frame[1])
            if not parsed.event.is_signed:
                raise ValueError("relay requires fully signed NIP-01 events")
            event_id = parsed.event.fingerprint()
            parsed_event_json = parsed.event.to_json_obj()
            self._events.append(parsed_event_json)
            await self._broadcast_event(parsed_event_json)
        except Exception as exc:
            accepted = False
            message = f"invalid: {exc}"
            if parsed_event_json is None:
                try:
                    event_id = NostrainEvent.from_json_obj(frame[1]).fingerprint()
                except Exception:
                    event_id = ""
        await websocket.send(_frame(["OK", event_id, accepted, message]))

    async def _handle_subscription(self, websocket, frame: list[Any]) -> None:
        subscription_id = str(frame[1])
        filters = tuple(frame[2:])
        self._subscriptions.append(
            {
                "websocket": websocket,
                "subscription_id": subscription_id,
                "filters": filters,
            }
        )
        for event in self._events:
            if any(self._matches_filter(event, relay_filter) for relay_filter in filters):
                await websocket.send(_frame(["EVENT", subscription_id, event]))
        await websocket.send(_frame(["EOSE", subscription_id]))

    async def _handle_close(self, websocket, frame: list[Any]) -> None:
        subscription_id = str(frame[1])
        self._subscriptions = [
            subscription
            for subscription in self._subscriptions
            if not (
                subscription["websocket"] is websocket
                and subscription["subscription_id"] == subscription_id
            )
        ]
        try:
            await websocket.send(_frame(["CLOSED", subscription_id, "closed"]))
        except websockets.ConnectionClosed:
            pass

    async def _broadcast_event(self, event: dict[str, Any]) -> None:
        stale: list[dict[str, Any]] = []
        for subscription in self._subscriptions:
            if not any(self._matches_filter(event, relay_filter) for relay_filter in subscription["filters"]):
                continue
            try:
                await subscription["websocket"].send(
                    _frame(["EVENT", subscription["subscription_id"], event])
                )
            except Exception:
                stale.append(subscription)
        if stale:
            stale_ids = {id(subscription) for subscription in stale}
            self._subscriptions = [
                subscription
                for subscription in self._subscriptions
                if id(subscription) not in stale_ids
            ]

    def _matches_filter(self, event: dict[str, Any], relay_filter: dict[str, Any]) -> bool:
        if "kinds" in relay_filter and int(event["kind"]) not in relay_filter["kinds"]:
            return False
        if "since" in relay_filter and int(event["created_at"]) < int(relay_filter["since"]):
            return False
        if "until" in relay_filter and int(event["created_at"]) > int(relay_filter["until"]):
            return False

        for key, values in relay_filter.items():
            if key == "authors":
                if str(event.get("pubkey")) not in {str(value) for value in values}:
                    return False
                continue
            if not key.startswith("#"):
                continue
            tag_name = key[1:]
            matching_values = {
                str(tag[1])
                for tag in event.get("tags", [])
                if isinstance(tag, list) and len(tag) >= 2 and tag[0] == tag_name
            }
            if not matching_values.intersection({str(value) for value in values}):
                return False
        return True


class RelayTransportTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        src_path = str(ROOT / "src")
        env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
        return subprocess.run(
            [sys.executable, "-m", "nostrain", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_collect_events_deduplicates_replays_and_ignores_invalid_or_mismatched_updates(self) -> None:
        older = _build_event(
            worker_id="worker-a",
            current_fixture="current_state.json",
            created_at=1_700_000_000,
        )
        newer = _build_event(
            worker_id="worker-a",
            current_fixture="current_state_peer.json",
            created_at=1_700_000_010,
        )
        wrong_round = _build_event(
            worker_id="worker-b",
            current_fixture="current_state.json",
            created_at=1_700_000_020,
            round_index=8,
        )
        invalid = _build_event(
            worker_id="worker-c",
            current_fixture="current_state.json",
            created_at=1_700_000_030,
        )
        invalid["sig"] = "00" * 64

        with MockRelay([older, wrong_round, invalid, newer]) as relay:
            collection = asyncio.run(
                collect_gradient_events(
                    relay.url,
                    run_name="demo-run",
                    round_index=7,
                    idle_timeout=0.2,
                )
            )

        self.assertEqual(len(collection.events), 1)
        self.assertEqual(collection.duplicates_discarded, 1)
        self.assertEqual(collection.invalid_events, 1)
        self.assertEqual(collection.events[0].parsed.event.created_at, 1_700_000_010)
        self.assertEqual(collection.worker_ids, ("worker-a",))
        self.assertTrue(collection.events[0].parsed.event.is_signed)

    def test_collect_heartbeat_events_discards_stale_workers_and_replays(self) -> None:
        older = _build_heartbeat(
            worker_id="worker-a",
            created_at=1_700_000_200,
            advertised_relays=("ws://127.0.0.1:8765",),
        )
        newer = _build_heartbeat(
            worker_id="worker-a",
            created_at=1_700_000_250,
            advertised_relays=("ws://127.0.0.1:9999",),
        )
        active_peer = _build_heartbeat(
            worker_id="worker-b",
            created_at=1_700_000_240,
        )
        stale = _build_heartbeat(
            worker_id="worker-c",
            created_at=1_700_000_000,
        )
        invalid = _build_heartbeat(
            worker_id="worker-c",
            created_at=1_700_000_260,
        )
        invalid["sig"] = "00" * 64

        with MockRelay([older, stale, invalid, newer, active_peer]) as relay:
            collection = asyncio.run(
                collect_heartbeat_events(
                    relay.url,
                    run_name="demo-run",
                    target_round=7,
                    idle_timeout=0.2,
                    reference_time=1_700_000_300,
                )
            )

        self.assertEqual(collection.worker_ids, ("worker-a", "worker-b"))
        self.assertEqual(collection.duplicates_discarded, 1)
        self.assertEqual(collection.invalid_events, 1)
        self.assertEqual(collection.stale_events, 1)
        self.assertEqual(
            collection.events[0].parsed.metadata.advertised_relays,
            ("ws://127.0.0.1:9999",),
        )

    def test_collect_events_can_stop_after_quorum_of_discovered_workers(self) -> None:
        heartbeat_a = _build_heartbeat(worker_id="worker-a", created_at=1_700_000_000)
        heartbeat_b = _build_heartbeat(worker_id="worker-b", created_at=1_700_000_001)
        heartbeat_c = _build_heartbeat(worker_id="worker-c", created_at=1_700_000_002)
        event_a = _build_event(
            worker_id="worker-a",
            current_fixture="current_state.json",
            created_at=1_700_000_010,
        )
        event_b = _build_event(
            worker_id="worker-b",
            current_fixture="current_state_peer.json",
            created_at=1_700_000_011,
        )

        with MockRelay([heartbeat_a, heartbeat_b, heartbeat_c, event_a, event_b]) as relay:
            collection = asyncio.run(
                collect_gradient_events(
                    relay.url,
                    run_name="demo-run",
                    round_index=7,
                    idle_timeout=5.0,
                    strategy="quorum",
                    discover_workers=True,
                    reference_time=1_700_000_020,
                )
            )

        self.assertEqual(len(collection.events), 2)
        self.assertEqual(collection.worker_ids, ("worker-a", "worker-b"))
        self.assertEqual(collection.known_workers, ("worker-a", "worker-b", "worker-c"))
        self.assertEqual(collection.sync_strategy, "quorum")
        self.assertEqual(collection.completion_reason, "quorum")

    def test_cli_publish_collect_and_aggregate_roundtrip(self) -> None:
        expected_average = {
            "parameters": {
                "encoder.bias": {
                    "shape": [3],
                    "values": [0.0115, -0.0225, 0.0325],
                },
                "encoder.weight": {
                    "shape": [2, 3],
                    "values": [0.17, -0.11, 0.295, 0.435, -0.535, 0.615],
                },
                "head.weight": {
                    "shape": [1, 3],
                    "values": [0.625, -0.25, 0.77],
                },
            }
        }

        with tempfile.TemporaryDirectory() as temporary_directory, MockRelay() as relay:
            tempdir = Path(temporary_directory)
            payload_a = tempdir / "payload-a.json"
            payload_b = tempdir / "payload-b.json"
            event_a = tempdir / "event-a.json"
            event_b = tempdir / "event-b.json"
            publish_a = tempdir / "publish-a.json"
            publish_b = tempdir / "publish-b.json"
            collected = tempdir / "collected.json"
            aggregated = tempdir / "aggregated.json"
            aggregated_from_events = tempdir / "aggregated-from-events.json"
            aggregation_summary = tempdir / "aggregation-summary.json"

            digest = self._run("hash-state", str(FIXTURES / "initial_state.json")).stdout.strip()

            self._run(
                "encode-delta",
                str(FIXTURES / "initial_state.json"),
                str(FIXTURES / "current_state.json"),
                "--topk",
                "1.0",
                "-o",
                str(payload_a),
            )
            self._run(
                "encode-delta",
                str(FIXTURES / "initial_state.json"),
                str(FIXTURES / "current_state_peer.json"),
                "--topk",
                "1.0",
                "-o",
                str(payload_b),
            )

            self._run(
                "build-event",
                str(payload_a),
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-a",
                "--model",
                digest,
                "--created-at",
                "1700000001",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "-o",
                str(event_a),
            )
            self._run(
                "build-event",
                str(payload_b),
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-b",
                "--model",
                digest,
                "--created-at",
                "1700000002",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-b"],
                "-o",
                str(event_b),
            )

            self._run(
                "publish-event",
                str(event_a),
                "--relay",
                relay.url,
                "--json",
                "-o",
                str(publish_a),
            )
            self._run(
                "publish-event",
                str(event_b),
                "--relay",
                relay.url,
                "--json",
                "-o",
                str(publish_b),
            )

            publish_result = json.loads(publish_a.read_text(encoding="utf-8"))
            self.assertTrue(publish_result["accepted"])
            self.assertEqual(publish_result["relay"], relay.url)
            self.assertEqual(len(publish_result["event_id"]), 64)

            self._run(
                "collect-events",
                "--relay",
                relay.url,
                "--run",
                "demo-run",
                "--round",
                "7",
                "--limit",
                "2",
                "--idle-timeout",
                "0.2",
                "--json",
                "-o",
                str(collected),
            )
            collected_json = json.loads(collected.read_text(encoding="utf-8"))
            self.assertEqual(collected_json["event_count"], 2)
            self.assertEqual(sorted(collected_json["workers"]), ["worker-a", "worker-b"])
            self.assertEqual(collected_json["duplicates_discarded"], 0)
            self.assertEqual(collected_json["invalid_events"], 0)
            self.assertTrue(all(event["signed"] for event in collected_json["events"]))

            self._run(
                "aggregate-payloads",
                str(event_a),
                str(event_b),
                "-o",
                str(aggregated_from_events),
            )

            self._run(
                "aggregate-round",
                "--relay",
                relay.url,
                "--run",
                "demo-run",
                "--round",
                "7",
                "--limit",
                "2",
                "--idle-timeout",
                "0.2",
                "--summary-out",
                str(aggregation_summary),
                "-o",
                str(aggregated),
            )

            aggregated_json = json.loads(aggregated.read_text(encoding="utf-8"))
            expected_delta = compute_delta(
                ModelState.from_path(FIXTURES / "initial_state.json"),
                ModelState.from_json_obj(expected_average),
            ).to_json_obj()
            assert_state_json_almost_equal(self, aggregated_json, expected_delta, places=2)
            assert_state_json_almost_equal(
                self,
                json.loads(aggregated_from_events.read_text(encoding="utf-8")),
                expected_delta,
                places=2,
            )

            aggregation_summary_json = json.loads(aggregation_summary.read_text(encoding="utf-8"))
            self.assertEqual(aggregation_summary_json["event_count"], 2)
            self.assertEqual(sorted(aggregation_summary_json["workers"]), ["worker-a", "worker-b"])
            self.assertEqual(aggregation_summary_json["completion_reason"], "limit")
            self.assertEqual(aggregation_summary_json["sync_strategy"], "timeout")

    def test_cli_build_publish_and_discover_heartbeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, MockRelay() as relay:
            tempdir = Path(temporary_directory)
            heartbeat_a = tempdir / "heartbeat-a.json"
            heartbeat_b = tempdir / "heartbeat-b.json"
            publish_a = tempdir / "publish-a.json"
            discovered = tempdir / "workers.json"
            current_timestamp = str(int(time.time()))
            current_timestamp_peer = str(int(time.time()) + 1)

            self._run(
                "build-heartbeat",
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-a",
                "--capability",
                "gradient-event",
                "--advertise-relay",
                relay.url,
                "--created-at",
                current_timestamp,
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "-o",
                str(heartbeat_a),
            )
            self._run(
                "build-heartbeat",
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-b",
                "--capability",
                "gradient-event",
                "--created-at",
                current_timestamp_peer,
                "--sec-key",
                WORKER_SECRET_KEYS["worker-b"],
                "-o",
                str(heartbeat_b),
            )

            self._run(
                "publish-event",
                str(heartbeat_a),
                "--relay",
                relay.url,
                "--json",
                "-o",
                str(publish_a),
            )
            self._run(
                "publish-event",
                str(heartbeat_b),
                "--relay",
                relay.url,
                "--json",
            )

            self._run(
                "discover-workers",
                "--relay",
                relay.url,
                "--run",
                "demo-run",
                "--round",
                "7",
                "--json",
                "-o",
                str(discovered),
            )
            discovered_json = json.loads(discovered.read_text(encoding="utf-8"))
            publish_json = json.loads(publish_a.read_text(encoding="utf-8"))

            self.assertTrue(publish_json["accepted"])
            self.assertEqual(discovered_json["event_count"], 2)
            self.assertEqual(sorted(discovered_json["workers"]), ["worker-a", "worker-b"])
            self.assertEqual(discovered_json["stale_events"], 0)
            self.assertEqual(
                discovered_json["events"][0]["capabilities"],
                ["gradient-event"],
            )

    def test_cli_publish_rejects_unsigned_event_on_signed_relay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, MockRelay() as relay:
            tempdir = Path(temporary_directory)
            payload_path = tempdir / "payload.json"
            event_path = tempdir / "event.json"

            digest = self._run("hash-state", str(FIXTURES / "initial_state.json")).stdout.strip()
            self._run(
                "encode-delta",
                str(FIXTURES / "initial_state.json"),
                str(FIXTURES / "current_state.json"),
                "--topk",
                "1.0",
                "-o",
                str(payload_path),
            )
            self._run(
                "build-event",
                str(payload_path),
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-a",
                "--model",
                digest,
                "--created-at",
                "1700000001",
                "-o",
                str(event_path),
            )

            with self.assertRaises(subprocess.CalledProcessError):
                self._run(
                    "publish-event",
                    str(event_path),
                    "--relay",
                    relay.url,
                )

    def test_cli_run_training_completes_after_timeout_with_missing_peer(self) -> None:
        seeded_peer_heartbeat = _build_heartbeat(
            worker_id="worker-b",
            created_at=int(time.time()) + 10,
            current_round=0,
            run_name="training-run",
        )

        with tempfile.TemporaryDirectory() as temporary_directory, MockRelay([seeded_peer_heartbeat]) as relay:
            tempdir = Path(temporary_directory)
            final_state = tempdir / "final-state.json"
            summary = tempdir / "summary.json"
            momentum = tempdir / "momentum.json"
            artifact_dir = tempdir / "artifacts"

            self._run(
                "run-training",
                str(FIXTURES / "linear_initial_state.json"),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--relay",
                relay.url,
                "--run",
                "training-run",
                "--worker",
                "worker-a",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "--rounds",
                "2",
                "--inner-steps",
                "30",
                "--local-learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--outer-learning-rate",
                "1.0",
                "--momentum",
                "0.5",
                "--round-timeout",
                "0.2",
                "--artifact-dir",
                str(artifact_dir),
                "--momentum-out",
                str(momentum),
                "--summary-out",
                str(summary),
                "-o",
                str(final_state),
            )

            final_state_json = json.loads(final_state.read_text(encoding="utf-8"))
            summary_json = json.loads(summary.read_text(encoding="utf-8"))

            self.assertIn("parameters", final_state_json)
            self.assertTrue(momentum.exists())
            self.assertEqual(summary_json["rounds_completed"], 2)
            self.assertEqual(len(summary_json["rounds"]), 2)
            self.assertEqual(summary_json["rounds"][0]["collected_event_count"], 1)
            self.assertEqual(summary_json["rounds"][0]["missing_workers"], ["worker-b"])
            self.assertEqual(summary_json["rounds"][0]["completion_reason"], "idle-timeout")
            self.assertTrue((artifact_dir / "round-0000" / "collection.json").exists())
            self.assertTrue((artifact_dir / "round-0001" / "next-state.json").exists())

    def test_cli_run_training_two_workers_converge_via_relay(self) -> None:
        seeded_heartbeats = [
            _build_heartbeat(
                worker_id="worker-a",
                created_at=int(time.time()) + 10,
                current_round=0,
                run_name="shared-run",
            ),
            _build_heartbeat(
                worker_id="worker-b",
                created_at=int(time.time()) + 11,
                current_round=0,
                run_name="shared-run",
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory, MockRelay(seeded_heartbeats) as relay:
            tempdir = Path(temporary_directory)
            state_a = tempdir / "state-a.json"
            state_b = tempdir / "state-b.json"
            summary_a = tempdir / "summary-a.json"
            summary_b = tempdir / "summary-b.json"

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            src_path = str(ROOT / "src")
            env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"

            process_a = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "nostrain",
                    "run-training",
                    str(FIXTURES / "linear_initial_state.json"),
                    str(FIXTURES / "linear_dataset_worker_a.json"),
                    "--relay",
                    relay.url,
                    "--run",
                    "shared-run",
                    "--worker",
                    "worker-a",
                    "--sec-key",
                    WORKER_SECRET_KEYS["worker-a"],
                    "--inner-steps",
                    "30",
                    "--local-learning-rate",
                    "0.05",
                    "--batch-size",
                    "2",
                    "--outer-learning-rate",
                    "1.0",
                    "--momentum",
                    "0.0",
                    "--round-timeout",
                    "0.4",
                    "--summary-out",
                    str(summary_a),
                    "-o",
                    str(state_a),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process_b = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "nostrain",
                    "run-training",
                    str(FIXTURES / "linear_initial_state.json"),
                    str(FIXTURES / "linear_dataset_worker_b.json"),
                    "--relay",
                    relay.url,
                    "--run",
                    "shared-run",
                    "--worker",
                    "worker-b",
                    "--sec-key",
                    WORKER_SECRET_KEYS["worker-b"],
                    "--inner-steps",
                    "30",
                    "--local-learning-rate",
                    "0.05",
                    "--batch-size",
                    "2",
                    "--outer-learning-rate",
                    "1.0",
                    "--momentum",
                    "0.0",
                    "--round-timeout",
                    "0.4",
                    "--summary-out",
                    str(summary_b),
                    "-o",
                    str(state_b),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdout_a, stderr_a = process_a.communicate(timeout=20)
            stdout_b, stderr_b = process_b.communicate(timeout=20)

            if process_a.returncode != 0:
                self.fail(f"worker-a failed: {stderr_a or stdout_a}")
            if process_b.returncode != 0:
                self.fail(f"worker-b failed: {stderr_b or stdout_b}")

            final_state_a = ModelState.from_path(state_a)
            final_state_b = ModelState.from_path(state_b)
            summary_json_a = json.loads(summary_a.read_text(encoding="utf-8"))
            summary_json_b = json.loads(summary_b.read_text(encoding="utf-8"))

            assert_model_state_almost_equal(self, final_state_a, final_state_b, places=8)
            self.assertEqual(summary_json_a["rounds"][0]["collected_event_count"], 2)
            self.assertEqual(summary_json_b["rounds"][0]["collected_event_count"], 2)
            self.assertEqual(
                sorted(summary_json_a["rounds"][0]["known_workers"]),
                ["worker-a", "worker-b"],
            )
            self.assertEqual(
                sorted(summary_json_b["rounds"][0]["collected_workers"]),
                ["worker-a", "worker-b"],
            )

    def test_collect_gradient_events_across_relays_deduplicates_replays(self) -> None:
        worker_a_event = _build_event(
            worker_id="worker-a",
            current_fixture="current_state.json",
            created_at=1700000000,
        )
        worker_b_event = _build_event(
            worker_id="worker-b",
            current_fixture="current_state_peer.json",
            created_at=1700000001,
        )

        with MockRelay([worker_a_event]) as relay_a, MockRelay([worker_a_event, worker_b_event]) as relay_b:
            collection = asyncio.run(
                collect_gradient_events_across_relays(
                    (relay_a.url, relay_b.url),
                    run_name="demo-run",
                    round_index=7,
                    idle_timeout=0.2,
                    open_timeout=1.0,
                )
            )

        self.assertEqual(sorted(collection.worker_ids), ["worker-a", "worker-b"])
        self.assertEqual(collection.duplicates_discarded, 1)
        self.assertEqual(
            sorted(collection.successful_relays),
            sorted([relay_a.url, relay_b.url]),
        )
        self.assertEqual(collection.aggregate_delta().parameter_count, 3)

    def test_cli_run_training_survives_partial_relay_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, MockRelay() as relay:
            tempdir = Path(temporary_directory)
            final_state = tempdir / "final-state.json"
            summary = tempdir / "summary.json"
            unreachable_relay = "ws://127.0.0.1:65534"

            self._run(
                "run-training",
                str(FIXTURES / "linear_initial_state.json"),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--relay",
                relay.url,
                "--relay",
                unreachable_relay,
                "--run",
                "resilient-run",
                "--worker",
                "worker-a",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "--rounds",
                "1",
                "--inner-steps",
                "20",
                "--local-learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--outer-learning-rate",
                "1.0",
                "--momentum",
                "0.0",
                "--round-timeout",
                "0.2",
                "--timeout",
                "0.2",
                "--summary-out",
                str(summary),
                "-o",
                str(final_state),
            )

            summary_json = json.loads(summary.read_text(encoding="utf-8"))
            self.assertIn("parameters", json.loads(final_state.read_text(encoding="utf-8")))
            self.assertEqual(summary_json["relays"], [relay.url, unreachable_relay])
            self.assertEqual(summary_json["rounds_completed"], 1)
            self.assertEqual(summary_json["rounds"][0]["configured_relays"], [relay.url, unreachable_relay])
            self.assertEqual(summary_json["rounds"][0]["published_gradient_relays"], [relay.url])
            self.assertEqual(summary_json["rounds"][0]["collected_from_relays"], [relay.url])
            self.assertIn(unreachable_relay, summary_json["rounds"][0]["failed_relays"])

    def test_cli_run_training_resume_from_checkpoint_matches_uninterrupted_run(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            MockRelay() as resume_relay,
            MockRelay() as uninterrupted_relay,
        ):
            tempdir = Path(temporary_directory)
            checkpoint = tempdir / "checkpoint.json"
            resumed_state_path = tempdir / "resumed-state.json"
            resumed_summary = tempdir / "resumed-summary.json"
            uninterrupted_state_path = tempdir / "full-state.json"
            uninterrupted_summary = tempdir / "full-summary.json"

            self._run(
                "run-training",
                str(FIXTURES / "linear_initial_state.json"),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--relay",
                resume_relay.url,
                "--run",
                "checkpoint-run",
                "--worker",
                "worker-a",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "--rounds",
                "1",
                "--inner-steps",
                "30",
                "--local-learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--outer-learning-rate",
                "1.0",
                "--momentum",
                "0.5",
                "--round-timeout",
                "0.2",
                "--checkpoint-out",
                str(checkpoint),
                "-o",
                str(tempdir / "partial-state.json"),
            )

            checkpoint_json = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_json["next_round"], 1)
            self.assertEqual(checkpoint_json["rounds_completed"], 1)

            self._run(
                "run-training",
                str(FIXTURES / "linear_initial_state.json"),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--relay",
                resume_relay.url,
                "--run",
                "checkpoint-run",
                "--worker",
                "worker-a",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "--rounds",
                "1",
                "--inner-steps",
                "30",
                "--local-learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--outer-learning-rate",
                "1.0",
                "--momentum",
                "0.5",
                "--round-timeout",
                "0.2",
                "--resume-from",
                str(checkpoint),
                "--summary-out",
                str(resumed_summary),
                "-o",
                str(resumed_state_path),
            )

            self._run(
                "run-training",
                str(FIXTURES / "linear_initial_state.json"),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--relay",
                uninterrupted_relay.url,
                "--run",
                "checkpoint-run-full",
                "--worker",
                "worker-a",
                "--sec-key",
                WORKER_SECRET_KEYS["worker-a"],
                "--rounds",
                "2",
                "--inner-steps",
                "30",
                "--local-learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--outer-learning-rate",
                "1.0",
                "--momentum",
                "0.5",
                "--round-timeout",
                "0.2",
                "--summary-out",
                str(uninterrupted_summary),
                "-o",
                str(uninterrupted_state_path),
            )

            resumed_state = ModelState.from_path(resumed_state_path)
            uninterrupted_state = ModelState.from_path(uninterrupted_state_path)
            resumed_summary_json = json.loads(resumed_summary.read_text(encoding="utf-8"))

            assert_model_state_almost_equal(self, resumed_state, uninterrupted_state, places=8)
            self.assertEqual(resumed_summary_json["rounds_completed"], 2)
            self.assertEqual(
                [round_summary["round"] for round_summary in resumed_summary_json["rounds"]],
                [0, 1],
            )


if __name__ == "__main__":
    unittest.main()
