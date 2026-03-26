from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any
import unittest

import websockets

from nostrain.model import ModelState, compute_delta, state_digest
from nostrain.protocol import GradientEventMetadata, NostrainEvent, build_gradient_event
from nostrain.relay import collect_gradient_events
from nostrain.compression import compress_delta
from tests.helpers import assert_state_json_almost_equal


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


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
        event = frame[1]
        event_id = NostrainEvent.from_json_obj(event).fingerprint()
        accepted = True
        message = "stored"
        try:
            self._events.append(event)
            await self._broadcast_event(event)
        except Exception as exc:  # pragma: no cover - not exercised in current tests
            accepted = False
            message = f"invalid: {exc}"
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
        invalid["content"] = "not-a-valid-payload"

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


if __name__ == "__main__":
    unittest.main()
