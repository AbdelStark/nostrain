from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import secrets
from typing import Any

from .aggregation import aggregate_deltas
from .compression import decompress_payload
from .model import ModelState
from .protocol import (
    NOSTRAIN_GRADIENT_KIND,
    NOSTRAIN_MARKER,
    NostrainEvent,
    ParsedGradientEvent,
    parse_gradient_event,
)

try:
    import websockets
except ImportError as exc:  # pragma: no cover - exercised in environments without the dependency
    websockets = None
    _WEBSOCKETS_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised implicitly by integration tests
    _WEBSOCKETS_IMPORT_ERROR = None


def _require_websocket_support() -> None:
    if websockets is None:
        raise RuntimeError(
            "relay transport requires the optional 'websockets' dependency to be installed"
        ) from _WEBSOCKETS_IMPORT_ERROR


def _encode_frame(items: list[Any]) -> str:
    return json.dumps(items, separators=(",", ":"), ensure_ascii=False)


def _decode_frame(raw_message: str) -> list[Any]:
    frame = json.loads(raw_message)
    if not isinstance(frame, list) or not frame:
        raise ValueError("relay frames must be non-empty JSON arrays")
    return frame


def _subscription_id() -> str:
    return f"nostrain-{secrets.token_hex(8)}"


@dataclass(frozen=True)
class RelayPublishResult:
    relay_url: str
    event_id: str
    accepted: bool
    message: str
    notices: tuple[str, ...] = ()

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "relay": self.relay_url,
            "event_id": self.event_id,
            "accepted": self.accepted,
            "message": self.message,
            "notices": list(self.notices),
        }


@dataclass(frozen=True)
class CollectedGradientEvent:
    event_id: str
    parsed: ParsedGradientEvent

    def to_summary_obj(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.parsed.event.kind,
            "created_at": self.parsed.event.created_at,
            "run": self.parsed.metadata.run_name,
            "round": self.parsed.metadata.round_index,
            "worker": self.parsed.metadata.worker_id,
            "model": self.parsed.metadata.model_hash,
            "steps": self.parsed.metadata.inner_steps,
            "compression": self.parsed.payload.compression_label,
            "parameter_count": self.parsed.payload.parameter_count,
            "total_values": self.parsed.payload.total_values,
            "selected_values": self.parsed.payload.selected_values,
            "density": self.parsed.payload.density,
            "wire_bytes": self.parsed.payload.wire_bytes,
            "compression_ratio": self.parsed.payload.compression_ratio,
        }

    def to_json_obj(self) -> dict[str, Any]:
        data = self.to_summary_obj()
        data["event"] = self.parsed.event.to_json_obj()
        return data


@dataclass(frozen=True)
class RelayCollectionResult:
    relay_url: str
    run_name: str
    round_index: int
    events: tuple[CollectedGradientEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    notices: tuple[str, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    def aggregate_delta(self) -> ModelState:
        if not self.events:
            raise ValueError("cannot aggregate an empty relay collection")
        return aggregate_deltas(
            decompress_payload(event.parsed.payload) for event in self.events
        )

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        data = {
            "relay": self.relay_url,
            "run": self.run_name,
            "round": self.round_index,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "notices": list(self.notices),
        }
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


def _prefer_candidate(
    existing: CollectedGradientEvent,
    candidate: CollectedGradientEvent,
) -> bool:
    if candidate.parsed.event.created_at != existing.parsed.event.created_at:
        return candidate.parsed.event.created_at > existing.parsed.event.created_at
    return candidate.event_id < existing.event_id


async def publish_gradient_event(
    relay_url: str,
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_gradient_event(nostrain_event)
    event_id = nostrain_event.fingerprint()
    notices: list[str] = []

    async with websockets.connect(  # type: ignore[union-attr]
        relay_url,
        open_timeout=open_timeout,
        close_timeout=reply_timeout,
    ) as websocket:
        await websocket.send(_encode_frame(["EVENT", nostrain_event.to_json_obj()]))
        while True:
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=reply_timeout)
            frame = _decode_frame(raw_message)
            message_type = frame[0]
            if message_type == "NOTICE":
                if len(frame) > 1:
                    notices.append(str(frame[1]))
                continue
            if message_type != "OK":
                raise ValueError(f"unexpected relay response while publishing: {message_type!r}")
            return RelayPublishResult(
                relay_url=relay_url,
                event_id=str(frame[1]) if len(frame) > 1 else event_id,
                accepted=bool(frame[2]) if len(frame) > 2 else False,
                message=str(frame[3]) if len(frame) > 3 else "",
                notices=tuple(notices),
            )


async def collect_gradient_events(
    relay_url: str,
    *,
    run_name: str,
    round_index: int,
    max_events: int | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
) -> RelayCollectionResult:
    _require_websocket_support()

    if round_index < 0:
        raise ValueError("round index must be non-negative")
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")

    subscription_id = _subscription_id()
    relay_filter: dict[str, Any] = {
        "kinds": [NOSTRAIN_GRADIENT_KIND],
        "#t": [NOSTRAIN_MARKER],
    }
    if since is not None:
        relay_filter["since"] = since
    if max_events is not None:
        relay_filter["limit"] = max_events

    selected: dict[str, CollectedGradientEvent] = {}
    duplicates_discarded = 0
    invalid_events = 0
    notices: list[str] = []

    async with websockets.connect(  # type: ignore[union-attr]
        relay_url,
        open_timeout=open_timeout,
        close_timeout=idle_timeout,
    ) as websocket:
        await websocket.send(_encode_frame(["REQ", subscription_id, relay_filter]))
        try:
            while True:
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=idle_timeout,
                    )
                except asyncio.TimeoutError:
                    break

                frame = _decode_frame(raw_message)
                message_type = frame[0]

                if message_type == "EVENT":
                    if len(frame) < 3 or frame[1] != subscription_id:
                        continue
                    try:
                        parsed = parse_gradient_event(frame[2])
                    except ValueError:
                        invalid_events += 1
                        continue
                    if parsed.metadata.run_name != run_name or parsed.metadata.round_index != round_index:
                        continue

                    collected_event = CollectedGradientEvent(
                        event_id=parsed.event.fingerprint(),
                        parsed=parsed,
                    )
                    identity = parsed.metadata.parameterized_identifier
                    existing = selected.get(identity)
                    if existing is None:
                        selected[identity] = collected_event
                    else:
                        duplicates_discarded += 1
                        if _prefer_candidate(existing, collected_event):
                            selected[identity] = collected_event
                    if max_events is not None and len(selected) >= max_events:
                        break
                    continue

                if message_type == "EOSE":
                    continue

                if message_type == "NOTICE":
                    if len(frame) > 1:
                        notices.append(str(frame[1]))
                    continue

                if message_type == "CLOSED" and len(frame) > 1 and frame[1] == subscription_id:
                    break
        finally:
            try:
                await websocket.send(_encode_frame(["CLOSE", subscription_id]))
            except Exception:
                pass

    ordered_events = tuple(
        sorted(
            selected.values(),
            key=lambda event: (
                event.parsed.metadata.worker_id,
                event.parsed.event.created_at,
                event.event_id,
            ),
        )
    )
    return RelayCollectionResult(
        relay_url=relay_url,
        run_name=run_name,
        round_index=round_index,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        notices=tuple(notices),
    )
