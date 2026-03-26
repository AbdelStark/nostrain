from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import secrets
import time
from typing import Any

from .aggregation import aggregate_deltas
from .compression import decompress_payload
from .model import ModelState
from .protocol import (
    NOSTRAIN_GRADIENT_KIND,
    NOSTRAIN_HEARTBEAT_KIND,
    NOSTRAIN_MARKER,
    NostrainEvent,
    ParsedGradientEvent,
    ParsedHeartbeatEvent,
    parse_gradient_event,
    parse_heartbeat_event,
    parse_nostrain_event,
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


def _validate_sync_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    if normalized not in {"timeout", "strict", "quorum", "async"}:
        raise ValueError("sync strategy must be one of: timeout, strict, quorum, async")
    return normalized


def _quorum_threshold(worker_count: int) -> int:
    return (worker_count // 2) + 1


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
            "signed": self.parsed.event.is_signed,
            "pubkey": self.parsed.event.pubkey,
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
class CollectedHeartbeatEvent:
    event_id: str
    parsed: ParsedHeartbeatEvent

    def to_summary_obj(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.parsed.event.kind,
            "created_at": self.parsed.event.created_at,
            "signed": self.parsed.event.is_signed,
            "pubkey": self.parsed.event.pubkey,
            "run": self.parsed.metadata.run_name,
            "worker": self.parsed.metadata.worker_id,
            "current_round": self.parsed.metadata.current_round,
            "heartbeat_interval": self.parsed.metadata.heartbeat_interval,
            "capabilities": list(self.parsed.metadata.capabilities),
            "advertised_relays": list(self.parsed.metadata.advertised_relays),
        }

    def to_json_obj(self) -> dict[str, Any]:
        data = self.to_summary_obj()
        data["event"] = self.parsed.event.to_json_obj()
        return data


@dataclass(frozen=True)
class HeartbeatCollectionResult:
    relay_url: str
    run_name: str
    target_round: int | None
    events: tuple[CollectedHeartbeatEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    stale_events: int
    notices: tuple[str, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        data = {
            "relay": self.relay_url,
            "run": self.run_name,
            "target_round": self.target_round,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "stale_events": self.stale_events,
            "notices": list(self.notices),
        }
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
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
    known_workers: tuple[str, ...] = ()
    sync_strategy: str = "timeout"
    completion_reason: str = "idle-timeout"

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
            "known_workers": list(self.known_workers),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "sync_strategy": self.sync_strategy,
            "completion_reason": self.completion_reason,
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
    return await _publish_validated_event(
        relay_url,
        nostrain_event,
        open_timeout=open_timeout,
        reply_timeout=reply_timeout,
    )


async def publish_heartbeat_event(
    relay_url: str,
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_heartbeat_event(nostrain_event)
    return await _publish_validated_event(
        relay_url,
        nostrain_event,
        open_timeout=open_timeout,
        reply_timeout=reply_timeout,
    )


async def publish_nostrain_event(
    relay_url: str,
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_nostrain_event(nostrain_event)
    return await _publish_validated_event(
        relay_url,
        nostrain_event,
        open_timeout=open_timeout,
        reply_timeout=reply_timeout,
    )


async def _publish_validated_event(
    relay_url: str,
    event: NostrainEvent,
    *,
    open_timeout: float,
    reply_timeout: float,
) -> RelayPublishResult:
    event_id = event.fingerprint()
    notices: list[str] = []

    async with websockets.connect(  # type: ignore[union-attr]
        relay_url,
        open_timeout=open_timeout,
        close_timeout=reply_timeout,
    ) as websocket:
        await websocket.send(_encode_frame(["EVENT", event.to_json_obj()]))
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


def _heartbeat_is_stale(
    parsed: ParsedHeartbeatEvent,
    *,
    reference_time: int,
    max_missed_heartbeats: int,
) -> bool:
    stale_after = parsed.metadata.heartbeat_interval * max_missed_heartbeats
    return parsed.event.created_at + stale_after < reference_time


async def collect_heartbeat_events(
    relay_url: str,
    *,
    run_name: str,
    target_round: int | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
    reference_time: int | None = None,
    max_missed_heartbeats: int = 3,
) -> HeartbeatCollectionResult:
    _require_websocket_support()

    if target_round is not None and target_round < 0:
        raise ValueError("target_round must be non-negative when provided")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")
    if max_missed_heartbeats <= 0:
        raise ValueError("max_missed_heartbeats must be positive")

    subscription_id = _subscription_id()
    relay_filter: dict[str, Any] = {
        "kinds": [NOSTRAIN_HEARTBEAT_KIND],
        "#t": [NOSTRAIN_MARKER],
    }
    if since is not None:
        relay_filter["since"] = since

    selected: dict[str, CollectedHeartbeatEvent] = {}
    duplicates_discarded = 0
    invalid_events = 0
    stale_events = 0
    notices: list[str] = []
    current_reference_time = reference_time if reference_time is not None else int(time.time())

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
                        parsed = parse_heartbeat_event(frame[2])
                    except ValueError:
                        invalid_events += 1
                        continue
                    if parsed.metadata.run_name != run_name:
                        continue
                    if target_round is not None and parsed.metadata.current_round < target_round:
                        continue
                    if _heartbeat_is_stale(
                        parsed,
                        reference_time=current_reference_time,
                        max_missed_heartbeats=max_missed_heartbeats,
                    ):
                        stale_events += 1
                        continue

                    collected_event = CollectedHeartbeatEvent(
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
                    continue

                if message_type == "EOSE" and len(frame) > 1 and frame[1] == subscription_id:
                    break

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
    return HeartbeatCollectionResult(
        relay_url=relay_url,
        run_name=run_name,
        target_round=target_round,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        stale_events=stale_events,
        notices=tuple(notices),
    )


def _completion_reason(
    *,
    selected_count: int,
    max_events: int | None,
    strategy: str,
    known_workers: tuple[str, ...],
    saw_eose: bool,
) -> str | None:
    if max_events is not None and selected_count >= max_events:
        return "limit"
    if strategy == "async" and saw_eose:
        return "async"
    if strategy == "strict" and known_workers and selected_count >= len(known_workers):
        return "strict"
    if strategy == "quorum" and known_workers and selected_count >= _quorum_threshold(len(known_workers)):
        return "quorum"
    return None


async def collect_gradient_events(
    relay_url: str,
    *,
    run_name: str,
    round_index: int,
    max_events: int | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
    strategy: str = "timeout",
    discover_workers: bool = False,
    heartbeat_idle_timeout: float | None = None,
    heartbeat_since: int | None = None,
    reference_time: int | None = None,
    max_missed_heartbeats: int = 3,
) -> RelayCollectionResult:
    _require_websocket_support()

    if round_index < 0:
        raise ValueError("round index must be non-negative")
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")
    if heartbeat_idle_timeout is not None and heartbeat_idle_timeout <= 0:
        raise ValueError("heartbeat_idle_timeout must be positive when provided")
    if max_missed_heartbeats <= 0:
        raise ValueError("max_missed_heartbeats must be positive")
    strategy = _validate_sync_strategy(strategy)

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
    known_workers: tuple[str, ...] = ()
    if discover_workers or strategy in {"strict", "quorum"}:
        heartbeats = await collect_heartbeat_events(
            relay_url,
            run_name=run_name,
            target_round=round_index,
            idle_timeout=heartbeat_idle_timeout or idle_timeout,
            open_timeout=open_timeout,
            since=heartbeat_since if heartbeat_since is not None else since,
            reference_time=reference_time,
            max_missed_heartbeats=max_missed_heartbeats,
        )
        known_workers = heartbeats.worker_ids
        notices.extend(heartbeats.notices)
    completion_reason = "idle-timeout"
    saw_eose = False

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
                    reason = _completion_reason(
                        selected_count=len(selected),
                        max_events=max_events,
                        strategy=strategy,
                        known_workers=known_workers,
                        saw_eose=saw_eose,
                    )
                    if reason is not None:
                        completion_reason = reason
                        break
                    continue

                if message_type == "EOSE" and len(frame) > 1 and frame[1] == subscription_id:
                    saw_eose = True
                    reason = _completion_reason(
                        selected_count=len(selected),
                        max_events=max_events,
                        strategy=strategy,
                        known_workers=known_workers,
                        saw_eose=saw_eose,
                    )
                    if reason is not None:
                        completion_reason = reason
                        break
                    continue

                if message_type == "NOTICE":
                    if len(frame) > 1:
                        notices.append(str(frame[1]))
                    continue

                if message_type == "CLOSED" and len(frame) > 1 and frame[1] == subscription_id:
                    completion_reason = "relay-closed"
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
        known_workers=known_workers,
        sync_strategy=strategy,
        completion_reason=completion_reason,
    )
