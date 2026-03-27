from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import secrets
import time
from typing import Any, Awaitable, Callable, Iterable, Sequence, TypeVar

from .aggregation import aggregate_deltas
from .compression import decompress_payload
from .model import ModelState
from .protocol import (
    NOSTRAIN_CHECKPOINT_KIND,
    NOSTRAIN_GRADIENT_KIND,
    NOSTRAIN_HEARTBEAT_KIND,
    NOSTRAIN_MARKER,
    NostrainEvent,
    ParsedCheckpointEvent,
    ParsedGradientEvent,
    ParsedHeartbeatEvent,
    parse_checkpoint_event,
    parse_gradient_event,
    parse_heartbeat_event,
    parse_nostrain_event,
)
from .retry import RelayRetryPolicy

try:
    import websockets
except ImportError as exc:  # pragma: no cover - exercised in environments without the dependency
    websockets = None
    _WEBSOCKETS_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised implicitly by integration tests
    _WEBSOCKETS_IMPORT_ERROR = None


T = TypeVar("T")


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


def _normalize_relay_urls(relay_urls: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(str(relay).strip() for relay in relay_urls if str(relay).strip())
    )
    if not normalized:
        raise ValueError("at least one relay URL is required")
    return normalized


def _validate_sync_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    if normalized not in {"timeout", "strict", "quorum", "async"}:
        raise ValueError("sync strategy must be one of: timeout, strict, quorum, async")
    return normalized


def _retry_count(attempt_count: int) -> int:
    return max(0, attempt_count - 1)


def _retry_json_fields(
    attempt_count: int,
    retry_delays: tuple[float, ...],
    *,
    recovered_after_retry: bool | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "attempt_count": attempt_count,
        "retry_count": _retry_count(attempt_count),
        "retry_delays": list(retry_delays),
    }
    if recovered_after_retry is not None:
        data["recovered_after_retry"] = recovered_after_retry
    return data


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


class _RelayOperationFailed(RuntimeError):
    def __init__(
        self,
        *,
        relay_url: str,
        operation: str,
        cause: Exception,
        attempt_count: int,
        retry_delays: tuple[float, ...],
    ) -> None:
        self.relay_url = relay_url
        self.operation = operation
        self.cause = cause
        self.attempt_count = attempt_count
        self.retry_delays = retry_delays
        super().__init__(
            f"{operation} failed on {relay_url} after {attempt_count} attempt(s): "
            f"{type(cause).__name__}: {cause}"
        )


async def _run_with_retry(
    relay_url: str,
    operation: str,
    operation_factory: Callable[[], Awaitable[T]],
    *,
    retry_policy: RelayRetryPolicy | None,
) -> tuple[T, int, tuple[float, ...]]:
    policy = retry_policy or RelayRetryPolicy()
    retry_delays: list[float] = []

    for attempt_count in range(1, policy.max_attempts + 1):
        try:
            result = await operation_factory()
            return result, attempt_count, tuple(retry_delays)
        except Exception as exc:
            if attempt_count >= policy.max_attempts:
                raise _RelayOperationFailed(
                    relay_url=relay_url,
                    operation=operation,
                    cause=exc,
                    attempt_count=attempt_count,
                    retry_delays=tuple(retry_delays),
                ) from exc
            retry_delay = policy.delay_for_retry(attempt_count)
            retry_delays.append(retry_delay)
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)

    raise AssertionError("retry loop exhausted without returning or raising")


def _is_idempotent_publish_rejection(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    return any(
        token in normalized
        for token in ("duplicate", "already have", "already exists", "already stored")
    )


def _quorum_threshold(worker_count: int) -> int:
    return (worker_count // 2) + 1


@dataclass(frozen=True)
class RelayPublishResult:
    relay_url: str
    event_id: str
    accepted: bool
    message: str
    notices: tuple[str, ...] = ()
    attempt_count: int = 1
    retry_delays: tuple[float, ...] = ()

    @property
    def retry_count(self) -> int:
        return _retry_count(self.attempt_count)

    def to_json_obj(self) -> dict[str, Any]:
        data = {
            "relay": self.relay_url,
            "event_id": self.event_id,
            "accepted": self.accepted,
            "message": self.message,
            "notices": list(self.notices),
        }
        data.update(
            _retry_json_fields(
                self.attempt_count,
                self.retry_delays,
                recovered_after_retry=self.accepted and self.attempt_count > 1,
            )
        )
        return data


@dataclass(frozen=True)
class RelayOperationError:
    relay_url: str
    operation: str
    message: str
    attempt_count: int = 1
    retry_delays: tuple[float, ...] = ()

    @property
    def retry_count(self) -> int:
        return _retry_count(self.attempt_count)

    def to_json_obj(self) -> dict[str, Any]:
        data = {
            "relay": self.relay_url,
            "operation": self.operation,
            "message": self.message,
        }
        data.update(_retry_json_fields(self.attempt_count, self.retry_delays))
        return data


def _relay_operation_error_from_exception(
    relay_url: str,
    operation: str,
    exc: Exception,
) -> RelayOperationError:
    if isinstance(exc, _RelayOperationFailed):
        return RelayOperationError(
            relay_url=exc.relay_url,
            operation=exc.operation,
            message=f"{type(exc.cause).__name__}: {exc.cause}",
            attempt_count=exc.attempt_count,
            retry_delays=exc.retry_delays,
        )
    return RelayOperationError(
        relay_url=relay_url,
        operation=operation,
        message=f"{type(exc).__name__}: {exc}",
    )


def _aggregate_retry_count(
    relay_results: Sequence[Any],
    failed_relays: Sequence[RelayOperationError],
) -> int:
    return sum(
        _retry_count(int(getattr(item, "attempt_count", 1)))
        for item in (*relay_results, *failed_relays)
    )


def _aggregate_max_attempt_count(
    relay_results: Sequence[Any],
    failed_relays: Sequence[RelayOperationError],
) -> int:
    attempt_counts = [
        int(getattr(item, "attempt_count", 1)) for item in (*relay_results, *failed_relays)
    ]
    if not attempt_counts:
        return 1
    return max(attempt_counts)


def _aggregate_retried_relays(
    relay_results: Sequence[Any],
    failed_relays: Sequence[RelayOperationError],
) -> tuple[str, ...]:
    return _unique_strings(
        str(getattr(item, "relay_url", ""))
        for item in (*relay_results, *failed_relays)
        if int(getattr(item, "attempt_count", 1)) > 1
    )


@dataclass(frozen=True)
class MultiRelayPublishResult:
    relay_urls: tuple[str, ...]
    event_id: str
    accepted_results: tuple[RelayPublishResult, ...]
    failed_relays: tuple[RelayOperationError, ...]
    require_all: bool = False

    @property
    def accepted(self) -> bool:
        if self.require_all:
            return (
                len(self.accepted_results) == len(self.relay_urls)
                and not self.failed_relays
                and all(result.accepted for result in self.accepted_results)
            )
        return any(result.accepted for result in self.accepted_results)

    @property
    def accepted_relays(self) -> tuple[str, ...]:
        return tuple(result.relay_url for result in self.accepted_results if result.accepted)

    @property
    def total_retry_count(self) -> int:
        return _aggregate_retry_count(self.accepted_results, self.failed_relays)

    @property
    def max_attempt_count(self) -> int:
        return _aggregate_max_attempt_count(self.accepted_results, self.failed_relays)

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return _aggregate_retried_relays(self.accepted_results, self.failed_relays)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "relays": list(self.relay_urls),
            "event_id": self.event_id,
            "accepted": self.accepted,
            "accepted_relays": list(self.accepted_relays),
            "accepted_count": len(self.accepted_relays),
            "retried_relays": list(self.retried_relays),
            "total_retry_count": self.total_retry_count,
            "max_attempt_count": self.max_attempt_count,
            "failed_relays": [failure.to_json_obj() for failure in self.failed_relays],
            "results": [result.to_json_obj() for result in self.accepted_results],
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
class CollectedCheckpointEvent:
    event_id: str
    parsed: ParsedCheckpointEvent

    def to_summary_obj(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.parsed.event.kind,
            "created_at": self.parsed.event.created_at,
            "signed": self.parsed.event.is_signed,
            "pubkey": self.parsed.event.pubkey,
            "run": self.parsed.metadata.run_name,
            "worker": self.parsed.metadata.worker_id,
            "round": self.parsed.metadata.round_index,
            "history_slot": self.parsed.metadata.history_slot,
            "next_round": self.parsed.metadata.next_round,
            "model": self.parsed.metadata.model_hash,
            "rounds_completed": self.parsed.metadata.rounds_completed,
            "checkpoint_updated_at": int(self.parsed.checkpoint_data.get("updated_at", 0)),
        }

    def to_json_obj(self) -> dict[str, Any]:
        data = self.to_summary_obj()
        data["event"] = self.parsed.event.to_json_obj()
        data["checkpoint"] = self.parsed.checkpoint_data
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
    attempt_count: int = 1
    retry_delays: tuple[float, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def retry_count(self) -> int:
        return _retry_count(self.attempt_count)

    @property
    def total_retry_count(self) -> int:
        return self.retry_count

    @property
    def max_attempt_count(self) -> int:
        return self.attempt_count

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return (self.relay_url,) if self.retry_count else ()

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
        data.update(
            _retry_json_fields(
                self.attempt_count,
                self.retry_delays,
                recovered_after_retry=self.attempt_count > 1,
            )
        )
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


def _latest_checkpoint_event(
    events: Sequence[CollectedCheckpointEvent],
) -> CollectedCheckpointEvent | None:
    if not events:
        return None
    return max(
        events,
        key=lambda event: (
            event.parsed.metadata.next_round,
            event.parsed.metadata.round_index,
            event.parsed.event.created_at,
            event.event_id,
        ),
    )


@dataclass(frozen=True)
class CheckpointCollectionResult:
    relay_url: str
    run_name: str
    events: tuple[CollectedCheckpointEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    notices: tuple[str, ...] = ()
    attempt_count: int = 1
    retry_delays: tuple[float, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def retry_count(self) -> int:
        return _retry_count(self.attempt_count)

    @property
    def total_retry_count(self) -> int:
        return self.retry_count

    @property
    def max_attempt_count(self) -> int:
        return self.attempt_count

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return (self.relay_url,) if self.retry_count else ()

    @property
    def latest_event(self) -> CollectedCheckpointEvent | None:
        return _latest_checkpoint_event(self.events)

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        latest_event = self.latest_event
        data: dict[str, Any] = {
            "relay": self.relay_url,
            "run": self.run_name,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "notices": list(self.notices),
        }
        data.update(
            _retry_json_fields(
                self.attempt_count,
                self.retry_delays,
                recovered_after_retry=self.attempt_count > 1,
            )
        )
        if latest_event is not None:
            data["latest"] = latest_event.to_summary_obj()
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


@dataclass(frozen=True)
class MultiRelayCheckpointCollectionResult:
    relay_urls: tuple[str, ...]
    run_name: str
    events: tuple[CollectedCheckpointEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    notices: tuple[str, ...] = ()
    failed_relays: tuple[RelayOperationError, ...] = ()
    relay_results: tuple[CheckpointCollectionResult, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def successful_relays(self) -> tuple[str, ...]:
        return tuple(result.relay_url for result in self.relay_results)

    @property
    def total_retry_count(self) -> int:
        return _aggregate_retry_count(self.relay_results, self.failed_relays)

    @property
    def max_attempt_count(self) -> int:
        return _aggregate_max_attempt_count(self.relay_results, self.failed_relays)

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return _aggregate_retried_relays(self.relay_results, self.failed_relays)

    @property
    def latest_event(self) -> CollectedCheckpointEvent | None:
        return _latest_checkpoint_event(self.events)

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        latest_event = self.latest_event
        data: dict[str, Any] = {
            "relays": list(self.relay_urls),
            "successful_relays": list(self.successful_relays),
            "run": self.run_name,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "retried_relays": list(self.retried_relays),
            "total_retry_count": self.total_retry_count,
            "max_attempt_count": self.max_attempt_count,
            "failed_relays": [failure.to_json_obj() for failure in self.failed_relays],
            "notices": list(self.notices),
            "relay_results": [
                result.to_json_obj(include_events=False) for result in self.relay_results
            ],
        }
        if latest_event is not None:
            data["latest"] = latest_event.to_summary_obj()
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


@dataclass(frozen=True)
class MultiRelayHeartbeatCollectionResult:
    relay_urls: tuple[str, ...]
    run_name: str
    target_round: int | None
    events: tuple[CollectedHeartbeatEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    stale_events: int
    notices: tuple[str, ...] = ()
    failed_relays: tuple[RelayOperationError, ...] = ()
    relay_results: tuple[HeartbeatCollectionResult, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def successful_relays(self) -> tuple[str, ...]:
        return tuple(result.relay_url for result in self.relay_results)

    @property
    def total_retry_count(self) -> int:
        return _aggregate_retry_count(self.relay_results, self.failed_relays)

    @property
    def max_attempt_count(self) -> int:
        return _aggregate_max_attempt_count(self.relay_results, self.failed_relays)

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return _aggregate_retried_relays(self.relay_results, self.failed_relays)

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        data = {
            "relays": list(self.relay_urls),
            "successful_relays": list(self.successful_relays),
            "run": self.run_name,
            "target_round": self.target_round,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "stale_events": self.stale_events,
            "retried_relays": list(self.retried_relays),
            "total_retry_count": self.total_retry_count,
            "max_attempt_count": self.max_attempt_count,
            "failed_relays": [failure.to_json_obj() for failure in self.failed_relays],
            "notices": list(self.notices),
            "relay_results": [
                result.to_json_obj(include_events=False) for result in self.relay_results
            ],
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
    heartbeat_collection: HeartbeatCollectionResult | None = None
    attempt_count: int = 1
    retry_delays: tuple[float, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def retry_count(self) -> int:
        return _retry_count(self.attempt_count)

    @property
    def total_retry_count(self) -> int:
        return self.retry_count + (
            self.heartbeat_collection.total_retry_count if self.heartbeat_collection is not None else 0
        )

    @property
    def max_attempt_count(self) -> int:
        heartbeat_attempts = (
            self.heartbeat_collection.max_attempt_count
            if self.heartbeat_collection is not None
            else 1
        )
        return max(self.attempt_count, heartbeat_attempts)

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return _unique_strings(
            (
                self.relay_url if self.retry_count else "",
                *(
                    self.heartbeat_collection.retried_relays
                    if self.heartbeat_collection is not None
                    else ()
                ),
            )
        )

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
            "retried_relays": list(self.retried_relays),
            "total_retry_count": self.total_retry_count,
            "max_attempt_count": self.max_attempt_count,
            "notices": list(self.notices),
        }
        data.update(
            _retry_json_fields(
                self.attempt_count,
                self.retry_delays,
                recovered_after_retry=self.attempt_count > 1,
            )
        )
        if self.heartbeat_collection is not None:
            data["heartbeat_discovery"] = self.heartbeat_collection.to_json_obj(include_events=False)
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


@dataclass(frozen=True)
class MultiRelayCollectionResult:
    relay_urls: tuple[str, ...]
    run_name: str
    round_index: int
    events: tuple[CollectedGradientEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    notices: tuple[str, ...] = ()
    known_workers: tuple[str, ...] = ()
    sync_strategy: str = "timeout"
    completion_reason: str = "idle-timeout"
    failed_relays: tuple[RelayOperationError, ...] = ()
    relay_results: tuple[RelayCollectionResult, ...] = ()
    heartbeat_collection: MultiRelayHeartbeatCollectionResult | None = None

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def successful_relays(self) -> tuple[str, ...]:
        return tuple(result.relay_url for result in self.relay_results)

    @property
    def total_retry_count(self) -> int:
        total = _aggregate_retry_count(self.relay_results, self.failed_relays)
        if self.heartbeat_collection is not None:
            total += sum(result.retry_count for result in self.heartbeat_collection.relay_results)
        return total

    @property
    def max_attempt_count(self) -> int:
        heartbeat_attempts = (
            max(
                (result.attempt_count for result in self.heartbeat_collection.relay_results),
                default=1,
            )
            if self.heartbeat_collection is not None
            else 1
        )
        return max(
            _aggregate_max_attempt_count(self.relay_results, self.failed_relays),
            heartbeat_attempts,
        )

    @property
    def retried_relays(self) -> tuple[str, ...]:
        heartbeat_relays = (
            self.heartbeat_collection.retried_relays if self.heartbeat_collection is not None else ()
        )
        return _unique_strings(
            (
                *_aggregate_retried_relays(self.relay_results, self.failed_relays),
                *heartbeat_relays,
            )
        )

    def aggregate_delta(self) -> ModelState:
        if not self.events:
            raise ValueError("cannot aggregate an empty relay collection")
        return aggregate_deltas(
            decompress_payload(event.parsed.payload) for event in self.events
        )

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        data = {
            "relays": list(self.relay_urls),
            "successful_relays": list(self.successful_relays),
            "run": self.run_name,
            "round": self.round_index,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "known_workers": list(self.known_workers),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "sync_strategy": self.sync_strategy,
            "completion_reason": self.completion_reason,
            "retried_relays": list(self.retried_relays),
            "total_retry_count": self.total_retry_count,
            "max_attempt_count": self.max_attempt_count,
            "failed_relays": [failure.to_json_obj() for failure in self.failed_relays],
            "notices": list(self.notices),
            "relay_results": [
                result.to_json_obj(include_events=False) for result in self.relay_results
            ],
        }
        if self.heartbeat_collection is not None:
            data["heartbeat_discovery"] = self.heartbeat_collection.to_json_obj(include_events=False)
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


@dataclass(frozen=True)
class LateGradientCollectionResult:
    relay_url: str
    run_name: str
    current_round: int
    events: tuple[CollectedGradientEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    notices: tuple[str, ...] = ()
    attempt_count: int = 1
    retry_delays: tuple[float, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def round_indices(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(event.parsed.metadata.round_index for event in self.events))

    @property
    def retry_count(self) -> int:
        return _retry_count(self.attempt_count)

    @property
    def total_retry_count(self) -> int:
        return self.retry_count

    @property
    def max_attempt_count(self) -> int:
        return self.attempt_count

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return (self.relay_url,) if self.retry_count else ()

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        data = {
            "relay": self.relay_url,
            "run": self.run_name,
            "current_round": self.current_round,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "rounds": list(self.round_indices),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "notices": list(self.notices),
        }
        data.update(
            _retry_json_fields(
                self.attempt_count,
                self.retry_delays,
                recovered_after_retry=self.attempt_count > 1,
            )
        )
        if include_events:
            data["events"] = [event.to_json_obj() for event in self.events]
        return data


@dataclass(frozen=True)
class MultiRelayLateGradientCollectionResult:
    relay_urls: tuple[str, ...]
    run_name: str
    current_round: int
    events: tuple[CollectedGradientEvent, ...]
    duplicates_discarded: int
    invalid_events: int
    notices: tuple[str, ...] = ()
    failed_relays: tuple[RelayOperationError, ...] = ()
    relay_results: tuple[LateGradientCollectionResult, ...] = ()

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(event.parsed.metadata.worker_id for event in self.events)

    @property
    def round_indices(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(event.parsed.metadata.round_index for event in self.events))

    @property
    def successful_relays(self) -> tuple[str, ...]:
        return tuple(result.relay_url for result in self.relay_results)

    @property
    def total_retry_count(self) -> int:
        return _aggregate_retry_count(self.relay_results, self.failed_relays)

    @property
    def max_attempt_count(self) -> int:
        return _aggregate_max_attempt_count(self.relay_results, self.failed_relays)

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return _aggregate_retried_relays(self.relay_results, self.failed_relays)

    def to_json_obj(self, *, include_events: bool = True) -> dict[str, Any]:
        data = {
            "relays": list(self.relay_urls),
            "successful_relays": list(self.successful_relays),
            "run": self.run_name,
            "current_round": self.current_round,
            "event_count": len(self.events),
            "workers": list(self.worker_ids),
            "rounds": list(self.round_indices),
            "duplicates_discarded": self.duplicates_discarded,
            "invalid_events": self.invalid_events,
            "retried_relays": list(self.retried_relays),
            "total_retry_count": self.total_retry_count,
            "max_attempt_count": self.max_attempt_count,
            "failed_relays": [failure.to_json_obj() for failure in self.failed_relays],
            "notices": list(self.notices),
            "relay_results": [
                result.to_json_obj(include_events=False) for result in self.relay_results
            ],
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
    retry_policy: RelayRetryPolicy | None = None,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_gradient_event(nostrain_event)
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "publish",
        lambda: _publish_validated_event_once(
            relay_url,
            nostrain_event,
            open_timeout=open_timeout,
            reply_timeout=reply_timeout,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def publish_heartbeat_event(
    relay_url: str,
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
    retry_policy: RelayRetryPolicy | None = None,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_heartbeat_event(nostrain_event)
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "publish",
        lambda: _publish_validated_event_once(
            relay_url,
            nostrain_event,
            open_timeout=open_timeout,
            reply_timeout=reply_timeout,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def publish_checkpoint_event(
    relay_url: str,
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
    retry_policy: RelayRetryPolicy | None = None,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_checkpoint_event(nostrain_event)
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "publish",
        lambda: _publish_validated_event_once(
            relay_url,
            nostrain_event,
            open_timeout=open_timeout,
            reply_timeout=reply_timeout,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def publish_nostrain_event(
    relay_url: str,
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
    retry_policy: RelayRetryPolicy | None = None,
) -> RelayPublishResult:
    _require_websocket_support()

    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_nostrain_event(nostrain_event)
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "publish",
        lambda: _publish_validated_event_once(
            relay_url,
            nostrain_event,
            open_timeout=open_timeout,
            reply_timeout=reply_timeout,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def publish_nostrain_events(
    relay_urls: Sequence[str],
    event: NostrainEvent | dict[str, Any],
    *,
    open_timeout: float = 10.0,
    reply_timeout: float = 10.0,
    require_all: bool = False,
    retry_policy: RelayRetryPolicy | None = None,
) -> MultiRelayPublishResult:
    normalized_relays = _normalize_relay_urls(relay_urls)
    nostrain_event = event if isinstance(event, NostrainEvent) else NostrainEvent.from_json_obj(event)
    parse_nostrain_event(nostrain_event)

    results = await asyncio.gather(
        *[
            publish_nostrain_event(
                relay_url,
                nostrain_event,
                open_timeout=open_timeout,
                reply_timeout=reply_timeout,
                retry_policy=retry_policy,
            )
            for relay_url in normalized_relays
        ],
        return_exceptions=True,
    )

    accepted_results: list[RelayPublishResult] = []
    failed_relays: list[RelayOperationError] = []
    for relay_url, result in zip(normalized_relays, results):
        if isinstance(result, Exception):
            failed_relays.append(_relay_operation_error_from_exception(relay_url, "publish", result))
            continue
        if not result.accepted:
            failed_relays.append(
                RelayOperationError(
                    relay_url=relay_url,
                    operation="publish",
                    message=result.message,
                    attempt_count=result.attempt_count,
                    retry_delays=result.retry_delays,
                )
            )
            continue
        accepted_results.append(result)

    return MultiRelayPublishResult(
        relay_urls=normalized_relays,
        event_id=nostrain_event.fingerprint(),
        accepted_results=tuple(accepted_results),
        failed_relays=tuple(failed_relays),
        require_all=require_all,
    )


async def _publish_validated_event_once(
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
            accepted = bool(frame[2]) if len(frame) > 2 else False
            message = str(frame[3]) if len(frame) > 3 else ""
            if not accepted and _is_idempotent_publish_rejection(message):
                accepted = True
            return RelayPublishResult(
                relay_url=relay_url,
                event_id=str(frame[1]) if len(frame) > 1 else event_id,
                accepted=accepted,
                message=message,
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
    retry_policy: RelayRetryPolicy | None = None,
) -> HeartbeatCollectionResult:
    _require_websocket_support()

    if target_round is not None and target_round < 0:
        raise ValueError("target_round must be non-negative when provided")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")
    if max_missed_heartbeats <= 0:
        raise ValueError("max_missed_heartbeats must be positive")
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "collect-heartbeats",
        lambda: _collect_heartbeat_events_once(
            relay_url,
            run_name=run_name,
            target_round=target_round,
            idle_timeout=idle_timeout,
            open_timeout=open_timeout,
            since=since,
            reference_time=reference_time,
            max_missed_heartbeats=max_missed_heartbeats,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def _collect_heartbeat_events_once(
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


def _merge_heartbeat_results(
    relay_urls: tuple[str, ...],
    *,
    run_name: str,
    target_round: int | None,
    relay_results: Sequence[HeartbeatCollectionResult],
    failed_relays: Sequence[RelayOperationError],
) -> MultiRelayHeartbeatCollectionResult:
    selected: dict[str, CollectedHeartbeatEvent] = {}
    duplicates_discarded = sum(result.duplicates_discarded for result in relay_results)
    invalid_events = sum(result.invalid_events for result in relay_results)
    stale_events = sum(result.stale_events for result in relay_results)
    notices: list[str] = []

    for result in relay_results:
        notices.extend(result.notices)
        for event in result.events:
            identity = event.parsed.metadata.parameterized_identifier
            existing = selected.get(identity)
            if existing is None:
                selected[identity] = event
            else:
                duplicates_discarded += 1
                if _prefer_candidate(existing, event):
                    selected[identity] = event

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
    return MultiRelayHeartbeatCollectionResult(
        relay_urls=relay_urls,
        run_name=run_name,
        target_round=target_round,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        stale_events=stale_events,
        notices=tuple(notices),
        failed_relays=tuple(failed_relays),
        relay_results=tuple(relay_results),
    )


async def collect_heartbeat_events_across_relays(
    relay_urls: Sequence[str],
    *,
    run_name: str,
    target_round: int | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
    reference_time: int | None = None,
    max_missed_heartbeats: int = 3,
    retry_policy: RelayRetryPolicy | None = None,
) -> MultiRelayHeartbeatCollectionResult:
    normalized_relays = _normalize_relay_urls(relay_urls)
    results = await asyncio.gather(
        *[
            collect_heartbeat_events(
                relay_url,
                run_name=run_name,
                target_round=target_round,
                idle_timeout=idle_timeout,
                open_timeout=open_timeout,
                since=since,
                reference_time=reference_time,
                max_missed_heartbeats=max_missed_heartbeats,
                retry_policy=retry_policy,
            )
            for relay_url in normalized_relays
        ],
        return_exceptions=True,
    )

    relay_results: list[HeartbeatCollectionResult] = []
    failed_relays: list[RelayOperationError] = []
    for relay_url, result in zip(normalized_relays, results):
        if isinstance(result, Exception):
            failed_relays.append(
                _relay_operation_error_from_exception(relay_url, "collect-heartbeats", result)
            )
            continue
        relay_results.append(result)

    return _merge_heartbeat_results(
        normalized_relays,
        run_name=run_name,
        target_round=target_round,
        relay_results=relay_results,
        failed_relays=failed_relays,
    )


async def collect_checkpoint_events(
    relay_url: str,
    *,
    run_name: str,
    min_round: int | None = None,
    worker_id: str | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
    retry_policy: RelayRetryPolicy | None = None,
) -> CheckpointCollectionResult:
    _require_websocket_support()

    if min_round is not None and min_round < 0:
        raise ValueError("min_round must be non-negative when provided")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "collect-checkpoints",
        lambda: _collect_checkpoint_events_once(
            relay_url,
            run_name=run_name,
            min_round=min_round,
            worker_id=worker_id,
            idle_timeout=idle_timeout,
            open_timeout=open_timeout,
            since=since,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def _collect_checkpoint_events_once(
    relay_url: str,
    *,
    run_name: str,
    min_round: int | None = None,
    worker_id: str | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
) -> CheckpointCollectionResult:
    subscription_id = _subscription_id()
    relay_filter: dict[str, Any] = {
        "kinds": [NOSTRAIN_CHECKPOINT_KIND],
        "#t": [NOSTRAIN_MARKER],
    }
    if since is not None:
        relay_filter["since"] = since

    selected: dict[str, CollectedCheckpointEvent] = {}
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
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=idle_timeout)
                except asyncio.TimeoutError:
                    break

                frame = _decode_frame(raw_message)
                message_type = frame[0]

                if message_type == "EVENT":
                    if len(frame) < 3 or frame[1] != subscription_id:
                        continue
                    try:
                        parsed = parse_checkpoint_event(frame[2])
                    except ValueError:
                        invalid_events += 1
                        continue
                    if parsed.metadata.run_name != run_name:
                        continue
                    if min_round is not None and parsed.metadata.round_index < min_round:
                        continue
                    if worker_id is not None and parsed.metadata.worker_id != worker_id:
                        continue

                    collected_event = CollectedCheckpointEvent(
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
                event.parsed.metadata.round_index,
                event.parsed.metadata.worker_id,
                event.parsed.event.created_at,
                event.event_id,
            ),
        )
    )
    return CheckpointCollectionResult(
        relay_url=relay_url,
        run_name=run_name,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        notices=tuple(notices),
    )


def _merge_checkpoint_results(
    relay_urls: tuple[str, ...],
    *,
    run_name: str,
    relay_results: Sequence[CheckpointCollectionResult],
    failed_relays: Sequence[RelayOperationError],
) -> MultiRelayCheckpointCollectionResult:
    selected: dict[str, CollectedCheckpointEvent] = {}
    duplicates_discarded = sum(result.duplicates_discarded for result in relay_results)
    invalid_events = sum(result.invalid_events for result in relay_results)
    notices: list[str] = []

    for result in relay_results:
        notices.extend(result.notices)
        for event in result.events:
            identity = event.parsed.metadata.parameterized_identifier
            existing = selected.get(identity)
            if existing is None:
                selected[identity] = event
            else:
                duplicates_discarded += 1
                if _prefer_candidate(existing, event):
                    selected[identity] = event

    ordered_events = tuple(
        sorted(
            selected.values(),
            key=lambda event: (
                event.parsed.metadata.round_index,
                event.parsed.metadata.worker_id,
                event.parsed.event.created_at,
                event.event_id,
            ),
        )
    )
    return MultiRelayCheckpointCollectionResult(
        relay_urls=relay_urls,
        run_name=run_name,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        notices=tuple(notices),
        failed_relays=tuple(failed_relays),
        relay_results=tuple(relay_results),
    )


async def collect_checkpoint_events_across_relays(
    relay_urls: Sequence[str],
    *,
    run_name: str,
    min_round: int | None = None,
    worker_id: str | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
    retry_policy: RelayRetryPolicy | None = None,
) -> MultiRelayCheckpointCollectionResult:
    normalized_relays = _normalize_relay_urls(relay_urls)
    results = await asyncio.gather(
        *[
            collect_checkpoint_events(
                relay_url,
                run_name=run_name,
                min_round=min_round,
                worker_id=worker_id,
                idle_timeout=idle_timeout,
                open_timeout=open_timeout,
                since=since,
                retry_policy=retry_policy,
            )
            for relay_url in normalized_relays
        ],
        return_exceptions=True,
    )

    relay_results: list[CheckpointCollectionResult] = []
    failed_relays: list[RelayOperationError] = []
    for relay_url, result in zip(normalized_relays, results):
        if isinstance(result, Exception):
            failed_relays.append(
                _relay_operation_error_from_exception(relay_url, "collect-checkpoints", result)
            )
            continue
        relay_results.append(result)

    return _merge_checkpoint_results(
        normalized_relays,
        run_name=run_name,
        relay_results=relay_results,
        failed_relays=failed_relays,
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
    retry_policy: RelayRetryPolicy | None = None,
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

    heartbeat_collection = None
    if discover_workers or strategy in {"strict", "quorum"}:
        heartbeat_collection = await collect_heartbeat_events(
            relay_url,
            run_name=run_name,
            target_round=round_index,
            idle_timeout=heartbeat_idle_timeout or idle_timeout,
            open_timeout=open_timeout,
            since=heartbeat_since if heartbeat_since is not None else since,
            reference_time=reference_time,
            max_missed_heartbeats=max_missed_heartbeats,
            retry_policy=retry_policy,
        )

    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "collect-gradients",
        lambda: _collect_gradient_events_once(
            relay_url,
            run_name=run_name,
            round_index=round_index,
            max_events=max_events,
            idle_timeout=idle_timeout,
            open_timeout=open_timeout,
            since=since,
            strategy=strategy,
            known_workers=heartbeat_collection.worker_ids if heartbeat_collection is not None else (),
            initial_notices=heartbeat_collection.notices if heartbeat_collection is not None else (),
        ),
        retry_policy=retry_policy,
    )
    return replace(
        result,
        heartbeat_collection=heartbeat_collection,
        attempt_count=attempt_count,
        retry_delays=retry_delays,
    )


async def _collect_gradient_events_once(
    relay_url: str,
    *,
    run_name: str,
    round_index: int,
    max_events: int | None = None,
    idle_timeout: float = 2.0,
    open_timeout: float = 10.0,
    since: int | None = None,
    strategy: str = "timeout",
    known_workers: tuple[str, ...] = (),
    initial_notices: tuple[str, ...] = (),
) -> RelayCollectionResult:
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
    notices: list[str] = list(initial_notices)
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


def _merge_gradient_results(
    relay_urls: tuple[str, ...],
    *,
    run_name: str,
    round_index: int,
    max_events: int | None,
    strategy: str,
    relay_results: Sequence[RelayCollectionResult],
    known_workers: tuple[str, ...],
    failed_relays: Sequence[RelayOperationError],
    notices: Sequence[str],
    heartbeat_collection: MultiRelayHeartbeatCollectionResult | None = None,
) -> MultiRelayCollectionResult:
    selected: dict[str, CollectedGradientEvent] = {}
    duplicates_discarded = sum(result.duplicates_discarded for result in relay_results)
    invalid_events = sum(result.invalid_events for result in relay_results)
    merged_notices = list(notices)

    for result in relay_results:
        merged_notices.extend(result.notices)
        for event in result.events:
            identity = event.parsed.metadata.parameterized_identifier
            existing = selected.get(identity)
            if existing is None:
                selected[identity] = event
            else:
                duplicates_discarded += 1
                if _prefer_candidate(existing, event):
                    selected[identity] = event

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
    completion_reason = _completion_reason(
        selected_count=len(ordered_events),
        max_events=max_events,
        strategy=strategy,
        known_workers=known_workers,
        saw_eose=(
            strategy == "async"
            and bool(relay_results)
            and all(result.completion_reason == "async" for result in relay_results)
        ),
    )
    if completion_reason is None:
        if relay_results and all(result.completion_reason == "relay-closed" for result in relay_results):
            completion_reason = "relay-closed"
        else:
            completion_reason = "idle-timeout"

    return MultiRelayCollectionResult(
        relay_urls=relay_urls,
        run_name=run_name,
        round_index=round_index,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        notices=tuple(merged_notices),
        known_workers=known_workers,
        sync_strategy=strategy,
        completion_reason=completion_reason,
        failed_relays=tuple(failed_relays),
        relay_results=tuple(relay_results),
        heartbeat_collection=heartbeat_collection,
    )


async def collect_gradient_events_across_relays(
    relay_urls: Sequence[str],
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
    retry_policy: RelayRetryPolicy | None = None,
) -> MultiRelayCollectionResult:
    normalized_relays = _normalize_relay_urls(relay_urls)
    strategy = _validate_sync_strategy(strategy)

    known_workers: tuple[str, ...] = ()
    notices: list[str] = []
    failed_relays: list[RelayOperationError] = []
    heartbeat_results = None
    if discover_workers or strategy in {"strict", "quorum"}:
        heartbeat_results = await collect_heartbeat_events_across_relays(
            normalized_relays,
            run_name=run_name,
            target_round=round_index,
            idle_timeout=heartbeat_idle_timeout or idle_timeout,
            open_timeout=open_timeout,
            since=heartbeat_since if heartbeat_since is not None else since,
            reference_time=reference_time,
            max_missed_heartbeats=max_missed_heartbeats,
            retry_policy=retry_policy,
        )
        known_workers = heartbeat_results.worker_ids
        notices.extend(heartbeat_results.notices)
        failed_relays.extend(heartbeat_results.failed_relays)

    per_relay_strategy = strategy if len(normalized_relays) == 1 else ("async" if strategy == "async" else "timeout")
    results = await asyncio.gather(
        *[
            collect_gradient_events(
                relay_url,
                run_name=run_name,
                round_index=round_index,
                max_events=max_events,
                idle_timeout=idle_timeout,
                open_timeout=open_timeout,
                since=since,
                strategy=per_relay_strategy,
                discover_workers=False,
                heartbeat_idle_timeout=heartbeat_idle_timeout,
                heartbeat_since=heartbeat_since,
                reference_time=reference_time,
                max_missed_heartbeats=max_missed_heartbeats,
                retry_policy=retry_policy,
            )
            for relay_url in normalized_relays
        ],
        return_exceptions=True,
    )

    relay_results: list[RelayCollectionResult] = []
    for relay_url, result in zip(normalized_relays, results):
        if isinstance(result, Exception):
            failed_relays.append(
                _relay_operation_error_from_exception(relay_url, "collect-gradients", result)
            )
            continue
        relay_results.append(result)

    return _merge_gradient_results(
        normalized_relays,
        run_name=run_name,
        round_index=round_index,
        max_events=max_events,
        strategy=strategy,
        relay_results=relay_results,
        known_workers=known_workers,
        failed_relays=failed_relays,
        notices=notices,
        heartbeat_collection=heartbeat_results,
    )


async def collect_late_gradient_events(
    relay_url: str,
    *,
    run_name: str,
    current_round: int,
    idle_timeout: float = 0.2,
    open_timeout: float = 10.0,
    since: int | None = None,
    retry_policy: RelayRetryPolicy | None = None,
) -> LateGradientCollectionResult:
    _require_websocket_support()

    if current_round < 0:
        raise ValueError("current_round must be non-negative")
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")
    if current_round == 0:
        return LateGradientCollectionResult(
            relay_url=relay_url,
            run_name=run_name,
            current_round=current_round,
            events=(),
            duplicates_discarded=0,
            invalid_events=0,
        )
    result, attempt_count, retry_delays = await _run_with_retry(
        relay_url,
        "collect-late-gradients",
        lambda: _collect_late_gradient_events_once(
            relay_url,
            run_name=run_name,
            current_round=current_round,
            idle_timeout=idle_timeout,
            open_timeout=open_timeout,
            since=since,
        ),
        retry_policy=retry_policy,
    )
    return replace(result, attempt_count=attempt_count, retry_delays=retry_delays)


async def _collect_late_gradient_events_once(
    relay_url: str,
    *,
    run_name: str,
    current_round: int,
    idle_timeout: float = 0.2,
    open_timeout: float = 10.0,
    since: int | None = None,
) -> LateGradientCollectionResult:
    subscription_id = _subscription_id()
    relay_filter: dict[str, Any] = {
        "kinds": [NOSTRAIN_GRADIENT_KIND],
        "#t": [NOSTRAIN_MARKER],
    }
    if since is not None:
        relay_filter["since"] = since

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
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=idle_timeout)
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
                    if parsed.metadata.run_name != run_name:
                        continue
                    if parsed.metadata.round_index >= current_round:
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
                event.parsed.metadata.round_index,
                event.parsed.metadata.worker_id,
                event.parsed.event.created_at,
                event.event_id,
            ),
        )
    )
    return LateGradientCollectionResult(
        relay_url=relay_url,
        run_name=run_name,
        current_round=current_round,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        notices=tuple(notices),
    )


def _merge_late_gradient_results(
    relay_urls: tuple[str, ...],
    *,
    run_name: str,
    current_round: int,
    relay_results: Sequence[LateGradientCollectionResult],
    failed_relays: Sequence[RelayOperationError],
) -> MultiRelayLateGradientCollectionResult:
    selected: dict[str, CollectedGradientEvent] = {}
    duplicates_discarded = sum(result.duplicates_discarded for result in relay_results)
    invalid_events = sum(result.invalid_events for result in relay_results)
    notices: list[str] = []

    for result in relay_results:
        notices.extend(result.notices)
        for event in result.events:
            identity = event.parsed.metadata.parameterized_identifier
            existing = selected.get(identity)
            if existing is None:
                selected[identity] = event
            else:
                duplicates_discarded += 1
                if _prefer_candidate(existing, event):
                    selected[identity] = event

    ordered_events = tuple(
        sorted(
            selected.values(),
            key=lambda event: (
                event.parsed.metadata.round_index,
                event.parsed.metadata.worker_id,
                event.parsed.event.created_at,
                event.event_id,
            ),
        )
    )
    return MultiRelayLateGradientCollectionResult(
        relay_urls=relay_urls,
        run_name=run_name,
        current_round=current_round,
        events=ordered_events,
        duplicates_discarded=duplicates_discarded,
        invalid_events=invalid_events,
        notices=tuple(notices),
        failed_relays=tuple(failed_relays),
        relay_results=tuple(relay_results),
    )


async def collect_late_gradient_events_across_relays(
    relay_urls: Sequence[str],
    *,
    run_name: str,
    current_round: int,
    idle_timeout: float = 0.2,
    open_timeout: float = 10.0,
    since: int | None = None,
    retry_policy: RelayRetryPolicy | None = None,
) -> MultiRelayLateGradientCollectionResult:
    normalized_relays = _normalize_relay_urls(relay_urls)
    results = await asyncio.gather(
        *[
            collect_late_gradient_events(
                relay_url,
                run_name=run_name,
                current_round=current_round,
                idle_timeout=idle_timeout,
                open_timeout=open_timeout,
                since=since,
                retry_policy=retry_policy,
            )
            for relay_url in normalized_relays
        ],
        return_exceptions=True,
    )

    relay_results: list[LateGradientCollectionResult] = []
    failed_relays: list[RelayOperationError] = []
    for relay_url, result in zip(normalized_relays, results):
        if isinstance(result, Exception):
            failed_relays.append(
                _relay_operation_error_from_exception(relay_url, "collect-late-gradients", result)
            )
            continue
        relay_results.append(result)

    return _merge_late_gradient_results(
        normalized_relays,
        run_name=run_name,
        current_round=current_round,
        relay_results=relay_results,
        failed_relays=failed_relays,
    )
