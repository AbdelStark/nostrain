from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

from .compression import CompressedGradientPayload, inspect_payload

NOSTRAIN_GRADIENT_KIND = 33333
NOSTRAIN_MARKER = "nostrain"


@dataclass(frozen=True)
class GradientEventMetadata:
    run_name: str
    round_index: int
    worker_id: str
    model_hash: str
    inner_steps: int = 500
    created_at: int | None = None

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run name cannot be empty")
        if self.round_index < 0:
            raise ValueError("round index must be non-negative")
        if not self.worker_id:
            raise ValueError("worker id cannot be empty")
        if not self.model_hash:
            raise ValueError("model hash cannot be empty")
        if self.inner_steps <= 0:
            raise ValueError("inner step count must be positive")

    @property
    def parameterized_identifier(self) -> str:
        return (
            f"run:{self.run_name}:worker:{self.worker_id}:round:{self.round_index}"
        )

    @property
    def resolved_created_at(self) -> int:
        return self.created_at if self.created_at is not None else int(time.time())


@dataclass(frozen=True)
class NostrainEvent:
    kind: int
    created_at: int
    tags: tuple[tuple[str, str], ...]
    content: str

    def tag_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for tag_name, value in self.tags:
            mapping[tag_name] = value
        return mapping

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                [self.kind, self.created_at, list(self.tags), self.content],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "created_at": self.created_at,
            "tags": [list(tag) for tag in self.tags],
            "content": self.content,
        }

    @classmethod
    def from_json_obj(cls, data: Any) -> "NostrainEvent":
        if not isinstance(data, dict):
            raise ValueError("event JSON must be an object")
        if "kind" not in data or "created_at" not in data or "tags" not in data or "content" not in data:
            raise ValueError("event JSON must contain kind, created_at, tags, and content")

        raw_tags = data["tags"]
        if not isinstance(raw_tags, list):
            raise ValueError("event tags must be a list")

        tags: list[tuple[str, str]] = []
        for raw_tag in raw_tags:
            if (
                not isinstance(raw_tag, list)
                or len(raw_tag) < 2
                or not isinstance(raw_tag[0], str)
                or not isinstance(raw_tag[1], str)
            ):
                raise ValueError("event tags must be 2-item string lists")
            tags.append((raw_tag[0], raw_tag[1]))

        return cls(
            kind=int(data["kind"]),
            created_at=int(data["created_at"]),
            tags=tuple(tags),
            content=str(data["content"]),
        )


@dataclass(frozen=True)
class ParsedGradientEvent:
    event: NostrainEvent
    metadata: GradientEventMetadata
    payload: CompressedGradientPayload


def build_gradient_event(
    metadata: GradientEventMetadata,
    payload: str | CompressedGradientPayload,
) -> NostrainEvent:
    payload_metadata = payload if isinstance(payload, CompressedGradientPayload) else inspect_payload(payload)
    tags = (
        ("d", metadata.parameterized_identifier),
        ("t", NOSTRAIN_MARKER),
        ("run", metadata.run_name),
        ("round", str(metadata.round_index)),
        ("worker", metadata.worker_id),
        ("model", metadata.model_hash),
        ("steps", str(metadata.inner_steps)),
        ("compression", payload_metadata.compression_label),
        ("params", str(payload_metadata.parameter_count)),
        ("values", str(payload_metadata.total_values)),
        ("selected", str(payload_metadata.selected_values)),
    )
    return NostrainEvent(
        kind=NOSTRAIN_GRADIENT_KIND,
        created_at=metadata.resolved_created_at,
        tags=tags,
        content=payload_metadata.payload,
    )


def parse_gradient_event(data: NostrainEvent | dict[str, Any]) -> ParsedGradientEvent:
    event = data if isinstance(data, NostrainEvent) else NostrainEvent.from_json_obj(data)
    if event.kind != NOSTRAIN_GRADIENT_KIND:
        raise ValueError(
            f"nostrain gradient events must use kind {NOSTRAIN_GRADIENT_KIND}, got {event.kind}"
        )

    tags = event.tag_map()
    required = [
        "d",
        "t",
        "run",
        "round",
        "worker",
        "model",
        "steps",
        "compression",
        "params",
        "values",
        "selected",
    ]
    missing = [tag for tag in required if tag not in tags]
    if missing:
        raise ValueError(f"event is missing required tags: {', '.join(missing)}")
    if tags["t"] != NOSTRAIN_MARKER:
        raise ValueError(f"event marker tag must be {NOSTRAIN_MARKER!r}")

    payload = inspect_payload(event.content)
    if tags["compression"] != payload.compression_label:
        raise ValueError("event compression tag does not match the embedded payload")
    if int(tags["params"]) != payload.parameter_count:
        raise ValueError("event params tag does not match the embedded payload")
    if int(tags["values"]) != payload.total_values:
        raise ValueError("event values tag does not match the embedded payload")
    if int(tags["selected"]) != payload.selected_values:
        raise ValueError("event selected tag does not match the embedded payload")

    metadata = GradientEventMetadata(
        run_name=tags["run"],
        round_index=int(tags["round"]),
        worker_id=tags["worker"],
        model_hash=tags["model"],
        inner_steps=int(tags["steps"]),
        created_at=event.created_at,
    )
    if tags["d"] != metadata.parameterized_identifier:
        raise ValueError("event d tag does not match the run/worker/round identity")

    return ParsedGradientEvent(event=event, metadata=metadata, payload=payload)
