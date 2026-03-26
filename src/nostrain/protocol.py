from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Any

from .compression import CompressedGradientPayload, inspect_payload
from .crypto import (
    normalize_hex_string,
    normalize_public_key_hex,
    normalize_signature_hex,
    schnorr_sign,
    schnorr_verify,
    secret_key_to_public_key,
)

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


NostrTag = tuple[str, ...]


@dataclass(frozen=True)
class NostrainEvent:
    kind: int
    created_at: int
    tags: tuple[NostrTag, ...]
    content: str
    pubkey: str | None = None
    event_id: str | None = None
    sig: str | None = None

    @property
    def is_signed(self) -> bool:
        return self.pubkey is not None and self.event_id is not None and self.sig is not None

    @property
    def is_signable(self) -> bool:
        return self.pubkey is not None

    @property
    def signing_state(self) -> str:
        if self.is_signed:
            return "signed"
        if self.is_signable:
            return "signable"
        return "unsigned"

    def tag_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for tag in self.tags:
            if len(tag) >= 2:
                mapping[tag[0]] = tag[1]
        return mapping

    def serialize_for_event_id(self) -> str:
        if self.pubkey is None:
            raise ValueError("cannot serialize an unsigned event for NIP-01 event id generation")
        normalized_pubkey = normalize_public_key_hex(self.pubkey)
        return json.dumps(
            [0, normalized_pubkey, self.created_at, self.kind, [list(tag) for tag in self.tags], self.content],
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def compute_event_id(self) -> str:
        if self.pubkey is None:
            raise ValueError("cannot compute a NIP-01 event id without a pubkey")
        return hashlib.sha256(self.serialize_for_event_id().encode("utf-8")).hexdigest()

    def validate_signature(self) -> None:
        _validate_signing_fields(self)

    def fingerprint(self) -> str:
        if self.pubkey is not None:
            computed = self.compute_event_id()
            if self.event_id is not None and self.event_id != computed:
                raise ValueError("event id does not match canonical NIP-01 serialization")
            return computed

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                [self.kind, self.created_at, [list(tag) for tag in self.tags], self.content],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def to_json_obj(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "created_at": self.created_at,
            "tags": [list(tag) for tag in self.tags],
            "content": self.content,
        }
        if self.pubkey is not None:
            data["pubkey"] = normalize_public_key_hex(self.pubkey)
            data["id"] = self.event_id or self.compute_event_id()
        if self.sig is not None:
            data["sig"] = normalize_signature_hex(self.sig)
        return data

    @classmethod
    def from_json_obj(cls, data: Any) -> "NostrainEvent":
        if not isinstance(data, dict):
            raise ValueError("event JSON must be an object")
        required_fields = {"kind", "created_at", "tags", "content"}
        if not required_fields.issubset(data):
            raise ValueError("event JSON must contain kind, created_at, tags, and content")

        raw_tags = data["tags"]
        if not isinstance(raw_tags, list):
            raise ValueError("event tags must be a list")

        tags: list[NostrTag] = []
        for raw_tag in raw_tags:
            if not isinstance(raw_tag, (list, tuple)) or not raw_tag:
                raise ValueError("event tags must be non-empty string lists")
            normalized_tag = tuple(str(value) for value in raw_tag)
            if any(value == "" for value in normalized_tag):
                raise ValueError("event tags cannot contain empty string items")
            tags.append(normalized_tag)

        pubkey = (
            normalize_public_key_hex(str(data["pubkey"]))
            if "pubkey" in data
            else None
        )
        event_id = (
            normalize_hex_string(str(data["id"]), byte_length=32, label="event id")
            if "id" in data
            else None
        )
        sig = (
            normalize_signature_hex(str(data["sig"]))
            if "sig" in data
            else None
        )

        return cls(
            kind=int(data["kind"]),
            created_at=int(data["created_at"]),
            tags=tuple(tags),
            content=str(data["content"]),
            pubkey=pubkey,
            event_id=event_id,
            sig=sig,
        )


@dataclass(frozen=True)
class ParsedGradientEvent:
    event: NostrainEvent
    metadata: GradientEventMetadata
    payload: CompressedGradientPayload


def build_gradient_event(
    metadata: GradientEventMetadata,
    payload: str | CompressedGradientPayload,
    *,
    secret_key_hex: str | None = None,
    public_key_hex: str | None = None,
    signature_hex: str | None = None,
    event_id: str | None = None,
    aux_rand: bytes | None = None,
) -> NostrainEvent:
    if secret_key_hex is not None and any(
        value is not None for value in (public_key_hex, signature_hex, event_id)
    ):
        raise ValueError("secret key signing cannot be combined with explicit pubkey/id/signature fields")
    if signature_hex is not None and public_key_hex is None:
        raise ValueError("a delegated signature requires an explicit public key")
    if event_id is not None and public_key_hex is None:
        raise ValueError("an explicit event id requires an explicit public key")

    payload_metadata = payload if isinstance(payload, CompressedGradientPayload) else inspect_payload(payload)
    tags: tuple[NostrTag, ...] = (
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
    resolved_pubkey = (
        secret_key_to_public_key(secret_key_hex)
        if secret_key_hex is not None
        else (
            normalize_public_key_hex(public_key_hex)
            if public_key_hex is not None
            else None
        )
    )
    event = NostrainEvent(
        kind=NOSTRAIN_GRADIENT_KIND,
        created_at=metadata.resolved_created_at,
        tags=tags,
        content=payload_metadata.payload,
        pubkey=resolved_pubkey,
    )

    if resolved_pubkey is None:
        return event

    computed_event_id = event.compute_event_id()
    if event_id is not None:
        explicit_event_id = normalize_hex_string(event_id, byte_length=32, label="event id")
        if explicit_event_id != computed_event_id:
            raise ValueError("explicit event id does not match canonical NIP-01 serialization")
    event = replace(event, event_id=computed_event_id)

    if secret_key_hex is not None:
        signature = schnorr_sign(bytes.fromhex(computed_event_id), secret_key_hex, aux_rand=aux_rand)
        return replace(event, sig=signature)

    if signature_hex is not None:
        signed_event = replace(event, sig=normalize_signature_hex(signature_hex))
        signed_event.validate_signature()
        return signed_event

    return event


def parse_gradient_event(data: NostrainEvent | dict[str, Any]) -> ParsedGradientEvent:
    event = data if isinstance(data, NostrainEvent) else NostrainEvent.from_json_obj(data)
    if event.kind != NOSTRAIN_GRADIENT_KIND:
        raise ValueError(
            f"nostrain gradient events must use kind {NOSTRAIN_GRADIENT_KIND}, got {event.kind}"
        )

    _validate_signing_fields(event)

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


def _validate_signing_fields(event: NostrainEvent) -> None:
    if event.pubkey is None:
        if event.event_id is not None or event.sig is not None:
            raise ValueError("event id/signature cannot be present without a pubkey")
        return

    pubkey = normalize_public_key_hex(event.pubkey)
    computed_event_id = event.compute_event_id()
    if event.event_id is not None:
        normalized_event_id = normalize_hex_string(event.event_id, byte_length=32, label="event id")
        if normalized_event_id != computed_event_id:
            raise ValueError("event id does not match canonical NIP-01 serialization")
    if event.sig is None:
        return
    if event.event_id is None:
        raise ValueError("signed events must include an event id")
    signature = normalize_signature_hex(event.sig)
    if not schnorr_verify(bytes.fromhex(computed_event_id), pubkey, signature):
        raise ValueError("event signature failed BIP340 verification")
