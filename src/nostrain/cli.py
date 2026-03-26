from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .aggregation import aggregate_deltas, nesterov_outer_step
from .compression import CompressionCodec, compress_delta, decompress_payload, inspect_payload
from .crypto import secret_key_to_public_key
from .model import ModelState, apply_delta, compute_delta, state_digest
from .protocol import (
    GradientEventMetadata,
    HeartbeatEventMetadata,
    ParsedGradientEvent,
    build_gradient_event,
    build_heartbeat_event,
    parse_gradient_event,
    parse_nostrain_event,
)
from .relay import (
    collect_gradient_events,
    collect_heartbeat_events,
    publish_nostrain_event,
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path | None, data: Any) -> None:
    payload = json.dumps(data, indent=2, sort_keys=False) + "\n"
    if path is None:
        print(payload, end="")
        return
    Path(path).write_text(payload, encoding="utf-8")


def _write_text(path: str | Path | None, data: str) -> None:
    if path is None:
        print(data)
        return
    Path(path).write_text(data + "\n", encoding="utf-8")


def _resolve_worker_id(args: argparse.Namespace) -> str:
    worker_id = getattr(args, "worker", None)
    if worker_id is None and getattr(args, "sec_key", None):
        worker_id = secret_key_to_public_key(args.sec_key)
    if worker_id is None and getattr(args, "pubkey", None):
        worker_id = args.pubkey.strip().lower()
    if worker_id is None:
        raise ValueError("--worker is required unless --sec-key or --pubkey is provided")
    return worker_id


def _load_payload_string(path: str | Path) -> str:
    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("payload input file is empty")
    if raw_text.startswith("{"):
        data = json.loads(raw_text)
        if not isinstance(data, dict) or "payload" not in data:
            raise ValueError("payload JSON must contain a top-level 'payload' key")
        return str(data["payload"])
    return raw_text


def _handle_hash_state(args: argparse.Namespace) -> int:
    state = ModelState.from_path(args.state)
    _write_text(args.output, state_digest(state))
    return 0


def _handle_derive_pubkey(args: argparse.Namespace) -> int:
    _write_text(args.output, secret_key_to_public_key(args.sec_key))
    return 0


def _handle_encode_delta(args: argparse.Namespace) -> int:
    initial = ModelState.from_path(args.initial)
    current = ModelState.from_path(args.current)
    delta = compute_delta(initial, current)
    payload = compress_delta(delta, topk_ratio=args.topk, codec=args.codec)
    if args.raw:
        _write_text(args.output, payload.payload)
    else:
        _write_json(args.output, payload.to_json_obj())
    return 0


def _handle_decode_payload(args: argparse.Namespace) -> int:
    payload = _load_payload_string(args.payload)
    delta = decompress_payload(payload)
    _write_json(args.output, delta.to_json_obj())
    return 0


def _handle_apply_payload(args: argparse.Namespace) -> int:
    base = ModelState.from_path(args.base)
    payload = _load_payload_string(args.payload)
    delta = decompress_payload(payload)
    reconstructed = apply_delta(base, delta)
    _write_json(args.output, reconstructed.to_json_obj())
    return 0


def _handle_build_event(args: argparse.Namespace) -> int:
    payload = _load_payload_string(args.payload)
    metadata = GradientEventMetadata(
        run_name=args.run,
        round_index=args.round,
        worker_id=_resolve_worker_id(args),
        model_hash=args.model,
        inner_steps=args.steps,
        created_at=args.created_at,
    )
    event = build_gradient_event(
        metadata,
        payload,
        secret_key_hex=args.sec_key,
        public_key_hex=args.pubkey,
        signature_hex=args.sig,
        event_id=args.event_id,
    )
    _write_json(args.output, event.to_json_obj())
    return 0


def _handle_build_heartbeat(args: argparse.Namespace) -> int:
    metadata = HeartbeatEventMetadata(
        run_name=args.run,
        worker_id=_resolve_worker_id(args),
        current_round=args.round,
        heartbeat_interval=args.heartbeat_interval,
        capabilities=tuple(args.capability or ("gradient-event",)),
        advertised_relays=tuple(args.advertise_relay or ()),
        created_at=args.created_at,
    )
    event = build_heartbeat_event(
        metadata,
        secret_key_hex=args.sec_key,
        public_key_hex=args.pubkey,
        signature_hex=args.sig,
        event_id=args.event_id,
    )
    _write_json(args.output, event.to_json_obj())
    return 0


def _handle_publish_event(args: argparse.Namespace) -> int:
    event = _load_json(args.event)
    result = asyncio.run(
        publish_nostrain_event(
            args.relay,
            event,
            open_timeout=args.timeout,
            reply_timeout=args.timeout,
        )
    )
    if not result.accepted:
        raise RuntimeError(f"relay rejected event publication: {result.message}")
    if args.json:
        _write_json(args.output, result.to_json_obj())
    else:
        _write_text(
            args.output,
            f"published {result.event_id} to {result.relay_url}",
        )
    return 0


def _load_delta_input(path: str | Path) -> ModelState:
    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("delta input file is empty")
    if raw_text.startswith("{"):
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError("delta input JSON must be an object")
        if "parameters" in data:
            return ModelState.from_json_obj(data)
        if {"kind", "created_at", "tags", "content"} <= data.keys():
            return decompress_payload(parse_gradient_event(data).payload)
        if "payload" in data:
            return decompress_payload(str(data["payload"]))
    return decompress_payload(raw_text)


def _handle_aggregate_payloads(args: argparse.Namespace) -> int:
    deltas = [_load_delta_input(path) for path in args.inputs]
    aggregated = aggregate_deltas(deltas)
    _write_json(args.output, aggregated.to_json_obj())
    return 0


def _handle_outer_step(args: argparse.Namespace) -> int:
    base = ModelState.from_path(args.base)
    aggregated_delta = _load_delta_input(args.delta)
    previous_momentum = (
        ModelState.from_path(args.momentum_state) if args.momentum_state else None
    )
    result = nesterov_outer_step(
        base,
        aggregated_delta,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        previous_momentum=previous_momentum,
    )
    _write_json(args.output, result.next_state.to_json_obj())
    if args.momentum_out:
        _write_json(args.momentum_out, result.momentum_state.to_json_obj())
    if args.summary_out:
        _write_json(
            args.summary_out,
            {
                "learning_rate": args.learning_rate,
                "momentum": args.momentum,
                "parameter_count": result.next_state.parameter_count,
                "aggregated_values": result.aggregated_delta.total_values,
            },
        )
    return 0


def _collection_summary(collection: Any) -> str:
    workers = ", ".join(collection.worker_ids) if collection.worker_ids else "-"
    known_workers = ", ".join(collection.known_workers) if collection.known_workers else "-"
    return "\n".join(
        [
            f"relay: {collection.relay_url}",
            f"run: {collection.run_name}",
            f"round: {collection.round_index}",
            f"events: {len(collection.events)}",
            f"workers: {workers}",
            f"known_workers: {known_workers}",
            f"duplicates_discarded: {collection.duplicates_discarded}",
            f"invalid_events: {collection.invalid_events}",
            f"sync_strategy: {collection.sync_strategy}",
            f"completion_reason: {collection.completion_reason}",
        ]
    )


def _heartbeat_collection_summary(collection: Any) -> str:
    workers = ", ".join(collection.worker_ids) if collection.worker_ids else "-"
    return "\n".join(
        [
            f"relay: {collection.relay_url}",
            f"run: {collection.run_name}",
            f"target_round: {collection.target_round if collection.target_round is not None else '-'}",
            f"events: {len(collection.events)}",
            f"workers: {workers}",
            f"duplicates_discarded: {collection.duplicates_discarded}",
            f"invalid_events: {collection.invalid_events}",
            f"stale_events: {collection.stale_events}",
        ]
    )


def _handle_discover_workers(args: argparse.Namespace) -> int:
    collection = asyncio.run(
        collect_heartbeat_events(
            args.relay,
            run_name=args.run,
            target_round=args.round,
            idle_timeout=args.idle_timeout,
            open_timeout=args.timeout,
            since=args.since,
            max_missed_heartbeats=args.max_missed_heartbeats,
        )
    )
    if args.json:
        _write_json(args.output, collection.to_json_obj())
    else:
        _write_text(args.output, _heartbeat_collection_summary(collection))
    return 0


def _handle_collect_events(args: argparse.Namespace) -> int:
    collection = asyncio.run(
        collect_gradient_events(
            args.relay,
            run_name=args.run,
            round_index=args.round,
            max_events=args.limit,
            idle_timeout=args.idle_timeout,
            open_timeout=args.timeout,
            since=args.since,
            strategy=args.sync,
            discover_workers=args.discover_workers,
            heartbeat_idle_timeout=args.heartbeat_idle_timeout,
            heartbeat_since=args.heartbeat_since,
            max_missed_heartbeats=args.max_missed_heartbeats,
        )
    )
    if args.json:
        _write_json(args.output, collection.to_json_obj())
    else:
        _write_text(args.output, _collection_summary(collection))
    return 0


def _handle_aggregate_round(args: argparse.Namespace) -> int:
    collection = asyncio.run(
        collect_gradient_events(
            args.relay,
            run_name=args.run,
            round_index=args.round,
            max_events=args.limit,
            idle_timeout=args.idle_timeout,
            open_timeout=args.timeout,
            since=args.since,
            strategy=args.sync,
            discover_workers=args.discover_workers,
            heartbeat_idle_timeout=args.heartbeat_idle_timeout,
            heartbeat_since=args.heartbeat_since,
            max_missed_heartbeats=args.max_missed_heartbeats,
        )
    )
    aggregated = collection.aggregate_delta()
    _write_json(args.output, aggregated.to_json_obj())
    if args.summary_out:
        _write_json(args.summary_out, collection.to_json_obj(include_events=False))
    return 0


def _event_summary(path: str | Path) -> dict[str, Any]:
    event = _load_json(path)
    parsed = parse_nostrain_event(event)
    summary = {
        "event_id": parsed.event.fingerprint(),
        "type": "gradient" if isinstance(parsed, ParsedGradientEvent) else "heartbeat",
        "kind": parsed.event.kind,
        "created_at": parsed.event.created_at,
        "signed": parsed.event.is_signed,
        "signing_state": parsed.event.signing_state,
        "pubkey": parsed.event.pubkey,
        "worker": parsed.metadata.worker_id,
    }
    if isinstance(parsed, ParsedGradientEvent):
        summary.update(
            {
                "run": parsed.metadata.run_name,
                "round": parsed.metadata.round_index,
                "model": parsed.metadata.model_hash,
                "steps": parsed.metadata.inner_steps,
                "compression": parsed.payload.compression_label,
                "parameter_count": parsed.payload.parameter_count,
                "total_values": parsed.payload.total_values,
                "selected_values": parsed.payload.selected_values,
                "density": parsed.payload.density,
                "wire_bytes": parsed.payload.wire_bytes,
                "compression_ratio": parsed.payload.compression_ratio,
            }
        )
    else:
        summary.update(
            {
                "run": parsed.metadata.run_name,
                "round": parsed.metadata.current_round,
                "heartbeat_interval": parsed.metadata.heartbeat_interval,
                "capabilities": list(parsed.metadata.capabilities),
                "advertised_relays": list(parsed.metadata.advertised_relays),
            }
        )
    return summary


def _handle_inspect_event(args: argparse.Namespace) -> int:
    summary = _event_summary(args.event)
    if args.json:
        _write_json(args.output, summary)
        return 0

    lines = [
        f"type: {summary['type']}",
        f"event_id: {summary['event_id']}",
        f"kind: {summary['kind']}",
        f"created_at: {summary['created_at']}",
        f"signed: {summary['signed']}",
        f"signing_state: {summary['signing_state']}",
        f"pubkey: {summary['pubkey'] or '-'}",
        f"run: {summary['run']}",
        f"round: {summary['round']}",
        f"worker: {summary['worker']}",
    ]
    if summary["type"] == "gradient":
        lines.extend(
            [
                f"model: {summary['model']}",
                f"steps: {summary['steps']}",
                f"compression: {summary['compression']}",
                f"parameter_count: {summary['parameter_count']}",
                f"selected_values: {summary['selected_values']}/{summary['total_values']}",
                f"density: {summary['density']:.4f}",
                f"wire_bytes: {summary['wire_bytes']}",
                f"compression_ratio: {summary['compression_ratio']:.2f}",
            ]
        )
    else:
        lines.extend(
            [
                f"heartbeat_interval: {summary['heartbeat_interval']}",
                "capabilities: "
                + (", ".join(summary["capabilities"]) if summary["capabilities"] else "-"),
                "advertised_relays: "
                + (
                    ", ".join(summary["advertised_relays"])
                    if summary["advertised_relays"]
                    else "-"
                ),
            ]
        )
    _write_text(args.output, "\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nostrain",
        description="Protocol, payload, and relay tooling for distributed training over Nostr relays.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_state = subparsers.add_parser(
        "hash-state",
        help="Compute a deterministic SHA-256 digest for a model state JSON file.",
    )
    hash_state.add_argument("state", help="Path to a model state JSON file.")
    hash_state.add_argument("-o", "--output", help="Optional output path.")
    hash_state.set_defaults(handler=_handle_hash_state)

    derive_pubkey = subparsers.add_parser(
        "derive-pubkey",
        help="Derive a Nostr x-only public key from a 32-byte secp256k1 secret key.",
    )
    derive_pubkey.add_argument("sec_key", help="Lowercase hex-encoded 32-byte secret key.")
    derive_pubkey.add_argument("-o", "--output", help="Optional output path.")
    derive_pubkey.set_defaults(handler=_handle_derive_pubkey)

    encode_delta = subparsers.add_parser(
        "encode-delta",
        help="Compute a pseudo-gradient between two model state snapshots and compress it.",
    )
    encode_delta.add_argument("initial", help="Path to the initial model state JSON.")
    encode_delta.add_argument("current", help="Path to the current model state JSON.")
    encode_delta.add_argument(
        "--topk",
        type=float,
        default=0.01,
        help="Fraction of values to retain by absolute magnitude (default: 0.01).",
    )
    encode_delta.add_argument(
        "--codec",
        type=CompressionCodec,
        choices=list(CompressionCodec),
        default=CompressionCodec.ZLIB,
        help="Wire codec to use for the payload container.",
    )
    encode_delta.add_argument("--raw", action="store_true", help="Emit raw base64 instead of JSON.")
    encode_delta.add_argument("-o", "--output", help="Optional output path.")
    encode_delta.set_defaults(handler=_handle_encode_delta)

    decode_payload = subparsers.add_parser(
        "decode-payload",
        help="Decode a compressed payload back into a pseudo-gradient model state JSON.",
    )
    decode_payload.add_argument("payload", help="Path to a raw payload or payload JSON file.")
    decode_payload.add_argument("-o", "--output", help="Optional output path.")
    decode_payload.set_defaults(handler=_handle_decode_payload)

    aggregate_payloads = subparsers.add_parser(
        "aggregate-payloads",
        help="Average one or more payloads or decoded delta JSON files into a single delta.",
    )
    aggregate_payloads.add_argument(
        "inputs",
        nargs="+",
        help="Paths to raw payloads, payload JSON files, or decoded delta JSON files.",
    )
    aggregate_payloads.add_argument("-o", "--output", help="Optional output path.")
    aggregate_payloads.set_defaults(handler=_handle_aggregate_payloads)

    apply_payload = subparsers.add_parser(
        "apply-payload",
        help="Apply a pseudo-gradient payload to a base model state to reconstruct the next state.",
    )
    apply_payload.add_argument("base", help="Path to the base model state JSON.")
    apply_payload.add_argument("payload", help="Path to a raw payload or payload JSON file.")
    apply_payload.add_argument("-o", "--output", help="Optional output path.")
    apply_payload.set_defaults(handler=_handle_apply_payload)

    build_event = subparsers.add_parser(
        "build-event",
        help="Wrap a compressed payload in a nostrain gradient event envelope, optionally signed for NIP-01 relays.",
    )
    build_event.add_argument("payload", help="Path to a raw payload or payload JSON file.")
    build_event.add_argument("--run", required=True, help="Training run name.")
    build_event.add_argument("--round", type=int, required=True, help="Outer round number.")
    build_event.add_argument(
        "--worker",
        help="Worker identity tag. Defaults to the derived/provided pubkey when signing inputs are supplied.",
    )
    build_event.add_argument("--model", required=True, help="Model snapshot hash.")
    build_event.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Inner training steps represented by this payload (default: 500).",
    )
    build_event.add_argument(
        "--sec-key",
        help="Lowercase hex-encoded 32-byte secret key used to sign the event locally.",
    )
    build_event.add_argument(
        "--pubkey",
        help="Lowercase hex-encoded x-only public key for delegated signing workflows.",
    )
    build_event.add_argument(
        "--sig",
        help="Lowercase hex-encoded Schnorr signature to attach to a delegated-signing event.",
    )
    build_event.add_argument(
        "--event-id",
        help="Optional explicit event id to verify against the canonical serialized event.",
    )
    build_event.add_argument(
        "--created-at",
        type=int,
        help="Optional explicit Unix timestamp for the Nostr event.",
    )
    build_event.add_argument("-o", "--output", help="Optional output path.")
    build_event.set_defaults(handler=_handle_build_event)

    build_heartbeat = subparsers.add_parser(
        "build-heartbeat",
        help="Build a signed nostrain worker heartbeat event for relay-based discovery.",
    )
    build_heartbeat.add_argument("--run", required=True, help="Training run name.")
    build_heartbeat.add_argument("--round", type=int, required=True, help="Current outer round number.")
    build_heartbeat.add_argument(
        "--worker",
        help="Worker identity tag. Defaults to the derived/provided pubkey when signing inputs are supplied.",
    )
    build_heartbeat.add_argument(
        "--heartbeat-interval",
        type=int,
        default=60,
        help="Expected heartbeat cadence in seconds (default: 60).",
    )
    build_heartbeat.add_argument(
        "--capability",
        action="append",
        help="Advertise a worker capability tag. Can be provided multiple times.",
    )
    build_heartbeat.add_argument(
        "--advertise-relay",
        action="append",
        help="Advertise a relay URL hint. Can be provided multiple times.",
    )
    build_heartbeat.add_argument(
        "--sec-key",
        help="Lowercase hex-encoded 32-byte secret key used to sign the event locally.",
    )
    build_heartbeat.add_argument(
        "--pubkey",
        help="Lowercase hex-encoded x-only public key for delegated signing workflows.",
    )
    build_heartbeat.add_argument(
        "--sig",
        help="Lowercase hex-encoded Schnorr signature to attach to a delegated-signing event.",
    )
    build_heartbeat.add_argument(
        "--event-id",
        help="Optional explicit event id to verify against the canonical serialized event.",
    )
    build_heartbeat.add_argument(
        "--created-at",
        type=int,
        help="Optional explicit Unix timestamp for the Nostr event.",
    )
    build_heartbeat.add_argument("-o", "--output", help="Optional output path.")
    build_heartbeat.set_defaults(handler=_handle_build_heartbeat)

    publish_event = subparsers.add_parser(
        "publish-event",
        help="Publish a nostrain gradient or heartbeat event JSON file to a websocket relay.",
    )
    publish_event.add_argument("event", help="Path to a nostrain event JSON file.")
    publish_event.add_argument("--relay", required=True, help="Websocket relay URL.")
    publish_event.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection and relay reply timeout in seconds (default: 10).",
    )
    publish_event.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    publish_event.add_argument("-o", "--output", help="Optional output path.")
    publish_event.set_defaults(handler=_handle_publish_event)

    discover_workers = subparsers.add_parser(
        "discover-workers",
        help="Collect active worker heartbeats for a run and optional target round.",
    )
    discover_workers.add_argument("--relay", required=True, help="Websocket relay URL.")
    discover_workers.add_argument("--run", required=True, help="Training run name.")
    discover_workers.add_argument(
        "--round",
        type=int,
        help="Only include workers advertising this round or later.",
    )
    discover_workers.add_argument(
        "--since",
        type=int,
        help="Optional lower bound for heartbeat timestamps.",
    )
    discover_workers.add_argument(
        "--max-missed-heartbeats",
        type=int,
        default=3,
        help="Discard workers missing more than this many heartbeats (default: 3).",
    )
    discover_workers.add_argument(
        "--idle-timeout",
        type=float,
        default=2.0,
        help="Stop after this many idle seconds without relay messages (default: 2).",
    )
    discover_workers.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10).",
    )
    discover_workers.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    discover_workers.add_argument("-o", "--output", help="Optional output path.")
    discover_workers.set_defaults(handler=_handle_discover_workers)

    outer_step = subparsers.add_parser(
        "outer-step",
        help="Apply a local DiLoCo-style outer update using an aggregated delta and optional momentum state.",
    )
    outer_step.add_argument("base", help="Path to the base model state JSON.")
    outer_step.add_argument(
        "delta",
        help="Path to an aggregated delta JSON file, payload JSON file, or raw payload file.",
    )
    outer_step.add_argument(
        "--learning-rate",
        type=float,
        default=0.7,
        help="Outer learning rate (default: 0.7).",
    )
    outer_step.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Nesterov momentum coefficient (default: 0.9).",
    )
    outer_step.add_argument(
        "--momentum-state",
        help="Optional prior momentum state JSON file.",
    )
    outer_step.add_argument(
        "--momentum-out",
        help="Optional path to write the next momentum state JSON.",
    )
    outer_step.add_argument(
        "--summary-out",
        help="Optional path to write a small JSON summary of the outer step.",
    )
    outer_step.add_argument("-o", "--output", help="Optional output path.")
    outer_step.set_defaults(handler=_handle_outer_step)

    collect_events = subparsers.add_parser(
        "collect-events",
        help="Subscribe to relay events for one nostrain run/round, validate them, and deduplicate replays.",
    )
    collect_events.add_argument("--relay", required=True, help="Websocket relay URL.")
    collect_events.add_argument("--run", required=True, help="Training run name.")
    collect_events.add_argument("--round", type=int, required=True, help="Outer round number.")
    collect_events.add_argument(
        "--limit",
        type=int,
        help="Stop after collecting this many unique worker events.",
    )
    collect_events.add_argument(
        "--since",
        type=int,
        help="Optional lower bound for event timestamps.",
    )
    collect_events.add_argument(
        "--sync",
        choices=("timeout", "strict", "quorum", "async"),
        default="timeout",
        help="Round stopping strategy (default: timeout).",
    )
    collect_events.add_argument(
        "--discover-workers",
        action="store_true",
        help="Collect active heartbeat snapshots before subscribing for gradient events.",
    )
    collect_events.add_argument(
        "--heartbeat-since",
        type=int,
        help="Optional lower bound for heartbeat timestamps used during discovery.",
    )
    collect_events.add_argument(
        "--heartbeat-idle-timeout",
        type=float,
        help="Optional idle timeout override for heartbeat discovery.",
    )
    collect_events.add_argument(
        "--max-missed-heartbeats",
        type=int,
        default=3,
        help="Discard workers missing more than this many heartbeats (default: 3).",
    )
    collect_events.add_argument(
        "--idle-timeout",
        type=float,
        default=2.0,
        help="Stop after this many idle seconds without new relay messages (default: 2).",
    )
    collect_events.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10).",
    )
    collect_events.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    collect_events.add_argument("-o", "--output", help="Optional output path.")
    collect_events.set_defaults(handler=_handle_collect_events)

    aggregate_round = subparsers.add_parser(
        "aggregate-round",
        help="Collect one run/round from a relay and aggregate the embedded worker deltas.",
    )
    aggregate_round.add_argument("--relay", required=True, help="Websocket relay URL.")
    aggregate_round.add_argument("--run", required=True, help="Training run name.")
    aggregate_round.add_argument("--round", type=int, required=True, help="Outer round number.")
    aggregate_round.add_argument(
        "--limit",
        type=int,
        help="Stop after collecting this many unique worker events.",
    )
    aggregate_round.add_argument(
        "--since",
        type=int,
        help="Optional lower bound for event timestamps.",
    )
    aggregate_round.add_argument(
        "--sync",
        choices=("timeout", "strict", "quorum", "async"),
        default="timeout",
        help="Round stopping strategy (default: timeout).",
    )
    aggregate_round.add_argument(
        "--discover-workers",
        action="store_true",
        help="Collect active heartbeat snapshots before subscribing for gradient events.",
    )
    aggregate_round.add_argument(
        "--heartbeat-since",
        type=int,
        help="Optional lower bound for heartbeat timestamps used during discovery.",
    )
    aggregate_round.add_argument(
        "--heartbeat-idle-timeout",
        type=float,
        help="Optional idle timeout override for heartbeat discovery.",
    )
    aggregate_round.add_argument(
        "--max-missed-heartbeats",
        type=int,
        default=3,
        help="Discard workers missing more than this many heartbeats (default: 3).",
    )
    aggregate_round.add_argument(
        "--idle-timeout",
        type=float,
        default=2.0,
        help="Stop after this many idle seconds without new relay messages (default: 2).",
    )
    aggregate_round.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10).",
    )
    aggregate_round.add_argument(
        "--summary-out",
        help="Optional path to write collection summary JSON.",
    )
    aggregate_round.add_argument("-o", "--output", help="Optional output path.")
    aggregate_round.set_defaults(handler=_handle_aggregate_round)

    inspect_event = subparsers.add_parser(
        "inspect-event",
        help="Validate a nostrain gradient or heartbeat event JSON file and print a summary.",
    )
    inspect_event.add_argument("event", help="Path to a nostrain event JSON file.")
    inspect_event.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    inspect_event.add_argument("-o", "--output", help="Optional output path.")
    inspect_event.set_defaults(handler=_handle_inspect_event)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
