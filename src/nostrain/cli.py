from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .aggregation import aggregate_deltas, nesterov_outer_step
from .compression import CompressionCodec, compress_delta, decompress_payload, inspect_payload
from .model import ModelState, apply_delta, compute_delta, state_digest
from .protocol import GradientEventMetadata, build_gradient_event, parse_gradient_event


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
        worker_id=args.worker,
        model_hash=args.model,
        inner_steps=args.steps,
        created_at=args.created_at,
    )
    event = build_gradient_event(metadata, payload)
    _write_json(args.output, event.to_json_obj())
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


def _event_summary(path: str | Path) -> dict[str, Any]:
    event = _load_json(path)
    parsed = parse_gradient_event(event)
    return {
        "kind": parsed.event.kind,
        "created_at": parsed.event.created_at,
        "run": parsed.metadata.run_name,
        "round": parsed.metadata.round_index,
        "worker": parsed.metadata.worker_id,
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


def _handle_inspect_event(args: argparse.Namespace) -> int:
    summary = _event_summary(args.event)
    if args.json:
        _write_json(args.output, summary)
        return 0

    lines = [
        f"kind: {summary['kind']}",
        f"created_at: {summary['created_at']}",
        f"run: {summary['run']}",
        f"round: {summary['round']}",
        f"worker: {summary['worker']}",
        f"model: {summary['model']}",
        f"steps: {summary['steps']}",
        f"compression: {summary['compression']}",
        f"parameter_count: {summary['parameter_count']}",
        f"selected_values: {summary['selected_values']}/{summary['total_values']}",
        f"density: {summary['density']:.4f}",
        f"wire_bytes: {summary['wire_bytes']}",
        f"compression_ratio: {summary['compression_ratio']:.2f}",
    ]
    _write_text(args.output, "\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nostrain",
        description="Protocol and payload tooling for distributed training over Nostr relays.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_state = subparsers.add_parser(
        "hash-state",
        help="Compute a deterministic SHA-256 digest for a model state JSON file.",
    )
    hash_state.add_argument("state", help="Path to a model state JSON file.")
    hash_state.add_argument("-o", "--output", help="Optional output path.")
    hash_state.set_defaults(handler=_handle_hash_state)

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
        help="Wrap a compressed payload in a nostrain gradient event envelope.",
    )
    build_event.add_argument("payload", help="Path to a raw payload or payload JSON file.")
    build_event.add_argument("--run", required=True, help="Training run name.")
    build_event.add_argument("--round", type=int, required=True, help="Outer round number.")
    build_event.add_argument("--worker", required=True, help="Worker identity or pubkey.")
    build_event.add_argument("--model", required=True, help="Model snapshot hash.")
    build_event.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Inner training steps represented by this payload (default: 500).",
    )
    build_event.add_argument(
        "--created-at",
        type=int,
        help="Optional explicit Unix timestamp for the Nostr event.",
    )
    build_event.add_argument("-o", "--output", help="Optional output path.")
    build_event.set_defaults(handler=_handle_build_event)

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

    inspect_event = subparsers.add_parser(
        "inspect-event",
        help="Validate a nostrain event JSON file and print a summary.",
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
