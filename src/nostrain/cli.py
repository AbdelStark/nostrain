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
    CheckpointEventMetadata,
    GradientEventMetadata,
    HeartbeatEventMetadata,
    ParsedCheckpointEvent,
    ParsedGradientEvent,
    build_checkpoint_event,
    build_gradient_event,
    build_heartbeat_event,
    parse_gradient_event,
    parse_nostrain_event,
)
from .relay import (
    collect_checkpoint_events,
    collect_checkpoint_events_across_relays,
    collect_gradient_events,
    collect_gradient_events_across_relays,
    collect_heartbeat_events,
    collect_heartbeat_events_across_relays,
    publish_nostrain_events,
    publish_nostrain_event,
)
from .retry import RelayRetryPolicy
from .runtime import (
    DEFAULT_TRAINING_BACKEND,
    DEFAULT_TRAINING_RUNTIME,
    RegressionDataset,
    SUPPORTED_TRAINING_BACKENDS,
    SUPPORTED_TRAINING_RUNTIMES,
    infer_training_runtime_from_state,
    initialize_training_state,
)
from .stateio import (
    JSON_STATE_FORMAT,
    STATE_FORMAT_CHOICES,
    load_model_state,
    load_model_state_document,
    resolve_state_format,
    write_model_state,
)
from .training import (
    LocalTrainingConfig,
    TrainingCheckpoint,
    TrainingWorkerConfig,
    run_training_session,
    train_regression,
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


def _infer_runtime_name(state: ModelState) -> str | None:
    try:
        return infer_training_runtime_from_state(state)
    except ValueError:
        return None


def _load_state(path: str | Path, state_format: str | None) -> ModelState:
    return load_model_state(path, state_format=state_format)


def _write_state(
    path: str | Path | None,
    state: ModelState,
    *,
    state_format: str | None,
    runtime_name: str | None = None,
) -> None:
    resolved_format = resolve_state_format(state_format, path)
    runtime_name = runtime_name or _infer_runtime_name(state)
    if path is None:
        if resolved_format != JSON_STATE_FORMAT:
            raise ValueError("binary model state output requires -o/--output")
        _write_json(None, state.to_json_obj())
        return
    write_model_state(
        path,
        state,
        state_format=resolved_format,
        runtime_name=runtime_name,
    )


def _resolve_worker_id(args: argparse.Namespace) -> str:
    worker_id = getattr(args, "worker", None)
    if worker_id is None and getattr(args, "sec_key", None):
        worker_id = secret_key_to_public_key(args.sec_key)
    if worker_id is None and getattr(args, "pubkey", None):
        worker_id = args.pubkey.strip().lower()
    if worker_id is None:
        raise ValueError("--worker is required unless --sec-key or --pubkey is provided")
    return worker_id


def _resolve_checkpoint_worker(
    args: argparse.Namespace,
    checkpoint: TrainingCheckpoint,
) -> str:
    if getattr(args, "worker", None) is not None:
        return _resolve_worker_id(args)
    if getattr(args, "sec_key", None) or getattr(args, "pubkey", None):
        return _resolve_worker_id(args)
    return checkpoint.worker_id


def _resolve_relay_urls(relay_args: str | list[str]) -> tuple[str, ...]:
    if isinstance(relay_args, str):
        raw_relays = [relay_args]
    else:
        raw_relays = list(relay_args or [])
    relay_urls = tuple(dict.fromkeys(str(relay).strip() for relay in raw_relays if str(relay).strip()))
    if not relay_urls:
        raise ValueError("at least one --relay is required")
    return relay_urls


def _relay_retry_policy_from_args(args: argparse.Namespace) -> RelayRetryPolicy:
    retries = int(getattr(args, "relay_retries", 0))
    if retries < 0:
        raise ValueError("--relay-retries must be non-negative")
    return RelayRetryPolicy(
        max_attempts=retries + 1,
        initial_backoff=float(getattr(args, "relay_retry_backoff", 0.0)),
        max_backoff=float(getattr(args, "relay_retry_backoff_max", 0.0)),
        backoff_multiplier=float(getattr(args, "relay_retry_backoff_multiplier", 2.0)),
    )


def _add_relay_retry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--relay-retries",
        type=int,
        default=0,
        help="Retry transient relay failures this many times per relay operation (default: 0).",
    )
    parser.add_argument(
        "--relay-retry-backoff",
        type=float,
        default=0.25,
        help="Initial backoff in seconds between relay retries (default: 0.25).",
    )
    parser.add_argument(
        "--relay-retry-backoff-max",
        type=float,
        default=2.0,
        help="Maximum backoff in seconds between relay retries (default: 2.0).",
    )
    parser.add_argument(
        "--relay-retry-backoff-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to each successive relay retry delay (default: 2.0).",
    )


def _add_state_format_argument(
    parser: argparse.ArgumentParser,
    *,
    flag_name: str = "--state-format",
    destination: str = "state_format",
    scope: str = "model state inputs",
) -> None:
    parser.add_argument(
        flag_name,
        dest=destination,
        choices=STATE_FORMAT_CHOICES,
        default="auto",
        help=(
            f"Interpret {scope} using this format. "
            "Defaults to auto-detection from the file extension."
        ),
    )


def _add_output_state_format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-format",
        choices=STATE_FORMAT_CHOICES,
        default="auto",
        help=(
            "Write model-state outputs using this format. "
            "Defaults to stdout JSON or infers from the output path extension."
        ),
    )


def _add_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_TRAINING_BACKENDS,
        default=DEFAULT_TRAINING_BACKEND,
        help=(
            "Training backend used for local evaluation and optimization "
            f"(default: {DEFAULT_TRAINING_BACKEND})."
        ),
    )


def _format_retry_delays(delays: tuple[float, ...]) -> str:
    if not delays:
        return "-"
    return ", ".join(f"{delay:.3f}" for delay in delays)


def _retry_summary_lines(result: Any) -> list[str]:
    if hasattr(result, "relay_url"):
        lines = [
            f"attempt_count: {getattr(result, 'attempt_count', 1)}",
            f"retry_count: {getattr(result, 'retry_count', 0)}",
            f"retry_delays: {_format_retry_delays(tuple(getattr(result, 'retry_delays', ())))}",
        ]
    else:
        retried_relays = tuple(getattr(result, "retried_relays", ()))
        lines = [
            f"total_retry_count: {getattr(result, 'total_retry_count', 0)}",
            f"max_attempt_count: {getattr(result, 'max_attempt_count', 1)}",
            f"retried_relays: {', '.join(retried_relays) if retried_relays else '-'}",
        ]
    heartbeat_collection = getattr(result, "heartbeat_collection", None)
    if heartbeat_collection is not None:
        lines.append(
            "heartbeat_discovery_retry_count: "
            f"{getattr(heartbeat_collection, 'total_retry_count', getattr(heartbeat_collection, 'retry_count', 0))}"
        )
    return lines


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
    state = _load_state(args.state, args.state_format)
    _write_text(args.output, state_digest(state))
    return 0


def _handle_derive_pubkey(args: argparse.Namespace) -> int:
    _write_text(args.output, secret_key_to_public_key(args.sec_key))
    return 0


def _handle_init_state(args: argparse.Namespace) -> int:
    state = initialize_training_state(
        args.runtime,
        feature_count=args.features,
        hidden_size=args.hidden_size,
        seed=args.seed,
        weight_scale=args.weight_scale,
    )
    _write_state(
        args.output,
        state,
        state_format=args.output_format,
        runtime_name=args.runtime,
    )
    return 0


def _handle_encode_delta(args: argparse.Namespace) -> int:
    initial = _load_state(args.initial, args.state_format)
    current = _load_state(args.current, args.state_format)
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
    base = _load_state(args.base, args.state_format)
    payload = _load_payload_string(args.payload)
    delta = decompress_payload(payload)
    reconstructed = apply_delta(base, delta)
    _write_state(
        args.output,
        reconstructed,
        state_format=args.output_format,
    )
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


def _handle_build_checkpoint(args: argparse.Namespace) -> int:
    checkpoint = TrainingCheckpoint.from_path(args.checkpoint)
    if checkpoint.next_round <= 0:
        raise ValueError("checkpoints must have next_round > 0 before they can be advertised")
    publisher = _resolve_checkpoint_worker(args, checkpoint)
    event = build_checkpoint_event(
        CheckpointEventMetadata(
            run_name=checkpoint.run_name,
            worker_id=publisher,
            round_index=checkpoint.next_round - 1,
            next_round=checkpoint.next_round,
            model_hash=state_digest(checkpoint.current_state),
            rounds_completed=checkpoint.rounds_completed,
            history_slot=args.history_slot,
            created_at=args.created_at,
        ),
        checkpoint.to_json_obj(),
        secret_key_hex=args.sec_key,
        public_key_hex=args.pubkey,
        signature_hex=args.sig,
        event_id=args.event_id,
    )
    _write_json(args.output, event.to_json_obj())
    return 0


def _handle_publish_event(args: argparse.Namespace) -> int:
    event = _load_json(args.event)
    relay_urls = _resolve_relay_urls(args.relay)
    retry_policy = _relay_retry_policy_from_args(args)
    if len(relay_urls) == 1:
        result = asyncio.run(
            publish_nostrain_event(
                relay_urls[0],
                event,
                open_timeout=args.timeout,
                reply_timeout=args.timeout,
                retry_policy=retry_policy,
            )
        )
        if not result.accepted:
            raise RuntimeError(f"relay rejected event publication: {result.message}")
        if args.json:
            _write_json(args.output, result.to_json_obj())
        else:
            _write_text(
                args.output,
                f"published {result.event_id} to {result.relay_url} after {result.attempt_count} attempt(s)",
            )
        return 0

    result = asyncio.run(
        publish_nostrain_events(
            relay_urls,
            event,
            open_timeout=args.timeout,
            reply_timeout=args.timeout,
            retry_policy=retry_policy,
        )
    )
    if not result.accepted:
        failures = "; ".join(
            f"{failure.relay_url}: {failure.message}" for failure in result.failed_relays
        )
        raise RuntimeError(f"event publication failed on every relay: {failures}")
    if args.json:
        _write_json(args.output, result.to_json_obj())
    else:
        _write_text(
            args.output,
            f"published {result.event_id} to {len(result.accepted_relays)}/{len(relay_urls)} relays "
            f"with {result.total_retry_count} relay retry(s)",
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
    base = _load_state(args.base, args.state_format)
    aggregated_delta = _load_delta_input(args.delta)
    previous_momentum = (
        _load_state(args.momentum_state, args.state_format) if args.momentum_state else None
    )
    result = nesterov_outer_step(
        base,
        aggregated_delta,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        previous_momentum=previous_momentum,
    )
    _write_state(
        args.output,
        result.next_state,
        state_format=args.output_format,
    )
    if args.momentum_out:
        _write_state(
            args.momentum_out,
            result.momentum_state,
            state_format=args.output_format,
        )
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


def _handle_train_local(args: argparse.Namespace) -> int:
    initial_state = _load_state(args.state, args.state_format)
    dataset = RegressionDataset.from_path(args.dataset)
    result = train_regression(
        initial_state,
        dataset,
        config=LocalTrainingConfig(
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        ),
        runtime_name=args.runtime,
        backend_name=args.backend,
    )
    _write_state(
        args.output,
        result.trained_state,
        state_format=args.output_format,
        runtime_name=result.runtime_name,
    )
    if args.metrics_out:
        _write_json(args.metrics_out, result.to_json_obj())
    return 0


def _handle_convert_state(args: argparse.Namespace) -> int:
    document = load_model_state_document(args.state, state_format=args.input_format)
    _write_state(
        args.output,
        document.state,
        state_format=args.output_format,
        runtime_name=document.runtime_name,
    )
    return 0


def _handle_run_training(args: argparse.Namespace) -> int:
    dataset = RegressionDataset.from_path(args.dataset)
    relay_urls = _resolve_relay_urls(args.relay)
    worker_id = _resolve_worker_id(args)
    runtime_name = args.runtime
    retry_policy = _relay_retry_policy_from_args(args)
    if args.resume_from and args.resume_latest_checkpoint:
        raise ValueError("--resume-from cannot be combined with --resume-latest-checkpoint")
    checkpoint = None
    if args.resume_from:
        checkpoint = TrainingCheckpoint.from_path(args.resume_from)
    elif args.resume_latest_checkpoint:
        checkpoint_collection = asyncio.run(
            collect_checkpoint_events_across_relays(
                relay_urls,
                run_name=args.run,
                worker_id=args.checkpoint_worker,
                idle_timeout=args.checkpoint_idle_timeout or args.round_timeout,
                open_timeout=args.timeout,
                since=args.checkpoint_since,
                retry_policy=retry_policy,
            )
        )
        latest_event = checkpoint_collection.latest_event
        if latest_event is None:
            raise ValueError(f"no relay checkpoint found for run {args.run!r}")
        checkpoint = TrainingCheckpoint.from_json_obj(latest_event.parsed.checkpoint_data)
    if checkpoint is not None:
        if checkpoint.run_name != args.run:
            raise ValueError(
                f"checkpoint run {checkpoint.run_name!r} does not match --run {args.run!r}"
            )
        if args.resume_from and checkpoint.worker_id != worker_id:
            raise ValueError(
                f"checkpoint worker {checkpoint.worker_id!r} does not match worker {worker_id!r}"
            )
        initial_state = checkpoint.current_state
        previous_momentum = checkpoint.momentum_state
        start_round = checkpoint.next_round
        prior_rounds = checkpoint.rounds
        prior_late_gradients = checkpoint.late_gradients
        prior_late_reconciliations = checkpoint.late_reconciliations
        late_gradient_since = checkpoint.late_gradient_since or checkpoint.updated_at
        runtime_name = runtime_name or checkpoint.runtime_name
    else:
        initial_state = _load_state(args.state, args.state_format)
        previous_momentum = (
            _load_state(args.momentum_state, args.state_format) if args.momentum_state else None
        )
        start_round = args.start_round
        prior_rounds = ()
        prior_late_gradients = ()
        prior_late_reconciliations = ()
        late_gradient_since = None
    if (args.resume_from or args.resume_latest_checkpoint) and args.momentum_state:
        raise ValueError("--momentum-state cannot be combined with checkpoint-based resume")
    session = asyncio.run(
        run_training_session(
            initial_state,
            dataset,
            config=TrainingWorkerConfig(
                run_name=args.run,
                relay_urls=relay_urls,
                worker_id=worker_id,
                secret_key_hex=args.sec_key,
                runtime_name=runtime_name,
                backend_name=args.backend,
                rounds=args.rounds,
                start_round=start_round,
                inner_steps=args.inner_steps,
                local_learning_rate=args.local_learning_rate,
                batch_size=args.batch_size,
                topk_ratio=args.topk,
                codec=args.codec,
                outer_learning_rate=args.outer_learning_rate,
                outer_momentum=args.momentum,
                round_timeout=args.round_timeout,
                open_timeout=args.timeout,
                heartbeat_interval=args.heartbeat_interval,
                max_missed_heartbeats=args.max_missed_heartbeats,
                late_gradient_timeout=args.late_gradient_timeout,
                late_gradient_strategy=args.late_gradient_strategy,
                late_reconciliation_learning_rate=args.late_gradient_learning_rate,
                late_reconciliation_momentum=args.late_gradient_momentum,
                advertised_relays=tuple(args.advertise_relay or ()),
                checkpoint_history=args.checkpoint_history,
                artifact_retention_rounds=args.artifact_retention_rounds,
                relay_retry_policy=retry_policy,
            ),
            previous_momentum=previous_momentum,
            artifact_dir=args.artifact_dir,
            prior_rounds=prior_rounds,
            prior_late_gradients=prior_late_gradients,
            prior_late_reconciliations=prior_late_reconciliations,
            late_gradient_since=late_gradient_since,
            checkpoint_out=args.checkpoint_out,
        )
    )
    _write_state(
        args.output,
        session.final_state,
        state_format=args.output_format,
        runtime_name=session.runtime_name,
    )
    if args.momentum_out and session.final_momentum_state is not None:
        _write_state(
            args.momentum_out,
            session.final_momentum_state,
            state_format=args.output_format,
            runtime_name=session.runtime_name,
        )
    if args.summary_out:
        _write_json(args.summary_out, session.to_json_obj())
    return 0


def _collection_summary(collection: Any) -> str:
    relays = getattr(collection, "relay_urls", (collection.relay_url,))
    relay_label = ", ".join(relays)
    workers = ", ".join(collection.worker_ids) if collection.worker_ids else "-"
    known_workers = ", ".join(collection.known_workers) if collection.known_workers else "-"
    return "\n".join(
        [
            f"relays: {relay_label}",
            f"run: {collection.run_name}",
            f"round: {collection.round_index}",
            f"events: {len(collection.events)}",
            f"workers: {workers}",
            f"known_workers: {known_workers}",
            f"duplicates_discarded: {collection.duplicates_discarded}",
            f"invalid_events: {collection.invalid_events}",
            f"sync_strategy: {collection.sync_strategy}",
            f"completion_reason: {collection.completion_reason}",
            *_retry_summary_lines(collection),
        ]
    )


def _heartbeat_collection_summary(collection: Any) -> str:
    relays = getattr(collection, "relay_urls", (collection.relay_url,))
    relay_label = ", ".join(relays)
    workers = ", ".join(collection.worker_ids) if collection.worker_ids else "-"
    return "\n".join(
        [
            f"relays: {relay_label}",
            f"run: {collection.run_name}",
            f"target_round: {collection.target_round if collection.target_round is not None else '-'}",
            f"events: {len(collection.events)}",
            f"workers: {workers}",
            f"duplicates_discarded: {collection.duplicates_discarded}",
            f"invalid_events: {collection.invalid_events}",
            f"stale_events: {collection.stale_events}",
            *_retry_summary_lines(collection),
        ]
    )


def _checkpoint_collection_summary(collection: Any) -> str:
    relays = getattr(collection, "relay_urls", (collection.relay_url,))
    relay_label = ", ".join(relays)
    workers = ", ".join(collection.worker_ids) if collection.worker_ids else "-"
    latest = collection.latest_event
    latest_label = (
        "round "
        f"{latest.parsed.metadata.round_index}"
        + (
            f" (slot {latest.parsed.metadata.history_slot})"
            if latest is not None and latest.parsed.metadata.history_slot is not None
            else ""
        )
        + f" -> next {latest.parsed.metadata.next_round}"
        if latest is not None
        else "-"
    )
    return "\n".join(
        [
            f"relays: {relay_label}",
            f"run: {collection.run_name}",
            f"events: {len(collection.events)}",
            f"workers: {workers}",
            f"latest: {latest_label}",
            f"duplicates_discarded: {collection.duplicates_discarded}",
            f"invalid_events: {collection.invalid_events}",
            *_retry_summary_lines(collection),
        ]
    )


def _handle_discover_workers(args: argparse.Namespace) -> int:
    relay_urls = _resolve_relay_urls(args.relay)
    retry_policy = _relay_retry_policy_from_args(args)
    if len(relay_urls) == 1:
        collection = asyncio.run(
            collect_heartbeat_events(
                relay_urls[0],
                run_name=args.run,
                target_round=args.round,
                idle_timeout=args.idle_timeout,
                open_timeout=args.timeout,
                since=args.since,
                max_missed_heartbeats=args.max_missed_heartbeats,
                retry_policy=retry_policy,
            )
        )
    else:
        collection = asyncio.run(
            collect_heartbeat_events_across_relays(
                relay_urls,
                run_name=args.run,
                target_round=args.round,
                idle_timeout=args.idle_timeout,
                open_timeout=args.timeout,
                since=args.since,
                max_missed_heartbeats=args.max_missed_heartbeats,
                retry_policy=retry_policy,
            )
        )
    if args.json:
        _write_json(args.output, collection.to_json_obj())
    else:
        _write_text(args.output, _heartbeat_collection_summary(collection))
    return 0


def _handle_discover_checkpoints(args: argparse.Namespace) -> int:
    relay_urls = _resolve_relay_urls(args.relay)
    retry_policy = _relay_retry_policy_from_args(args)
    if len(relay_urls) == 1:
        collection = asyncio.run(
            collect_checkpoint_events(
                relay_urls[0],
                run_name=args.run,
                min_round=args.min_round,
                worker_id=args.worker,
                idle_timeout=args.idle_timeout,
                open_timeout=args.timeout,
                since=args.since,
                retry_policy=retry_policy,
            )
        )
    else:
        collection = asyncio.run(
            collect_checkpoint_events_across_relays(
                relay_urls,
                run_name=args.run,
                min_round=args.min_round,
                worker_id=args.worker,
                idle_timeout=args.idle_timeout,
                open_timeout=args.timeout,
                since=args.since,
                retry_policy=retry_policy,
            )
        )
    if args.json:
        _write_json(args.output, collection.to_json_obj())
    else:
        _write_text(args.output, _checkpoint_collection_summary(collection))
    return 0


def _handle_collect_events(args: argparse.Namespace) -> int:
    relay_urls = _resolve_relay_urls(args.relay)
    retry_policy = _relay_retry_policy_from_args(args)
    if len(relay_urls) == 1:
        collection = asyncio.run(
            collect_gradient_events(
                relay_urls[0],
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
                retry_policy=retry_policy,
            )
        )
    else:
        collection = asyncio.run(
            collect_gradient_events_across_relays(
                relay_urls,
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
                retry_policy=retry_policy,
            )
        )
    if args.json:
        _write_json(args.output, collection.to_json_obj())
    else:
        _write_text(args.output, _collection_summary(collection))
    return 0


def _handle_aggregate_round(args: argparse.Namespace) -> int:
    relay_urls = _resolve_relay_urls(args.relay)
    retry_policy = _relay_retry_policy_from_args(args)
    if len(relay_urls) == 1:
        collection = asyncio.run(
            collect_gradient_events(
                relay_urls[0],
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
                retry_policy=retry_policy,
            )
        )
    else:
        collection = asyncio.run(
            collect_gradient_events_across_relays(
                relay_urls,
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
                retry_policy=retry_policy,
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
    if isinstance(parsed, ParsedGradientEvent):
        event_type = "gradient"
    elif isinstance(parsed, ParsedCheckpointEvent):
        event_type = "checkpoint"
    else:
        event_type = "heartbeat"
    summary = {
        "event_id": parsed.event.fingerprint(),
        "type": event_type,
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
    elif isinstance(parsed, ParsedCheckpointEvent):
        summary.update(
            {
                "run": parsed.metadata.run_name,
                "round": parsed.metadata.round_index,
                "history_slot": parsed.metadata.history_slot,
                "next_round": parsed.metadata.next_round,
                "model": parsed.metadata.model_hash,
                "rounds_completed": parsed.metadata.rounds_completed,
                "checkpoint_updated_at": int(parsed.checkpoint_data.get("updated_at", 0)),
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
    elif summary["type"] == "checkpoint":
        lines.extend(
            [
                "history_slot: "
                + (
                    str(summary["history_slot"])
                    if summary["history_slot"] is not None
                    else "-"
                ),
                f"next_round: {summary['next_round']}",
                f"model: {summary['model']}",
                f"rounds_completed: {summary['rounds_completed']}",
                f"checkpoint_updated_at: {summary['checkpoint_updated_at']}",
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
        help="Compute a deterministic SHA-256 digest for a model state file.",
    )
    hash_state.add_argument("state", help="Path to a model state file.")
    _add_state_format_argument(hash_state, scope="the model state input")
    hash_state.add_argument("-o", "--output", help="Optional output path.")
    hash_state.set_defaults(handler=_handle_hash_state)

    derive_pubkey = subparsers.add_parser(
        "derive-pubkey",
        help="Derive a Nostr x-only public key from a 32-byte secp256k1 secret key.",
    )
    derive_pubkey.add_argument("sec_key", help="Lowercase hex-encoded 32-byte secret key.")
    derive_pubkey.add_argument("-o", "--output", help="Optional output path.")
    derive_pubkey.set_defaults(handler=_handle_derive_pubkey)

    init_state = subparsers.add_parser(
        "init-state",
        help="Generate a deterministic initial model state for a supported training runtime.",
    )
    init_state.add_argument(
        "--runtime",
        choices=SUPPORTED_TRAINING_RUNTIMES,
        default=DEFAULT_TRAINING_RUNTIME,
        help="Training runtime to initialize (default: linear-regression).",
    )
    init_state.add_argument(
        "--features",
        type=int,
        required=True,
        help="Number of input features in the initialized model.",
    )
    init_state.add_argument(
        "--hidden-size",
        type=int,
        help="Hidden width for mlp-regression initial states.",
    )
    init_state.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed for non-linear initializers (default: 0).",
    )
    init_state.add_argument(
        "--weight-scale",
        type=float,
        default=0.1,
        help="Base initialization scale for mlp-regression weights (default: 0.1).",
    )
    _add_output_state_format_argument(init_state)
    init_state.add_argument("-o", "--output", help="Optional output path.")
    init_state.set_defaults(handler=_handle_init_state)

    convert_state = subparsers.add_parser(
        "convert-state",
        help="Convert a model state between canonical JSON, numpy-npz, and PyTorch state-dict formats.",
    )
    convert_state.add_argument("state", help="Path to the source model state file.")
    _add_state_format_argument(
        convert_state,
        flag_name="--input-format",
        destination="input_format",
        scope="the source model state",
    )
    _add_output_state_format_argument(convert_state)
    convert_state.add_argument("-o", "--output", help="Optional output path.")
    convert_state.set_defaults(handler=_handle_convert_state)

    encode_delta = subparsers.add_parser(
        "encode-delta",
        help="Compute a pseudo-gradient between two model state snapshots and compress it.",
    )
    encode_delta.add_argument("initial", help="Path to the initial model state file.")
    encode_delta.add_argument("current", help="Path to the current model state file.")
    _add_state_format_argument(encode_delta)
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
    apply_payload.add_argument("base", help="Path to the base model state file.")
    apply_payload.add_argument("payload", help="Path to a raw payload or payload JSON file.")
    _add_state_format_argument(apply_payload, scope="the base model state")
    _add_output_state_format_argument(apply_payload)
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

    build_checkpoint = subparsers.add_parser(
        "build-checkpoint",
        help="Wrap a training checkpoint JSON file in a signed nostrain checkpoint event envelope.",
    )
    build_checkpoint.add_argument(
        "checkpoint",
        help="Path to a training checkpoint JSON file produced by run-training.",
    )
    build_checkpoint.add_argument(
        "--worker",
        help="Publisher identity tag. Defaults to the checkpoint worker or the derived/provided pubkey.",
    )
    build_checkpoint.add_argument(
        "--sec-key",
        help="Lowercase hex-encoded 32-byte secret key used to sign the event locally.",
    )
    build_checkpoint.add_argument(
        "--pubkey",
        help="Lowercase hex-encoded x-only public key for delegated signing workflows.",
    )
    build_checkpoint.add_argument(
        "--sig",
        help="Lowercase hex-encoded Schnorr signature to attach to a delegated-signing event.",
    )
    build_checkpoint.add_argument(
        "--event-id",
        help="Optional explicit event id to verify against the canonical serialized event.",
    )
    build_checkpoint.add_argument(
        "--created-at",
        type=int,
        help="Optional explicit Unix timestamp for the Nostr event.",
    )
    build_checkpoint.add_argument(
        "--history-slot",
        type=int,
        help="Optional bounded checkpoint history slot used for parameterized replacement on relays.",
    )
    build_checkpoint.add_argument("-o", "--output", help="Optional output path.")
    build_checkpoint.set_defaults(handler=_handle_build_checkpoint)

    publish_event = subparsers.add_parser(
        "publish-event",
        help="Publish a nostrain gradient, heartbeat, or checkpoint event JSON file to a websocket relay.",
    )
    publish_event.add_argument("event", help="Path to a nostrain event JSON file.")
    publish_event.add_argument(
        "--relay",
        action="append",
        required=True,
        help="Websocket relay URL. Can be provided multiple times for redundant publication.",
    )
    publish_event.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection and relay reply timeout in seconds (default: 10).",
    )
    _add_relay_retry_arguments(publish_event)
    publish_event.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    publish_event.add_argument("-o", "--output", help="Optional output path.")
    publish_event.set_defaults(handler=_handle_publish_event)

    discover_workers = subparsers.add_parser(
        "discover-workers",
        help="Collect active worker heartbeats for a run and optional target round.",
    )
    discover_workers.add_argument(
        "--relay",
        action="append",
        required=True,
        help="Websocket relay URL. Can be provided multiple times for cross-relay discovery.",
    )
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
    _add_relay_retry_arguments(discover_workers)
    discover_workers.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    discover_workers.add_argument("-o", "--output", help="Optional output path.")
    discover_workers.set_defaults(handler=_handle_discover_workers)

    discover_checkpoints = subparsers.add_parser(
        "discover-checkpoints",
        help="Collect distributed training checkpoints for a run and surface the latest recoverable state.",
    )
    discover_checkpoints.add_argument(
        "--relay",
        action="append",
        required=True,
        help="Websocket relay URL. Can be provided multiple times for cross-relay discovery.",
    )
    discover_checkpoints.add_argument("--run", required=True, help="Training run name.")
    discover_checkpoints.add_argument(
        "--worker",
        help="Optional checkpoint publisher filter.",
    )
    discover_checkpoints.add_argument(
        "--min-round",
        type=int,
        help="Only include checkpoints at or after this completed round.",
    )
    discover_checkpoints.add_argument(
        "--since",
        type=int,
        help="Optional lower bound for checkpoint timestamps.",
    )
    discover_checkpoints.add_argument(
        "--idle-timeout",
        type=float,
        default=2.0,
        help="Stop after this many idle seconds without relay messages (default: 2).",
    )
    discover_checkpoints.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10).",
    )
    _add_relay_retry_arguments(discover_checkpoints)
    discover_checkpoints.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    discover_checkpoints.add_argument("-o", "--output", help="Optional output path.")
    discover_checkpoints.set_defaults(handler=_handle_discover_checkpoints)

    outer_step = subparsers.add_parser(
        "outer-step",
        help="Apply a local DiLoCo-style outer update using an aggregated delta and optional momentum state.",
    )
    outer_step.add_argument("base", help="Path to the base model state file.")
    outer_step.add_argument(
        "delta",
        help="Path to an aggregated delta JSON file, payload JSON file, or raw payload file.",
    )
    _add_state_format_argument(outer_step, scope="the base and momentum model states")
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
        help="Optional prior momentum state file.",
    )
    outer_step.add_argument(
        "--momentum-out",
        help="Optional path to write the next momentum state.",
    )
    outer_step.add_argument(
        "--summary-out",
        help="Optional path to write a small JSON summary of the outer step.",
    )
    _add_output_state_format_argument(outer_step)
    outer_step.add_argument("-o", "--output", help="Optional output path.")
    outer_step.set_defaults(handler=_handle_outer_step)

    train_local = subparsers.add_parser(
        "train-local",
        help="Run a local training inner loop against a runtime-compatible model state and dataset JSON.",
    )
    train_local.add_argument("state", help="Path to the initial model state file.")
    train_local.add_argument("dataset", help="Path to a regression dataset JSON file.")
    _add_state_format_argument(train_local, scope="the initial model state")
    _add_output_state_format_argument(train_local)
    train_local.add_argument(
        "--runtime",
        choices=SUPPORTED_TRAINING_RUNTIMES,
        help="Optional runtime override. Defaults to the dataset task or the model state layout.",
    )
    _add_backend_argument(train_local)
    train_local.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of local SGD steps to run (default: 500).",
    )
    train_local.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Inner learning rate (default: 0.01).",
    )
    train_local.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Mini-batch size for local SGD (default: 1).",
    )
    train_local.add_argument(
        "--metrics-out",
        help="Optional path to write a JSON summary of the local training run.",
    )
    train_local.add_argument("-o", "--output", help="Optional output path.")
    train_local.set_defaults(handler=_handle_train_local)

    run_training = subparsers.add_parser(
        "run-training",
        help="Run local training rounds, publish updates to one or more relays, and apply relay-aggregated outer steps.",
    )
    run_training.add_argument(
        "state",
        help="Path to the initial global model state file. Ignored when resuming from a checkpoint.",
    )
    run_training.add_argument("dataset", help="Path to a regression dataset JSON file.")
    _add_state_format_argument(run_training, scope="the initial and momentum model states")
    _add_output_state_format_argument(run_training)
    run_training.add_argument(
        "--runtime",
        choices=SUPPORTED_TRAINING_RUNTIMES,
        help="Optional runtime override. Defaults to the dataset task or checkpoint/model state layout.",
    )
    _add_backend_argument(run_training)
    run_training.add_argument(
        "--relay",
        action="append",
        required=True,
        help="Websocket relay URL. Can be provided multiple times for redundant publish/collect.",
    )
    run_training.add_argument("--run", required=True, help="Training run name.")
    run_training.add_argument(
        "--worker",
        help="Worker identity tag. Defaults to the derived pubkey from --sec-key.",
    )
    run_training.add_argument(
        "--sec-key",
        required=True,
        help="Lowercase hex-encoded 32-byte secret key used to sign heartbeat and gradient events.",
    )
    run_training.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of outer rounds to execute (default: 1).",
    )
    run_training.add_argument(
        "--start-round",
        type=int,
        default=0,
        help="Round index assigned to the first executed round (default: 0).",
    )
    run_training.add_argument(
        "--inner-steps",
        type=int,
        default=500,
        help="Number of local inner-loop steps per round (default: 500).",
    )
    run_training.add_argument(
        "--local-learning-rate",
        type=float,
        default=0.01,
        help="Local inner-loop learning rate (default: 0.01).",
    )
    run_training.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Mini-batch size for local SGD (default: 1).",
    )
    run_training.add_argument(
        "--topk",
        type=float,
        default=1.0,
        help="Fraction of local delta values to retain in the published payload (default: 1.0).",
    )
    run_training.add_argument(
        "--codec",
        type=CompressionCodec,
        choices=list(CompressionCodec),
        default=CompressionCodec.ZLIB,
        help="Wire codec to use for published payloads.",
    )
    run_training.add_argument(
        "--outer-learning-rate",
        type=float,
        default=0.7,
        help="Outer learning rate applied after relay aggregation (default: 0.7).",
    )
    run_training.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Outer Nesterov momentum coefficient (default: 0.9).",
    )
    run_training.add_argument(
        "--momentum-state",
        help="Optional prior momentum state file.",
    )
    run_training.add_argument(
        "--resume-from",
        help="Optional training checkpoint JSON file written by a prior run-training session.",
    )
    run_training.add_argument(
        "--resume-latest-checkpoint",
        action="store_true",
        help="Resume from the latest distributed relay checkpoint for this run.",
    )
    run_training.add_argument(
        "--checkpoint-worker",
        help="Optional relay checkpoint publisher filter used with --resume-latest-checkpoint.",
    )
    run_training.add_argument(
        "--checkpoint-since",
        type=int,
        help="Optional lower bound for relay checkpoint timestamps when resuming from relays.",
    )
    run_training.add_argument(
        "--checkpoint-idle-timeout",
        type=float,
        help="Optional idle timeout override for relay checkpoint discovery.",
    )
    run_training.add_argument(
        "--momentum-out",
        help="Optional path to write the final momentum state.",
    )
    run_training.add_argument(
        "--checkpoint-out",
        help="Optional path to write a resumable training checkpoint after each completed round.",
    )
    run_training.add_argument(
        "--checkpoint-history",
        type=int,
        default=4,
        help="Number of rolling checkpoint slots to retain per worker on relays and in artifact checkpoints (default: 4).",
    )
    run_training.add_argument(
        "--round-timeout",
        type=float,
        default=2.0,
        help="Stop waiting for peer gradients after this many idle seconds (default: 2).",
    )
    run_training.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Relay connection timeout in seconds (default: 10).",
    )
    _add_relay_retry_arguments(run_training)
    run_training.add_argument(
        "--heartbeat-interval",
        type=int,
        default=60,
        help="Advertised heartbeat interval in seconds (default: 60).",
    )
    run_training.add_argument(
        "--max-missed-heartbeats",
        type=int,
        default=3,
        help="Discard discovered workers missing more than this many heartbeats (default: 3).",
    )
    run_training.add_argument(
        "--late-gradient-timeout",
        type=float,
        default=0.2,
        help="Idle timeout for scanning late gradients from older rounds (default: 0.2).",
    )
    run_training.add_argument(
        "--late-gradient-strategy",
        choices=("deferred", "discard"),
        default="deferred",
        help="How to handle late gradients from completed rounds: fold them into the next round or only record them (default: deferred).",
    )
    run_training.add_argument(
        "--late-gradient-learning-rate",
        type=float,
        help="Optional learning rate override for deferred late-gradient reconciliation. Defaults to --outer-learning-rate.",
    )
    run_training.add_argument(
        "--late-gradient-momentum",
        type=float,
        help="Optional momentum override for deferred late-gradient reconciliation. Defaults to --momentum.",
    )
    run_training.add_argument(
        "--advertise-relay",
        action="append",
        help="Advertise an additional relay URL hint. Can be provided multiple times.",
    )
    run_training.add_argument(
        "--artifact-dir",
        help="Optional directory for per-round artifacts such as events, payloads, and summaries.",
    )
    run_training.add_argument(
        "--artifact-retention-rounds",
        type=int,
        help="Optional limit for how many per-round artifact directories to keep under --artifact-dir.",
    )
    run_training.add_argument(
        "--summary-out",
        help="Optional path to write a JSON session summary.",
    )
    run_training.add_argument("-o", "--output", help="Optional output path.")
    run_training.set_defaults(handler=_handle_run_training)

    collect_events = subparsers.add_parser(
        "collect-events",
        help="Subscribe to relay events for one nostrain run/round, validate them, and deduplicate replays.",
    )
    collect_events.add_argument(
        "--relay",
        action="append",
        required=True,
        help="Websocket relay URL. Can be provided multiple times for cross-relay collection.",
    )
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
    _add_relay_retry_arguments(collect_events)
    collect_events.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    collect_events.add_argument("-o", "--output", help="Optional output path.")
    collect_events.set_defaults(handler=_handle_collect_events)

    aggregate_round = subparsers.add_parser(
        "aggregate-round",
        help="Collect one run/round from a relay and aggregate the embedded worker deltas.",
    )
    aggregate_round.add_argument(
        "--relay",
        action="append",
        required=True,
        help="Websocket relay URL. Can be provided multiple times for cross-relay aggregation.",
    )
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
    _add_relay_retry_arguments(aggregate_round)
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
