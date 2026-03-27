#!/usr/bin/env python3
"""Worker wrapper with live progress logging for the nostrain demo.

Usage: python demo/worker.py --name alice --color 31 --sec-key HEX ...
"""

import argparse
import asyncio
import json
import sys
import time

from nostrain import (
    HeartbeatEventMetadata,
    GradientEventMetadata,
    LocalTrainingConfig,
    RegressionDataset,
    RelayRetryPolicy,
    build_gradient_event,
    build_heartbeat_event,
    compress_delta,
    compute_delta,
    evaluate_regression,
    nesterov_outer_step,
    publish_nostrain_events,
    collect_gradient_events_across_relays,
    state_digest,
    train_regression,
)
from nostrain.crypto import secret_key_to_public_key
from nostrain.runtime import resolve_training_runtime
from nostrain.stateio import load_model_state


def log(name, color, icon, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\033[90m{ts}\033[0m \033[1;{color}m{icon} [{name}]\033[0m {msg}", flush=True)


async def run_worker(
    name: str,
    color: str,
    sec_key: str,
    state_path: str,
    dataset_path: str,
    relay_url: str,
    run_name: str,
    rounds: int,
    inner_steps: int,
    local_lr: float,
    outer_lr: float,
    momentum: float,
    batch_size: int,
    topk: float,
    round_timeout: float,
    heartbeat_interval: int,
    result_out: str | None = None,
):
    dataset = RegressionDataset.from_path(dataset_path)
    current_state = load_model_state(state_path)
    runtime_name = resolve_training_runtime(None, dataset, state=current_state)
    pubkey = secret_key_to_public_key(sec_key)
    relay_urls = (relay_url,)
    retry_policy = RelayRetryPolicy(max_attempts=3, initial_backoff=0.5, max_backoff=2.0)
    current_momentum = None

    initial_loss = evaluate_regression(current_state, dataset, runtime_name=runtime_name)
    log(name, color, ">>", f"Online | params={current_state.total_values} examples={len(dataset.examples)} loss={initial_loss:.6f}")

    for round_index in range(rounds):
        round_start = int(time.time())
        t0 = time.monotonic()

        # Heartbeat
        heartbeat = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name=run_name,
                worker_id=pubkey,
                current_round=round_index,
                heartbeat_interval=heartbeat_interval,
                capabilities=("gradient-event", runtime_name),
                advertised_relays=relay_urls,
                created_at=round_start,
            ),
            secret_key_hex=sec_key,
        )
        hb_pub = await publish_nostrain_events(
            relay_urls, heartbeat, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry_policy,
        )
        if not hb_pub.accepted:
            log(name, color, "!!", "Heartbeat FAILED")
            return

        # Inner training
        loss_before = evaluate_regression(current_state, dataset, runtime_name=runtime_name)
        local = train_regression(
            current_state, dataset,
            config=LocalTrainingConfig(steps=inner_steps, learning_rate=local_lr, batch_size=batch_size),
            runtime_name=runtime_name,
        )
        log(name, color, "..", f"R{round_index} inner: {loss_before:.6f} -> {local.loss_after:.6f}")

        # Compress + publish
        delta = compute_delta(current_state, local.trained_state)
        payload = compress_delta(delta, topk_ratio=topk)
        grad_event = build_gradient_event(
            GradientEventMetadata(
                run_name=run_name,
                round_index=round_index,
                worker_id=pubkey,
                model_hash=state_digest(current_state),
                inner_steps=inner_steps,
                created_at=max(round_start, int(time.time())),
            ),
            payload,
            secret_key_hex=sec_key,
        )
        gp = await publish_nostrain_events(
            relay_urls, grad_event, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry_policy,
        )
        if not gp.accepted:
            log(name, color, "!!", "Gradient publish FAILED")
            return
        log(name, color, "^^", f"R{round_index} gradient published ({payload.wire_bytes}B)")

        # Collect peers
        collection = await collect_gradient_events_across_relays(
            relay_urls,
            run_name=run_name,
            round_index=round_index,
            idle_timeout=round_timeout,
            open_timeout=10.0,
            since=round_start - 1,
            strategy="timeout",
            discover_workers=True,
            heartbeat_idle_timeout=round_timeout,
            heartbeat_since=round_start - 1,
            max_missed_heartbeats=3,
            retry_policy=retry_policy,
        )
        peers = len(collection.events)
        log(name, color, "vv", f"R{round_index} collected {peers} gradient(s)")

        # Outer step
        outer = nesterov_outer_step(
            current_state,
            collection.aggregate_delta(),
            learning_rate=outer_lr,
            momentum=momentum,
            previous_momentum=current_momentum,
        )
        loss_after = evaluate_regression(outer.next_state, dataset, runtime_name=runtime_name)
        elapsed = time.monotonic() - t0
        current_state = outer.next_state
        current_momentum = outer.momentum_state

        log(name, color, "OK", f"R{round_index} done {elapsed:.1f}s | loss: {loss_before:.6f} -> {loss_after:.6f}")

    # Final
    final_loss = evaluate_regression(current_state, dataset, runtime_name=runtime_name)
    log(name, color, "##", f"DONE | loss: {initial_loss:.6f} -> {final_loss:.6f}")
    for p in current_state.parameters:
        vals = ", ".join(f"{v:.4f}" for v in p.values)
        log(name, color, "  ", f"  {p.name}: [{vals}]")

    # Write result file for summary script
    if result_out:
        pmap = current_state.parameter_map()
        result = {
            "name": name,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "parameters": {p.name: list(p.values) for p in current_state.parameters},
        }
        with open(result_out, "w") as f:
            json.dump(result, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--color", default="37")
    p.add_argument("--state", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--relay", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--sec-key", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--inner-steps", type=int, default=80)
    p.add_argument("--local-lr", type=float, default=0.03)
    p.add_argument("--outer-lr", type=float, default=0.7)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--topk", type=float, default=1.0)
    p.add_argument("--round-timeout", type=float, default=8.0)
    p.add_argument("--heartbeat-interval", type=int, default=10)
    p.add_argument("--result-out", default=None)
    args = p.parse_args()

    asyncio.run(run_worker(
        name=args.name,
        color=args.color,
        sec_key=args.sec_key,
        state_path=args.state,
        dataset_path=args.dataset,
        relay_url=args.relay,
        run_name=args.run,
        rounds=args.rounds,
        inner_steps=args.inner_steps,
        local_lr=args.local_lr,
        outer_lr=args.outer_lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        topk=args.topk,
        round_timeout=args.round_timeout,
        heartbeat_interval=args.heartbeat_interval,
        result_out=args.result_out,
    ))


if __name__ == "__main__":
    main()
