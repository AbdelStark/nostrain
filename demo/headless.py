#!/usr/bin/env python3
"""Headless end-to-end nostrain demo — 4 workers, 1 local relay, no tmux.

Runs everything in-process with asyncio tasks. Validates training converges.
"""

import asyncio
import json
import sys
import time
import threading

sys.path.insert(0, "src")

from nostrain import (
    HeartbeatEventMetadata,
    GradientEventMetadata,
    LocalTrainingConfig,
    ModelState,
    RegressionDataset,
    RelayRetryPolicy,
    build_gradient_event,
    build_heartbeat_event,
    collect_gradient_events_across_relays,
    compress_delta,
    compute_delta,
    evaluate_regression,
    nesterov_outer_step,
    publish_nostrain_events,
    state_digest,
    train_regression,
)
from nostrain.crypto import secret_key_to_public_key
from nostrain.stateio import load_model_state

# Import MockRelay from tests
from tests.test_relay import MockRelay


# ── Config ───────────────────────────────────────────────────
ROUNDS = 5
INNER_STEPS = 80
LOCAL_LR = 0.03
OUTER_LR = 0.7
MOMENTUM = 0.9
BATCH_SIZE = 4
TOPK = 1.0
ROUND_TIMEOUT = 6.0
HEARTBEAT_INTERVAL = 10

WORKER_KEYS = [
    "c1719dd06171c7911953655fa6c3db97ffb76779e96037ed870fd320a4ab17bb",
    "0d12877240f2ed60ee104db0d664a9dd95f28e8050e4478f41d840c79fa2edaf",
    "26bcb3fd23513829b5fe387ccab7a8665089e71596d741587c1fd183fdd9abd5",
    "62ad120ffabf1832062aa74dd4bfd9180080fb2aea45e23c994ff5594f5a45c6",
]
WORKER_NAMES = ["alice", "bob", "carol", "dave"]
COLORS = ["31", "32", "33", "34"]


def log(name, color, icon, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\033[90m{ts}\033[0m \033[1;{color}m{icon} [{name}]\033[0m {msg}", flush=True)


async def run_worker(
    name: str,
    color: str,
    sec_key: str,
    state: ModelState,
    dataset: RegressionDataset,
    relay_url: str,
    run_name: str,
    runtime_name: str,
) -> ModelState:
    """Run one worker for ROUNDS rounds. Returns final state."""
    pubkey = secret_key_to_public_key(sec_key)
    relay_urls = (relay_url,)
    retry_policy = RelayRetryPolicy(max_attempts=3, initial_backoff=0.5, max_backoff=2.0)
    current_state = state
    current_momentum = None

    initial_loss = evaluate_regression(current_state, dataset, runtime_name=runtime_name)
    param_count = current_state.total_values
    log(name, color, ">>", f"Online | params={param_count} examples={len(dataset.examples)} loss={initial_loss:.6f}")

    for round_index in range(ROUNDS):
        round_start = int(time.time())
        t0 = time.monotonic()

        # Heartbeat
        heartbeat = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name=run_name,
                worker_id=pubkey,
                current_round=round_index,
                heartbeat_interval=HEARTBEAT_INTERVAL,
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
            log(name, color, "!!", f"Heartbeat publish FAILED")
            break

        # Inner training
        loss_before = evaluate_regression(current_state, dataset, runtime_name=runtime_name)
        local = train_regression(
            current_state, dataset,
            config=LocalTrainingConfig(steps=INNER_STEPS, learning_rate=LOCAL_LR, batch_size=BATCH_SIZE),
            runtime_name=runtime_name,
        )
        log(name, color, "..", f"R{round_index} inner: {loss_before:.6f} -> {local.loss_after:.6f}")

        # Compress + publish gradient
        delta = compute_delta(current_state, local.trained_state)
        payload = compress_delta(delta, topk_ratio=TOPK)
        grad_event = build_gradient_event(
            GradientEventMetadata(
                run_name=run_name,
                round_index=round_index,
                worker_id=pubkey,
                model_hash=state_digest(current_state),
                inner_steps=INNER_STEPS,
                created_at=max(round_start, int(time.time())),
            ),
            payload,
            secret_key_hex=sec_key,
        )
        gp = await publish_nostrain_events(
            relay_urls, grad_event, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry_policy,
        )
        if not gp.accepted:
            log(name, color, "!!", f"Gradient publish FAILED")
            break
        log(name, color, "^^", f"R{round_index} gradient published ({payload.wire_bytes}B)")

        # Collect peer gradients
        collection = await collect_gradient_events_across_relays(
            relay_urls,
            run_name=run_name,
            round_index=round_index,
            idle_timeout=ROUND_TIMEOUT,
            open_timeout=10.0,
            since=round_start - 1,
            strategy="timeout",
            discover_workers=True,
            heartbeat_idle_timeout=ROUND_TIMEOUT,
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
            learning_rate=OUTER_LR,
            momentum=MOMENTUM,
            previous_momentum=current_momentum,
        )
        loss_after = evaluate_regression(outer.next_state, dataset, runtime_name=runtime_name)
        elapsed = time.monotonic() - t0
        current_state = outer.next_state
        current_momentum = outer.momentum_state

        log(name, color, "OK", f"R{round_index} done {elapsed:.1f}s | loss: {loss_before:.6f} -> {loss_after:.6f}")

    final_loss = evaluate_regression(current_state, dataset, runtime_name=runtime_name)
    log(name, color, "##", f"DONE | loss: {initial_loss:.6f} -> {final_loss:.6f}")

    # Print learned params
    for p in current_state.parameters:
        vals = ", ".join(f"{v:.4f}" for v in p.values)
        log(name, color, "  ", f"  {p.name}: [{vals}]")

    return current_state


async def main():
    print("\033[1;36m")
    print("  NOSTRAIN HEADLESS E2E DEMO")
    print("  4 workers, local relay, DiLoCo outer loop")
    print("\033[0m")

    # Generate data shards
    print("  Generating data shards...", flush=True)
    import subprocess
    subprocess.run([sys.executable, "demo/generate_data.py", "demo"], check=True)

    # Init model
    from nostrain.training import initialize_training_state
    state = initialize_training_state(
        runtime_name="linear-regression",
        feature_count=3,
        seed=0,
    )
    print(f"  Model: {state.total_values} params, hash={state_digest(state)[:16]}...")
    print()

    # Load datasets
    datasets = [
        RegressionDataset.from_path(f"demo/shard_{i+1}.json")
        for i in range(4)
    ]

    run_name = f"headless-{int(time.time())}"

    # Start local relay
    with MockRelay() as relay:
        print(f"  Relay: {relay.url}")
        print()

        # Launch all 4 workers as concurrent tasks
        tasks = []
        for i in range(4):
            tasks.append(
                run_worker(
                    name=WORKER_NAMES[i],
                    color=COLORS[i],
                    sec_key=WORKER_KEYS[i],
                    state=state,
                    dataset=datasets[i],
                    relay_url=relay.url,
                    run_name=run_name,
                    runtime_name="linear-regression",
                )
            )
        results = await asyncio.gather(*tasks)

    # ── Validate ─────────────────────────────────────────────
    print()
    print("\033[1;36m  === RESULTS ===\033[0m")
    print()

    # True weights: y = 3.0*x1 - 1.5*x2 + 0.5*x3 + 1.0
    true_weights = [3.0, -1.5, 0.5]
    true_bias = 1.0

    all_converged = True
    for i, final_state in enumerate(results):
        pmap = final_state.parameter_map()
        weight = pmap["linear.weight"]
        bias = pmap["linear.bias"]

        w_err = sum((a - b) ** 2 for a, b in zip(weight.values, true_weights)) ** 0.5
        b_err = abs(bias.values[0] - true_bias)

        status = "\033[1;32mPASS\033[0m" if w_err < 1.0 and b_err < 1.0 else "\033[1;31mFAIL\033[0m"
        if w_err >= 1.0 or b_err >= 1.0:
            all_converged = False

        w_str = ", ".join(f"{v:.4f}" for v in weight.values)
        print(f"  {WORKER_NAMES[i]:>5}: w=[{w_str}] b={bias.values[0]:.4f}  err={w_err:.4f}  {status}")

    print(f"\n  True: w=[{', '.join(f'{w:.1f}' for w in true_weights)}] b={true_bias:.1f}")
    print()

    if all_converged:
        print("\033[1;32m  All workers converged!\033[0m")
    else:
        print("\033[1;33m  Some workers did not fully converge (may need more rounds).\033[0m")

    return 0 if all_converged else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
