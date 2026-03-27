#!/usr/bin/env python3
"""Headless end-to-end nostrain GPT demo.

4 workers train a character-level GPT on different Shakespeare shards,
exchanging compressed pseudo-gradients through a local Nostr relay.

Usage: python demo/gpt/train.py [--rounds 5] [--inner-steps 100]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import torch

sys.path.insert(0, "src")

from nostrain import (
    HeartbeatEventMetadata,
    GradientEventMetadata,
    RelayRetryPolicy,
    build_gradient_event,
    build_heartbeat_event,
    collect_gradient_events_across_relays,
    compress_delta,
    compute_delta,
    nesterov_outer_step,
    publish_nostrain_events,
    state_digest,
)
from nostrain.pytorch import model_state_from_module, load_state_into_module
from nostrain.crypto import secret_key_to_public_key

# Must be importable from demo/gpt/
from demo.gpt.model import CharGPT, GPTConfig
from demo.gpt.data import (
    VOCAB_SIZE,
    ShardDataset,
    decode,
    download_shakespeare,
    encode,
    make_shards,
)

# Reuse the test relay
from tests.test_relay import MockRelay

WORKER_KEYS = [
    "c1719dd06171c7911953655fa6c3db97ffb76779e96037ed870fd320a4ab17bb",
    "0d12877240f2ed60ee104db0d664a9dd95f28e8050e4478f41d840c79fa2edaf",
    "26bcb3fd23513829b5fe387ccab7a8665089e71596d741587c1fd183fdd9abd5",
    "62ad120ffabf1832062aa74dd4bfd9180080fb2aea45e23c994ff5594f5a45c6",
]
WORKER_NAMES = ["alice", "bob", "carol", "dave"]
COLORS = ["31", "32", "33", "34"]

SEED_TEXT = "ROMEO:"


def log(name: str, color: str, icon: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\033[90m{ts}\033[0m \033[1;{color}m{icon} [{name}]\033[0m {msg}", flush=True)


def sample_text(model: CharGPT, prompt: str = SEED_TEXT, max_tokens: int = 120) -> str:
    """Generate a text sample from the model."""
    model.eval()
    tokens = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_tokens, temperature=0.8)
    model.train()
    text = decode(out[0])
    # Clean up: single line, collapse whitespace
    text = " ".join(text.split())
    return text


@torch.no_grad()
def eval_loss(model: CharGPT, dataset: ShardDataset, num_batches: int = 8, batch_size: int = 16) -> float:
    """Estimate loss over several random batches."""
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = dataset.get_batch(batch_size)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / num_batches


def train_inner(
    model: CharGPT,
    dataset: ShardDataset,
    steps: int,
    lr: float,
    batch_size: int,
) -> float:
    """Run local SGD steps. Returns final loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
    for step in range(steps):
        x, y = dataset.get_batch(batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()
    return last_loss


async def run_worker(
    name: str,
    color: str,
    sec_key: str,
    config: GPTConfig,
    initial_state_dict: dict,
    dataset: ShardDataset,
    relay_url: str,
    run_name: str,
    rounds: int,
    inner_steps: int,
    lr: float,
    outer_lr: float,
    momentum: float,
    batch_size: int,
    topk: float,
) -> CharGPT:
    pubkey = secret_key_to_public_key(sec_key)
    relay_urls = (relay_url,)
    retry = RelayRetryPolicy(max_attempts=3, initial_backoff=0.5, max_backoff=2.0)

    model = CharGPT(config)
    model.load_state_dict(initial_state_dict)
    current_state = model_state_from_module(model)
    current_momentum = None

    initial_loss = eval_loss(model, dataset)
    param_count = model.param_count()
    log(name, color, ">>", f"Online | {param_count:,} params | loss={initial_loss:.3f}")

    sample = sample_text(model)
    log(name, color, "  ", f"\033[90m\"{sample[:80]}...\"\033[0m")

    for round_idx in range(rounds):
        t0 = time.monotonic()
        round_start = int(time.time())

        # Heartbeat
        hb = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name=run_name, worker_id=pubkey, current_round=round_idx,
                heartbeat_interval=30, capabilities=("gradient-event",),
                advertised_relays=relay_urls, created_at=round_start,
            ),
            secret_key_hex=sec_key,
        )
        hb_pub = await publish_nostrain_events(relay_urls, hb, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry)
        if not hb_pub.accepted:
            log(name, color, "!!", "Heartbeat FAILED")
            break

        # Inner training
        loss_before = eval_loss(model, dataset)
        load_state_into_module(current_state, model)
        final_loss = train_inner(model, dataset, inner_steps, lr, batch_size)
        trained_state = model_state_from_module(model)

        log(name, color, "..", f"R{round_idx} inner {inner_steps} steps | loss: {loss_before:.3f} -> {final_loss:.3f}")

        # Compress + publish gradient
        delta = compute_delta(current_state, trained_state)
        payload = compress_delta(delta, topk_ratio=topk)
        grad_event = build_gradient_event(
            GradientEventMetadata(
                run_name=run_name, round_index=round_idx, worker_id=pubkey,
                model_hash=state_digest(current_state), inner_steps=inner_steps,
                created_at=max(round_start, int(time.time())),
            ),
            payload, secret_key_hex=sec_key,
        )
        gp = await publish_nostrain_events(relay_urls, grad_event, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry)
        if not gp.accepted:
            log(name, color, "!!", "Gradient FAILED")
            break

        wire_kb = payload.wire_bytes / 1024
        log(name, color, "^^", f"R{round_idx} gradient published ({wire_kb:.1f}KB)")

        # Collect peers
        collection = await collect_gradient_events_across_relays(
            relay_urls, run_name=run_name, round_index=round_idx,
            idle_timeout=12.0, open_timeout=10.0, since=round_start - 1,
            strategy="timeout", discover_workers=True,
            heartbeat_idle_timeout=12.0, heartbeat_since=round_start - 1,
            max_missed_heartbeats=3, retry_policy=retry,
        )
        peers = len(collection.events)
        log(name, color, "vv", f"R{round_idx} collected {peers} gradient(s)")

        # Outer step
        outer = nesterov_outer_step(
            current_state, collection.aggregate_delta(),
            learning_rate=outer_lr, momentum=momentum,
            previous_momentum=current_momentum,
        )
        current_state = outer.next_state
        current_momentum = outer.momentum_state

        # Load updated state into model for next round + text sample
        load_state_into_module(current_state, model)
        new_loss = eval_loss(model, dataset)
        elapsed = time.monotonic() - t0

        log(name, color, "OK", f"R{round_idx} done {elapsed:.1f}s | loss: {loss_before:.3f} -> {new_loss:.3f}")

        # Show text sample every round
        sample = sample_text(model)
        log(name, color, "  ", f"\033[90m\"{sample[:100]}...\"\033[0m")

    # Final
    final_loss = eval_loss(model, dataset)
    log(name, color, "##", f"DONE | loss: {initial_loss:.3f} -> {final_loss:.3f}")
    sample = sample_text(model, max_tokens=200)
    log(name, color, "  ", f"\033[93m\"{sample[:160]}\"\033[0m")

    return model


async def main(rounds: int = 5, inner_steps: int = 100, topk: float = 0.3) -> int:
    print("\033[1;36m")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   NOSTRAIN GPT DEMO — Shakespeare over Nostr        ║")
    print("  ║                                                      ║")
    print("  ║   4 workers · char-level GPT · DiLoCo outer loop     ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print("\033[0m")

    # Data
    text = download_shakespeare()
    shards = make_shards(text, 4)
    datasets = [ShardDataset(s, block_size=128) for s in shards]
    print(f"  Shakespeare: {len(text):,} chars, 4 shards of ~{len(shards[0]):,} chars")

    # Model
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = CharGPT(config)
    param_count = model.param_count()
    print(f"  Model: {param_count:,} params ({param_count/1e6:.1f}M)")
    print(f"  Config: {config.n_layer}L / {config.n_head}H / {config.n_embd}E")
    print(f"  Training: {rounds} rounds x {inner_steps} inner steps, topk={topk}")

    # Save initial state dict for all workers
    initial_sd = {k: v.clone() for k, v in model.state_dict().items()}

    run_name = f"gpt-{int(time.time())}"

    with MockRelay() as relay:
        print(f"  Relay: {relay.url}")
        print()

        tasks = [
            run_worker(
                name=WORKER_NAMES[i], color=COLORS[i], sec_key=WORKER_KEYS[i],
                config=config, initial_state_dict=initial_sd, dataset=datasets[i],
                relay_url=relay.url, run_name=run_name, rounds=rounds,
                inner_steps=inner_steps, lr=3e-4, outer_lr=0.7, momentum=0.9,
                batch_size=32, topk=topk,
            )
            for i in range(4)
        ]
        models = await asyncio.gather(*tasks)

    # Final comparison
    print()
    print("\033[1;36m  === FINAL TEXT SAMPLES ===\033[0m")
    print()
    for i, m in enumerate(models):
        sample = sample_text(m, max_tokens=250)
        print(f"  \033[1;{COLORS[i]}m{WORKER_NAMES[i]}\033[0m:")
        # Word-wrap at ~80 chars
        words = sample.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 80:
                print(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        if line.strip():
            print(line)
        print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=100)
    parser.add_argument("--topk", type=float, default=0.3)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(rounds=args.rounds, inner_steps=args.inner_steps, topk=args.topk)))
