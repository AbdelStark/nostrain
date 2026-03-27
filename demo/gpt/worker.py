#!/usr/bin/env python3
"""GPT worker for the tmux demo. Runs one worker with live progress + text samples.

Usage: python demo/gpt/worker.py --name alice --color 31 --sec-key HEX --shard-id 0 ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import torch

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

from demo.gpt.model import CharGPT, GPTConfig
from demo.gpt.data import VOCAB_SIZE, ShardDataset, decode, download_shakespeare, encode, make_shards


SEED_TEXT = "ROMEO:"


def log(name: str, color: str, icon: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\033[90m{ts}\033[0m \033[1;{color}m{icon} [{name}]\033[0m {msg}", flush=True)


def sample_text(model: CharGPT, prompt: str = SEED_TEXT, max_tokens: int = 120) -> str:
    model.eval()
    tokens = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_tokens, temperature=0.8)
    model.train()
    return " ".join(decode(out[0]).split())


@torch.no_grad()
def eval_loss(model: CharGPT, dataset: ShardDataset, num_batches: int = 8, batch_size: int = 16) -> float:
    model.eval()
    total = sum(model(x, y)[1].item() for x, y in (dataset.get_batch(batch_size) for _ in range(num_batches)))
    model.train()
    return total / num_batches


def train_inner(model: CharGPT, dataset: ShardDataset, steps: int, lr: float, batch_size: int) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
    for _ in range(steps):
        x, y = dataset.get_batch(batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()
    return last_loss


async def run(args: argparse.Namespace) -> None:
    text = download_shakespeare()
    shards = make_shards(text, 4)
    dataset = ShardDataset(shards[args.shard_id], block_size=128)

    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = CharGPT(config)
    # All workers start from same seed
    torch.manual_seed(42)
    model = CharGPT(config)

    pubkey = secret_key_to_public_key(args.sec_key)
    relay_urls = (args.relay,)
    retry = RelayRetryPolicy(max_attempts=3, initial_backoff=0.5, max_backoff=2.0)

    current_state = model_state_from_module(model)
    current_momentum = None

    initial_loss = eval_loss(model, dataset)
    log(args.name, args.color, ">>", f"Online | {model.param_count():,} params | shard {args.shard_id}")
    log(args.name, args.color, ">>", f"Initial loss: {initial_loss:.3f}")
    sample = sample_text(model)
    log(args.name, args.color, "  ", f"\033[90m\"{sample[:80]}...\"\033[0m")
    print(flush=True)

    for round_idx in range(args.rounds):
        t0 = time.monotonic()
        round_start = int(time.time())

        hb = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name=args.run, worker_id=pubkey, current_round=round_idx,
                heartbeat_interval=30, capabilities=("gradient-event",),
                advertised_relays=relay_urls, created_at=round_start,
            ),
            secret_key_hex=args.sec_key,
        )
        hb_pub = await publish_nostrain_events(relay_urls, hb, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry)
        if not hb_pub.accepted:
            log(args.name, args.color, "!!", "Heartbeat FAILED")
            return

        loss_before = eval_loss(model, dataset)
        load_state_into_module(current_state, model)
        final_loss = train_inner(model, dataset, args.inner_steps, args.lr, args.batch_size)
        trained_state = model_state_from_module(model)
        log(args.name, args.color, "..", f"R{round_idx} inner | loss: {loss_before:.3f} -> {final_loss:.3f}")

        delta = compute_delta(current_state, trained_state)
        payload = compress_delta(delta, topk_ratio=args.topk)
        grad_event = build_gradient_event(
            GradientEventMetadata(
                run_name=args.run, round_index=round_idx, worker_id=pubkey,
                model_hash=state_digest(current_state), inner_steps=args.inner_steps,
                created_at=max(round_start, int(time.time())),
            ),
            payload, secret_key_hex=args.sec_key,
        )
        gp = await publish_nostrain_events(relay_urls, grad_event, open_timeout=10.0, reply_timeout=10.0, retry_policy=retry)
        if not gp.accepted:
            log(args.name, args.color, "!!", "Gradient FAILED")
            return
        log(args.name, args.color, "^^", f"R{round_idx} published ({payload.wire_bytes/1024:.1f}KB)")

        collection = await collect_gradient_events_across_relays(
            relay_urls, run_name=args.run, round_index=round_idx,
            idle_timeout=15.0, open_timeout=10.0, since=round_start - 1,
            strategy="timeout", discover_workers=True,
            heartbeat_idle_timeout=15.0, heartbeat_since=round_start - 1,
            max_missed_heartbeats=3, retry_policy=retry,
        )
        log(args.name, args.color, "vv", f"R{round_idx} collected {len(collection.events)} gradient(s)")

        outer = nesterov_outer_step(
            current_state, collection.aggregate_delta(),
            learning_rate=args.outer_lr, momentum=args.momentum,
            previous_momentum=current_momentum,
        )
        current_state = outer.next_state
        current_momentum = outer.momentum_state
        load_state_into_module(current_state, model)

        new_loss = eval_loss(model, dataset)
        elapsed = time.monotonic() - t0
        log(args.name, args.color, "OK", f"R{round_idx} done {elapsed:.1f}s | loss: {loss_before:.3f} -> {new_loss:.3f}")

        sample = sample_text(model)
        log(args.name, args.color, "  ", f"\033[93m\"{sample[:100]}\"\033[0m")
        print(flush=True)

    final_loss = eval_loss(model, dataset)
    print(flush=True)
    log(args.name, args.color, "##", f"TRAINING COMPLETE | loss: {initial_loss:.3f} -> {final_loss:.3f}")
    sample = sample_text(model, max_tokens=200)
    log(args.name, args.color, "  ", f"\033[1;93m\"{sample[:160]}\"\033[0m")

    if args.result_out:
        result = {
            "name": args.name,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "sample": sample_text(model, max_tokens=300),
        }
        with open(args.result_out, "w") as f:
            json.dump(result, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--color", default="37")
    p.add_argument("--sec-key", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--relay", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--inner-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--outer-lr", type=float, default=0.7)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=float, default=0.3)
    p.add_argument("--result-out", default=None)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
