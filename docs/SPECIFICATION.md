# nostrain: Specification v0.1.0

## Overview
Distributed ML training over Nostr relays using DiLoCo (Distributed Low-Communication). Workers train independently, publish compressed pseudo-gradients as signed Nostr events, subscribe to other workers' gradients, and apply outer optimizer steps. No coordinator. No central server. Global, censorship-resistant, permissionless.

## Core Algorithm (DiLoCo over Nostr)

Each worker independently runs the inner/outer optimization loop:

1. Inner training: 500 local AdamW steps on local data shard
2. Compute pseudo-gradient: delta between current and initial weights
3. Compress: top-k sparsification (1%) + int8 quantization + zstd
4. Publish: signed Nostr event (kind 33333) to relay
5. Subscribe: collect other workers' pseudo-gradients for this round
6. Aggregate: average all pseudo-gradients (including own)
7. Outer step: Nesterov momentum optimizer (lr=0.7, momentum=0.9)
8. Continue from step 1

## Nostr Event Schema

Kind: 33333 (parameterized replaceable). Tags: d (run identifier), t (nostrain), run, round, worker (hex pubkey), model (SHA-256 of initial weights), steps, compression method, params count. Content: base64 compressed pseudo-gradients.

## Compression Pipeline

1. Top-k sparsification: keep top 1% by magnitude
2. Sparse representation: (indices, values) pairs
3. Int8 quantization: scale to [-127, 127] + scale factor
4. zstd byte compression
5. Base64 encode

Expected: 50-200x compression. 150M model: ~600MB raw to ~6-18MB compressed.

## CLI

```bash
nostrain start train.py --relay wss://relay.damus.io --name my-run
nostrain join --relay wss://relay.damus.io --name my-run
nostrain monitor --relay wss://relay.damus.io --name my-run
nostrain list --relay wss://relay.damus.io
```

## Dependencies

torch >= 2.0, nostr-sdk >= 0.30 (Python bindings for rust-nostr), zstandard >= 0.22, click >= 8.0, rich >= 13.0
