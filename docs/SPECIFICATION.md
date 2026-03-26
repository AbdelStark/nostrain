# nostrain: Specification v0.3.0

## Overview

nostrain is a protocol for distributed ML training over Nostr relays. The implemented repository milestone now covers the payload layer plus a signed transport slice: deterministic model snapshots, pseudo-gradient generation, sparse quantized wire encoding, NIP-01-compatible event envelopes for those updates, and websocket collection/aggregation for one round.

Live training orchestration remains out of scope for this version.

## Implemented core

1. Load model snapshots from deterministic JSON
2. Compute pseudo-gradient deltas (`current - initial`)
3. Flatten the full model into a single ordered value stream
4. Keep the top-k values by absolute magnitude
5. Quantize retained values to int8 with a shared scale factor
6. Compress the wire payload (`zlib` by default, optional `zstd`)
7. Base64-wrap the payload for Nostr event content
8. Average multiple worker deltas into one aggregated update
9. Apply a local Nesterov-style outer update with persistent momentum state
10. Build and validate nostrain gradient events (kind `33333`)
11. Canonically serialize events, derive deterministic IDs, and sign/verify them with BIP340 Schnorr signatures
12. Publish signed gradient events to a relay-compatible websocket endpoint
13. Subscribe to one run/round, reject malformed events, and deduplicate replayed worker updates
14. Aggregate collected relay events directly into a model delta

## Model state schema

```json
{
  "parameters": {
    "<parameter-name>": {
      "shape": [2, 3],
      "values": [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    }
  }
}
```

Rules:

- parameter names are unique
- parameters are canonically ordered by name when loaded
- `len(values)` must equal the product of `shape`
- model digests are SHA-256 over the canonical parameter stream

## Gradient event schema

Kind: `33333`

Top-level NIP-01 fields:

- `id`
- `pubkey`
- `created_at`
- `kind`
- `tags`
- `content`
- `sig`

Tags:

- `d`: `run:<run-name>:worker:<worker-id>:round:<round>`
- `t`: `nostrain`
- `run`
- `round`
- `worker`
- `model`
- `steps`
- `compression`
- `params`
- `values`
- `selected`

Content:

- base64 nostrain payload container

Transport note:

- current repository events may be emitted as unsigned local envelopes, signable envelopes (`pubkey` + `id`), or fully signed NIP-01 events
- signed relay publication uses canonical event serialization and BIP340 Schnorr signatures over the event id
- transport is validated against local/mock relay endpoints and public-relay-compatible signed publication flows
- collection subscribes using `kinds=[33333]` and `#t=["nostrain"]`, then narrows `run` and `round` client-side because relay-side tag indexing is only standardized for single-letter tags in NIP-01

## Payload wire format

### Container

- magic: `NSCP`
- version: `1`
- codec id: `1=zlib`, `2=zstd`
- compressed raw payload bytes

### Raw payload

- magic: `NSTR`
- version: `1`
- `topk_ratio`
- `total_values`
- `selected_values`
- `parameter_count`
- `scale`
- parameter layout manifest:
  - parameter name
  - tensor rank
  - tensor shape
- sparse index list (`uint32`)
- quantized value list (`int8`)

Reconstruction is exact with respect to the sparse support and approximate with respect to quantization.

## CLI surface

Implemented commands:

```bash
nostrain hash-state <state.json>
nostrain derive-pubkey <sec-key>
nostrain encode-delta <initial.json> <current.json> [--topk 0.01] [--codec zlib|zstd]
nostrain decode-payload <payload.json>
nostrain aggregate-payloads <payload-a.json> <payload-b.json> [...]
nostrain apply-payload <base.json> <payload.json>
nostrain outer-step <base.json> <aggregated.json> [--learning-rate 0.7] [--momentum 0.9]
nostrain build-event <payload.json> --run <name> --round <n> --worker <id> --model <sha256> [--sec-key <hex> | --pubkey <hex> [--sig <hex>]]
nostrain publish-event <event.json> --relay <ws://...>
nostrain collect-events --relay <ws://...> --run <name> --round <n>
nostrain aggregate-round --relay <ws://...> --run <name> --round <n>
nostrain inspect-event <event.json> [--json]
```

## Local outer step semantics

The repository now supports local round simulation over decoded worker deltas.

Given worker deltas `d_1 ... d_n`:

1. aggregate with an arithmetic mean
2. update velocity: `v_t = momentum * v_(t-1) + d_avg`
3. compute Nesterov-style update: `u_t = learning_rate * (d_avg + momentum * v_t)`
4. apply `u_t` to the base model state

This treats the aggregated pseudo-gradient as an update direction inferred from local training drift rather than as a raw loss gradient.

## Deferred from v0.3.0

- worker discovery heartbeats
- multi-relay deduplication
- DiLoCo training loop integration with PyTorch or MLX
- live dashboarding
