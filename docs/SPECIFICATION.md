# nostrain: Specification v0.5.0

## Overview

nostrain is a protocol for distributed ML training over Nostr relays. The implemented repository milestone now covers the payload layer, a signed transport slice, and a first end-to-end runner: deterministic model snapshots, pseudo-gradient generation, sparse quantized wire encoding, NIP-01-compatible gradient and heartbeat event envelopes, websocket discovery/collection for one round, and a built-in linear-regression worker loop that trains locally and synchronizes through a relay.

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
15. Build and validate signed worker heartbeat discovery events (kind `33334`)
16. Discover active workers from relay heartbeats while discarding stale workers
17. Stop round collection via `timeout`, `strict`, `quorum`, or `async` strategies using discovered workers
18. Load deterministic JSON datasets for built-in linear-regression workloads
19. Import/export linear-regression weights through the shared `ModelState` schema
20. Run configurable local SGD inner loops over JSON datasets
21. Publish heartbeat + gradient events for each round and apply the aggregated outer step locally
22. Persist machine-readable per-round/session artifacts for relay-backed training runs

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

Built-in runner state:

- `linear.weight`: shape `[1, feature_count]`
- `linear.bias`: shape `[1]`

The protocol remains generic, but the built-in training runtime currently targets this linear schema so the full loop stays dependency-light and deterministic.

## Linear regression dataset schema

```json
{
  "task": "linear-regression",
  "examples": [
    {
      "inputs": [1.0, 0.0],
      "target": 2.5
    }
  ]
}
```

Rules:

- `task` must be `linear-regression`
- `examples` must be non-empty
- every example must provide the same number of input features
- targets are scalar floating-point regression values

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

## Heartbeat event schema

Kind: `33334`

Top-level NIP-01 fields:

- `id`
- `pubkey`
- `created_at`
- `kind`
- `tags`
- `content`
- `sig`

Tags:

- `d`: `run:<run-name>:worker:<worker-id>`
- `t`: `nostrain`
- `run`
- `worker`
- `round`
- `heartbeat`
- repeated `capability`
- repeated `relay`

Content:

- empty string

Transport note:

- heartbeats are signed with the same canonical NIP-01 serialization rules as gradient events
- discovery keeps only the newest heartbeat per worker
- workers are considered stale after missing more than three advertised heartbeat intervals
- round collection can optionally snapshot current workers first, then stop when a `strict` or `quorum` target is satisfied

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
nostrain build-heartbeat --run <name> --round <n> --worker <id> [--capability gradient-event] [--advertise-relay ws://...]
nostrain publish-event <event.json> --relay <ws://...>
nostrain discover-workers --relay <ws://...> --run <name> [--round <n>]
nostrain collect-events --relay <ws://...> --run <name> --round <n> [--sync timeout|strict|quorum|async] [--discover-workers]
nostrain aggregate-round --relay <ws://...> --run <name> --round <n> [--sync timeout|strict|quorum|async] [--discover-workers]
nostrain train-local <state.json> <dataset.json> [--steps 500] [--learning-rate 0.01] [--batch-size 1]
nostrain run-training <state.json> <dataset.json> --relay <ws://...> --run <name> --sec-key <hex> [--rounds 1]
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

## Training runner semantics

The built-in runner executes the following loop for each round:

1. snapshot the current global model state
2. publish a signed worker heartbeat for the target round
3. run local SGD for `H` inner steps on the worker's JSON dataset
4. compute `theta_local - theta_base`, compress it, and publish it as a signed gradient event
5. collect same-round worker updates from the relay until the timeout path goes idle
6. average the collected deltas and apply the configured outer Nesterov step
7. persist round/session artifacts when output paths are configured

Fault-tolerance boundary:

- missing peer gradients are tolerated when at least one valid gradient event is collected before timeout
- stale or invalid events are discarded by the existing transport validators
- total relay failure or zero collected gradients still aborts the round to avoid silent model divergence

## Deferred from v0.5.0

- multi-relay deduplication
- checkpoint distribution and recovery
- DiLoCo training loop integration with PyTorch or MLX
- live dashboarding
