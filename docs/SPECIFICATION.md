# nostrain: Specification v0.7.0

## Overview

nostrain is a protocol for distributed ML training over Nostr relays. The implemented repository milestone now covers the payload layer, a signed transport slice, and a resilient end-to-end runner: deterministic model snapshots, pseudo-gradient generation, sparse quantized wire encoding, NIP-01-compatible gradient/heartbeat/checkpoint event envelopes, websocket discovery/collection across one or more relays, resumable checkpoints, rolling checkpoint-slot retention, late-gradient discard tracking, and a built-in linear-regression worker loop that trains locally and synchronizes through relays.

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
23. Publish the same heartbeat/gradient events to multiple relays and continue when only a subset succeeds
24. Collect/discover across multiple relays and deduplicate replayed worker updates by worker identity
25. Persist resumable training checkpoints containing model state, momentum state, and completed round summaries
26. Resume `run-training` from a prior checkpoint without replaying completed rounds
27. Build and validate signed checkpoint distribution events (kind `33335`)
28. Discover the latest recoverable checkpoint across one or more relays
29. Resume `run-training` from the latest relay checkpoint for a run without local state handoff
30. Scan for late gradients from earlier rounds after advancing and report them as discarded
31. Bound relay-visible checkpoint history per worker by publishing checkpoints into configurable rolling slots
32. Bound local checkpoint artifacts and optionally prune old per-round artifact directories under `run-training`

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
- publication can target multiple relays; the current runner proceeds when at least one relay accepts the event
- collection subscribes using `kinds=[33333]` and `#t=["nostrain"]`, then narrows `run` and `round` client-side because relay-side tag indexing is only standardized for single-letter tags in NIP-01
- cross-relay collection deduplicates worker updates by the parameterized event identity and keeps the freshest candidate when duplicates disagree

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
- multi-relay discovery unions the active worker set across all queried relays while deduplicating replayed heartbeats

## Checkpoint event schema

Kind: `33335`

Top-level NIP-01 fields:

- `id`
- `pubkey`
- `created_at`
- `kind`
- `tags`
- `content`
- `sig`

Tags:

- `d`: `run:<run-name>:worker:<worker-id>:checkpoint:slot:<slot>` for bounded retention, or the legacy `...:checkpoint:round:<round>` form
- `t`: `nostrain`
- `run`
- `worker`
- `round`
- optional `slot`
- `next-round`
- `model`
- `rounds-completed`
- `checkpoint`

Content:

- canonical JSON training checkpoint document

Transport note:

- checkpoints are signed with the same canonical NIP-01 serialization rules as gradient and heartbeat events
- the `checkpoint` tag is the SHA-256 digest of the canonical checkpoint JSON content
- `run-training` emits checkpoints into `round % checkpoint_history` rolling slots by default so each worker keeps a bounded set of replaceable relay identities
- discovery deduplicates replaceable checkpoint identities per worker/slot and selects the highest `next-round` checkpoint as the latest recoverable state
- legacy round-based checkpoint identities are still parsed for compatibility with older runs
- relay-based resume restores `current_state`, `momentum_state`, prior round history, and previously observed late gradients from the embedded checkpoint document

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
nostrain build-checkpoint <checkpoint.json> [--worker <id>] [--history-slot <n>] [--sec-key <hex> | --pubkey <hex> [--sig <hex>]]
nostrain publish-event <event.json> --relay <ws://...> [--relay <ws://...> ...]
nostrain discover-workers --relay <ws://...> [--relay <ws://...> ...] --run <name> [--round <n>]
nostrain discover-checkpoints --relay <ws://...> [--relay <ws://...> ...] --run <name> [--worker <id>]
nostrain collect-events --relay <ws://...> [--relay <ws://...> ...] --run <name> --round <n> [--sync timeout|strict|quorum|async] [--discover-workers]
nostrain aggregate-round --relay <ws://...> [--relay <ws://...> ...] --run <name> --round <n> [--sync timeout|strict|quorum|async] [--discover-workers]
nostrain train-local <state.json> <dataset.json> [--steps 500] [--learning-rate 0.01] [--batch-size 1]
nostrain run-training <state.json> <dataset.json> --relay <ws://...> [--relay <ws://...> ...] --run <name> --sec-key <hex> [--rounds 1] [--checkpoint-out path.json] [--checkpoint-history 4] [--artifact-retention-rounds N] [--resume-from checkpoint.json | --resume-latest-checkpoint]
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
4. compute `theta_local - theta_base`, compress it, and publish it as a signed gradient event to one or more relays
5. collect same-round worker updates from the configured relays until the timeout path goes idle, deduplicating cross-relay replays
6. average the collected deltas and apply the configured outer Nesterov step
7. build and publish a signed checkpoint event containing the latest recoverable run state, reusing a rolling checkpoint slot when retention is enabled
8. persist round/session artifacts, bounded checkpoint-slot artifacts, and an optional resumable checkpoint when output paths are configured
9. before the next round, scan for newly arrived gradients from earlier rounds and record them as discarded late arrivals
10. on resume, restore model state, momentum state, round history, and prior late-gradient observations from either a local checkpoint file or the latest relay checkpoint before continuing at `next_round`

Fault-tolerance boundary:

- missing peer gradients are tolerated when at least one valid gradient event is collected before timeout
- partial relay outages are tolerated when at least one configured relay accepts publication and later yields a valid gradient collection
- stale or invalid events are discarded by the existing transport validators
- late gradients from older rounds are surfaced separately and excluded from the current outer step
- total relay failure or zero collected gradients still aborts the round to avoid silent model divergence

## Deferred from v0.7.0

- reconciliation of discarded late gradients after a worker has already advanced
- DiLoCo training loop integration with PyTorch or MLX
- live dashboarding
