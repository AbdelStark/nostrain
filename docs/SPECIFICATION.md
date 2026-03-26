# nostrain: Specification v0.1.0

## Overview

nostrain is a protocol for distributed ML training over Nostr relays. The implemented repository milestone covers the payload layer: deterministic model snapshots, pseudo-gradient generation, sparse quantized wire encoding, and Nostr event envelopes for those updates.

Relay I/O and live training orchestration are intentionally out of scope for this version.

## Implemented core

1. Load model snapshots from deterministic JSON
2. Compute pseudo-gradient deltas (`current - initial`)
3. Flatten the full model into a single ordered value stream
4. Keep the top-k values by absolute magnitude
5. Quantize retained values to int8 with a shared scale factor
6. Compress the wire payload (`zlib` by default, optional `zstd`)
7. Base64-wrap the payload for Nostr event content
8. Build and validate nostrain gradient events (kind `33333`)

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
nostrain encode-delta <initial.json> <current.json> [--topk 0.01] [--codec zlib|zstd]
nostrain decode-payload <payload.json>
nostrain apply-payload <base.json> <payload.json>
nostrain build-event <payload.json> --run <name> --round <n> --worker <id> --model <sha256>
nostrain inspect-event <event.json> [--json]
```

## Deferred from v0.1.0

- Nostr relay publishing/subscribing
- event signing
- worker discovery heartbeats
- multi-relay deduplication
- DiLoCo training loop integration with PyTorch or MLX
- live dashboarding
