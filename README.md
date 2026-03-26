# nostrain

Distributed ML training over Nostr relays.

The project vision is still the same: no coordinator, no central server, just workers exchanging sparse pseudo-gradients through Nostr. The repository now ships the first executable milestone behind that vision: the protocol and payload toolkit that model snapshots, computes DiLoCo-style deltas, compresses them into transport payloads, and wraps them in nostrain event envelopes.

## Current status

`nostrain` v0.1.0 is a local protocol toolchain. It implements:

- canonical model-state JSON loading and hashing
- pseudo-gradient computation (`current - initial`)
- top-k sparsification + int8 quantization + wire compression
- multi-worker delta aggregation
- momentum-backed local outer updates
- nostrain gradient event construction and validation
- a CLI for encoding, decoding, applying, and inspecting payloads

It does **not** yet implement live relay connectivity, worker discovery, or training-script orchestration. Those are the next milestones on top of this foundation.

## Install

```bash
python3 -m pip install -e .
```

Optional `zstd` support:

```bash
python3 -m pip install -e ".[zstd]"
```

## Model state format

The current toolchain uses a deterministic JSON representation for model snapshots:

```json
{
  "parameters": {
    "encoder.weight": {
      "shape": [2, 3],
      "values": [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    },
    "encoder.bias": {
      "shape": [3],
      "values": [0.01, -0.02, 0.03]
    }
  }
}
```

This keeps the protocol testable today without depending on PyTorch serialization. A future training runner can export/import this structure at the edge.

## CLI

Hash a model snapshot:

```bash
nostrain hash-state initial.json
```

Create a compressed pseudo-gradient payload:

```bash
nostrain encode-delta initial.json current.json --topk 0.25 -o payload.json
```

Decode the payload back to a sparse delta:

```bash
nostrain decode-payload payload.json
```

Apply a payload to reconstruct the updated model state:

```bash
nostrain apply-payload initial.json payload.json -o reconstructed.json
```

Average several worker payloads into one aggregated delta:

```bash
nostrain aggregate-payloads worker-a.json worker-b.json -o aggregated.json
```

Apply a local DiLoCo-style outer step:

```bash
nostrain outer-step initial.json aggregated.json \
  --learning-rate 0.7 \
  --momentum 0.9 \
  --momentum-out next-momentum.json \
  -o next-state.json
```

Build a nostrain event envelope:

```bash
nostrain build-event payload.json \
  --run demo-run \
  --round 7 \
  --worker worker-pubkey \
  --model "$(nostrain hash-state initial.json)" \
  -o event.json
```

Inspect and validate an event:

```bash
nostrain inspect-event event.json --json
```

## Protocol summary

Gradient events still target Nostr kind `33333`. The implemented envelope includes:

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

The event content is a base64 wire payload containing:

1. a nostrain container header
2. a sparse tensor layout manifest
3. top-k indices
4. int8 quantized values + scale
5. compressed bytes (`zlib` now, optional `zstd` when installed)

## Roadmap

- [x] Canonical model-state format and hashing
- [x] Pseudo-gradient delta computation
- [x] Top-k + int8 compressed payload wire format
- [x] Multi-worker delta aggregation
- [x] Local outer-step simulation with momentum state
- [x] nostrain gradient event builder and validator
- [x] CLI for local encode/decode/apply/inspect workflows
- [ ] Relay publish/subscribe transport
- [ ] Worker discovery and heartbeat events
- [ ] Training-script integration with DiLoCo inner/outer loops
- [ ] Multi-relay redundancy
- [ ] Live monitoring dashboard
