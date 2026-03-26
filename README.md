# nostrain

Distributed ML training over Nostr relays.

The project vision is still the same: no coordinator, no central server, just workers exchanging sparse pseudo-gradients through Nostr. The repository now ships the protocol/payload toolkit plus the first relay transport slice: model snapshots, DiLoCo-style deltas, sparse transport payloads, nostrain event envelopes, and relay collection for one round.

## Current status

`nostrain` v0.2.0 is a protocol and relay toolchain. It implements:

- canonical model-state JSON loading and hashing
- pseudo-gradient computation (`current - initial`)
- top-k sparsification + int8 quantization + wire compression
- multi-worker delta aggregation
- momentum-backed local outer updates
- nostrain gradient event construction and validation
- websocket relay publish/subscribe for gradient events
- replay-safe event collection and round aggregation over a relay
- a CLI for encoding, decoding, applying, publishing, collecting, and inspecting payloads

It does **not** yet implement Nostr event signing, worker discovery, or training-script orchestration. Public relays that enforce signed NIP-01 events will reject these envelopes today; the transport milestone in this repository targets local or permissive relay endpoints while signing remains the next protocol upgrade.

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

Publish the event to a relay-compatible websocket endpoint:

```bash
nostrain publish-event event.json --relay ws://127.0.0.1:8765 --json
```

Collect and validate one round from a relay:

```bash
nostrain collect-events \
  --relay ws://127.0.0.1:8765 \
  --run demo-run \
  --round 7 \
  --limit 2 \
  --json
```

Aggregate the collected worker updates directly from the relay:

```bash
nostrain aggregate-round \
  --relay ws://127.0.0.1:8765 \
  --run demo-run \
  --round 7 \
  --limit 2 \
  -o aggregated.json
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

Relay collection currently subscribes on `kind=33333` and `#t=["nostrain"]`, then narrows `run` and `round` client-side. This avoids relying on non-standard multi-character tag indexing at the relay layer.

## Roadmap

- [x] Canonical model-state format and hashing
- [x] Pseudo-gradient delta computation
- [x] Top-k + int8 compressed payload wire format
- [x] Multi-worker delta aggregation
- [x] Local outer-step simulation with momentum state
- [x] nostrain gradient event builder and validator
- [x] CLI for local encode/decode/apply/inspect workflows
- [x] Relay publish/subscribe transport
- [x] Replay-safe relay round collection and aggregation
- [ ] Event signing for public relays
- [ ] Worker discovery and heartbeat events
- [ ] Training-script integration with DiLoCo inner/outer loops
- [ ] Multi-relay redundancy
- [ ] Live monitoring dashboard
