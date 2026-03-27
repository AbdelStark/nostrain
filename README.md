# nostrain

Distributed ML training over Nostr relays.

The project vision is still the same: no coordinator, no central server, just workers exchanging sparse pseudo-gradients through Nostr. The repository now ships the protocol/payload toolkit, a signed relay transport slice, and a resilient end-to-end training runner: model snapshots, DiLoCo-style deltas, sparse transport payloads, NIP-01-compatible nostrain gradient/heartbeat/checkpoint events, relay collection with active-worker discovery, cross-relay deduplication, resumable checkpoints, relay-visible checkpoint distribution, deferred late-gradient reconciliation, configurable relay retry/backoff, runtime-pluggable built-in workers for both linear and non-linear regression over one or more relays, plus a NumPy edge-runtime path for model-state interchange and local optimization.

## Current status

`nostrain` v0.9.0 is a protocol, relay, and training toolchain. It implements:

- canonical model-state JSON loading and hashing
- pseudo-gradient computation (`current - initial`)
- top-k sparsification + int8 quantization + wire compression
- multi-worker delta aggregation
- momentum-backed local outer updates
- deterministic NIP-01 event IDs plus BIP340 Schnorr signing/verification
- nostrain gradient event construction and validation
- signed worker heartbeat event construction and validation
- websocket relay publish/subscribe for gradient events
- active worker discovery with stale-heartbeat filtering
- replay-safe event collection and round aggregation over a relay
- cross-relay publish redundancy and replay-safe collection deduplication
- configurable relay retry/backoff across publish, discovery, and collection operations
- idempotent duplicate-publish handling plus retry telemetry in CLI summaries and training artifacts
- round sync strategies (`timeout`, `strict`, `quorum`, `async`) driven by discovered workers
- a deterministic JSON dataset format for built-in regression workloads
- pluggable built-in runtimes for `linear-regression` and `mlp-regression`
- backend-selectable local SGD/evaluation paths (`python`, `numpy`) for both runtimes
- canonical model-state interchange between JSON and `numpy-npz`
- CLI state-format auto-detection plus `convert-state` for explicit format conversion
- a relay-backed training runner that publishes heartbeats and gradients to one or more relays, tolerates partial relay outages, and applies the outer step locally
- resumable training checkpoints carrying model state, momentum state, and prior round history
- signed checkpoint distribution events carrying the latest recoverable training state
- relay checkpoint discovery plus `run-training --resume-latest-checkpoint` for rejoining workers
- rolling checkpoint-slot retention that bounds relay-visible checkpoint history per worker
- checkpointed late-gradient payload tracking plus deferred reconciliation before the next round
- configurable `run-training --late-gradient-strategy discard` accounting for record-only stale updates
- bounded checkpoint artifacts plus optional per-round artifact pruning under `run-training`
- deterministic `init-state` generation for supported built-in runtimes
- a CLI for converting, encoding, decoding, applying, publishing, collecting, and inspecting payloads

It does **not** yet implement framework-specific adapters such as PyTorch/MLX modules or state-dicts. The signed transport path now targets public NIP-01 relays as well as local/mock websocket endpoints, and the built-in runners stay dependency-light and testable while exercising a real external runtime boundary through NumPy state interchange and backend execution.

## Install

```bash
python3 -m pip install -e .
```

Optional `zstd` support:

```bash
python3 -m pip install -e ".[zstd]"
```

Optional `numpy` support for `numpy-npz` state interchange and the NumPy training backend:

```bash
python3 -m pip install -e ".[numpy]"
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

This keeps the protocol testable today without depending on framework-specific serialization. Built-in runtimes, the current NumPy edge adapter, and future PyTorch/MLX adapters can export/import this structure at the edge.

The built-in `linear-regression` runtime expects this minimal state:

```json
{
  "parameters": {
    "linear.bias": {
      "shape": [1],
      "values": [0.0]
    },
    "linear.weight": {
      "shape": [1, 2],
      "values": [0.0, 0.0]
    }
  }
}
```

The built-in `mlp-regression` runtime uses a one-hidden-layer state:

```json
{
  "parameters": {
    "mlp.hidden.bias": {
      "shape": [4],
      "values": [0.0, 0.0, 0.0, 0.0]
    },
    "mlp.hidden.weight": {
      "shape": [4, 2],
      "values": [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08]
    },
    "mlp.output.bias": {
      "shape": [1],
      "values": [0.0]
    },
    "mlp.output.weight": {
      "shape": [1, 4],
      "values": [0.02, -0.01, 0.03, -0.02]
    }
  }
}
```

## NumPy state interop

`nostrain` can also read and write model states as `numpy-npz` archives. Each parameter is stored as a named array in a `.npz` file, with small metadata entries describing the nostrain schema and optional runtime.

Convert a canonical JSON state into an archive:

```bash
nostrain convert-state linear-initial.json -o linear-initial.npz
```

Round-trip it back to JSON:

```bash
nostrain convert-state linear-initial.npz -o linear-initial-roundtrip.json
```

## Dataset format

The built-in worker loop consumes deterministic JSON regression datasets:

```json
{
  "task": "linear-regression",
  "examples": [
    {
      "inputs": [1.0, 0.0],
      "target": 2.5
    },
    {
      "inputs": [0.0, 1.0],
      "target": -0.5
    }
  ]
}
```

Each example must use the same input width. `task` currently supports `linear-regression` and `mlp-regression`, both with scalar regression targets.

## CLI

Initialize a deterministic built-in model state:

```bash
nostrain init-state --runtime linear-regression --features 2 -o linear-initial.json
nostrain init-state --runtime mlp-regression --features 2 --hidden-size 4 -o mlp-initial.json
nostrain init-state --runtime linear-regression --features 2 -o linear-initial.npz
```

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

Run the built-in local training inner loop:

```bash
nostrain train-local linear-initial.json worker-a-dataset.json \
  --steps 50 \
  --learning-rate 0.05 \
  --batch-size 2 \
  --metrics-out local-training.json \
  -o local-state.json
```

Run the same local training loop through the NumPy backend and emit `.npz` weights:

```bash
nostrain train-local linear-initial.npz worker-a-dataset.json \
  --backend numpy \
  --steps 50 \
  --learning-rate 0.05 \
  --batch-size 2 \
  --metrics-out local-training.json \
  -o local-state.npz
```

Run the non-linear MLP runtime against an `mlp-regression` dataset:

```bash
nostrain train-local mlp-initial.json worker-a-mlp-dataset.json \
  --runtime mlp-regression \
  --steps 40 \
  --learning-rate 0.05 \
  --batch-size 2 \
  --metrics-out mlp-training.json \
  -o mlp-state.json
```

Build a nostrain event envelope:

```bash
nostrain build-event payload.json \
  --run demo-run \
  --round 7 \
  --worker worker-pubkey \
  --model "$(nostrain hash-state initial.json)" \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  -o event.json
```

Build a signed heartbeat for worker discovery:

```bash
nostrain build-heartbeat \
  --run demo-run \
  --round 7 \
  --worker worker-pubkey \
  --capability gradient-event \
  --advertise-relay ws://127.0.0.1:8765 \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  -o heartbeat.json
```

Derive an x-only Nostr pubkey from a worker secret key:

```bash
nostrain derive-pubkey 0000000000000000000000000000000000000000000000000000000000000003
```

Publish the event to a relay-compatible websocket endpoint:

```bash
nostrain publish-event event.json --relay ws://127.0.0.1:8765 --json
```

Publish the same event redundantly to multiple relays:

```bash
nostrain publish-event event.json \
  --relay ws://127.0.0.1:8765 \
  --relay wss://relay.example.com \
  --json
```

Relay-facing commands can retry transient websocket failures with backoff:

```bash
nostrain collect-events \
  --relay ws://127.0.0.1:8765 \
  --run demo-run \
  --round 7 \
  --relay-retries 2 \
  --relay-retry-backoff 0.25 \
  --relay-retry-backoff-max 1.0 \
  --json
```

The shared retry flags are available on `publish-event`, `discover-workers`, `discover-checkpoints`, `collect-events`, `aggregate-round`, and `run-training`. JSON outputs and training summaries include per-relay attempt counts, retry delays, and the relays that needed retries.

Build a signed checkpoint event from a saved worker checkpoint:

```bash
nostrain build-checkpoint worker-a-checkpoint.json \
  --history-slot 0 \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  -o checkpoint-event.json
```

Discover the latest recoverable checkpoint for a run:

```bash
nostrain discover-checkpoints \
  --relay ws://127.0.0.1:8765 \
  --run linear-demo \
  --json
```

Collect and validate one round from a relay:

```bash
nostrain collect-events \
  --relay ws://127.0.0.1:8765 \
  --run demo-run \
  --round 7 \
  --discover-workers \
  --sync quorum \
  --limit 2 \
  --json
```

Aggregate the collected worker updates directly from the relay:

```bash
nostrain aggregate-round \
  --relay ws://127.0.0.1:8765 \
  --run demo-run \
  --round 7 \
  --discover-workers \
  --sync strict \
  --limit 2 \
  -o aggregated.json
```

Run an end-to-end worker session over multiple relays with checkpoint output:

```bash
nostrain run-training linear-initial.json worker-a-dataset.json \
  --relay ws://127.0.0.1:8765 \
  --relay wss://relay.example.com \
  --run linear-demo \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  --inner-steps 50 \
  --local-learning-rate 0.05 \
  --batch-size 2 \
  --topk 1.0 \
  --outer-learning-rate 1.0 \
  --momentum 0.0 \
  --round-timeout 2.0 \
  --checkpoint-history 4 \
  --artifact-dir artifacts/worker-a \
  --artifact-retention-rounds 2 \
  --checkpoint-out worker-a-checkpoint.json \
  --summary-out session-summary.json \
  -o final-state.json
```

The same command accepts `.npz` model states and `--backend numpy` while keeping checkpoints, payloads, and relay envelopes in the canonical nostrain formats.

Resume a worker from its last saved checkpoint:

```bash
nostrain run-training linear-initial.json worker-a-dataset.json \
  --relay ws://127.0.0.1:8765 \
  --relay wss://relay.example.com \
  --run linear-demo \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  --resume-from worker-a-checkpoint.json \
  --rounds 2 \
  --summary-out resumed-session-summary.json \
  -o resumed-final-state.json
```

Resume from the latest relay-distributed checkpoint instead of a local file:

```bash
nostrain run-training linear-initial.json worker-a-dataset.json \
  --relay ws://127.0.0.1:8765 \
  --run linear-demo \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  --resume-latest-checkpoint \
  --rounds 1 \
  --summary-out relay-resume-summary.json \
  -o relay-resume-state.json
```

Keep late gradients for audit only instead of folding them into the next round:

```bash
nostrain run-training linear-initial.json worker-a-dataset.json \
  --relay ws://127.0.0.1:8765 \
  --run linear-demo \
  --sec-key 0000000000000000000000000000000000000000000000000000000000000003 \
  --resume-latest-checkpoint \
  --late-gradient-strategy discard \
  --summary-out relay-resume-summary.json \
  -o relay-resume-state.json
```

List active workers advertising a given run/round:

```bash
nostrain discover-workers \
  --relay ws://127.0.0.1:8765 \
  --run demo-run \
  --round 7 \
  --json
```

Inspect and validate an event:

```bash
nostrain inspect-event event.json --json
```

## Protocol summary

Gradient events still target Nostr kind `33333`. The implemented envelope includes:

- NIP-01 top-level fields: `id`, `pubkey`, `created_at`, `kind`, `tags`, `content`, `sig`

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

Relay collection currently subscribes on `kind=33333` and `#t=["nostrain"]`, then narrows `run` and `round` client-side. This avoids relying on non-standard multi-character tag indexing at the relay layer while remaining compatible with NIP-01 relay indexing rules.

Heartbeat discovery events target Nostr kind `33334`. They use:

- `d`: `run:<run-name>:worker:<worker-id>`
- `t`: `nostrain`
- `run`
- `worker`
- `round`
- `heartbeat`
- repeated `capability` tags
- repeated `relay` tags

Heartbeat content is intentionally empty; worker capabilities and relay hints live in the tag set so relays and clients can index them directly. Discovery keeps only the newest heartbeat per worker and drops workers that have missed more than three advertised heartbeat intervals.

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
- [x] Event signing for public relays
- [x] Worker discovery and heartbeat events
- [x] Runtime-aware built-in training runners for `linear-regression` and `mlp-regression`
- [x] Multi-relay redundancy
- [x] Local checkpoint recovery for resumed workers
- [x] Relay checkpoint advertisement/distribution
- [x] Checkpoint retention/pruning policy
- [x] Late-gradient reconciliation across advanced rounds
- [ ] Live monitoring dashboard
