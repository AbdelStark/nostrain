# RFC-001: DiLoCo Training Loop

Inner optimizer: AdamW (lr=4e-4, weight_decay=0.1, warmup=1000 steps). Outer optimizer: Nesterov SGD (lr=0.7, momentum=0.9). Inner steps H=500 (configurable). Each worker maintains theta (current weights) and theta_initial (snapshot at start of round). Pseudo-gradient = theta - theta_initial. After outer step, all workers have the same theta and begin next round. Reference implementation: OpenDiLoCo by Prime Intellect.

Implementation status: the current codebase implements the payload math, multi-worker averaging, a local Nesterov-style outer step, and a built-in linear-regression inner/outer training runner. Richer runtimes such as PyTorch/MLX are still pending.

# RFC-002: Nostr Event Protocol

Kind `33333` (parameterized replaceable). Workers publish one event per outer round. The implemented envelope tags are:

- `d = run:<run-name>:worker:<worker-id>:round:<round>`
- `t = nostrain`
- `run`, `round`, `worker`, `model`, `steps`
- `compression`, `params`, `values`, `selected`

Event content is a base64-encoded compressed pseudo-gradient payload. The current transport subscribes with relay filter `kinds=[33333], #t=["nostrain"]` and narrows `run` and `round` client-side, since NIP-01 standardizes relay indexing only for single-letter tags. Publication now supports canonical NIP-01 event ids and BIP340 Schnorr signatures so the same envelope can be accepted by public relays.

# RFC-003: Compression Pipeline

Implemented wire steps:

1. Top-k sparsification over the flattened model value stream
2. Sparse representation as a global index list plus quantized values
3. Int8 quantization with a shared scale factor
4. Container compression with `zlib` today and optional `zstd` support when installed
5. Base64 encoding for Nostr event content

The payload also stores a tensor layout manifest so sparse values can be reconstructed into named parameters.

# RFC-004: Worker Discovery and Synchronization

Discovery: workers publish a "heartbeat" event (kind `33334`) every 60 seconds with worker metadata (pubkey, capabilities, current round). Other workers subscribe to heartbeats to know who is active. Round sync: configurable strategy. "strict" means wait for all known workers. "quorum" means wait for >50% of known workers. "async" means don't wait. "timeout" means wait up to `T` seconds, then proceed with whatever arrived. Default target remains timeout with `T=120s`.

Implementation status: heartbeat events, active-worker discovery, stale-worker filtering, and `strict`/`quorum`/`async`/`timeout` round collection are implemented.

# RFC-005: Fault Tolerance

Worker crash: other workers detect via missing heartbeat (>3 missed = considered offline). Training continues with remaining workers. Crashed workers can now rejoin by discovering the latest signed checkpoint event on the relay and resuming from `next_round`. Relay failure currently fails over across configured relays; richer retry/backoff policy is still pending. Stale gradients that arrive after the outer step are collected as late arrivals and discarded from the current update path.

Implementation status: distributed checkpoint discovery/resume and late-gradient discard tracking are implemented; richer retry/backoff policy remains planned.

# RFC-006: CLI and User Interface

Implemented commands:

- `hash-state`
- `encode-delta`
- `decode-payload`
- `aggregate-payloads`
- `apply-payload`
- `outer-step`
- `build-event`
- `build-heartbeat`
- `build-checkpoint`
- `inspect-event`
- `publish-event`
- `discover-workers`
- `discover-checkpoints`
- `collect-events`
- `aggregate-round`
- `train-local`
- `run-training`

Deferred commands:

- `start`
- `join`
- `monitor`
- `list`

The current CLI is intentionally protocol-first. Training orchestration and relay UX will be added once the transport layer exists.
