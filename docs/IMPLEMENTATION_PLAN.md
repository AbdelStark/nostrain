# nostrain: Implementation Plan

## Completed milestone: protocol engine

Delivered in the current repository state:

- Python package scaffolding with installable `nostrain` CLI
- canonical model-state JSON parsing and hashing
- pseudo-gradient delta computation and application
- sparse payload compression: top-k + int8 + container codec
- nostrain gradient event builder and validator
- automated tests for state math, payload round-trips, protocol validation, and CLI workflow

This moves the project from idea/docs to executable protocol tooling.

## Completed milestone: local round simulation

Delivered in the current repository state:

- aggregation of multiple worker payloads into one delta
- a local Nesterov-style outer update with persisted momentum state
- CLI commands for `aggregate-payloads` and `outer-step`
- tests covering multi-worker averaging and local round execution

This makes it possible to simulate a DiLoCo outer round entirely offline.

## Completed milestone: relay transport

Delivered in the current repository state:

- websocket relay client for publishing nostrain event envelopes
- relay collection for one run/round with client-side run filtering and replay-safe deduplication
- CLI commands for `publish-event`, `collect-events`, and `aggregate-round`
- integration tests with a local mock relay covering publish/collect/aggregate flows
- documentation updates describing the relay transport boundary

This moves the project from offline payload tooling to a relay-connected prototype that can exchange and aggregate updates over a websocket transport.

## Completed milestone: event signing + public relay compatibility

Delivered in the current repository state:

- canonical NIP-01 event serialization and deterministic event IDs
- BIP340 Schnorr signing support for worker keypairs
- relay publish path validated against signed-event expectations
- tests covering official BIP340 vectors plus signed-event relay roundtrips
- CLI support for `derive-pubkey`, local `--sec-key` signing, and delegated `--pubkey`/`--sig` event assembly

This moves the project from a permissive/unsigned relay prototype to a public-relay-compatible signed transport.

## Completed milestone: worker heartbeat discovery + round sync

Delivered in the current repository state:

- signed worker heartbeat events (kind `33334`) with capability and relay-hint tags
- relay heartbeat discovery with replay deduplication and stale-worker filtering
- round collection strategies driven by discovered workers: `timeout`, `strict`, `quorum`, `async`
- CLI support for `build-heartbeat`, `discover-workers`, and strategy-aware `collect-events` / `aggregate-round`
- tests covering heartbeat parsing, discovery, and quorum-based collection

This moves the project from blind relay collection to relay-aware coordination with an explicit picture of which workers are currently alive for a round.

## Completed milestone: training runner

Delivered in the current repository state:

- deterministic JSON datasets for a built-in linear-regression workload
- linear-model import/export through the shared `ModelState` schema
- configurable local SGD inner loops via `train-local`
- relay-backed worker orchestration via `run-training`
- timeout-based round completion with partial gradient aggregation
- per-round artifacts plus session/momentum outputs for inspection and continuation
- tests covering local training, missing-peer timeouts, and two-worker relay convergence

This moves the project from protocol-first tooling to an executable distributed-training prototype that can actually perform local work, publish updates, and converge across workers through a relay.

## Completed milestone: multi-relay resilience + local checkpoint recovery

Delivered in the current repository state:

- redundant heartbeat/gradient publication to multiple relays
- cross-relay worker discovery and gradient collection with replay-safe deduplication
- partial-relay failure tolerance for `run-training`, `publish-event`, `discover-workers`, `collect-events`, and `aggregate-round`
- resumable training checkpoints containing model state, momentum state, relay list, and completed round summaries
- CLI support for repeated `--relay`, `--checkpoint-out`, and `--resume-from`
- integration tests covering cross-relay deduplication, relay failover, and resumed-vs-uninterrupted training equivalence

This moves the project from a single-relay prototype to a more fault-tolerant training system that can survive relay loss and resume worker progress without manual state reconstruction.

## Completed milestone: late-gradient handling + checkpoint distribution

Delivered in the current repository state:

- signed checkpoint events (kind `33335`) carrying the latest recoverable training state
- relay checkpoint discovery plus `run-training --resume-latest-checkpoint` for rejoining workers
- per-round checkpoint publication artifacts alongside the existing local checkpoint files
- explicit late-gradient scans that surface/discard stale updates from already-completed rounds
- checkpoint/session summaries that persist previously observed late gradients across resume boundaries
- protocol, relay, CLI, and training tests covering checkpoint publish/discovery, relay resume, and late-gradient accounting

This moves the project from local-only checkpoint recovery to distributed checkpoint recovery with explicit stale-update handling when workers advance at different times.

## Completed milestone: bounded checkpoint retention

Delivered in the current repository state:

- rolling checkpoint-slot publication so each worker reuses a bounded set of parameterized checkpoint identities on relays
- CLI support for `build-checkpoint --history-slot`, `run-training --checkpoint-history`, and `run-training --artifact-retention-rounds`
- bounded checkpoint artifact snapshots under the worker artifact root plus a machine-readable retention manifest
- optional pruning of old per-round artifact directories without affecting latest-checkpoint resume
- protocol, relay, and runner tests covering slot replacement, bounded relay checkpoint history, and local artifact pruning

This moves the project from unbounded checkpoint accumulation to a retention-aware training system that can keep distributed recovery state compact without losing resumability.

## Completed milestone: deferred late-gradient reconciliation

Delivered in the current repository state:

- checkpoint/session persistence for late-gradient payloads plus an explicit late-gradient watermark for resume-safe scanning
- deferred late-gradient reconciliation before the next round using a separate configurable outer step
- CLI support for `run-training --late-gradient-strategy`, `--late-gradient-learning-rate`, and `--late-gradient-momentum`
- per-round/session reporting that distinguishes reconciled, pending, and failed late-gradient records
- protocol, relay, CLI, and training tests covering deferred reconciliation, record-only late-gradient mode, and checkpoint-resume equivalence

This moves the project from stale-update accounting to a training loop that can actually reincorporate compatible late updates without replaying earlier rounds.

## Completed milestone: richer runtimes

Delivered in the current repository state:

- shared runtime resolution across datasets, model states, checkpoints, and `run-training`
- pluggable built-in runtimes for `linear-regression` and one-hidden-layer `mlp-regression`
- deterministic `init-state` CLI support for built-in runtime initialization
- pure-Python MLP local training plus relay-backed two-worker convergence coverage
- checkpoint/session metadata that preserves runtime identity across resume boundaries

This moves the project from a linear-only training prototype to a runtime-aware training system that can exercise a non-linear model through the same resilient relay/checkpoint path.

## Completed milestone: relay retry and backoff

Delivered in the current repository state:

- configurable retry/backoff policy for publish, discovery, collection, and late-gradient scans
- idempotent duplicate-publish handling for retry-safe event publication after dropped relay ACKs
- CLI flags for retry attempts and backoff shape across relay-facing commands
- retry telemetry in relay JSON outputs plus `run-training` round/session summaries
- integration tests covering transient publish disconnects, transient collection disconnects, CLI retry flags, and retry-aware training summaries

This moves the project from partial-relay fault tolerance to transport flows that can recover from transient websocket failures without hiding what happened.

## Completed milestone: NumPy runtime boundary

Delivered in the current repository state:

- canonical model-state interchange between JSON and `numpy-npz`
- CLI support for `convert-state` plus auto-detected state formats across initialization, hashing, payload application, local training, and relay-backed training
- backend-selectable local training/evaluation through `python` and `numpy`
- relay-backed `run-training` support for NumPy-backed workers without changing protocol/checkpoint formats
- conformance tests that compare NumPy-vs-Python training behavior plus CLI/relay coverage for `.npz` state I/O

This moves the project from built-in-only runtime execution to a real external runtime boundary that can exchange and optimize model states outside the canonical JSON edge without weakening the protocol contracts.

## Completed milestone: PyTorch state-dict adapter

Delivered in the current repository state:

- PyTorch-compatible state-dict archive conversion via the shared `convert-state` and auto-detected state-format path
- direct-module export layouts (`weight` / `bias`, `hidden.*`, `output.*`) for the built-in runtimes
- import normalization for nested wrapper prefixes such as `module.model.*`
- local training and state hashing over `.pt.npz` / `.pth.npz` archives without changing the canonical protocol payloads
- conformance tests covering adapter round-trips, prefixed import, and CLI training over PyTorch-style archives

This moves the project from an abstract external-runtime boundary to a concrete framework-facing bridge that can exchange model weights with PyTorch-style state dicts while preserving the existing protocol and training contracts.

## Completed milestone: native PyTorch checkpoints + torch backend

Delivered in the current repository state:

- direct `.pt` / `.pth` `torch.save` checkpoint import/export alongside the existing `.pt.npz` archive bridge
- wrapped checkpoint loading for common `state_dict` / `model_state_dict` payload layouts
- backend-selectable local training/evaluation through `python`, `numpy`, and `torch`
- relay-backed `run-training` support for torch-backed workers without changing protocol/checkpoint semantics
- conformance tests covering native checkpoint round-trips plus CLI/relay torch-backend execution

This moves the project from a state-dict archive bridge to a framework-aware runtime path that can exchange and optimize built-in models through native PyTorch checkpoint files and tensor execution.

## Next milestone: module-native adapters

Goal: move beyond built-in tensor layouts and support richer framework-native module workflows.

Deliverables:

- optional module wrappers for built-in runtimes so `nostrain` can materialize `torch.nn.Module` graphs directly
- broader checkpoint import support for common framework training bundles without requiring manual state-dict extraction

## Deferred polish

- rich terminal UX
- visualization/dashboard
