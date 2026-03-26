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
- documentation updates describing the unsigned transport boundary

This moves the project from offline payload tooling to a relay-connected prototype that can exchange and aggregate updates over a websocket transport.

## Next milestone: event signing + public relay compatibility

Goal: upgrade the current unsigned transport envelope into a real signed Nostr event flow that interoperates with public relays.

Deliverables:

- canonical event serialization and deterministic event IDs
- Schnorr signing support for worker keypairs
- relay publish path that satisfies NIP-01 event requirements
- validation tests against signed-event fixtures and the local mock relay
- CLI support for worker key material or delegated signing inputs

## Milestone after that: training runner

Goal: connect the protocol layer to an actual DiLoCo inner/outer training loop.

Deliverables:

- adapter for exporting/importing model state from a training runtime
- configurable inner step count
- outer aggregation and update application wired to live worker traffic
- timeout-based round completion
- fault-tolerant handling of missing gradients

## Deferred polish

- rich terminal UX
- multi-relay redundancy
- heartbeat discovery events
- checkpoint publishing
- visualization/dashboard
