# AGENTS.md

## Project identity

`nostrain` is a Python library and CLI for coordinator-free distributed ML
training over Nostr relays using signed, compressed pseudo-gradients.

## Architecture map

- `src/nostrain/model.py`: canonical tensor and model-state representation
- `src/nostrain/compression.py`: top-k sparsification, int8 quantization, and
  payload framing
- `src/nostrain/protocol.py`: Nostr event metadata, signing, parsing, and
  checkpoint validation
- `src/nostrain/relay.py`: websocket publishing, collection, retries,
  heartbeat discovery, and multi-relay merging
- `src/nostrain/runtime.py`: built-in regression runtimes and backend-specific
  local training/evaluation
- `src/nostrain/training.py`: end-to-end round orchestration, checkpoints,
  late-gradient reconciliation, and artifact emission
- `src/nostrain/stateio.py` and `src/nostrain/pytorch.py`: JSON/NPZ/PyTorch
  state import/export
- `src/nostrain/cli.py`: public CLI surface

## Tech stack

- Python 3.10+
- setuptools packaging
- `websockets` for relay transport
- optional `numpy`, `torch`, and `zstandard`
- `pytest`, `pytest-cov`, `coverage`, and `ruff` for development

## Verified commands

```bash
python -m pip install -e ".[dev,numpy]"
python -m ruff check src tests demo
python -m pytest -q
python -m pytest tests/test_relay.py -q
python -m pytest --cov=nostrain --cov-report=term-missing -q
python -m build
python -m nostrain --help
```

## Conventions

- Preserve deterministic parameter ordering and hashing.
- Keep wire-format, event-tag, and checkpoint changes explicit and documented.
- Keep optional dependencies at the edges; transport/protocol code must remain
  usable without numpy or torch.
- New CLI flags or public API behavior require tests and README updates.
- Prefer result objects that surface retry and relay-failure context instead of
  swallowing errors.

## Critical constraints

- Do not change Nostr event kinds (`33333` / `33334` / `33335`) casually.
- Do not introduce hidden runtime coupling between the transport core and the
  built-in regression backends.
- Do not remove retry telemetry or checkpoint metadata fields that current
  tests and recovery flows depend on.
- Do not reintroduce a blanket JSON ignore rule; fixtures and source JSON must
  remain reviewable.

## Gotchas

- `tests/test_relay.py` is the slowest suite in the repo at roughly 50 seconds.
- CLI coverage under-reports when exercised only through subprocesses; the repo
  has good integration coverage, but raw line coverage remains lower than it
  should be.
- `tests/fake_torch` is intentionally used for most PyTorch compatibility tests
  so the full test suite does not require a real torch install.
- The GPT demo is built on the generic state transport and is not a first-class
  runtime in the library itself.

## Current state

As of 2026-03-27, the project is alpha: suitable for local experiments,
protocol prototyping, and relay-backed demos, but not yet suitable for
unattended production ML pipelines or hostile relay environments.
