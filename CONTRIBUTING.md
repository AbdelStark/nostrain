# Contributing

## Scope

`nostrain` is a Python library and CLI for coordinator-free distributed
training over Nostr relays. Contributions should preserve three properties:

- deterministic model state and hashing behavior
- stable Nostr event semantics and signature verification
- optional dependency boundaries between the transport core and numeric backends

## Development setup

```bash
python -m pip install -e ".[dev,numpy]"
```

If you need to exercise the real torch integration instead of the fake torch
test double, install the optional torch extra as well:

```bash
python -m pip install -e ".[dev,numpy,torch]"
```

## Daily commands

```bash
make lint
make test
make coverage
python -m build
```

The relay integration suite is slower than the rest of the repo and should be
run explicitly when touching transport, checkpoint, retry, or training-session
code:

```bash
python -m pytest tests/test_relay.py -q
```

## Change expectations

- Every public behavior change needs a test.
- Every bug fix needs a regression test or a documented reason a regression
  test is not possible.
- README, architecture notes, and `AGENTS.md` must be updated when public
  behavior or contributor workflow changes.
- Keep optional dependencies optional. `model.py`, `compression.py`,
  `protocol.py`, and `relay.py` must not start requiring numpy or torch.
- Do not change event kinds, tag names, checkpoint fields, or payload encoding
  without explicitly documenting the compatibility impact.

## Review checklist

Before opening or merging a change, confirm all of the following:

- `make lint` passes.
- `make test` passes.
- Relay-specific changes also pass `python -m pytest tests/test_relay.py -q`.
- Public docs are still accurate.
- New generated artifacts are not being hidden by ignore rules.

## Reporting bugs

Use the repository issue tracker and include:

- the exact command or Python entrypoint
- runtime and backend (`python`, `numpy`, or `torch`)
- relay URLs and timeout/retry settings
- any generated summary, checkpoint, or collection JSON relevant to the failure
