# Changelog

All notable user-visible changes to `nostrain` will be documented in this file.

The format follows Keep a Changelog, and versioning remains pre-1.0 while the
public API and wire-level ergonomics continue to harden.

## Unreleased

### Added

- Added a real repository contract: `LICENSE`, `CONTRIBUTING.md`,
  `docs/ARCHITECTURE.md`, `AGENTS.md`, `.env.example`, `Makefile`, and a
  GitHub Actions CI workflow.
- Declared a `dev` extra for the supported local toolchain: `pytest`,
  `pytest-cov`, `coverage`, and `ruff`.
- Added example-count metadata to gradient and heartbeat events so relay
  aggregation can weight uneven worker shards correctly.

### Changed

- Fixed the README Python API example so it matches the actual
  `build_gradient_event()` signature.
- Changed payload and relay aggregation to use example-count weighting when the
  metadata is present, while remaining backward compatible with older
  equal-weight events.
- Narrowed `.gitignore` from a blanket `*.json` rule to explicit generated
  artifacts, so real JSON fixtures and source files do not disappear from
  review accidentally.
- Cleaned unused imports and other lint noise across the library, tests, and
  demos so lint can be enforced in CI.

## 0.12.0

Initial pre-changelog baseline.
