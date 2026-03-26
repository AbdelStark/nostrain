# nostrain: Implementation Plan

## Phase 1: Core (Week 1-2)
- DiLoCo training loop (inner AdamW + outer Nesterov)
- Pseudo-gradient computation (theta - theta_initial)
- Compression pipeline (top-k + int8 + zstd)
- Nostr event publishing via nostr-sdk Python bindings
- Nostr event subscribing and filtering by run/round tags
- Single-relay, 2-worker proof of concept
- Test with TinyStories dataset + small GPT-2 (125M)

## Phase 2: Protocol (Week 3)
- Event schema finalization (kind 33333, all tags)
- Run discovery (list active runs on relay)
- Round synchronization (configurable: wait for N workers or timeout)
- Async mode (use whatever gradients arrived, don't block)
- Worker join/leave (dynamic worker count, fault tolerance)
- Multi-relay redundancy (publish to N relays, subscribe from all)

## Phase 3: Polish (Week 4)
- CLI with click (start, join, monitor, list)
- Rich terminal output (progress bars, worker table, loss curves)
- Training script interface (user provides standard PyTorch script)
- pip installable package (nostrain on PyPI)
- README with demo recording
- Example scripts (GPT-2 on TinyStories, MNIST distributed)

## Phase 4: Viral Launch
- Record demo: 2-3 machines in different locations training together via public Nostr relay
- Write blog post on architecture and design decisions
- Craft viral tweet (see VIRAL_TWEETS.md)
- Post simultaneously on Twitter, Nostr, Hacker News, r/MachineLearning, r/nostr
- Tag: DiLoCo authors, Nostr core devs, grove author, ML influencers

## Why Python + PyTorch (not Rust + Burn)
For viral potential, accessibility is everything. The ML community is Python-first. PyTorch means anyone can try it in 2 minutes with pip install. We use rust-nostr via Python bindings for the Nostr layer (Rust performance where it matters, Python ergonomics everywhere else). A Rust/Burn backend is a natural follow-up that plays to the freedom tech audience.
