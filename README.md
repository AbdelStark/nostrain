# nostrain

Distributed ML training over Nostr relays. No servers, no coordinators, no cloud. Just relays.

```bash
pip install nostrain

# Machine 1 — start training and publish gradients
nostrain start train.py --relay wss://relay.damus.io --name my-run

# Machine 2 (anywhere in the world) — join the run
nostrain join --relay wss://relay.damus.io --name my-run
```

Workers discover each other via Nostr. Each trains independently using [DiLoCo](https://arxiv.org/abs/2311.08105), then publishes compressed pseudo-gradients as Nostr events. Other workers subscribe, aggregate, and continue. No coordinator. No central server. Workers can join and leave at any time.

Training runs are identified by Nostr event tags. Anyone can observe a run's progress by subscribing to the relay. The entire training history is a public, verifiable log of signed Nostr events.

## How it works

```
┌──────────────┐     Nostr Relay     ┌──────────────┐
│  Worker A     │    (any relay)      │  Worker B     │
│              │                      │              │
│  Local train  │ ──publish──────►   │  Local train  │
│  500 steps    │   pseudo-grads      │  500 steps    │
│  (AdamW)      │   as NIP-XX events  │  (AdamW)      │
│              │                      │              │
│  Subscribe    │ ◄──subscribe────   │  Subscribe    │
│  aggregate    │   other workers'    │  aggregate    │
│  outer step   │   pseudo-grads      │  outer step   │
└──────────────┘                      └──────────────┘
        │                                     │
        └──── both can join/leave anytime ────┘
```

Every 500 local training steps (configurable), each worker:

1. Computes pseudo-gradients (difference between current and initial weights)
2. Compresses them (top-k sparsification + quantization, ~1-3% of full size)
3. Signs and publishes as a Nostr event (kind: 33333)
4. Subscribes to other workers' events for the same run
5. Aggregates received pseudo-gradients
6. Applies outer optimizer step (Nesterov momentum)
7. Continues local training

## Why Nostr?

DiLoCo communicates once every ~500 steps. The payload is small (pseudo-gradients compressed to a few MB). Nostr relays handle this easily.

| Property | What it means for training |
|----------|--------------------------|
| **Global** | Workers anywhere in the world. No VPN, no firewall config. |
| **Censorship-resistant** | Training can't be shut down by blocking a server. Use multiple relays. |
| **Serverless** | No coordinator process. Relays are stateless message brokers. |
| **Signed** | Every gradient update is signed by the worker's Nostr keypair. Verifiable provenance. |
| **Observable** | Anyone can subscribe to a training run and watch progress in real-time. |
| **Zappable** | Contributors can receive Lightning zaps for their compute contributions. |

## What grove did with AirDrop, nostrain does with Nostr

[grove](https://github.com/swarnim-j/grove) showed that DiLoCo works beautifully over local radio (Apple's AWDL). nostrain takes the same idea and makes it global. grove is local mesh. nostrain is a worldwide mesh over Nostr relays.

## Protocol

Training events use Nostr event kind `33333` (parameterized replaceable) with the following structure:

```json
{
  "kind": 33333,
  "content": "<compressed pseudo-gradients (base64)>",
  "tags": [
    ["d", "run:<run-name>"],
    ["t", "nostrain"],
    ["run", "<run-name>"],
    ["round", "42"],
    ["worker", "<worker-id>"],
    ["model", "<model-hash>"],
    ["steps", "500"],
    ["compression", "topk-0.01-int8"]
  ]
}
```

Workers discover runs by subscribing to events with tag `["t", "nostrain"]` and filtering by run name.

## Compression

Pseudo-gradients are large (full model size). We compress aggressively:

1. **Top-k sparsification:** Keep only the top 1% of gradient values by magnitude
2. **Int8 quantization:** Quantize remaining values to 8-bit integers
3. **Delta encoding:** Only send changes from the last communicated state
4. **zstd compression:** Final byte-level compression

Result: ~1-3% of the original payload size. A 150M parameter model's pseudo-gradients compress from ~600MB to ~6-18MB per sync round.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (or MLX for Apple Silicon)
- A Nostr relay (public relays work fine)

## Roadmap

- [x] DiLoCo core (inner/outer optimizer loop)
- [x] Nostr event publishing/subscribing
- [x] Top-k + int8 compression
- [ ] Async worker join/leave (fault tolerance)
- [ ] Multi-relay redundancy
- [ ] Lightning zaps for compute contributors
- [ ] Training progress dashboard (subscribe and visualize)
- [ ] Model checkpoint publishing as Nostr events
- [ ] Burn (Rust) backend support

## Citation

```
nostrain: Distributed ML training over Nostr relays
Abdel Bakhta, 2026
https://github.com/AbdelStark/nostrain
```

Built with conviction that the future of AI training is open, permissionless, and unstoppable.

---

**Bitcoin: A Peer-to-Peer Electronic Cash System**
**nostrain: A Peer-to-Peer Neural Network Training System**
