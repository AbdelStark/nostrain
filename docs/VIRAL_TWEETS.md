# nostrain: Viral Tweet Drafts

## Option A — The direct grove parallel (recommended)

I trained ML models across continents using Nostr relays.

nostrain is a distributed training library that uses the Nostr protocol as its communication layer. Workers discover each other on relays, train locally using DiLoCo, then publish compressed pseudo-gradients as signed Nostr events. No cloud, no coordinator, no VPN.

nostrain start train.py --relay wss://relay.damus.io
nostrain join --relay wss://relay.damus.io

Nostr relays are perfect for DiLoCo. The algorithm only syncs every ~500 steps. The payload compresses to a few MB. Relays handle this trivially.

Every gradient update is a signed Nostr event. The entire training history is a public, verifiable log. Anyone can subscribe and watch the training happen in real-time.

Workers can join and leave at any time. No registration. No API keys. Just connect to a relay and start training. The protocol handles the rest.

Built on PyTorch + nostr-sdk. Top-k + int8 compression brings payloads down to ~1-3% of full size.

What grove did with AirDrop, nostrain does with Nostr. Local mesh becomes global mesh.

GitHub: github.com/AbdelStark/nostrain
pip install nostrain


## Option B — The freedom tech angle

Censorship-resistant ML training. I built it.

nostrain uses Nostr relays as the communication layer for distributed model training. Workers anywhere in the world can join a training run by subscribing to a relay. No servers. No accounts. No way to shut it down.

The protocol: DiLoCo (from DeepMind) trains locally for 500 steps, then syncs compressed pseudo-gradients via Nostr events. Each update is signed by the worker's keypair. The entire training run is a public, verifiable, cryptographically signed log.

pip install nostrain
nostrain start train.py --relay wss://relay.damus.io

Try to stop this training run. You can't. The relays are replaceable. The workers are anonymous. The gradients are signed. The model improves regardless.

GitHub: github.com/AbdelStark/nostrain


## Option C — The technical deep-dive

DiLoCo communicates once every ~500 training steps. The payload is pseudo-gradients, compressed to a few MB. The latency tolerance is minutes, not milliseconds.

Know what else handles infrequent, small, latency-tolerant messages between anonymous parties? Nostr relays.

nostrain publishes pseudo-gradients as kind:33333 Nostr events. Workers subscribe by run name. Round synchronization happens via event tags. Fault tolerance comes free — DiLoCo tolerates 30% gradient drop, and Nostr relays can fail without stopping training.

The compression stack: top-k sparsification (1%) + int8 quantization + zstd. A 150M model's gradients go from 600MB to ~10MB per round.

Result: globally distributed ML training on public infrastructure that already exists. No new servers needed.

pip install nostrain
github.com/AbdelStark/nostrain


## Posting Strategy

1. Post on Twitter first (main audience: ML + crypto intersection)
2. Cross-post on Nostr immediately (the Nostr community will share it reflexively)
3. Submit to Hacker News (title: "nostrain — Distributed ML training over Nostr relays")
4. Post on r/MachineLearning and r/nostr
5. Tag in tweet: @fabordes (DiLoCo author), @jlopp (Bitcoin/Nostr), @jack (Nostr investor), grove author
6. Include a 30-second screen recording of two terminals training together

## Key Viral Mechanics

- **grove parallel**: People who saw grove go viral will immediately understand the concept
- **Two-community bridge**: ML people learn about Nostr, Nostr people learn about ML training
- **Bitcoin/freedom tech signal**: The cypherpunk crowd shares anything that makes censorship harder
- **Tryable in 2 min**: pip install + 2 commands. Low barrier to "holy shit this actually works"
- **Observable**: Anyone can subscribe to a relay and WATCH training happen. This is the demo.
- **Absurd but real**: "I trained a neural network using the protocol behind Nostr" is inherently shareable
