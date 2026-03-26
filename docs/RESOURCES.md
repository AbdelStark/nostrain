# nostrain: Resources

## Core Papers
| Paper | Why |
|-------|-----|
| DiLoCo (arXiv:2311.08105) | Core algorithm. 500x communication reduction via inner/outer optimization. |
| OpenDiLoCo (Prime Intellect) | Open source implementation. Trained 1B model across continents. |
| Streaming DiLoCo (arXiv:2501.18512) | Reduced peak bandwidth via streamed gradient chunks. |
| SPARTA + DiLoCo (ICLR 2025) | Sparse parameter communication. Extends sync interval 100x. |

## Nostr Protocol
| Resource | URL |
|----------|-----|
| NIP-01 (basic protocol) | https://github.com/nostr-protocol/nips/blob/master/01.md |
| nostr-sdk (Rust + Python) | https://github.com/rust-nostr/nostr |
| Public relay list | https://nostr.watch |

## Inspiration
| Resource | URL |
|----------|-----|
| grove (AirDrop training) | https://github.com/swarnim-j/grove |
| grove architecture blog | https://swarnimjain.com/grove |
| Petals (distributed inference) | https://github.com/bigscience-workshop/petals |
| Hivemind (decentralized training) | https://github.com/learning-at-home/hivemind |

## Python Dependencies
torch >= 2.0, nostr-sdk >= 0.30, zstandard >= 0.22, click >= 8.0, rich >= 13.0
