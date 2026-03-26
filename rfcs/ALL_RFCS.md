# RFC-001: DiLoCo Training Loop

Inner optimizer: AdamW (lr=4e-4, weight_decay=0.1, warmup=1000 steps). Outer optimizer: Nesterov SGD (lr=0.7, momentum=0.9). Inner steps H=500 (configurable). Each worker maintains theta (current weights) and theta_initial (snapshot at start of round). Pseudo-gradient = theta - theta_initial. After outer step, all workers have the same theta and begin next round. Reference implementation: OpenDiLoCo by Prime Intellect.

# RFC-002: Nostr Event Protocol

Kind 33333 (parameterized replaceable). Workers publish one event per outer round. Tags encode run metadata (name, round, worker pubkey, model hash, compression method). Content is base64-encoded compressed pseudo-gradients. Workers subscribe with filter: kinds=[33333], #t=["nostrain"], #run=[run_name], #round=[current_round]. Round completion: worker waits until it has events from N-1 other workers (or timeout). Multi-relay: publish to all configured relays, subscribe from all, deduplicate by event ID.

# RFC-003: Compression Pipeline

Step 1 — Top-k sparsification: compute magnitudes of all gradient values, keep top k% (default 1%), zero the rest. Store as sparse tensor (indices as uint32 + values as float32). Step 2 — Int8 quantization: find max absolute value, compute scale = max_abs / 127, quantize values to int8, store scale as float32 header. Step 3 — zstd compression: compress the byte buffer (indices + quantized values + scale header). Step 4 — Base64 encode for Nostr event content field. Decompression is the exact reverse. Expected ratio: 50-200x for typical models.

# RFC-004: Worker Discovery and Synchronization

Discovery: workers publish a "heartbeat" event (kind 33334) every 60 seconds with worker metadata (pubkey, capabilities, current round). Other workers subscribe to heartbeats to know who is active. Round sync: configurable strategy. "strict" — wait for all known workers. "quorum" — wait for >50% of known workers. "async" — don't wait, use whatever arrived, apply outer step immediately. "timeout" — wait up to T seconds, then proceed with whatever arrived. Default: timeout with T=120s.

# RFC-005: Fault Tolerance

Worker crash: other workers detect via missing heartbeat (>3 missed = considered offline). Training continues with remaining workers. Crashed worker can rejoin by: (1) downloading latest model checkpoint from relay, (2) joining the current round. Relay failure: if publishing fails, retry with exponential backoff. If configured with multiple relays, failover to next relay. Stale gradients: if a worker's gradient arrives late (after outer step already applied), it is discarded. Consistency: since DiLoCo is robust to dropped gradients (paper shows tolerance up to 30% drop rate), occasional message loss does not break training.

# RFC-006: CLI and User Interface

Commands: `start` (create run, begin training), `join` (join existing run), `monitor` (read-only dashboard), `list` (show active runs on relay). Training script interface: user provides a standard PyTorch training script that defines model, dataset, and loss function. nostrain wraps it with the DiLoCo outer loop and Nostr communication. Rich terminal UI shows: current round, loss, number of active workers, bytes published/received, compression ratio, time per round.
