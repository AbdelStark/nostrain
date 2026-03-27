#!/usr/bin/env python3
"""Generate 4 data shards for the nostrain demo.

True function: y = 3.0*x1 - 1.5*x2 + 0.5*x3 + 1.0
Each worker gets a non-overlapping shard with noise, so no single worker
can learn the full function alone — they need to collaborate.
"""

import json
import math
import os
import sys

FEATURES = 3
TRUE_WEIGHTS = [3.0, -1.5, 0.5]
TRUE_BIAS = 1.0
EXAMPLES_PER_SHARD = 32
NOISE_SCALE = 0.3
NUM_SHARDS = 4
SEED = 42


def lcg(state):
    """Simple LCG PRNG — no numpy needed."""
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    return state, (state / 0x7FFFFFFF)


def generate_shard(shard_id, seed):
    """Each shard samples from a different region of input space."""
    examples = []
    state = seed + shard_id * 10000

    # Each shard covers a different quadrant-ish region
    offsets = [
        (-2.0, 0.0),  # shard 0: negative x1
        (0.0, 2.0),   # shard 1: positive x1
        (-1.0, 1.0),  # shard 2: mixed
        (1.0, 3.0),   # shard 3: high x1
    ]
    lo, hi = offsets[shard_id]

    for _ in range(EXAMPLES_PER_SHARD):
        inputs = []
        for f in range(FEATURES):
            state, r = lcg(state)
            inputs.append(round(lo + r * (hi - lo), 4))

        target = TRUE_BIAS
        for w, x in zip(TRUE_WEIGHTS, inputs):
            target += w * x

        # Add noise
        state, r1 = lcg(state)
        state, r2 = lcg(state)
        # Box-Muller for normal noise
        noise = math.sqrt(-2.0 * math.log(max(r1, 1e-10))) * math.cos(2.0 * math.pi * r2)
        target = round(target + noise * NOISE_SCALE, 4)

        examples.append({"inputs": inputs, "target": target})

    return {"task": "linear-regression", "examples": examples}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "demo"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(NUM_SHARDS):
        shard = generate_shard(i, SEED)
        path = os.path.join(out_dir, f"shard_{i + 1}.json")
        with open(path, "w") as f:
            json.dump(shard, f, indent=2)
        print(f"  Shard {i + 1}: {len(shard['examples'])} examples -> {path}")

    print(f"\n  True function: y = {TRUE_BIAS}", end="")
    for i, w in enumerate(TRUE_WEIGHTS):
        sign = "+" if w >= 0 else "-"
        print(f" {sign} {abs(w)}*x{i + 1}", end="")
    print()


if __name__ == "__main__":
    main()
