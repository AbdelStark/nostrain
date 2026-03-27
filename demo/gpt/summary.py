#!/usr/bin/env python3
"""Display visual summary for the GPT demo run."""

import json
import sys
import time
from pathlib import Path

NAMES = ["alice", "bob", "carol", "dave"]
COLORS = ["31", "32", "33", "34"]


def bar(value: float, max_val: float, width: int = 25) -> str:
    filled = int(min(value / max_val, 1.0) * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def wait_for_results(result_dir: Path, n: int, timeout: float = 600.0) -> list[dict]:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        files = list(result_dir.glob("*_gpt_result.json"))
        if len(files) >= n:
            return [json.loads(f.read_text()) for f in sorted(files)]
        time.sleep(2)
    return [json.loads(f.read_text()) for f in sorted(result_dir.glob("*_gpt_result.json"))]


def main():
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/artifacts")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print("\033[2J\033[H", end="", flush=True)
    print("\033[1;36m")
    print("  \u2554" + "\u2550" * 60 + "\u2557")
    print("  \u2551  NOSTRAIN GPT DEMO \u2014 Shakespeare over Nostr              \u2551")
    print("  \u255a" + "\u2550" * 60 + "\u255d")
    print("\033[0m")
    print("  \033[90mWaiting for all workers to finish...\033[0m", flush=True)

    results = wait_for_results(result_dir, n)
    if not results:
        print("\n  \033[1;31mNo results found.\033[0m")
        return 1

    name_order = {name: i for i, name in enumerate(NAMES)}
    results.sort(key=lambda r: name_order.get(r["name"], 99))
    max_initial = max(r["initial_loss"] for r in results)

    print()
    print("  \033[1mLoss Reduction\033[0m")
    print("  " + "\u2500" * 60)
    print()

    for r in results:
        idx = NAMES.index(r["name"]) if r["name"] in NAMES else 0
        color = COLORS[idx]
        il, fl = r["initial_loss"], r["final_loss"]
        reduction = (1 - fl / il) * 100 if il > 0 else 0
        print(f"  \033[1;{color}m{r['name']:>5}\033[0m  initial  {bar(il, max_initial)} {il:.3f}")
        print(f"         final    {bar(fl, max_initial)} {fl:.3f}  \033[1;32m-{reduction:.0f}%\033[0m")
        print()

    print("  \033[1mGenerated Text Samples\033[0m")
    print("  " + "\u2500" * 60)
    print()

    for r in results:
        idx = NAMES.index(r["name"]) if r["name"] in NAMES else 0
        color = COLORS[idx]
        sample = r.get("sample", "")
        print(f"  \033[1;{color}m{r['name']}\033[0m:")
        words = sample.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        if line.strip():
            print(line)
        print()

    print("  \033[90mTrained on different Shakespeare text shards — no worker saw the full corpus.\033[0m")
    print("  \033[90mGradients exchanged as Schnorr-signed Nostr events through a local relay.\033[0m")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
