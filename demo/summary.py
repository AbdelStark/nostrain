#!/usr/bin/env python3
"""Display visual summary of a nostrain demo run.

Waits for all result files to appear, then prints a formatted summary.
Usage: python demo/summary.py <result_dir> <num_workers>
"""

import json
import sys
import time
from pathlib import Path

NAMES = ["alice", "bob", "carol", "dave"]
COLORS = ["31", "32", "33", "34"]
TRUE_WEIGHTS = [3.0, -1.5, 0.5]
TRUE_BIAS = 1.0


def wait_for_results(result_dir: Path, n: int, timeout: float = 300.0) -> list[dict]:
    """Poll until all N result files exist."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        files = list(result_dir.glob("*_result.json"))
        if len(files) >= n:
            results = []
            for f in sorted(files):
                with open(f) as fh:
                    results.append(json.load(fh))
            return results
        time.sleep(1)
    # Return whatever we have
    results = []
    for f in sorted(result_dir.glob("*_result.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def bar(value: float, max_val: float, width: int = 30) -> str:
    """Render a horizontal bar."""
    filled = int(min(value / max_val, 1.0) * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def main():
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/artifacts")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print("\033[2J\033[H", end="", flush=True)  # clear screen

    print("\033[1;36m")
    print("  \u2554" + "\u2550" * 56 + "\u2557")
    print("  \u2551     NOSTRAIN DISTRIBUTED TRAINING — RESULTS          \u2551")
    print("  \u255a" + "\u2550" * 56 + "\u255d")
    print("\033[0m")

    print("  \033[90mWaiting for all workers to finish...\033[0m", flush=True)
    results = wait_for_results(result_dir, n)

    if not results:
        print("\n  \033[1;31mNo results found.\033[0m")
        return 1

    # Sort by name order
    name_order = {name: i for i, name in enumerate(NAMES)}
    results.sort(key=lambda r: name_order.get(r["name"], 99))

    max_initial = max(r["initial_loss"] for r in results)

    print()
    print("  \033[1mLoss Reduction\033[0m")
    print("  " + "\u2500" * 56)
    print()

    for r in results:
        idx = NAMES.index(r["name"]) if r["name"] in NAMES else 0
        color = COLORS[idx]
        name = r["name"]
        il = r["initial_loss"]
        fl = r["final_loss"]
        reduction = (1 - fl / il) * 100 if il > 0 else 0

        before_bar = bar(il, max_initial, 20)
        after_bar = bar(fl, max_initial, 20)

        print(f"  \033[1;{color}m{name:>5}\033[0m  initial  {before_bar} {il:.4f}")
        print(f"         final    {after_bar} {fl:.4f}  \033[1;32m-{reduction:.0f}%\033[0m")
        print()

    # Learned parameters vs truth
    print("  \033[1mLearned Parameters\033[0m")
    print("  " + "\u2500" * 56)
    print()
    print(f"  \033[90m{'':>5}  {'bias':>8}    {'w1':>8}  {'w2':>8}  {'w3':>8}    err\033[0m")

    for r in results:
        idx = NAMES.index(r["name"]) if r["name"] in NAMES else 0
        color = COLORS[idx]
        name = r["name"]
        params = r["parameters"]
        bias = params.get("linear.bias", [0])[0]
        weights = params.get("linear.weight", [0, 0, 0])

        w_err = sum((a - b) ** 2 for a, b in zip(weights, TRUE_WEIGHTS)) ** 0.5
        b_err = abs(bias - TRUE_BIAS)
        total_err = (w_err ** 2 + b_err ** 2) ** 0.5

        status = "\033[1;32m\u2713\033[0m" if total_err < 1.0 else "\033[1;31m\u2717\033[0m"
        print(
            f"  \033[1;{color}m{name:>5}\033[0m  "
            f"{bias:>8.4f}    {weights[0]:>8.4f}  {weights[1]:>8.4f}  {weights[2]:>8.4f}"
            f"    {total_err:.4f} {status}"
        )

    print(
        f"  \033[1m{'TRUE':>5}\033[0m  "
        f"{TRUE_BIAS:>8.4f}    {TRUE_WEIGHTS[0]:>8.4f}  {TRUE_WEIGHTS[1]:>8.4f}  {TRUE_WEIGHTS[2]:>8.4f}"
    )

    # Convergence check
    all_same = True
    if len(results) > 1:
        ref = results[0]["parameters"]
        for r in results[1:]:
            for key in ref:
                if r["parameters"].get(key) != ref[key]:
                    all_same = False
                    break

    print()
    if all_same and len(results) > 1:
        print("  \033[1;32m\u2713 All workers converged to the same model.\033[0m")
    else:
        print("  \033[1;33m~ Workers converged to different models (may need more rounds).\033[0m")

    all_pass = all(
        sum((a - b) ** 2 for a, b in zip(r["parameters"].get("linear.weight", []), TRUE_WEIGHTS)) ** 0.5 < 1.0
        for r in results
    )
    if all_pass:
        print("  \033[1;32m\u2713 All within tolerance of true function.\033[0m")

    print()
    print("  \033[90my = 3.0*x1 - 1.5*x2 + 0.5*x3 + 1.0\033[0m")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
