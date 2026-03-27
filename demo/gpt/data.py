"""Shakespeare dataset loading, tokenization, and sharding for the nostrain GPT demo.

Uses the tiny-shakespeare dataset (~1MB). Characters are mapped to a printable ASCII
vocabulary (ordinals 32-127, 96 tokens). Text is split into non-overlapping sections
so each worker trains on different parts of the corpus.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"

# Printable ASCII: space (32) through tilde (126) = 95 chars, plus newline = 96
VOCAB_START = 32
VOCAB_END = 127
VOCAB_SIZE = VOCAB_END - VOCAB_START


def encode(text: str) -> list[int]:
    """Encode text to token indices (printable ASCII range)."""
    return [min(max(ord(c) - VOCAB_START, 0), VOCAB_SIZE - 1) for c in text]


def decode(tokens: list[int] | torch.Tensor) -> str:
    """Decode token indices back to text."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return "".join(chr(t + VOCAB_START) for t in tokens)


def download_shakespeare() -> str:
    """Download tiny-shakespeare if not cached, return text."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SHAKESPEARE_PATH.exists():
        print(f"  Downloading tiny-shakespeare...", flush=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
    text = SHAKESPEARE_PATH.read_text(encoding="utf-8")
    # Filter to printable ASCII
    text = "".join(c if VOCAB_START <= ord(c) < VOCAB_END else " " for c in text)
    return text


def make_shards(text: str, num_shards: int = 4) -> list[str]:
    """Split text into non-overlapping contiguous sections."""
    shard_size = len(text) // num_shards
    return [text[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]


class ShardDataset:
    """Character-level dataset for one worker's text shard."""

    def __init__(self, text: str, block_size: int = 128):
        self.tokens = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size - 1)

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) pairs."""
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([self.tokens[i : i + self.block_size] for i in ix])
        y = torch.stack([self.tokens[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y
