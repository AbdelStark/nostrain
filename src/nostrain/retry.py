from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelayRetryPolicy:
    max_attempts: int = 1
    initial_backoff: float = 0.0
    max_backoff: float = 0.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff < 0:
            raise ValueError("initial_backoff must be non-negative")
        if self.max_backoff < 0:
            raise ValueError("max_backoff must be non-negative")
        if self.max_backoff and self.max_backoff < self.initial_backoff:
            raise ValueError("max_backoff must be >= initial_backoff when provided")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")

    @property
    def retry_count(self) -> int:
        return max(0, self.max_attempts - 1)

    def delay_for_retry(self, retry_index: int) -> float:
        if retry_index <= 0:
            raise ValueError("retry_index must be positive")
        if self.retry_count == 0:
            return 0.0
        delay = self.initial_backoff * (self.backoff_multiplier ** (retry_index - 1))
        if self.max_backoff:
            delay = min(delay, self.max_backoff)
        return delay

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "retry_count": self.retry_count,
            "initial_backoff": self.initial_backoff,
            "max_backoff": self.max_backoff,
            "backoff_multiplier": self.backoff_multiplier,
        }
