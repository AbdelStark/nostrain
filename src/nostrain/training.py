from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aggregation import aggregate_deltas, nesterov_outer_step
from .compression import CompressionCodec, compress_delta, decompress_payload
from .model import (
    ModelState,
    add_states,
    compute_delta,
    state_digest,
    zeros_like,
)
from .protocol import (
    CheckpointEventMetadata,
    GradientEventMetadata,
    HeartbeatEventMetadata,
    build_checkpoint_event,
    build_gradient_event,
    build_heartbeat_event,
)
from .relay import (
    collect_gradient_events_across_relays,
    collect_late_gradient_events_across_relays,
    publish_nostrain_events,
)
from .retry import RelayRetryPolicy
from .runtime import (
    DEFAULT_TRAINING_BACKEND,
    DEFAULT_TRAINING_RUNTIME,
    LINEAR_BIAS_PARAMETER,
    LINEAR_REGRESSION_RUNTIME,
    LINEAR_WEIGHT_PARAMETER,
    MLP_HIDDEN_BIAS_PARAMETER,
    MLP_HIDDEN_WEIGHT_PARAMETER,
    MLP_OUTPUT_BIAS_PARAMETER,
    MLP_OUTPUT_WEIGHT_PARAMETER,
    MLP_REGRESSION_RUNTIME,
    NUMPY_TRAINING_BACKEND,
    PYTHON_TRAINING_BACKEND,
    SUPPORTED_TRAINING_BACKENDS,
    SUPPORTED_TRAINING_RUNTIMES,
    TORCH_TRAINING_BACKEND,
    LinearModel,
    LinearModelAdapter,
    LinearRegressionDataset,
    LocalTrainingConfig,
    LocalTrainingResult,
    MLPModel,
    MLPModelAdapter,
    RegressionDataset,
    RegressionExample,
    RuntimeAdapter,
    evaluate_linear_regression,
    evaluate_mlp_regression,
    evaluate_regression,
    infer_training_runtime_from_state,
    initialize_linear_regression_state,
    initialize_mlp_regression_state,
    initialize_training_state,
    resolve_training_backend,
    resolve_training_runtime,
    train_linear_regression,
    train_mlp_regression,
    train_regression,
)

LATE_GRADIENT_STRATEGIES = {"discard", "deferred"}

__all__ = [
    "DEFAULT_TRAINING_BACKEND",
    "DEFAULT_TRAINING_RUNTIME",
    "LATE_GRADIENT_STRATEGIES",
    "LINEAR_BIAS_PARAMETER",
    "LINEAR_REGRESSION_RUNTIME",
    "LINEAR_WEIGHT_PARAMETER",
    "LateGradientRecord",
    "LateGradientReconciliationSummary",
    "LocalTrainingConfig",
    "LocalTrainingResult",
    "LinearModel",
    "LinearModelAdapter",
    "LinearRegressionDataset",
    "MLP_HIDDEN_BIAS_PARAMETER",
    "MLP_HIDDEN_WEIGHT_PARAMETER",
    "MLP_OUTPUT_BIAS_PARAMETER",
    "MLP_OUTPUT_WEIGHT_PARAMETER",
    "MLP_REGRESSION_RUNTIME",
    "MLPModel",
    "MLPModelAdapter",
    "NUMPY_TRAINING_BACKEND",
    "PYTHON_TRAINING_BACKEND",
    "RegressionDataset",
    "RegressionExample",
    "SUPPORTED_TRAINING_BACKENDS",
    "SUPPORTED_TRAINING_RUNTIMES",
    "TORCH_TRAINING_BACKEND",
    "TrainingCheckpoint",
    "TrainingRoundSummary",
    "TrainingSessionResult",
    "TrainingWorkerConfig",
    "evaluate_linear_regression",
    "evaluate_mlp_regression",
    "evaluate_regression",
    "infer_training_runtime_from_state",
    "initialize_linear_regression_state",
    "initialize_mlp_regression_state",
    "initialize_training_state",
    "resolve_training_backend",
    "resolve_training_runtime",
    "run_training_session",
    "train_linear_regression",
    "train_mlp_regression",
    "train_regression",
]


@dataclass(frozen=True)
class TrainingWorkerConfig:
    run_name: str
    relay_urls: tuple[str, ...]
    worker_id: str
    secret_key_hex: str
    runtime_name: str | None = None
    backend_name: str = DEFAULT_TRAINING_BACKEND
    rounds: int = 1
    start_round: int = 0
    inner_steps: int = 500
    local_learning_rate: float = 0.01
    batch_size: int = 1
    topk_ratio: float = 1.0
    codec: CompressionCodec = CompressionCodec.ZLIB
    outer_learning_rate: float = 0.7
    outer_momentum: float = 0.9
    round_timeout: float = 2.0
    open_timeout: float = 10.0
    heartbeat_interval: int = 60
    max_missed_heartbeats: int = 3
    late_gradient_timeout: float = 0.2
    late_gradient_strategy: str = "deferred"
    late_reconciliation_learning_rate: float | None = None
    late_reconciliation_momentum: float | None = None
    advertised_relays: tuple[str, ...] = ()
    checkpoint_history: int = 4
    artifact_retention_rounds: int | None = None
    relay_retry_policy: RelayRetryPolicy = field(default_factory=RelayRetryPolicy)

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run name cannot be empty")
        if not self.worker_id:
            raise ValueError("worker id cannot be empty")
        if not self.secret_key_hex:
            raise ValueError("secret key cannot be empty")
        if self.runtime_name is not None and self.runtime_name not in SUPPORTED_TRAINING_RUNTIMES:
            raise ValueError(
                "runtime must be one of: " + ", ".join(SUPPORTED_TRAINING_RUNTIMES)
            )
        if self.backend_name not in SUPPORTED_TRAINING_BACKENDS:
            raise ValueError(
                "backend must be one of: " + ", ".join(SUPPORTED_TRAINING_BACKENDS)
            )
        if self.rounds <= 0:
            raise ValueError("round count must be positive")
        if self.start_round < 0:
            raise ValueError("start round must be non-negative")
        if self.inner_steps <= 0:
            raise ValueError("inner step count must be positive")
        if self.local_learning_rate <= 0:
            raise ValueError("local learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not 0 < self.topk_ratio <= 1:
            raise ValueError("top-k ratio must be within (0, 1]")
        if self.outer_learning_rate <= 0:
            raise ValueError("outer learning rate must be positive")
        if not 0 <= self.outer_momentum < 1:
            raise ValueError("outer momentum must be within [0, 1)")
        if self.round_timeout <= 0:
            raise ValueError("round timeout must be positive")
        if self.open_timeout <= 0:
            raise ValueError("open timeout must be positive")
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        if self.max_missed_heartbeats <= 0:
            raise ValueError("max missed heartbeats must be positive")
        if self.late_gradient_timeout <= 0:
            raise ValueError("late gradient timeout must be positive")
        if self.late_gradient_strategy not in LATE_GRADIENT_STRATEGIES:
            raise ValueError(
                "late gradient strategy must be one of: "
                + ", ".join(sorted(LATE_GRADIENT_STRATEGIES))
            )
        if (
            self.late_reconciliation_learning_rate is not None
            and self.late_reconciliation_learning_rate <= 0
        ):
            raise ValueError("late reconciliation learning rate must be positive when provided")
        if (
            self.late_reconciliation_momentum is not None
            and not 0 <= self.late_reconciliation_momentum < 1
        ):
            raise ValueError("late reconciliation momentum must be within [0, 1)")
        if self.checkpoint_history <= 0:
            raise ValueError("checkpoint history must be positive")
        if (
            self.artifact_retention_rounds is not None
            and self.artifact_retention_rounds <= 0
        ):
            raise ValueError("artifact retention rounds must be positive when provided")
        if not isinstance(self.relay_retry_policy, RelayRetryPolicy):
            raise ValueError("relay_retry_policy must be a RelayRetryPolicy instance")
        normalized_relays = tuple(
            dict.fromkeys(str(relay).strip() for relay in self.relay_urls if str(relay).strip())
        )
        if not normalized_relays:
            raise ValueError("at least one relay URL is required")
        if self.runtime_name is not None:
            object.__setattr__(self, "runtime_name", resolve_training_runtime(self.runtime_name))
        object.__setattr__(self, "backend_name", resolve_training_backend(self.backend_name))
        object.__setattr__(self, "relay_urls", normalized_relays)
        object.__setattr__(
            self,
            "advertised_relays",
            tuple(dict.fromkeys(str(relay).strip() for relay in self.advertised_relays if str(relay).strip())),
        )

    @property
    def relay_url(self) -> str:
        return self.relay_urls[0]

    @property
    def effective_late_reconciliation_learning_rate(self) -> float:
        if self.late_reconciliation_learning_rate is None:
            return self.outer_learning_rate
        return self.late_reconciliation_learning_rate

    @property
    def effective_late_reconciliation_momentum(self) -> float:
        if self.late_reconciliation_momentum is None:
            return self.outer_momentum
        return self.late_reconciliation_momentum


@dataclass(frozen=True)
class LateGradientRecord:
    round_index: int
    worker_id: str
    event_id: str
    created_at: int
    model_hash: str
    payload: str | None = None
    reconciliation_round: int | None = None
    reconciliation_model_hash_before: str | None = None
    reconciliation_model_hash_after: str | None = None
    reconciliation_error: str | None = None

    @property
    def is_reconciled(self) -> bool:
        return self.reconciliation_round is not None

    @property
    def is_reconcilable(self) -> bool:
        return (
            self.payload is not None
            and self.reconciliation_round is None
            and self.reconciliation_error is None
        )

    def to_json_obj(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "round": self.round_index,
            "worker": self.worker_id,
            "event_id": self.event_id,
            "created_at": self.created_at,
            "model_hash": self.model_hash,
        }
        if self.payload is not None:
            data["payload"] = self.payload
        if self.reconciliation_round is not None:
            data["reconciliation_round"] = self.reconciliation_round
        if self.reconciliation_model_hash_before is not None:
            data["reconciliation_model_hash_before"] = self.reconciliation_model_hash_before
        if self.reconciliation_model_hash_after is not None:
            data["reconciliation_model_hash_after"] = self.reconciliation_model_hash_after
        if self.reconciliation_error is not None:
            data["reconciliation_error"] = self.reconciliation_error
        return data

    @classmethod
    def from_json_obj(cls, data: Any) -> "LateGradientRecord":
        if not isinstance(data, dict):
            raise ValueError("late gradient record JSON must be an object")
        return cls(
            round_index=int(data["round"]),
            worker_id=str(data["worker"]),
            event_id=str(data["event_id"]),
            created_at=int(data["created_at"]),
            model_hash=str(data["model_hash"]),
            payload=str(data["payload"]) if data.get("payload") is not None else None,
            reconciliation_round=(
                int(data["reconciliation_round"])
                if data.get("reconciliation_round") is not None
                else None
            ),
            reconciliation_model_hash_before=(
                str(data["reconciliation_model_hash_before"])
                if data.get("reconciliation_model_hash_before") is not None
                else None
            ),
            reconciliation_model_hash_after=(
                str(data["reconciliation_model_hash_after"])
                if data.get("reconciliation_model_hash_after") is not None
                else None
            ),
            reconciliation_error=(
                str(data["reconciliation_error"])
                if data.get("reconciliation_error") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class LateGradientReconciliationSummary:
    current_round: int
    event_count: int
    worker_ids: tuple[str, ...]
    late_rounds: tuple[int, ...]
    event_ids: tuple[str, ...]
    learning_rate: float
    momentum: float
    model_hash_before: str
    model_hash_after: str
    applied_at: int

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "current_round": self.current_round,
            "event_count": self.event_count,
            "workers": list(self.worker_ids),
            "late_rounds": list(self.late_rounds),
            "event_ids": list(self.event_ids),
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "model_hash_before": self.model_hash_before,
            "model_hash_after": self.model_hash_after,
            "applied_at": self.applied_at,
        }

    @classmethod
    def from_json_obj(cls, data: Any) -> "LateGradientReconciliationSummary":
        if not isinstance(data, dict):
            raise ValueError("late gradient reconciliation JSON must be an object")
        return cls(
            current_round=int(data["current_round"]),
            event_count=int(data["event_count"]),
            worker_ids=tuple(str(worker) for worker in data.get("workers", [])),
            late_rounds=tuple(int(round_index) for round_index in data.get("late_rounds", [])),
            event_ids=tuple(str(event_id) for event_id in data.get("event_ids", [])),
            learning_rate=float(data["learning_rate"]),
            momentum=float(data["momentum"]),
            model_hash_before=str(data["model_hash_before"]),
            model_hash_after=str(data["model_hash_after"]),
            applied_at=int(data["applied_at"]),
        )


@dataclass(frozen=True)
class TrainingRoundSummary:
    round_index: int
    model_hash_before: str
    model_hash_after: str
    local_loss_before: float
    local_loss_after_inner: float
    local_loss_after_outer: float
    collected_event_count: int
    known_workers: tuple[str, ...]
    collected_workers: tuple[str, ...]
    completion_reason: str
    published_gradient_event_id: str
    published_heartbeat_event_id: str
    published_checkpoint_event_id: str
    configured_relays: tuple[str, ...]
    published_heartbeat_relays: tuple[str, ...]
    published_gradient_relays: tuple[str, ...]
    published_checkpoint_relays: tuple[str, ...]
    collected_from_relays: tuple[str, ...]
    failed_relays: tuple[str, ...]
    relay_retry_count: int = 0
    max_relay_attempt_count: int = 1
    retried_relays: tuple[str, ...] = ()
    reconciled_late_gradient_count: int = 0
    reconciled_late_workers: tuple[str, ...] = ()
    reconciled_late_rounds: tuple[int, ...] = ()
    late_reconciliation_model_hash_before: str = ""
    late_reconciliation_model_hash_after: str = ""
    late_reconciliation_error_count: int = 0

    @property
    def missing_workers(self) -> tuple[str, ...]:
        return tuple(worker for worker in self.known_workers if worker not in self.collected_workers)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "model_hash_before": self.model_hash_before,
            "model_hash_after": self.model_hash_after,
            "local_loss_before": self.local_loss_before,
            "local_loss_after_inner": self.local_loss_after_inner,
            "local_loss_after_outer": self.local_loss_after_outer,
            "collected_event_count": self.collected_event_count,
            "known_workers": list(self.known_workers),
            "collected_workers": list(self.collected_workers),
            "missing_workers": list(self.missing_workers),
            "completion_reason": self.completion_reason,
            "published_gradient_event_id": self.published_gradient_event_id,
            "published_heartbeat_event_id": self.published_heartbeat_event_id,
            "published_checkpoint_event_id": self.published_checkpoint_event_id,
            "configured_relays": list(self.configured_relays),
            "published_heartbeat_relays": list(self.published_heartbeat_relays),
            "published_gradient_relays": list(self.published_gradient_relays),
            "published_checkpoint_relays": list(self.published_checkpoint_relays),
            "collected_from_relays": list(self.collected_from_relays),
            "failed_relays": list(self.failed_relays),
            "relay_retry_count": self.relay_retry_count,
            "max_relay_attempt_count": self.max_relay_attempt_count,
            "retried_relays": list(self.retried_relays),
            "reconciled_late_gradient_count": self.reconciled_late_gradient_count,
            "reconciled_late_workers": list(self.reconciled_late_workers),
            "reconciled_late_rounds": list(self.reconciled_late_rounds),
            "late_reconciliation_model_hash_before": self.late_reconciliation_model_hash_before,
            "late_reconciliation_model_hash_after": self.late_reconciliation_model_hash_after,
            "late_reconciliation_error_count": self.late_reconciliation_error_count,
        }

    @classmethod
    def from_json_obj(cls, data: Any) -> "TrainingRoundSummary":
        if not isinstance(data, dict):
            raise ValueError("training round summary JSON must be an object")
        return cls(
            round_index=int(data["round"]),
            model_hash_before=str(data["model_hash_before"]),
            model_hash_after=str(data["model_hash_after"]),
            local_loss_before=float(data["local_loss_before"]),
            local_loss_after_inner=float(data["local_loss_after_inner"]),
            local_loss_after_outer=float(data["local_loss_after_outer"]),
            collected_event_count=int(data["collected_event_count"]),
            known_workers=tuple(str(worker) for worker in data.get("known_workers", [])),
            collected_workers=tuple(str(worker) for worker in data.get("collected_workers", [])),
            completion_reason=str(data["completion_reason"]),
            published_gradient_event_id=str(data["published_gradient_event_id"]),
            published_heartbeat_event_id=str(data["published_heartbeat_event_id"]),
            published_checkpoint_event_id=str(data.get("published_checkpoint_event_id", "")),
            configured_relays=tuple(str(relay) for relay in data.get("configured_relays", [])),
            published_heartbeat_relays=tuple(
                str(relay) for relay in data.get("published_heartbeat_relays", [])
            ),
            published_gradient_relays=tuple(
                str(relay) for relay in data.get("published_gradient_relays", [])
            ),
            published_checkpoint_relays=tuple(
                str(relay) for relay in data.get("published_checkpoint_relays", [])
            ),
            collected_from_relays=tuple(
                str(relay) for relay in data.get("collected_from_relays", [])
            ),
            failed_relays=tuple(str(relay) for relay in data.get("failed_relays", [])),
            relay_retry_count=int(data.get("relay_retry_count", 0)),
            max_relay_attempt_count=int(data.get("max_relay_attempt_count", 1)),
            retried_relays=tuple(str(relay) for relay in data.get("retried_relays", [])),
            reconciled_late_gradient_count=int(data.get("reconciled_late_gradient_count", 0)),
            reconciled_late_workers=tuple(
                str(worker) for worker in data.get("reconciled_late_workers", [])
            ),
            reconciled_late_rounds=tuple(
                int(round_index) for round_index in data.get("reconciled_late_rounds", [])
            ),
            late_reconciliation_model_hash_before=str(
                data.get("late_reconciliation_model_hash_before", "")
            ),
            late_reconciliation_model_hash_after=str(
                data.get("late_reconciliation_model_hash_after", "")
            ),
            late_reconciliation_error_count=int(data.get("late_reconciliation_error_count", 0)),
        )


@dataclass(frozen=True)
class TrainingSessionResult:
    run_name: str
    worker_id: str
    relay_urls: tuple[str, ...]
    start_round: int
    rounds_completed: int
    final_state: ModelState
    final_momentum_state: ModelState | None
    rounds: tuple[TrainingRoundSummary, ...]
    runtime_name: str = DEFAULT_TRAINING_RUNTIME
    backend_name: str = DEFAULT_TRAINING_BACKEND
    checkpoint_history: int = 1
    artifact_retention_rounds: int | None = None
    late_gradients: tuple[LateGradientRecord, ...] = ()
    late_reconciliations: tuple[LateGradientReconciliationSummary, ...] = ()

    @property
    def final_model_hash(self) -> str:
        return state_digest(self.final_state)

    @property
    def reconciled_late_gradient_count(self) -> int:
        return sum(1 for record in self.late_gradients if record.is_reconciled)

    @property
    def pending_late_gradient_count(self) -> int:
        return sum(1 for record in self.late_gradients if record.is_reconcilable)

    @property
    def late_gradient_error_count(self) -> int:
        return sum(1 for record in self.late_gradients if record.reconciliation_error is not None)

    @property
    def relay_retry_count(self) -> int:
        return sum(round_summary.relay_retry_count for round_summary in self.rounds)

    @property
    def max_relay_attempt_count(self) -> int:
        if not self.rounds:
            return 1
        return max(round_summary.max_relay_attempt_count for round_summary in self.rounds)

    @property
    def retried_relays(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                relay
                for round_summary in self.rounds
                for relay in round_summary.retried_relays
            )
        )

    def to_json_obj(
        self,
        *,
        include_rounds: bool = True,
        include_final_state: bool = False,
        include_final_momentum: bool = False,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run": self.run_name,
            "worker": self.worker_id,
            "relays": list(self.relay_urls),
            "runtime": self.runtime_name,
            "backend": self.backend_name,
            "start_round": self.start_round,
            "rounds_completed": self.rounds_completed,
            "final_model_hash": self.final_model_hash,
            "checkpoint_history": self.checkpoint_history,
            "artifact_retention_rounds": self.artifact_retention_rounds,
            "late_gradient_count": len(self.late_gradients),
            "reconciled_late_gradient_count": self.reconciled_late_gradient_count,
            "pending_late_gradient_count": self.pending_late_gradient_count,
            "late_gradient_error_count": self.late_gradient_error_count,
            "relay_retry_count": self.relay_retry_count,
            "max_relay_attempt_count": self.max_relay_attempt_count,
            "retried_relays": list(self.retried_relays),
        }
        if include_rounds:
            data["rounds"] = [round_summary.to_json_obj() for round_summary in self.rounds]
        if self.late_gradients:
            data["late_gradients"] = [
                late_gradient.to_json_obj() for late_gradient in self.late_gradients
            ]
        if self.late_reconciliations:
            data["late_reconciliations"] = [
                reconciliation.to_json_obj() for reconciliation in self.late_reconciliations
            ]
        if include_final_state:
            data["final_state"] = self.final_state.to_json_obj()
        if include_final_momentum and self.final_momentum_state is not None:
            data["final_momentum_state"] = self.final_momentum_state.to_json_obj()
        return data


@dataclass(frozen=True)
class TrainingCheckpoint:
    run_name: str
    worker_id: str
    relay_urls: tuple[str, ...]
    next_round: int
    current_state: ModelState
    momentum_state: ModelState | None
    rounds: tuple[TrainingRoundSummary, ...]
    late_gradients: tuple[LateGradientRecord, ...]
    late_reconciliations: tuple[LateGradientReconciliationSummary, ...]
    updated_at: int
    late_gradient_since: int = 0
    runtime_name: str = DEFAULT_TRAINING_RUNTIME

    @property
    def rounds_completed(self) -> int:
        return len(self.rounds)

    @property
    def start_round(self) -> int:
        if self.rounds:
            return self.rounds[0].round_index
        return self.next_round

    def to_json_obj(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format": "nostrain-training-checkpoint",
            "version": 2,
            "run": self.run_name,
            "worker": self.worker_id,
            "relays": list(self.relay_urls),
            "runtime": self.runtime_name,
            "next_round": self.next_round,
            "rounds_completed": self.rounds_completed,
            "updated_at": self.updated_at,
            "current_model_hash": state_digest(self.current_state),
            "current_state": self.current_state.to_json_obj(),
            "rounds": [round_summary.to_json_obj() for round_summary in self.rounds],
            "late_gradients": [late_gradient.to_json_obj() for late_gradient in self.late_gradients],
            "late_reconciliations": [
                reconciliation.to_json_obj() for reconciliation in self.late_reconciliations
            ],
            "late_gradient_since": self.late_gradient_since,
        }
        if self.momentum_state is not None:
            data["momentum_state"] = self.momentum_state.to_json_obj()
        return data

    @classmethod
    def from_json_obj(cls, data: Any) -> "TrainingCheckpoint":
        if not isinstance(data, dict):
            raise ValueError("training checkpoint JSON must be an object")
        if str(data.get("format", "")).strip() != "nostrain-training-checkpoint":
            raise ValueError("unsupported training checkpoint format")
        current_state = ModelState.from_json_obj(data["current_state"])
        raw_momentum = data.get("momentum_state")
        momentum_state = ModelState.from_json_obj(raw_momentum) if raw_momentum is not None else None
        return cls(
            run_name=str(data["run"]),
            worker_id=str(data["worker"]),
            relay_urls=tuple(str(relay) for relay in data.get("relays", [])),
            next_round=int(data["next_round"]),
            current_state=current_state,
            momentum_state=momentum_state,
            rounds=tuple(
                TrainingRoundSummary.from_json_obj(round_data)
                for round_data in data.get("rounds", [])
            ),
            late_gradients=tuple(
                LateGradientRecord.from_json_obj(record)
                for record in data.get("late_gradients", [])
            ),
            late_reconciliations=tuple(
                LateGradientReconciliationSummary.from_json_obj(record)
                for record in data.get("late_reconciliations", [])
            ),
            late_gradient_since=int(data.get("late_gradient_since", 0)),
            updated_at=int(data.get("updated_at", 0)),
            runtime_name=resolve_training_runtime(
                str(data.get("runtime", DEFAULT_TRAINING_RUNTIME))
            ),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainingCheckpoint":
        return cls.from_json_obj(json.loads(Path(path).read_text(encoding="utf-8")))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_model_state(path: Path, state: ModelState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state.write_json(path)


def _format_failed_relays(failed_relays: tuple[Any, ...]) -> str:
    if not failed_relays:
        return "no relay failures recorded"
    return "; ".join(
        f"{failure.relay_url} ({failure.operation}, attempts={getattr(failure, 'attempt_count', 1)}): {failure.message}"
        for failure in failed_relays
    )


def _result_retry_count(result: Any) -> int:
    return int(getattr(result, "retry_count", max(0, int(getattr(result, "attempt_count", 1)) - 1)))


def _result_attempt_count(result: Any) -> int:
    return int(getattr(result, "attempt_count", 1))


def _retry_stats(*results: Any) -> tuple[int, int, tuple[str, ...]]:
    retry_count = 0
    max_attempt_count = 1
    retried_relays: list[str] = []

    for result in results:
        if result is None:
            continue
        retry_count += int(getattr(result, "total_retry_count", _result_retry_count(result)))
        max_attempt_count = max(
            max_attempt_count,
            int(getattr(result, "max_attempt_count", _result_attempt_count(result))),
        )
        for relay in getattr(result, "retried_relays", ()):
            if relay not in retried_relays:
                retried_relays.append(str(relay))

    return retry_count, max_attempt_count, tuple(retried_relays)


def _checkpoint_history_slot(round_index: int, checkpoint_history: int) -> int:
    return round_index % checkpoint_history


def _round_directory_index(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("round-"):
        return None
    suffix = path.name.removeprefix("round-")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _write_checkpoint_artifacts(
    artifact_root: Path,
    *,
    checkpoint: TrainingCheckpoint,
    checkpoint_event: Any,
    checkpoint_publish: Any,
    checkpoint_history: int,
) -> int:
    latest_round = checkpoint.next_round - 1
    history_slot = _checkpoint_history_slot(latest_round, checkpoint_history)
    checkpoints_dir = artifact_root / "checkpoints"
    slot_label = f"slot-{history_slot:04d}"

    _write_json(checkpoints_dir / "latest.json", checkpoint.to_json_obj())
    _write_json(checkpoints_dir / "latest-event.json", checkpoint_event.to_json_obj())
    _write_json(checkpoints_dir / "latest-publish.json", checkpoint_publish.to_json_obj())
    _write_json(checkpoints_dir / f"{slot_label}.json", checkpoint.to_json_obj())
    _write_json(checkpoints_dir / f"{slot_label}-event.json", checkpoint_event.to_json_obj())
    _write_json(checkpoints_dir / f"{slot_label}-publish.json", checkpoint_publish.to_json_obj())
    return history_slot


def _apply_artifact_retention(
    artifact_root: Path,
    *,
    checkpoint: TrainingCheckpoint,
    checkpoint_history: int,
    artifact_retention_rounds: int | None,
    latest_checkpoint_slot: int,
) -> None:
    round_directories = sorted(
        (
            (round_index, path)
            for path in artifact_root.iterdir()
            if (round_index := _round_directory_index(path)) is not None
        ),
        key=lambda item: item[0],
    )
    pruned_rounds: list[int] = []
    if artifact_retention_rounds is not None and len(round_directories) > artifact_retention_rounds:
        for round_index, path in round_directories[:-artifact_retention_rounds]:
            shutil.rmtree(path)
            pruned_rounds.append(round_index)
        round_directories = round_directories[-artifact_retention_rounds:]

    active_checkpoint_rounds = checkpoint.rounds[-checkpoint_history:]
    _write_json(
        artifact_root / "retention.json",
        {
            "checkpoint_history": checkpoint_history,
            "artifact_retention_rounds": artifact_retention_rounds,
            "latest_round": checkpoint.next_round - 1,
            "latest_checkpoint_slot": latest_checkpoint_slot,
            "retained_rounds": [round_index for round_index, _ in round_directories],
            "pruned_rounds": pruned_rounds,
            "checkpoint_slots": [
                {
                    "slot": _checkpoint_history_slot(round_summary.round_index, checkpoint_history),
                    "round": round_summary.round_index,
                    "next_round": round_summary.round_index + 1,
                    "model_hash_after": round_summary.model_hash_after,
                }
                for round_summary in active_checkpoint_rounds
            ],
        },
    )


def _merge_late_gradient_record(
    existing: LateGradientRecord,
    incoming: LateGradientRecord,
) -> LateGradientRecord:
    return LateGradientRecord(
        round_index=existing.round_index,
        worker_id=existing.worker_id,
        event_id=existing.event_id,
        created_at=max(existing.created_at, incoming.created_at),
        model_hash=existing.model_hash or incoming.model_hash,
        payload=existing.payload if existing.payload is not None else incoming.payload,
        reconciliation_round=(
            existing.reconciliation_round
            if existing.reconciliation_round is not None
            else incoming.reconciliation_round
        ),
        reconciliation_model_hash_before=(
            existing.reconciliation_model_hash_before
            if existing.reconciliation_model_hash_before is not None
            else incoming.reconciliation_model_hash_before
        ),
        reconciliation_model_hash_after=(
            existing.reconciliation_model_hash_after
            if existing.reconciliation_model_hash_after is not None
            else incoming.reconciliation_model_hash_after
        ),
        reconciliation_error=(
            existing.reconciliation_error
            if existing.reconciliation_error is not None
            else incoming.reconciliation_error
        ),
    )


def _merge_late_gradient_records(
    existing_records: list[LateGradientRecord],
    new_records: list[LateGradientRecord],
) -> list[LateGradientRecord]:
    merged: dict[str, LateGradientRecord] = {}
    order: list[str] = []
    for record in (*existing_records, *new_records):
        current = merged.get(record.event_id)
        if current is None:
            merged[record.event_id] = record
            order.append(record.event_id)
            continue
        merged[record.event_id] = _merge_late_gradient_record(current, record)
    return [merged[event_id] for event_id in order]


@dataclass(frozen=True)
class _LateGradientReconciliationOutcome:
    current_state: ModelState
    momentum_state: ModelState | None
    late_gradients: tuple[LateGradientRecord, ...]
    summary: LateGradientReconciliationSummary | None
    error_count: int


def _reconcile_late_gradients(
    current_state: ModelState,
    current_momentum: ModelState | None,
    late_gradients: list[LateGradientRecord],
    *,
    current_round: int,
    config: TrainingWorkerConfig,
) -> _LateGradientReconciliationOutcome:
    if config.late_gradient_strategy != "deferred":
        return _LateGradientReconciliationOutcome(
            current_state=current_state,
            momentum_state=current_momentum,
            late_gradients=tuple(late_gradients),
            summary=None,
            error_count=0,
        )

    pending_records = [record for record in late_gradients if record.is_reconcilable]
    if not pending_records:
        return _LateGradientReconciliationOutcome(
            current_state=current_state,
            momentum_state=current_momentum,
            late_gradients=tuple(late_gradients),
            summary=None,
            error_count=0,
        )

    updated_records = list(late_gradients)
    record_indexes = {
        record.event_id: index for index, record in enumerate(updated_records)
    }
    validated_records: list[LateGradientRecord] = []
    validated_deltas: list[ModelState] = []
    error_count = 0

    for record in pending_records:
        try:
            delta = decompress_payload(record.payload or "")
            add_states(zeros_like(current_state), delta)
        except Exception as exc:
            error_count += 1
            updated_records[record_indexes[record.event_id]] = LateGradientRecord(
                round_index=record.round_index,
                worker_id=record.worker_id,
                event_id=record.event_id,
                created_at=record.created_at,
                model_hash=record.model_hash,
                payload=record.payload,
                reconciliation_round=record.reconciliation_round,
                reconciliation_model_hash_before=record.reconciliation_model_hash_before,
                reconciliation_model_hash_after=record.reconciliation_model_hash_after,
                reconciliation_error=f"{type(exc).__name__}: {exc}",
            )
            continue
        validated_records.append(record)
        validated_deltas.append(delta)

    if not validated_records:
        return _LateGradientReconciliationOutcome(
            current_state=current_state,
            momentum_state=current_momentum,
            late_gradients=tuple(updated_records),
            summary=None,
            error_count=error_count,
        )

    learning_rate = config.effective_late_reconciliation_learning_rate
    momentum = config.effective_late_reconciliation_momentum
    model_hash_before = state_digest(current_state)
    outer_result = nesterov_outer_step(
        current_state,
        aggregate_deltas(validated_deltas),
        learning_rate=learning_rate,
        momentum=momentum,
        previous_momentum=current_momentum,
    )
    model_hash_after = state_digest(outer_result.next_state)

    for record in validated_records:
        updated_records[record_indexes[record.event_id]] = LateGradientRecord(
            round_index=record.round_index,
            worker_id=record.worker_id,
            event_id=record.event_id,
            created_at=record.created_at,
            model_hash=record.model_hash,
            payload=record.payload,
            reconciliation_round=current_round,
            reconciliation_model_hash_before=model_hash_before,
            reconciliation_model_hash_after=model_hash_after,
            reconciliation_error=None,
        )

    return _LateGradientReconciliationOutcome(
        current_state=outer_result.next_state,
        momentum_state=outer_result.momentum_state,
        late_gradients=tuple(updated_records),
        summary=LateGradientReconciliationSummary(
            current_round=current_round,
            event_count=len(validated_records),
            worker_ids=tuple(
                sorted({record.worker_id for record in validated_records})
            ),
            late_rounds=tuple(
                sorted({record.round_index for record in validated_records})
            ),
            event_ids=tuple(record.event_id for record in validated_records),
            learning_rate=learning_rate,
            momentum=momentum,
            model_hash_before=model_hash_before,
            model_hash_after=model_hash_after,
            applied_at=int(time.time()),
        ),
        error_count=error_count,
    )


async def run_training_session(
    initial_state: ModelState,
    dataset: RegressionDataset,
    *,
    config: TrainingWorkerConfig,
    adapter: RuntimeAdapter | None = None,
    previous_momentum: ModelState | None = None,
    artifact_dir: str | Path | None = None,
    prior_rounds: tuple[TrainingRoundSummary, ...] = (),
    prior_late_gradients: tuple[LateGradientRecord, ...] = (),
    prior_late_reconciliations: tuple[LateGradientReconciliationSummary, ...] = (),
    late_gradient_since: int | None = None,
    checkpoint_out: str | Path | None = None,
) -> TrainingSessionResult:
    runtime_name = resolve_training_runtime(
        config.runtime_name,
        dataset,
        state=initial_state,
        adapter=adapter,
    )
    backend_name = resolve_training_backend(config.backend_name)
    current_state = initial_state
    current_momentum = previous_momentum
    round_summaries: list[TrainingRoundSummary] = list(prior_rounds)
    late_gradients: list[LateGradientRecord] = list(prior_late_gradients)
    late_reconciliations: list[LateGradientReconciliationSummary] = list(
        prior_late_reconciliations
    )
    late_scan_since = late_gradient_since
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None

    for round_offset in range(config.rounds):
        round_index = config.start_round + round_offset
        round_start = int(time.time())
        round_dir = (
            artifact_root / f"round-{round_index:04d}" if artifact_root is not None else None
        )
        late_collection = None
        late_reconciliation = None
        late_reconciliation_error_count = 0

        if late_scan_since is not None:
            late_collection = await collect_late_gradient_events_across_relays(
                config.relay_urls,
                run_name=config.run_name,
                current_round=round_index,
                idle_timeout=min(config.late_gradient_timeout, config.round_timeout),
                open_timeout=config.open_timeout,
                since=late_scan_since + 1,
                retry_policy=config.relay_retry_policy,
            )
            late_gradients = _merge_late_gradient_records(
                late_gradients,
                [
                    LateGradientRecord(
                        round_index=event.parsed.metadata.round_index,
                        worker_id=event.parsed.metadata.worker_id,
                        event_id=event.event_id,
                        created_at=event.parsed.event.created_at,
                        model_hash=event.parsed.metadata.model_hash,
                        payload=event.parsed.event.content,
                    )
                    for event in late_collection.events
                ],
            )
            if round_dir is not None:
                _write_json(round_dir / "late-gradients.json", late_collection.to_json_obj())

        late_reconciliation = _reconcile_late_gradients(
            current_state,
            current_momentum,
            late_gradients,
            current_round=round_index,
            config=config,
        )
        current_state = late_reconciliation.current_state
        current_momentum = late_reconciliation.momentum_state
        late_gradients = list(late_reconciliation.late_gradients)
        late_reconciliation_error_count = late_reconciliation.error_count
        if late_reconciliation.summary is not None:
            late_reconciliations.append(late_reconciliation.summary)
            if round_dir is not None:
                _write_json(
                    round_dir / "late-reconciliation.json",
                    late_reconciliation.summary.to_json_obj(),
                )

        heartbeat_event = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name=config.run_name,
                worker_id=config.worker_id,
                current_round=round_index,
                heartbeat_interval=config.heartbeat_interval,
                capabilities=("gradient-event", runtime_name),
                advertised_relays=config.advertised_relays or config.relay_urls,
                created_at=round_start,
            ),
            secret_key_hex=config.secret_key_hex,
        )
        heartbeat_publish = await publish_nostrain_events(
            config.relay_urls,
            heartbeat_event,
            open_timeout=config.open_timeout,
            reply_timeout=config.open_timeout,
            retry_policy=config.relay_retry_policy,
        )
        if not heartbeat_publish.accepted:
            raise RuntimeError(
                "failed to publish heartbeat to any relay: "
                + _format_failed_relays(heartbeat_publish.failed_relays)
            )

        local_training = train_regression(
            current_state,
            dataset,
            config=LocalTrainingConfig(
                steps=config.inner_steps,
                learning_rate=config.local_learning_rate,
                batch_size=config.batch_size,
            ),
            runtime_name=runtime_name,
            adapter=adapter,
            backend_name=backend_name,
        )
        local_delta = compute_delta(current_state, local_training.trained_state)
        payload = compress_delta(local_delta, topk_ratio=config.topk_ratio, codec=config.codec)
        gradient_event = build_gradient_event(
            GradientEventMetadata(
                run_name=config.run_name,
                round_index=round_index,
                worker_id=config.worker_id,
                model_hash=state_digest(current_state),
                inner_steps=config.inner_steps,
                created_at=max(round_start, int(time.time())),
            ),
            payload,
            secret_key_hex=config.secret_key_hex,
        )
        gradient_publish = await publish_nostrain_events(
            config.relay_urls,
            gradient_event,
            open_timeout=config.open_timeout,
            reply_timeout=config.open_timeout,
            retry_policy=config.relay_retry_policy,
        )
        if not gradient_publish.accepted:
            raise RuntimeError(
                "failed to publish gradient to any relay: "
                + _format_failed_relays(gradient_publish.failed_relays)
            )

        collection = await collect_gradient_events_across_relays(
            config.relay_urls,
            run_name=config.run_name,
            round_index=round_index,
            idle_timeout=config.round_timeout,
            open_timeout=config.open_timeout,
            since=round_start - 1,
            strategy="timeout",
            discover_workers=True,
            heartbeat_idle_timeout=config.round_timeout,
            heartbeat_since=round_start - 1,
            max_missed_heartbeats=config.max_missed_heartbeats,
            retry_policy=config.relay_retry_policy,
        )
        if not collection.events:
            raise RuntimeError(
                f"round {round_index} completed without any collected gradient events across relays; "
                + _format_failed_relays(collection.failed_relays)
            )

        outer_result = nesterov_outer_step(
            current_state,
            collection.aggregate_delta(),
            learning_rate=config.outer_learning_rate,
            momentum=config.outer_momentum,
            previous_momentum=current_momentum,
        )
        local_loss_after_outer = evaluate_regression(
            outer_result.next_state,
            dataset,
            runtime_name=runtime_name,
            adapter=adapter,
            backend_name=backend_name,
        )

        if round_dir is not None:
            _write_model_state(round_dir / "base-state.json", current_state)
            _write_model_state(round_dir / "local-state.json", local_training.trained_state)
            _write_model_state(round_dir / "local-delta.json", local_delta)
            _write_json(round_dir / "payload.json", payload.to_json_obj())
            _write_json(round_dir / "heartbeat.json", heartbeat_event.to_json_obj())
            _write_json(round_dir / "heartbeat-publish.json", heartbeat_publish.to_json_obj())
            _write_json(round_dir / "event.json", gradient_event.to_json_obj())
            _write_json(round_dir / "event-publish.json", gradient_publish.to_json_obj())
            _write_json(round_dir / "collection.json", collection.to_json_obj())
            _write_model_state(round_dir / "aggregated-delta.json", outer_result.aggregated_delta)
            _write_model_state(round_dir / "next-state.json", outer_result.next_state)
            _write_model_state(round_dir / "momentum-state.json", outer_result.momentum_state)
            _write_json(round_dir / "local-training.json", local_training.to_json_obj())

        current_state = outer_result.next_state
        current_momentum = outer_result.momentum_state
        round_retry_count, round_max_attempt_count, round_retried_relays = _retry_stats(
            heartbeat_publish,
            gradient_publish,
            collection,
            late_collection,
        )

        checkpoint_round_summary = TrainingRoundSummary(
            round_index=round_index,
            model_hash_before=state_digest(local_training.initial_state),
            model_hash_after=state_digest(outer_result.next_state),
            local_loss_before=local_training.loss_before,
            local_loss_after_inner=local_training.loss_after,
            local_loss_after_outer=local_loss_after_outer,
            collected_event_count=len(collection.events),
            known_workers=collection.known_workers,
            collected_workers=collection.worker_ids,
            completion_reason=collection.completion_reason,
            published_gradient_event_id=gradient_publish.event_id,
            published_heartbeat_event_id=heartbeat_publish.event_id,
            published_checkpoint_event_id="",
            configured_relays=config.relay_urls,
            published_heartbeat_relays=heartbeat_publish.accepted_relays,
            published_gradient_relays=gradient_publish.accepted_relays,
            published_checkpoint_relays=(),
            collected_from_relays=collection.successful_relays,
            failed_relays=(),
            relay_retry_count=round_retry_count,
            max_relay_attempt_count=round_max_attempt_count,
            retried_relays=round_retried_relays,
            reconciled_late_gradient_count=(
                late_reconciliation.summary.event_count
                if late_reconciliation.summary is not None
                else 0
            ),
            reconciled_late_workers=(
                late_reconciliation.summary.worker_ids
                if late_reconciliation.summary is not None
                else ()
            ),
            reconciled_late_rounds=(
                late_reconciliation.summary.late_rounds
                if late_reconciliation.summary is not None
                else ()
            ),
            late_reconciliation_model_hash_before=(
                late_reconciliation.summary.model_hash_before
                if late_reconciliation.summary is not None
                else ""
            ),
            late_reconciliation_model_hash_after=(
                late_reconciliation.summary.model_hash_after
                if late_reconciliation.summary is not None
                else ""
            ),
            late_reconciliation_error_count=late_reconciliation_error_count,
        )
        checkpoint_rounds = tuple((*round_summaries, checkpoint_round_summary))
        checkpoint = TrainingCheckpoint(
            run_name=config.run_name,
            worker_id=config.worker_id,
            relay_urls=config.relay_urls,
            next_round=round_index + 1,
            current_state=current_state,
            momentum_state=current_momentum,
            rounds=checkpoint_rounds,
            late_gradients=tuple(late_gradients),
            late_reconciliations=tuple(late_reconciliations),
            updated_at=int(time.time()),
            late_gradient_since=max(
                value
                for value in (
                    late_scan_since,
                    gradient_event.created_at,
                    *(event.parsed.event.created_at for event in collection.events),
                    *(
                        event.parsed.event.created_at
                        for event in (late_collection.events if late_collection is not None else ())
                    ),
                )
                if value is not None
            ),
            runtime_name=runtime_name,
        )
        checkpoint_event = build_checkpoint_event(
            CheckpointEventMetadata(
                run_name=config.run_name,
                worker_id=config.worker_id,
                round_index=round_index,
                next_round=checkpoint.next_round,
                model_hash=state_digest(checkpoint.current_state),
                rounds_completed=checkpoint.rounds_completed,
                history_slot=_checkpoint_history_slot(round_index, config.checkpoint_history),
                created_at=checkpoint.updated_at,
            ),
            checkpoint.to_json_obj(),
            secret_key_hex=config.secret_key_hex,
        )
        checkpoint_publish = await publish_nostrain_events(
            config.relay_urls,
            checkpoint_event,
            open_timeout=config.open_timeout,
            reply_timeout=config.open_timeout,
            retry_policy=config.relay_retry_policy,
        )
        final_round_retry_count, final_max_attempt_count, final_retried_relays = _retry_stats(
            heartbeat_publish,
            gradient_publish,
            collection,
            checkpoint_publish,
            late_collection,
        )

        failed_relay_urls = {
            failure.relay_url
            for failure in (
                *heartbeat_publish.failed_relays,
                *gradient_publish.failed_relays,
                *collection.failed_relays,
                *checkpoint_publish.failed_relays,
                *(late_collection.failed_relays if late_collection is not None else ()),
            )
        }

        round_summaries.append(
            TrainingRoundSummary(
                round_index=round_index,
                model_hash_before=state_digest(local_training.initial_state),
                model_hash_after=state_digest(outer_result.next_state),
                local_loss_before=local_training.loss_before,
                local_loss_after_inner=local_training.loss_after,
                local_loss_after_outer=local_loss_after_outer,
                collected_event_count=len(collection.events),
                known_workers=collection.known_workers,
                collected_workers=collection.worker_ids,
                completion_reason=collection.completion_reason,
                published_gradient_event_id=gradient_publish.event_id,
                published_heartbeat_event_id=heartbeat_publish.event_id,
                published_checkpoint_event_id=checkpoint_publish.event_id,
                configured_relays=config.relay_urls,
                published_heartbeat_relays=heartbeat_publish.accepted_relays,
                published_gradient_relays=gradient_publish.accepted_relays,
                published_checkpoint_relays=checkpoint_publish.accepted_relays,
                collected_from_relays=collection.successful_relays,
                failed_relays=tuple(
                    relay_url for relay_url in config.relay_urls if relay_url in failed_relay_urls
                ),
                relay_retry_count=final_round_retry_count,
                max_relay_attempt_count=final_max_attempt_count,
                retried_relays=final_retried_relays,
                reconciled_late_gradient_count=(
                    late_reconciliation.summary.event_count
                    if late_reconciliation.summary is not None
                    else 0
                ),
                reconciled_late_workers=(
                    late_reconciliation.summary.worker_ids
                    if late_reconciliation.summary is not None
                    else ()
                ),
                reconciled_late_rounds=(
                    late_reconciliation.summary.late_rounds
                    if late_reconciliation.summary is not None
                    else ()
                ),
                late_reconciliation_model_hash_before=(
                    late_reconciliation.summary.model_hash_before
                    if late_reconciliation.summary is not None
                    else ""
                ),
                late_reconciliation_model_hash_after=(
                    late_reconciliation.summary.model_hash_after
                    if late_reconciliation.summary is not None
                    else ""
                ),
                late_reconciliation_error_count=late_reconciliation_error_count,
            )
        )

        if artifact_root is not None:
            _write_json(artifact_root / "checkpoint.json", checkpoint.to_json_obj())
            _write_json(artifact_root / "checkpoint-event.json", checkpoint_event.to_json_obj())
            _write_json(round_dir / "checkpoint.json", checkpoint.to_json_obj())
            _write_json(round_dir / "checkpoint-event.json", checkpoint_event.to_json_obj())
            _write_json(round_dir / "checkpoint-publish.json", checkpoint_publish.to_json_obj())
            checkpoint_slot = _write_checkpoint_artifacts(
                artifact_root,
                checkpoint=checkpoint,
                checkpoint_event=checkpoint_event,
                checkpoint_publish=checkpoint_publish,
                checkpoint_history=config.checkpoint_history,
            )
            _apply_artifact_retention(
                artifact_root,
                checkpoint=checkpoint,
                checkpoint_history=config.checkpoint_history,
                artifact_retention_rounds=config.artifact_retention_rounds,
                latest_checkpoint_slot=checkpoint_slot,
            )
        if checkpoint_out is not None:
            _write_json(Path(checkpoint_out), checkpoint.to_json_obj())
        late_scan_since = checkpoint.late_gradient_since

    return TrainingSessionResult(
        run_name=config.run_name,
        worker_id=config.worker_id,
        relay_urls=config.relay_urls,
        start_round=prior_rounds[0].round_index if prior_rounds else config.start_round,
        rounds_completed=len(round_summaries),
        final_state=current_state,
        final_momentum_state=current_momentum,
        rounds=tuple(round_summaries),
        runtime_name=runtime_name,
        backend_name=backend_name,
        checkpoint_history=config.checkpoint_history,
        artifact_retention_rounds=config.artifact_retention_rounds,
        late_gradients=tuple(late_gradients),
        late_reconciliations=tuple(late_reconciliations),
    )
