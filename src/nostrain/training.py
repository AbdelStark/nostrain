from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from .aggregation import nesterov_outer_step
from .compression import CompressionCodec, compress_delta
from .model import ModelState, TensorState, compute_delta, state_digest
from .protocol import (
    GradientEventMetadata,
    HeartbeatEventMetadata,
    build_gradient_event,
    build_heartbeat_event,
)
from .relay import collect_gradient_events, publish_nostrain_event

LINEAR_WEIGHT_PARAMETER = "linear.weight"
LINEAR_BIAS_PARAMETER = "linear.bias"


@dataclass(frozen=True)
class RegressionExample:
    inputs: tuple[float, ...]
    target: float

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("regression examples must contain at least one input feature")

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "target": self.target,
        }


@dataclass(frozen=True)
class LinearRegressionDataset:
    examples: tuple[RegressionExample, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("linear regression datasets must contain at least one example")
        feature_count = len(self.examples[0].inputs)
        if feature_count <= 0:
            raise ValueError("linear regression datasets must contain at least one feature")
        for example in self.examples:
            if len(example.inputs) != feature_count:
                raise ValueError("all regression examples must use the same feature count")

    @property
    def feature_count(self) -> int:
        return len(self.examples[0].inputs)

    @property
    def example_count(self) -> int:
        return len(self.examples)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "task": "linear-regression",
            "examples": [example.to_json_obj() for example in self.examples],
        }

    @classmethod
    def from_json_obj(cls, data: Any) -> "LinearRegressionDataset":
        if not isinstance(data, dict):
            raise ValueError("dataset JSON must be an object")
        task = str(data.get("task", "linear-regression")).strip().lower()
        if task != "linear-regression":
            raise ValueError("only the 'linear-regression' dataset task is supported")
        raw_examples = data.get("examples")
        if not isinstance(raw_examples, list) or not raw_examples:
            raise ValueError("dataset JSON must contain a non-empty 'examples' array")

        examples: list[RegressionExample] = []
        for raw_example in raw_examples:
            if not isinstance(raw_example, dict):
                raise ValueError("dataset examples must be objects")
            if "inputs" not in raw_example or "target" not in raw_example:
                raise ValueError("dataset examples must contain 'inputs' and 'target'")
            raw_inputs = raw_example["inputs"]
            if not isinstance(raw_inputs, list) or not raw_inputs:
                raise ValueError("dataset example 'inputs' must be a non-empty array")
            examples.append(
                RegressionExample(
                    inputs=tuple(float(value) for value in raw_inputs),
                    target=float(raw_example["target"]),
                )
            )
        return cls(examples=tuple(examples))

    @classmethod
    def from_path(cls, path: str | Path) -> "LinearRegressionDataset":
        return cls.from_json_obj(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LinearModel:
    weights: tuple[float, ...]
    bias: float

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("linear models must contain at least one weight")

    @property
    def feature_count(self) -> int:
        return len(self.weights)

    def predict(self, inputs: tuple[float, ...]) -> float:
        if len(inputs) != self.feature_count:
            raise ValueError(
                f"example feature count {len(inputs)} does not match model width {self.feature_count}"
            )
        return sum(weight * value for weight, value in zip(self.weights, inputs)) + self.bias


@dataclass(frozen=True)
class LinearModelAdapter:
    weight_parameter: str = LINEAR_WEIGHT_PARAMETER
    bias_parameter: str = LINEAR_BIAS_PARAMETER

    def export_state(self, model: LinearModel) -> ModelState:
        return ModelState(
            parameters=(
                TensorState(
                    name=self.bias_parameter,
                    shape=(1,),
                    values=(model.bias,),
                ),
                TensorState(
                    name=self.weight_parameter,
                    shape=(1, model.feature_count),
                    values=model.weights,
                ),
            )
        )

    def import_state(self, state: ModelState, *, feature_count: int | None = None) -> LinearModel:
        parameter_map = state.parameter_map()
        if self.weight_parameter not in parameter_map or self.bias_parameter not in parameter_map:
            raise ValueError(
                "linear model state must contain "
                f"{self.weight_parameter!r} and {self.bias_parameter!r} parameters"
            )
        extra_parameters = sorted(
            name
            for name in parameter_map
            if name not in {self.weight_parameter, self.bias_parameter}
        )
        if extra_parameters:
            raise ValueError(
                "linear model state contains unexpected parameters: "
                + ", ".join(extra_parameters)
            )

        weight_tensor = parameter_map[self.weight_parameter]
        if weight_tensor.shape == (len(weight_tensor.values),):
            weights = weight_tensor.values
        elif len(weight_tensor.shape) == 2 and weight_tensor.shape[0] == 1:
            weights = weight_tensor.values
        else:
            raise ValueError(
                f"{self.weight_parameter!r} must use shape [features] or [1, features]"
            )

        if feature_count is not None and len(weights) != feature_count:
            raise ValueError(
                f"state feature count {len(weights)} does not match dataset feature count {feature_count}"
            )

        bias_tensor = parameter_map[self.bias_parameter]
        if len(bias_tensor.values) != 1 or bias_tensor.shape not in {(), (1,)}:
            raise ValueError(f"{self.bias_parameter!r} must contain exactly one scalar bias value")

        return LinearModel(weights=tuple(weights), bias=float(bias_tensor.values[0]))


def evaluate_linear_regression(
    state: ModelState,
    dataset: LinearRegressionDataset,
    *,
    adapter: LinearModelAdapter | None = None,
) -> float:
    adapter = adapter or LinearModelAdapter()
    model = adapter.import_state(state, feature_count=dataset.feature_count)
    total_squared_error = 0.0
    for example in dataset.examples:
        error = model.predict(example.inputs) - example.target
        total_squared_error += error * error
    return total_squared_error / dataset.example_count


@dataclass(frozen=True)
class LocalTrainingConfig:
    steps: int = 500
    learning_rate: float = 0.01
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("inner training steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("inner learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")


@dataclass(frozen=True)
class LocalTrainingResult:
    initial_state: ModelState
    trained_state: ModelState
    loss_before: float
    loss_after: float
    steps: int
    learning_rate: float
    batch_size: int
    example_count: int
    feature_count: int

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "task": "linear-regression",
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "example_count": self.example_count,
            "feature_count": self.feature_count,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
        }


def train_linear_regression(
    initial_state: ModelState,
    dataset: LinearRegressionDataset,
    *,
    config: LocalTrainingConfig | None = None,
    adapter: LinearModelAdapter | None = None,
) -> LocalTrainingResult:
    config = config or LocalTrainingConfig()
    adapter = adapter or LinearModelAdapter()

    model = adapter.import_state(initial_state, feature_count=dataset.feature_count)
    weights = list(model.weights)
    bias = model.bias
    loss_before = evaluate_linear_regression(initial_state, dataset, adapter=adapter)
    example_index = 0

    for _ in range(config.steps):
        batch: list[RegressionExample] = []
        for offset in range(config.batch_size):
            batch.append(dataset.examples[(example_index + offset) % dataset.example_count])
        example_index = (example_index + config.batch_size) % dataset.example_count

        gradients = [0.0 for _ in weights]
        bias_gradient = 0.0
        batch_scale = 1.0 / len(batch)

        for example in batch:
            prediction = sum(weight * value for weight, value in zip(weights, example.inputs)) + bias
            error = prediction - example.target
            factor = 2.0 * error * batch_scale
            for index, feature_value in enumerate(example.inputs):
                gradients[index] += factor * feature_value
            bias_gradient += factor

        for index, gradient in enumerate(gradients):
            weights[index] -= config.learning_rate * gradient
        bias -= config.learning_rate * bias_gradient

    trained_state = adapter.export_state(LinearModel(weights=tuple(weights), bias=bias))
    loss_after = evaluate_linear_regression(trained_state, dataset, adapter=adapter)
    return LocalTrainingResult(
        initial_state=initial_state,
        trained_state=trained_state,
        loss_before=loss_before,
        loss_after=loss_after,
        steps=config.steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        example_count=dataset.example_count,
        feature_count=dataset.feature_count,
    )


@dataclass(frozen=True)
class TrainingWorkerConfig:
    run_name: str
    relay_url: str
    worker_id: str
    secret_key_hex: str
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
    advertised_relays: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run name cannot be empty")
        if not self.relay_url:
            raise ValueError("relay URL cannot be empty")
        if not self.worker_id:
            raise ValueError("worker id cannot be empty")
        if not self.secret_key_hex:
            raise ValueError("secret key cannot be empty")
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
        object.__setattr__(
            self,
            "advertised_relays",
            tuple(dict.fromkeys(str(relay).strip() for relay in self.advertised_relays if str(relay).strip())),
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
        }


@dataclass(frozen=True)
class TrainingSessionResult:
    run_name: str
    worker_id: str
    start_round: int
    rounds_completed: int
    final_state: ModelState
    final_momentum_state: ModelState | None
    rounds: tuple[TrainingRoundSummary, ...]

    @property
    def final_model_hash(self) -> str:
        return state_digest(self.final_state)

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
            "start_round": self.start_round,
            "rounds_completed": self.rounds_completed,
            "final_model_hash": self.final_model_hash,
        }
        if include_rounds:
            data["rounds"] = [round_summary.to_json_obj() for round_summary in self.rounds]
        if include_final_state:
            data["final_state"] = self.final_state.to_json_obj()
        if include_final_momentum and self.final_momentum_state is not None:
            data["final_momentum_state"] = self.final_momentum_state.to_json_obj()
        return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_model_state(path: Path, state: ModelState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state.write_json(path)


async def run_training_session(
    initial_state: ModelState,
    dataset: LinearRegressionDataset,
    *,
    config: TrainingWorkerConfig,
    adapter: LinearModelAdapter | None = None,
    previous_momentum: ModelState | None = None,
    artifact_dir: str | Path | None = None,
) -> TrainingSessionResult:
    adapter = adapter or LinearModelAdapter()
    current_state = initial_state
    current_momentum = previous_momentum
    round_summaries: list[TrainingRoundSummary] = []
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None

    for round_offset in range(config.rounds):
        round_index = config.start_round + round_offset
        round_start = int(time.time())

        heartbeat_event = build_heartbeat_event(
            HeartbeatEventMetadata(
                run_name=config.run_name,
                worker_id=config.worker_id,
                current_round=round_index,
                heartbeat_interval=config.heartbeat_interval,
                capabilities=("gradient-event", "linear-regression"),
                advertised_relays=config.advertised_relays or (config.relay_url,),
                created_at=round_start,
            ),
            secret_key_hex=config.secret_key_hex,
        )
        heartbeat_publish = await publish_nostrain_event(
            config.relay_url,
            heartbeat_event,
            open_timeout=config.open_timeout,
            reply_timeout=config.open_timeout,
        )
        if not heartbeat_publish.accepted:
            raise RuntimeError(f"relay rejected heartbeat publication: {heartbeat_publish.message}")

        local_training = train_linear_regression(
            current_state,
            dataset,
            config=LocalTrainingConfig(
                steps=config.inner_steps,
                learning_rate=config.local_learning_rate,
                batch_size=config.batch_size,
            ),
            adapter=adapter,
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
        gradient_publish = await publish_nostrain_event(
            config.relay_url,
            gradient_event,
            open_timeout=config.open_timeout,
            reply_timeout=config.open_timeout,
        )
        if not gradient_publish.accepted:
            raise RuntimeError(f"relay rejected gradient publication: {gradient_publish.message}")

        collection = await collect_gradient_events(
            config.relay_url,
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
        )
        if not collection.events:
            raise RuntimeError(
                f"round {round_index} completed without any collected gradient events; relay state is inconsistent"
            )

        outer_result = nesterov_outer_step(
            current_state,
            collection.aggregate_delta(),
            learning_rate=config.outer_learning_rate,
            momentum=config.outer_momentum,
            previous_momentum=current_momentum,
        )
        local_loss_after_outer = evaluate_linear_regression(
            outer_result.next_state,
            dataset,
            adapter=adapter,
        )

        if artifact_root is not None:
            round_dir = artifact_root / f"round-{round_index:04d}"
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

        round_summaries.append(
            TrainingRoundSummary(
                round_index=round_index,
                model_hash_before=state_digest(current_state),
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
            )
        )

        current_state = outer_result.next_state
        current_momentum = outer_result.momentum_state

    return TrainingSessionResult(
        run_name=config.run_name,
        worker_id=config.worker_id,
        start_round=config.start_round,
        rounds_completed=len(round_summaries),
        final_state=current_state,
        final_momentum_state=current_momentum,
        rounds=tuple(round_summaries),
    )
