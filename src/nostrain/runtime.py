from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ModelState, TensorState

LINEAR_WEIGHT_PARAMETER = "linear.weight"
LINEAR_BIAS_PARAMETER = "linear.bias"
MLP_HIDDEN_WEIGHT_PARAMETER = "mlp.hidden.weight"
MLP_HIDDEN_BIAS_PARAMETER = "mlp.hidden.bias"
MLP_OUTPUT_WEIGHT_PARAMETER = "mlp.output.weight"
MLP_OUTPUT_BIAS_PARAMETER = "mlp.output.bias"

LINEAR_REGRESSION_RUNTIME = "linear-regression"
MLP_REGRESSION_RUNTIME = "mlp-regression"
DEFAULT_TRAINING_RUNTIME = LINEAR_REGRESSION_RUNTIME
SUPPORTED_TRAINING_RUNTIMES = (
    LINEAR_REGRESSION_RUNTIME,
    MLP_REGRESSION_RUNTIME,
)
PYTHON_TRAINING_BACKEND = "python"
NUMPY_TRAINING_BACKEND = "numpy"
TORCH_TRAINING_BACKEND = "torch"
DEFAULT_TRAINING_BACKEND = PYTHON_TRAINING_BACKEND
SUPPORTED_TRAINING_BACKENDS = (
    PYTHON_TRAINING_BACKEND,
    NUMPY_TRAINING_BACKEND,
    TORCH_TRAINING_BACKEND,
)


def _normalize_runtime_name(runtime_name: str) -> str:
    normalized = str(runtime_name).strip().lower()
    if normalized not in SUPPORTED_TRAINING_RUNTIMES:
        raise ValueError(
            "runtime must be one of: " + ", ".join(SUPPORTED_TRAINING_RUNTIMES)
        )
    return normalized


def resolve_training_backend(backend_name: str | None) -> str:
    normalized = (
        DEFAULT_TRAINING_BACKEND
        if backend_name is None
        else str(backend_name).strip().lower() or DEFAULT_TRAINING_BACKEND
    )
    if normalized not in SUPPORTED_TRAINING_BACKENDS:
        raise ValueError(
            "training backend must be one of: "
            + ", ".join(SUPPORTED_TRAINING_BACKENDS)
        )
    return normalized


def _require_numpy(feature_name: str):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in environments without numpy.
        raise RuntimeError(
            f"{feature_name} requires the optional numpy dependency; "
            'install it with `python3 -m pip install -e ".[numpy]"`'
        ) from exc
    return np


def _dataset_arrays(dataset: RegressionDataset):
    np = _require_numpy("numpy training backend")
    inputs = np.asarray(
        [example.inputs for example in dataset.examples],
        dtype=np.float64,
    )
    targets = np.asarray(
        [example.target for example in dataset.examples],
        dtype=np.float64,
    )
    return np, inputs, targets


def _require_torch(feature_name: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in environments without torch.
        raise RuntimeError(
            f"{feature_name} requires the optional torch dependency; "
            'install it with `python3 -m pip install -e ".[torch]"`'
        ) from exc
    return torch


def _dataset_tensors(dataset: RegressionDataset):
    torch = _require_torch("torch training backend")
    inputs = torch.tensor(
        [example.inputs for example in dataset.examples],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [example.target for example in dataset.examples],
        dtype=torch.float64,
    )
    return torch, inputs, targets


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
class RegressionDataset:
    examples: tuple[RegressionExample, ...]
    task: str = DEFAULT_TRAINING_RUNTIME

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("regression datasets must contain at least one example")
        _normalize_runtime_name(self.task)
        feature_count = len(self.examples[0].inputs)
        if feature_count <= 0:
            raise ValueError("regression datasets must contain at least one feature")
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
            "task": self.task,
            "examples": [example.to_json_obj() for example in self.examples],
        }

    @classmethod
    def from_json_obj(cls, data: Any) -> "RegressionDataset":
        if not isinstance(data, dict):
            raise ValueError("dataset JSON must be an object")
        task = _normalize_runtime_name(str(data.get("task", DEFAULT_TRAINING_RUNTIME)))
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
        return cls(examples=tuple(examples), task=task)

    @classmethod
    def from_path(cls, path: str | Path) -> "RegressionDataset":
        return cls.from_json_obj(json.loads(Path(path).read_text(encoding="utf-8")))


LinearRegressionDataset = RegressionDataset


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


@dataclass(frozen=True)
class MLPModel:
    feature_count: int
    hidden_size: int
    hidden_weights: tuple[float, ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float

    def __post_init__(self) -> None:
        if self.feature_count <= 0:
            raise ValueError("MLP models must contain at least one input feature")
        if self.hidden_size <= 0:
            raise ValueError("MLP models must contain at least one hidden unit")
        expected_hidden_values = self.feature_count * self.hidden_size
        if len(self.hidden_weights) != expected_hidden_values:
            raise ValueError(
                "MLP hidden weights length does not match "
                f"{self.hidden_size} x {self.feature_count}"
            )
        if len(self.hidden_bias) != self.hidden_size:
            raise ValueError("MLP hidden bias length must match hidden size")
        if len(self.output_weights) != self.hidden_size:
            raise ValueError("MLP output weight length must match hidden size")

    def forward(self, inputs: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
        if len(inputs) != self.feature_count:
            raise ValueError(
                f"example feature count {len(inputs)} does not match model width {self.feature_count}"
            )
        hidden_activations: list[float] = []
        for hidden_index in range(self.hidden_size):
            offset = hidden_index * self.feature_count
            total = self.hidden_bias[hidden_index]
            for feature_index, feature_value in enumerate(inputs):
                total += self.hidden_weights[offset + feature_index] * feature_value
            hidden_activations.append(math.tanh(total))
        prediction = self.output_bias + sum(
            weight * activation
            for weight, activation in zip(self.output_weights, hidden_activations)
        )
        return tuple(hidden_activations), prediction

    def predict(self, inputs: tuple[float, ...]) -> float:
        return self.forward(inputs)[1]


@dataclass(frozen=True)
class MLPModelAdapter:
    hidden_weight_parameter: str = MLP_HIDDEN_WEIGHT_PARAMETER
    hidden_bias_parameter: str = MLP_HIDDEN_BIAS_PARAMETER
    output_weight_parameter: str = MLP_OUTPUT_WEIGHT_PARAMETER
    output_bias_parameter: str = MLP_OUTPUT_BIAS_PARAMETER

    def export_state(self, model: MLPModel) -> ModelState:
        return ModelState(
            parameters=(
                TensorState(
                    name=self.hidden_bias_parameter,
                    shape=(model.hidden_size,),
                    values=model.hidden_bias,
                ),
                TensorState(
                    name=self.hidden_weight_parameter,
                    shape=(model.hidden_size, model.feature_count),
                    values=model.hidden_weights,
                ),
                TensorState(
                    name=self.output_bias_parameter,
                    shape=(1,),
                    values=(model.output_bias,),
                ),
                TensorState(
                    name=self.output_weight_parameter,
                    shape=(1, model.hidden_size),
                    values=model.output_weights,
                ),
            )
        )

    def import_state(
        self,
        state: ModelState,
        *,
        feature_count: int | None = None,
        hidden_size: int | None = None,
    ) -> MLPModel:
        parameter_map = state.parameter_map()
        required_parameters = {
            self.hidden_weight_parameter,
            self.hidden_bias_parameter,
            self.output_weight_parameter,
            self.output_bias_parameter,
        }
        if not required_parameters <= set(parameter_map):
            missing = sorted(required_parameters - set(parameter_map))
            raise ValueError(
                "MLP model state is missing required parameters: " + ", ".join(missing)
            )
        extra_parameters = sorted(set(parameter_map) - required_parameters)
        if extra_parameters:
            raise ValueError(
                "MLP model state contains unexpected parameters: "
                + ", ".join(extra_parameters)
            )

        hidden_weight_tensor = parameter_map[self.hidden_weight_parameter]
        if len(hidden_weight_tensor.shape) != 2:
            raise ValueError(f"{self.hidden_weight_parameter!r} must use shape [hidden, features]")
        inferred_hidden_size, inferred_feature_count = hidden_weight_tensor.shape
        if inferred_hidden_size <= 0 or inferred_feature_count <= 0:
            raise ValueError("MLP hidden weight shape must contain positive dimensions")
        if feature_count is not None and inferred_feature_count != feature_count:
            raise ValueError(
                f"state feature count {inferred_feature_count} does not match dataset feature count {feature_count}"
            )
        if hidden_size is not None and inferred_hidden_size != hidden_size:
            raise ValueError(
                f"state hidden size {inferred_hidden_size} does not match expected hidden size {hidden_size}"
            )

        hidden_bias_tensor = parameter_map[self.hidden_bias_parameter]
        if hidden_bias_tensor.shape != (inferred_hidden_size,):
            raise ValueError(
                f"{self.hidden_bias_parameter!r} must use shape [{inferred_hidden_size}]"
            )

        output_weight_tensor = parameter_map[self.output_weight_parameter]
        if output_weight_tensor.shape == (inferred_hidden_size,):
            output_weights = output_weight_tensor.values
        elif output_weight_tensor.shape == (1, inferred_hidden_size):
            output_weights = output_weight_tensor.values
        else:
            raise ValueError(
                f"{self.output_weight_parameter!r} must use shape [hidden] or [1, hidden]"
            )

        output_bias_tensor = parameter_map[self.output_bias_parameter]
        if len(output_bias_tensor.values) != 1 or output_bias_tensor.shape not in {(), (1,)}:
            raise ValueError(
                f"{self.output_bias_parameter!r} must contain exactly one scalar bias value"
            )

        return MLPModel(
            feature_count=inferred_feature_count,
            hidden_size=inferred_hidden_size,
            hidden_weights=tuple(hidden_weight_tensor.values),
            hidden_bias=tuple(hidden_bias_tensor.values),
            output_weights=tuple(output_weights),
            output_bias=float(output_bias_tensor.values[0]),
        )


RuntimeAdapter = LinearModelAdapter | MLPModelAdapter


def infer_training_runtime_from_adapter(adapter: RuntimeAdapter) -> str:
    if isinstance(adapter, LinearModelAdapter):
        return LINEAR_REGRESSION_RUNTIME
    if isinstance(adapter, MLPModelAdapter):
        return MLP_REGRESSION_RUNTIME
    raise ValueError(f"unsupported runtime adapter: {type(adapter).__name__}")


def infer_training_runtime_from_state(state: ModelState) -> str:
    matches: list[str] = []
    for runtime_name, adapter in (
        (LINEAR_REGRESSION_RUNTIME, LinearModelAdapter()),
        (MLP_REGRESSION_RUNTIME, MLPModelAdapter()),
    ):
        try:
            if runtime_name == LINEAR_REGRESSION_RUNTIME:
                adapter.import_state(state)
            else:
                adapter.import_state(state)
        except ValueError:
            continue
        matches.append(runtime_name)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("could not infer runtime from model state parameter layout")
    raise ValueError("model state matches multiple runtimes: " + ", ".join(matches))


def resolve_training_runtime(
    runtime_name: str | None,
    dataset: RegressionDataset | None = None,
    *,
    state: ModelState | None = None,
    adapter: RuntimeAdapter | None = None,
) -> str:
    explicit_runtime = (
        _normalize_runtime_name(runtime_name)
        if runtime_name is not None
        else None
    )
    dataset_runtime = _normalize_runtime_name(dataset.task) if dataset is not None else None
    adapter_runtime = (
        infer_training_runtime_from_adapter(adapter) if adapter is not None else None
    )
    state_runtime = (
        infer_training_runtime_from_state(state) if state is not None else None
    )

    resolved_runtime = (
        explicit_runtime
        or adapter_runtime
        or dataset_runtime
        or state_runtime
        or DEFAULT_TRAINING_RUNTIME
    )

    for source, candidate in (
        ("dataset", dataset_runtime),
        ("adapter", adapter_runtime),
        ("state", state_runtime),
    ):
        if candidate is not None and candidate != resolved_runtime:
            raise ValueError(
                f"{source} runtime {candidate!r} does not match resolved runtime {resolved_runtime!r}"
            )
    return resolved_runtime


def initialize_linear_regression_state(
    feature_count: int,
    *,
    bias: float = 0.0,
) -> ModelState:
    if feature_count <= 0:
        raise ValueError("feature count must be positive")
    return LinearModelAdapter().export_state(
        LinearModel(weights=tuple(0.0 for _ in range(feature_count)), bias=bias)
    )


def initialize_mlp_regression_state(
    feature_count: int,
    hidden_size: int,
    *,
    seed: int = 0,
    weight_scale: float = 0.1,
) -> ModelState:
    if feature_count <= 0:
        raise ValueError("feature count must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden size must be positive")
    if weight_scale <= 0:
        raise ValueError("weight scale must be positive")

    rng = random.Random(seed)
    hidden_scale = weight_scale / max(1.0, math.sqrt(feature_count))
    output_scale = weight_scale / max(1.0, math.sqrt(hidden_size))
    hidden_weights = tuple(
        rng.uniform(-hidden_scale, hidden_scale)
        for _ in range(feature_count * hidden_size)
    )
    output_weights = tuple(
        rng.uniform(-output_scale, output_scale)
        for _ in range(hidden_size)
    )
    return MLPModelAdapter().export_state(
        MLPModel(
            feature_count=feature_count,
            hidden_size=hidden_size,
            hidden_weights=hidden_weights,
            hidden_bias=tuple(0.0 for _ in range(hidden_size)),
            output_weights=output_weights,
            output_bias=0.0,
        )
    )


def initialize_training_state(
    runtime_name: str,
    *,
    feature_count: int,
    hidden_size: int | None = None,
    seed: int = 0,
    weight_scale: float = 0.1,
) -> ModelState:
    resolved_runtime = _normalize_runtime_name(runtime_name)
    if resolved_runtime == LINEAR_REGRESSION_RUNTIME:
        return initialize_linear_regression_state(feature_count)
    if hidden_size is None:
        raise ValueError("--hidden-size is required for mlp-regression state initialization")
    return initialize_mlp_regression_state(
        feature_count,
        hidden_size,
        seed=seed,
        weight_scale=weight_scale,
    )


def evaluate_linear_regression(
    state: ModelState,
    dataset: RegressionDataset,
    *,
    adapter: LinearModelAdapter | None = None,
    backend_name: str | None = None,
) -> float:
    resolve_training_runtime(LINEAR_REGRESSION_RUNTIME, dataset, state=state, adapter=adapter)
    resolved_backend = resolve_training_backend(backend_name)
    adapter = adapter or LinearModelAdapter()
    model = adapter.import_state(state, feature_count=dataset.feature_count)
    if resolved_backend == NUMPY_TRAINING_BACKEND:
        np, inputs, targets = _dataset_arrays(dataset)
        weights = np.asarray(model.weights, dtype=np.float64)
        predictions = inputs @ weights + float(model.bias)
        errors = predictions - targets
        return float(np.mean(errors * errors))
    if resolved_backend == TORCH_TRAINING_BACKEND:
        torch, inputs, targets = _dataset_tensors(dataset)
        weights = torch.tensor(model.weights, dtype=torch.float64)
        bias = torch.tensor(model.bias, dtype=torch.float64)
        predictions = inputs @ weights + bias
        errors = predictions - targets
        return float((errors * errors).mean().item())
    total_squared_error = 0.0
    for example in dataset.examples:
        error = model.predict(example.inputs) - example.target
        total_squared_error += error * error
    return total_squared_error / dataset.example_count


def evaluate_mlp_regression(
    state: ModelState,
    dataset: RegressionDataset,
    *,
    adapter: MLPModelAdapter | None = None,
    backend_name: str | None = None,
) -> float:
    resolve_training_runtime(MLP_REGRESSION_RUNTIME, dataset, state=state, adapter=adapter)
    resolved_backend = resolve_training_backend(backend_name)
    adapter = adapter or MLPModelAdapter()
    model = adapter.import_state(state, feature_count=dataset.feature_count)
    if resolved_backend == NUMPY_TRAINING_BACKEND:
        np, inputs, targets = _dataset_arrays(dataset)
        hidden_weights = np.asarray(model.hidden_weights, dtype=np.float64).reshape(
            model.hidden_size,
            model.feature_count,
        )
        hidden_bias = np.asarray(model.hidden_bias, dtype=np.float64)
        output_weights = np.asarray(model.output_weights, dtype=np.float64)
        hidden_activations = np.tanh(inputs @ hidden_weights.T + hidden_bias)
        predictions = hidden_activations @ output_weights + float(model.output_bias)
        errors = predictions - targets
        return float(np.mean(errors * errors))
    if resolved_backend == TORCH_TRAINING_BACKEND:
        torch, inputs, targets = _dataset_tensors(dataset)
        hidden_weights = torch.tensor(model.hidden_weights, dtype=torch.float64).reshape(
            model.hidden_size,
            model.feature_count,
        )
        hidden_bias = torch.tensor(model.hidden_bias, dtype=torch.float64)
        output_weights = torch.tensor(model.output_weights, dtype=torch.float64)
        output_bias = torch.tensor(model.output_bias, dtype=torch.float64)
        hidden_activations = torch.tanh(inputs @ hidden_weights.transpose(0, 1) + hidden_bias)
        predictions = hidden_activations @ output_weights + output_bias
        errors = predictions - targets
        return float((errors * errors).mean().item())
    total_squared_error = 0.0
    for example in dataset.examples:
        error = model.predict(example.inputs) - example.target
        total_squared_error += error * error
    return total_squared_error / dataset.example_count


def evaluate_regression(
    state: ModelState,
    dataset: RegressionDataset,
    *,
    runtime_name: str | None = None,
    adapter: RuntimeAdapter | None = None,
    backend_name: str | None = None,
) -> float:
    resolved_runtime = resolve_training_runtime(
        runtime_name,
        dataset,
        state=state,
        adapter=adapter,
    )
    if resolved_runtime == LINEAR_REGRESSION_RUNTIME:
        return evaluate_linear_regression(
            state,
            dataset,
            adapter=adapter if isinstance(adapter, LinearModelAdapter) else None,
            backend_name=backend_name,
        )
    return evaluate_mlp_regression(
        state,
        dataset,
        adapter=adapter if isinstance(adapter, MLPModelAdapter) else None,
        backend_name=backend_name,
    )


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
    runtime_name: str = DEFAULT_TRAINING_RUNTIME
    backend_name: str = DEFAULT_TRAINING_BACKEND

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "task": self.runtime_name,
            "runtime": self.runtime_name,
            "backend": self.backend_name,
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
    dataset: RegressionDataset,
    *,
    config: LocalTrainingConfig | None = None,
    adapter: LinearModelAdapter | None = None,
    backend_name: str | None = None,
) -> LocalTrainingResult:
    config = config or LocalTrainingConfig()
    resolve_training_runtime(LINEAR_REGRESSION_RUNTIME, dataset, state=initial_state, adapter=adapter)
    resolved_backend = resolve_training_backend(backend_name)
    adapter = adapter or LinearModelAdapter()

    model = adapter.import_state(initial_state, feature_count=dataset.feature_count)
    loss_before = evaluate_linear_regression(
        initial_state,
        dataset,
        adapter=adapter,
        backend_name=resolved_backend,
    )

    if resolved_backend == NUMPY_TRAINING_BACKEND:
        np, all_inputs, all_targets = _dataset_arrays(dataset)
        weights = np.asarray(model.weights, dtype=np.float64)
        bias = float(model.bias)
        example_index = 0

        for _ in range(config.steps):
            batch_indexes = [
                (example_index + offset) % dataset.example_count
                for offset in range(config.batch_size)
            ]
            example_index = (example_index + config.batch_size) % dataset.example_count
            batch_inputs = all_inputs[batch_indexes]
            batch_targets = all_targets[batch_indexes]

            errors = batch_inputs @ weights + bias - batch_targets
            factors = (2.0 / len(batch_indexes)) * errors
            gradients = batch_inputs.T @ factors
            bias_gradient = float(factors.sum())

            weights = weights - config.learning_rate * gradients
            bias -= config.learning_rate * bias_gradient

        trained_state = adapter.export_state(
            LinearModel(
                weights=tuple(float(value) for value in weights.tolist()),
                bias=float(bias),
            )
        )
    elif resolved_backend == TORCH_TRAINING_BACKEND:
        torch, all_inputs, all_targets = _dataset_tensors(dataset)
        weights = torch.tensor(model.weights, dtype=torch.float64)
        bias = torch.tensor(model.bias, dtype=torch.float64)
        example_index = 0

        for _ in range(config.steps):
            batch_indexes = [
                (example_index + offset) % dataset.example_count
                for offset in range(config.batch_size)
            ]
            example_index = (example_index + config.batch_size) % dataset.example_count
            batch_inputs = all_inputs[batch_indexes]
            batch_targets = all_targets[batch_indexes]

            errors = batch_inputs @ weights + bias - batch_targets
            factors = errors * (2.0 / len(batch_indexes))
            gradients = batch_inputs.transpose(0, 1) @ factors
            bias_gradient = factors.sum()

            weights = weights - config.learning_rate * gradients
            bias = bias - config.learning_rate * bias_gradient

        trained_state = adapter.export_state(
            LinearModel(
                weights=tuple(float(value) for value in weights.tolist()),
                bias=float(bias.item()),
            )
        )
    else:
        weights = list(model.weights)
        bias = model.bias
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

    loss_after = evaluate_linear_regression(
        trained_state,
        dataset,
        adapter=adapter,
        backend_name=resolved_backend,
    )
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
        runtime_name=LINEAR_REGRESSION_RUNTIME,
        backend_name=resolved_backend,
    )


def train_mlp_regression(
    initial_state: ModelState,
    dataset: RegressionDataset,
    *,
    config: LocalTrainingConfig | None = None,
    adapter: MLPModelAdapter | None = None,
    backend_name: str | None = None,
) -> LocalTrainingResult:
    config = config or LocalTrainingConfig()
    resolve_training_runtime(MLP_REGRESSION_RUNTIME, dataset, state=initial_state, adapter=adapter)
    resolved_backend = resolve_training_backend(backend_name)
    adapter = adapter or MLPModelAdapter()

    model = adapter.import_state(initial_state, feature_count=dataset.feature_count)
    loss_before = evaluate_mlp_regression(
        initial_state,
        dataset,
        adapter=adapter,
        backend_name=resolved_backend,
    )

    if resolved_backend == NUMPY_TRAINING_BACKEND:
        np, all_inputs, all_targets = _dataset_arrays(dataset)
        hidden_weights = np.asarray(model.hidden_weights, dtype=np.float64).reshape(
            model.hidden_size,
            model.feature_count,
        )
        hidden_bias = np.asarray(model.hidden_bias, dtype=np.float64)
        output_weights = np.asarray(model.output_weights, dtype=np.float64)
        output_bias = float(model.output_bias)
        example_index = 0

        for _ in range(config.steps):
            batch_indexes = [
                (example_index + offset) % dataset.example_count
                for offset in range(config.batch_size)
            ]
            example_index = (example_index + config.batch_size) % dataset.example_count
            batch_inputs = all_inputs[batch_indexes]
            batch_targets = all_targets[batch_indexes]

            hidden_activations = np.tanh(batch_inputs @ hidden_weights.T + hidden_bias)
            predictions = hidden_activations @ output_weights + output_bias
            errors = predictions - batch_targets
            output_delta = (2.0 / len(batch_indexes)) * errors

            output_weight_gradients = hidden_activations.T @ output_delta
            output_bias_gradient = float(output_delta.sum())
            hidden_delta = (
                output_delta[:, None] * output_weights[None, :]
            ) * (1.0 - hidden_activations * hidden_activations)
            hidden_bias_gradients = hidden_delta.sum(axis=0)
            hidden_weight_gradients = hidden_delta.T @ batch_inputs

            hidden_weights = hidden_weights - config.learning_rate * hidden_weight_gradients
            hidden_bias = hidden_bias - config.learning_rate * hidden_bias_gradients
            output_weights = output_weights - config.learning_rate * output_weight_gradients
            output_bias -= config.learning_rate * output_bias_gradient

        trained_state = adapter.export_state(
            MLPModel(
                feature_count=model.feature_count,
                hidden_size=model.hidden_size,
                hidden_weights=tuple(
                    float(value) for value in hidden_weights.reshape(-1).tolist()
                ),
                hidden_bias=tuple(float(value) for value in hidden_bias.tolist()),
                output_weights=tuple(float(value) for value in output_weights.tolist()),
                output_bias=float(output_bias),
            )
        )
    elif resolved_backend == TORCH_TRAINING_BACKEND:
        torch, all_inputs, all_targets = _dataset_tensors(dataset)
        hidden_weights = torch.tensor(model.hidden_weights, dtype=torch.float64).reshape(
            model.hidden_size,
            model.feature_count,
        )
        hidden_bias = torch.tensor(model.hidden_bias, dtype=torch.float64)
        output_weights = torch.tensor(model.output_weights, dtype=torch.float64)
        output_bias = torch.tensor(model.output_bias, dtype=torch.float64)
        example_index = 0

        for _ in range(config.steps):
            batch_indexes = [
                (example_index + offset) % dataset.example_count
                for offset in range(config.batch_size)
            ]
            example_index = (example_index + config.batch_size) % dataset.example_count
            batch_inputs = all_inputs[batch_indexes]
            batch_targets = all_targets[batch_indexes]

            hidden_activations = torch.tanh(
                batch_inputs @ hidden_weights.transpose(0, 1) + hidden_bias
            )
            predictions = hidden_activations @ output_weights + output_bias
            errors = predictions - batch_targets
            output_delta = errors * (2.0 / len(batch_indexes))

            output_weight_gradients = hidden_activations.transpose(0, 1) @ output_delta
            output_bias_gradient = output_delta.sum()
            hidden_delta = (
                output_delta.reshape(len(batch_indexes), 1)
                * output_weights.reshape(1, model.hidden_size)
            ) * (1.0 - hidden_activations * hidden_activations)
            hidden_bias_gradients = hidden_delta.sum(axis=0)
            hidden_weight_gradients = hidden_delta.transpose(0, 1) @ batch_inputs

            hidden_weights = hidden_weights - config.learning_rate * hidden_weight_gradients
            hidden_bias = hidden_bias - config.learning_rate * hidden_bias_gradients
            output_weights = output_weights - config.learning_rate * output_weight_gradients
            output_bias = output_bias - config.learning_rate * output_bias_gradient

        trained_state = adapter.export_state(
            MLPModel(
                feature_count=model.feature_count,
                hidden_size=model.hidden_size,
                hidden_weights=tuple(
                    float(value) for value in hidden_weights.reshape(-1).tolist()
                ),
                hidden_bias=tuple(float(value) for value in hidden_bias.tolist()),
                output_weights=tuple(float(value) for value in output_weights.tolist()),
                output_bias=float(output_bias.item()),
            )
        )
    else:
        hidden_weights = list(model.hidden_weights)
        hidden_bias = list(model.hidden_bias)
        output_weights = list(model.output_weights)
        output_bias = model.output_bias
        example_index = 0

        for _ in range(config.steps):
            batch: list[RegressionExample] = []
            for offset in range(config.batch_size):
                batch.append(dataset.examples[(example_index + offset) % dataset.example_count])
            example_index = (example_index + config.batch_size) % dataset.example_count

            hidden_weight_gradients = [0.0 for _ in hidden_weights]
            hidden_bias_gradients = [0.0 for _ in hidden_bias]
            output_weight_gradients = [0.0 for _ in output_weights]
            output_bias_gradient = 0.0
            batch_scale = 1.0 / len(batch)

            for example in batch:
                hidden_activations: list[float] = []
                for hidden_index in range(model.hidden_size):
                    offset = hidden_index * model.feature_count
                    total = hidden_bias[hidden_index]
                    for feature_index, feature_value in enumerate(example.inputs):
                        total += hidden_weights[offset + feature_index] * feature_value
                    hidden_activations.append(math.tanh(total))

                prediction = output_bias + sum(
                    weight * activation
                    for weight, activation in zip(output_weights, hidden_activations)
                )
                error = prediction - example.target
                output_delta = 2.0 * error * batch_scale

                for hidden_index, activation in enumerate(hidden_activations):
                    output_weight_gradients[hidden_index] += output_delta * activation
                output_bias_gradient += output_delta

                for hidden_index, activation in enumerate(hidden_activations):
                    hidden_delta = (
                        output_delta
                        * output_weights[hidden_index]
                        * (1.0 - activation * activation)
                    )
                    hidden_bias_gradients[hidden_index] += hidden_delta
                    offset = hidden_index * model.feature_count
                    for feature_index, feature_value in enumerate(example.inputs):
                        hidden_weight_gradients[offset + feature_index] += (
                            hidden_delta * feature_value
                        )

            for index, gradient in enumerate(hidden_weight_gradients):
                hidden_weights[index] -= config.learning_rate * gradient
            for index, gradient in enumerate(hidden_bias_gradients):
                hidden_bias[index] -= config.learning_rate * gradient
            for index, gradient in enumerate(output_weight_gradients):
                output_weights[index] -= config.learning_rate * gradient
            output_bias -= config.learning_rate * output_bias_gradient

        trained_state = adapter.export_state(
            MLPModel(
                feature_count=model.feature_count,
                hidden_size=model.hidden_size,
                hidden_weights=tuple(hidden_weights),
                hidden_bias=tuple(hidden_bias),
                output_weights=tuple(output_weights),
                output_bias=output_bias,
            )
        )

    loss_after = evaluate_mlp_regression(
        trained_state,
        dataset,
        adapter=adapter,
        backend_name=resolved_backend,
    )
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
        runtime_name=MLP_REGRESSION_RUNTIME,
        backend_name=resolved_backend,
    )


def train_regression(
    initial_state: ModelState,
    dataset: RegressionDataset,
    *,
    config: LocalTrainingConfig | None = None,
    runtime_name: str | None = None,
    adapter: RuntimeAdapter | None = None,
    backend_name: str | None = None,
) -> LocalTrainingResult:
    resolved_runtime = resolve_training_runtime(
        runtime_name,
        dataset,
        state=initial_state,
        adapter=adapter,
    )
    if resolved_runtime == LINEAR_REGRESSION_RUNTIME:
        return train_linear_regression(
            initial_state,
            dataset,
            config=config,
            adapter=adapter if isinstance(adapter, LinearModelAdapter) else None,
            backend_name=backend_name,
        )
    return train_mlp_regression(
        initial_state,
        dataset,
        config=config,
        adapter=adapter if isinstance(adapter, MLPModelAdapter) else None,
        backend_name=backend_name,
    )
