from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _shape_size(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _coerce_shape(raw_shape: Any) -> tuple[int, ...]:
    if raw_shape is None:
        raise ValueError("tensor shape is required")
    if isinstance(raw_shape, int):
        shape = (raw_shape,)
    else:
        shape = tuple(int(value) for value in raw_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"tensor shape cannot contain negative dimensions: {shape}")
    return shape


@dataclass(frozen=True)
class TensorState:
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name cannot be empty")
        expected = _shape_size(self.shape)
        if expected != len(self.values):
            raise ValueError(
                f"tensor {self.name!r} expected {expected} values for shape {self.shape}, "
                f"got {len(self.values)}"
            )

    @property
    def size(self) -> int:
        return len(self.values)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True)
class ModelState:
    parameters: tuple[TensorState, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("model state must contain at least one parameter")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("model state contains duplicate parameter names")

    @property
    def parameter_count(self) -> int:
        return len(self.parameters)

    @property
    def total_values(self) -> int:
        return sum(parameter.size for parameter in self.parameters)

    def parameter_map(self) -> dict[str, TensorState]:
        return {parameter.name: parameter for parameter in self.parameters}

    def flatten(self) -> tuple[tuple[float, ...], tuple[TensorLayout, ...]]:
        flat_values: list[float] = []
        layouts: list[TensorLayout] = []
        offset = 0
        for parameter in self.parameters:
            length = parameter.size
            layouts.append(
                TensorLayout(
                    name=parameter.name,
                    shape=parameter.shape,
                    offset=offset,
                    length=length,
                )
            )
            flat_values.extend(parameter.values)
            offset += length
        return tuple(flat_values), tuple(layouts)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "parameters": {
                parameter.name: parameter.to_json_obj() for parameter in self.parameters
            }
        }

    def to_json(self, *, pretty: bool = True) -> str:
        if pretty:
            return json.dumps(self.to_json_obj(), indent=2, sort_keys=False)
        return json.dumps(self.to_json_obj(), separators=(",", ":"), sort_keys=False)

    def write_json(self, path: str | Path, *, pretty: bool = True) -> None:
        Path(path).write_text(self.to_json(pretty=pretty) + "\n", encoding="utf-8")

    @classmethod
    def from_json_obj(cls, data: Any) -> "ModelState":
        if not isinstance(data, dict) or "parameters" not in data:
            raise ValueError("model state JSON must contain a top-level 'parameters' object")
        raw_parameters = data["parameters"]
        if not isinstance(raw_parameters, dict) or not raw_parameters:
            raise ValueError("'parameters' must be a non-empty object")

        parameters: list[TensorState] = []
        for name in sorted(raw_parameters):
            raw_tensor = raw_parameters[name]
            if not isinstance(raw_tensor, dict):
                raise ValueError(f"parameter {name!r} must be an object")
            if "shape" not in raw_tensor or "values" not in raw_tensor:
                raise ValueError(f"parameter {name!r} must contain 'shape' and 'values'")
            shape = _coerce_shape(raw_tensor["shape"])
            values = tuple(float(value) for value in raw_tensor["values"])
            parameters.append(TensorState(name=name, shape=shape, values=values))
        return cls(parameters=tuple(parameters))

    @classmethod
    def from_json(cls, raw_json: str) -> "ModelState":
        return cls.from_json_obj(json.loads(raw_json))

    @classmethod
    def from_path(cls, path: str | Path) -> "ModelState":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_flat_values(
        cls,
        layouts: tuple[TensorLayout, ...],
        flat_values: tuple[float, ...],
    ) -> "ModelState":
        parameters: list[TensorState] = []
        for layout in layouts:
            chunk = flat_values[layout.offset : layout.offset + layout.length]
            if len(chunk) != layout.length:
                raise ValueError(
                    f"flat value buffer ended before parameter {layout.name!r} could be reconstructed"
                )
            parameters.append(
                TensorState(
                    name=layout.name,
                    shape=layout.shape,
                    values=tuple(chunk),
                )
            )
        return cls(parameters=tuple(parameters))


def _check_same_structure(left: ModelState, right: ModelState) -> tuple[TensorState, ...]:
    right_map = right.parameter_map()
    matched: list[TensorState] = []
    for parameter in left.parameters:
        other = right_map.get(parameter.name)
        if other is None:
            raise ValueError(f"missing parameter {parameter.name!r} in model state")
        if other.shape != parameter.shape:
            raise ValueError(
                f"shape mismatch for parameter {parameter.name!r}: "
                f"{parameter.shape} != {other.shape}"
            )
        matched.append(other)
    extra = sorted(set(right_map) - {parameter.name for parameter in left.parameters})
    if extra:
        raise ValueError(f"unexpected parameters in model state: {', '.join(extra)}")
    return tuple(matched)


def _combine_states(
    left: ModelState,
    right: ModelState,
    op,
) -> ModelState:
    right_parameters = _check_same_structure(left, right)
    combined: list[TensorState] = []
    for left_parameter, right_parameter in zip(left.parameters, right_parameters):
        values = tuple(
            op(left_value, right_value)
            for left_value, right_value in zip(left_parameter.values, right_parameter.values)
        )
        combined.append(
            TensorState(
                name=left_parameter.name,
                shape=left_parameter.shape,
                values=values,
            )
        )
    return ModelState(parameters=tuple(combined))


def compute_delta(initial: ModelState, current: ModelState) -> ModelState:
    return _combine_states(initial, current, lambda initial_value, current_value: current_value - initial_value)


def apply_delta(base: ModelState, delta: ModelState) -> ModelState:
    return _combine_states(base, delta, lambda base_value, delta_value: base_value + delta_value)


def add_states(left: ModelState, right: ModelState) -> ModelState:
    return _combine_states(left, right, lambda left_value, right_value: left_value + right_value)


def subtract_states(left: ModelState, right: ModelState) -> ModelState:
    return _combine_states(left, right, lambda left_value, right_value: left_value - right_value)


def scale_state(state: ModelState, factor: float) -> ModelState:
    scaled = [
        TensorState(
            name=parameter.name,
            shape=parameter.shape,
            values=tuple(value * factor for value in parameter.values),
        )
        for parameter in state.parameters
    ]
    return ModelState(parameters=tuple(scaled))


def zeros_like(state: ModelState) -> ModelState:
    zeroed = [
        TensorState(
            name=parameter.name,
            shape=parameter.shape,
            values=tuple(0.0 for _ in parameter.values),
        )
        for parameter in state.parameters
    ]
    return ModelState(parameters=tuple(zeroed))


def state_digest(state: ModelState) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", state.parameter_count))
    for parameter in state.parameters:
        name_bytes = parameter.name.encode("utf-8")
        digest.update(struct.pack("<H", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<H", len(parameter.shape)))
        for dimension in parameter.shape:
            digest.update(struct.pack("<I", dimension))
        for value in parameter.values:
            digest.update(struct.pack("<d", float(value)))
    return digest.hexdigest()
