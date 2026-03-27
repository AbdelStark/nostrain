from __future__ import annotations

from collections import OrderedDict
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

float64 = np.float64


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value._array
    return np.asarray(value, dtype=np.float64)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, value: Any):
        self._array = np.asarray(value, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dimension) for dimension in self._array.shape)

    @property
    def T(self) -> "Tensor":
        return Tensor(self._array.T)

    def detach(self) -> "Tensor":
        return self

    def cpu(self) -> "Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array.copy()

    def reshape(self, *shape: int) -> "Tensor":
        return Tensor(self._array.reshape(*shape))

    def squeeze(self, axis: int | None = None) -> "Tensor":
        return Tensor(np.squeeze(self._array, axis=axis))

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        return Tensor(np.swapaxes(self._array, dim0, dim1))

    def sum(self, axis: int | None = None) -> "Tensor":
        return Tensor(self._array.sum(axis=axis))

    def mean(self) -> "Tensor":
        return Tensor(self._array.mean())

    def item(self) -> float:
        return float(self._array.item())

    def tolist(self) -> list[Any]:
        return self._array.tolist()

    def __getitem__(self, item: Any) -> "Tensor":
        return Tensor(self._array[item])

    def __add__(self, other: Any) -> "Tensor":
        return Tensor(self._array + _as_array(other))

    def __radd__(self, other: Any) -> "Tensor":
        return Tensor(_as_array(other) + self._array)

    def __sub__(self, other: Any) -> "Tensor":
        return Tensor(self._array - _as_array(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return Tensor(_as_array(other) - self._array)

    def __mul__(self, other: Any) -> "Tensor":
        return Tensor(self._array * _as_array(other))

    def __rmul__(self, other: Any) -> "Tensor":
        return Tensor(_as_array(other) * self._array)

    def __matmul__(self, other: Any) -> "Tensor":
        return Tensor(self._array @ _as_array(other))

    def __rmatmul__(self, other: Any) -> "Tensor":
        return Tensor(_as_array(other) @ self._array)

    def __truediv__(self, other: Any) -> "Tensor":
        return Tensor(self._array / _as_array(other))

    def __repr__(self) -> str:
        return f"Tensor({self._array!r})"


class Parameter(Tensor):
    pass


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_parameters", "_modules"}:
            object.__setattr__(self, name, value)
            return

        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if parameters is not None and name in parameters and not isinstance(value, Parameter):
            parameters.pop(name, None)
        if modules is not None and name in modules and not isinstance(value, Module):
            modules.pop(name, None)

        if isinstance(value, Parameter) and parameters is not None:
            parameters[name] = value
            if modules is not None:
                modules.pop(name, None)
        elif isinstance(value, Module) and modules is not None:
            modules[name] = value
            if parameters is not None:
                parameters.pop(name, None)

        object.__setattr__(self, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def state_dict(self) -> OrderedDict[str, Tensor]:
        state = OrderedDict()
        for name, parameter in self._parameters.items():
            state[name] = Tensor(parameter._array.copy())
        for name, module in self._modules.items():
            for child_name, value in module.state_dict().items():
                state[f"{name}.{child_name}"] = value
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> dict[str, list[str]]:
        expected_keys = tuple(self.state_dict())
        incoming_keys = tuple(str(key) for key in state_dict)
        missing_keys = [key for key in expected_keys if key not in state_dict]
        unexpected_keys = [key for key in incoming_keys if key not in expected_keys]
        if missing_keys or unexpected_keys:
            raise ValueError(
                "state_dict keys do not match module layout "
                f"(missing={missing_keys}, unexpected={unexpected_keys})"
            )
        for name, value in state_dict.items():
            self._assign_state_dict_value(str(name).split("."), value)
        return {"missing_keys": [], "unexpected_keys": []}

    def _assign_state_dict_value(self, path: list[str], value: Any) -> None:
        head = path[0]
        if len(path) == 1:
            setattr(self, head, Parameter(_as_array(value)))
            return
        module = getattr(self, head, None)
        if not isinstance(module, Module):
            raise ValueError(f"unknown child module path {'.'.join(path)!r}")
        module._assign_state_dict_value(path[1:], value)

    def eval(self) -> "Module":
        return self

    def train(self, mode: bool = True) -> "Module":
        del mode
        return self


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: Any | None = None,
    ) -> None:
        super().__init__()
        dtype = dtype or np.float64
        self.weight = Parameter(np.zeros((out_features, in_features), dtype=dtype))
        if bias:
            self.bias = Parameter(np.zeros((out_features,), dtype=dtype))

    def forward(self, inputs: Any) -> Tensor:
        output = _as_array(inputs) @ self.weight._array.T
        if hasattr(self, "bias"):
            output = output + self.bias._array
        return Tensor(output)


class Tanh(Module):
    def forward(self, value: Any) -> Tensor:
        return tanh(value)


class Sequential(Module):
    def __init__(self, *modules: Any) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (dict, OrderedDict)):
            items = list(modules[0].items())
        else:
            items = [(str(index), module) for index, module in enumerate(modules)]
        self._sequence_names = [str(name) for name, _ in items]
        for name, module in items:
            setattr(self, str(name), module)

    def forward(self, value: Any) -> Any:
        result = value
        for name in self._sequence_names:
            result = getattr(self, name)(result)
        return result


def tensor(data: Any, *, dtype: Any | None = None) -> Tensor:
    return Tensor(np.asarray(data, dtype=dtype or np.float64))


def from_numpy(array: np.ndarray) -> Tensor:
    return Tensor(np.asarray(array, dtype=np.float64))


def tanh(value: Any) -> Tensor:
    return Tensor(np.tanh(_as_array(value)))


nn = SimpleNamespace(
    Module=Module,
    Parameter=Parameter,
    Linear=Linear,
    Sequential=Sequential,
    Tanh=Tanh,
)


def save(obj: Any, destination: Any) -> None:
    if hasattr(destination, "write"):
        pickle.dump(obj, destination)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load(source: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    if hasattr(source, "read"):
        return pickle.load(source)
    with Path(source).open("rb") as handle:
        return pickle.load(handle)
