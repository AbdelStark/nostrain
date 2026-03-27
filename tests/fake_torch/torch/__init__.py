from __future__ import annotations

import pickle
from pathlib import Path
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


def tensor(data: Any, *, dtype: Any | None = None) -> Tensor:
    return Tensor(np.asarray(data, dtype=dtype or np.float64))


def from_numpy(array: np.ndarray) -> Tensor:
    return Tensor(np.asarray(array, dtype=np.float64))


def tanh(value: Any) -> Tensor:
    return Tensor(np.tanh(_as_array(value)))


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
