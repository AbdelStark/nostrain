from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import ModelState, TensorState
from .pytorch import (
    export_pytorch_state_dict,
    import_pytorch_state_dict,
    load_torch_checkpoint,
    write_torch_checkpoint,
)
from .runtime import infer_training_runtime_from_state

JSON_STATE_FORMAT = "json"
NUMPY_NPZ_STATE_FORMAT = "numpy-npz"
PYTORCH_STATE_DICT_STATE_FORMAT = "pytorch-state-dict"
DEFAULT_STATE_FORMAT = JSON_STATE_FORMAT
STATE_FORMAT_AUTO = "auto"
SUPPORTED_STATE_FORMATS = (
    JSON_STATE_FORMAT,
    NUMPY_NPZ_STATE_FORMAT,
    PYTORCH_STATE_DICT_STATE_FORMAT,
)
STATE_FORMAT_CHOICES = (
    STATE_FORMAT_AUTO,
    *SUPPORTED_STATE_FORMATS,
)

_NPZ_SCHEMA_KEY = "__nostrain_schema__"
_NPZ_VERSION_KEY = "__nostrain_version__"
_NPZ_RUNTIME_KEY = "__nostrain_runtime__"
_MODEL_STATE_NPZ_SCHEMA_VALUE = "nostrain-model-state"
_PYTORCH_STATE_DICT_NPZ_SCHEMA_VALUE = "nostrain-pytorch-state-dict"
_NPZ_METADATA_KEYS = {
    _NPZ_SCHEMA_KEY,
    _NPZ_VERSION_KEY,
    _NPZ_RUNTIME_KEY,
}


@dataclass(frozen=True)
class ModelStateDocument:
    state: ModelState
    runtime_name: str | None = None


def _require_numpy(feature_name: str):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in environments without numpy.
        raise RuntimeError(
            f"{feature_name} requires the optional numpy dependency; "
            'install it with `python3 -m pip install -e ".[numpy]"`'
        ) from exc
    return np


def infer_state_format_from_path(path: str | Path) -> str:
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] in ([".pt", ".npz"], [".pth", ".npz"]):
        return PYTORCH_STATE_DICT_STATE_FORMAT
    if suffixes and suffixes[-1] in {".pt", ".pth"}:
        return PYTORCH_STATE_DICT_STATE_FORMAT
    if suffixes and suffixes[-1] == ".npz":
        return NUMPY_NPZ_STATE_FORMAT
    return JSON_STATE_FORMAT


def resolve_state_format(
    state_format: str | None,
    path: str | Path | None = None,
    *,
    default: str = DEFAULT_STATE_FORMAT,
) -> str:
    normalized = (
        STATE_FORMAT_AUTO
        if state_format is None
        else str(state_format).strip().lower() or STATE_FORMAT_AUTO
    )
    if normalized == STATE_FORMAT_AUTO:
        if path is None:
            return default
        return infer_state_format_from_path(path)
    if normalized not in SUPPORTED_STATE_FORMATS:
        raise ValueError(
            "state format must be one of: " + ", ".join(STATE_FORMAT_CHOICES)
        )
    return normalized


def _load_archive_string(value, *, key: str) -> str:
    np = _require_numpy("numpy-npz state loading")
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    flat = array.reshape(-1)
    if flat.size != 1:
        raise ValueError(f"numpy-npz metadata entry {key!r} must contain exactly one value")
    return str(flat[0].item())


def _load_npz_document(
    path: str | Path,
    *,
    feature_name: str,
    expected_schema: str,
) -> ModelStateDocument:
    np = _require_numpy(feature_name)
    with np.load(Path(path), allow_pickle=False) as archive:
        if _NPZ_SCHEMA_KEY in archive.files:
            schema = _load_archive_string(archive[_NPZ_SCHEMA_KEY], key=_NPZ_SCHEMA_KEY)
            if schema != expected_schema:
                raise ValueError(
                    f"unsupported numpy-npz state schema {schema!r}; expected {expected_schema!r}"
                )

        runtime_name = None
        if _NPZ_RUNTIME_KEY in archive.files:
            runtime_name = _load_archive_string(
                archive[_NPZ_RUNTIME_KEY],
                key=_NPZ_RUNTIME_KEY,
            )

        parameters: list[TensorState] = []
        for name in sorted(entry for entry in archive.files if entry not in _NPZ_METADATA_KEYS):
            array = np.asarray(archive[name], dtype=np.float64)
            parameters.append(
                TensorState(
                    name=name,
                    shape=tuple(int(dimension) for dimension in array.shape),
                    values=tuple(float(value) for value in array.reshape(-1)),
                )
            )
    if not parameters:
        raise ValueError("numpy-npz state archives must contain at least one parameter array")
    return ModelStateDocument(
        state=ModelState(parameters=tuple(parameters)),
        runtime_name=runtime_name,
    )


def _write_npz_document(
    path: str | Path,
    document: ModelStateDocument,
    *,
    feature_name: str,
    schema_value: str,
) -> None:
    np = _require_numpy(feature_name)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {
        _NPZ_SCHEMA_KEY: np.array(schema_value),
        _NPZ_VERSION_KEY: np.array(1, dtype=np.int64),
    }
    if document.runtime_name is not None:
        arrays[_NPZ_RUNTIME_KEY] = np.array(document.runtime_name)
    for parameter in document.state.parameters:
        values = np.asarray(parameter.values, dtype=np.float64)
        arrays[parameter.name] = values.reshape(parameter.shape)
    with target.open("wb") as handle:
        np.savez(handle, **arrays)


def _uses_native_torch_checkpoint(path: str | Path) -> bool:
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    return bool(suffixes and suffixes[-1] in {".pt", ".pth"})


def load_model_state_document(
    path: str | Path,
    *,
    state_format: str | None = None,
) -> ModelStateDocument:
    resolved_format = resolve_state_format(state_format, path)
    if resolved_format == JSON_STATE_FORMAT:
        return ModelStateDocument(state=ModelState.from_path(path))
    if resolved_format == NUMPY_NPZ_STATE_FORMAT:
        return _load_npz_document(
            path,
            feature_name="numpy-npz state loading",
            expected_schema=_MODEL_STATE_NPZ_SCHEMA_VALUE,
        )

    if _uses_native_torch_checkpoint(path):
        state, runtime_name = load_torch_checkpoint(path)
        return ModelStateDocument(state=state, runtime_name=runtime_name)

    document = _load_npz_document(
        path,
        feature_name="PyTorch state-dict loading",
        expected_schema=_PYTORCH_STATE_DICT_NPZ_SCHEMA_VALUE,
    )
    state = import_pytorch_state_dict(document.state, runtime_name=document.runtime_name)
    runtime_name = document.runtime_name or infer_training_runtime_from_state(state)
    return ModelStateDocument(state=state, runtime_name=runtime_name)


def load_model_state(
    path: str | Path,
    *,
    state_format: str | None = None,
) -> ModelState:
    return load_model_state_document(path, state_format=state_format).state


def write_model_state_document(
    path: str | Path,
    document: ModelStateDocument,
    *,
    state_format: str | None = None,
) -> None:
    target = Path(path)
    resolved_format = resolve_state_format(state_format, target)
    if resolved_format == JSON_STATE_FORMAT:
        document.state.write_json(target)
        return
    if resolved_format == NUMPY_NPZ_STATE_FORMAT:
        _write_npz_document(
            target,
            document,
            feature_name="numpy-npz state writing",
            schema_value=_MODEL_STATE_NPZ_SCHEMA_VALUE,
        )
        return

    runtime_name = document.runtime_name or infer_training_runtime_from_state(document.state)
    if _uses_native_torch_checkpoint(target):
        write_torch_checkpoint(
            target,
            document.state,
            runtime_name=runtime_name,
        )
        return

    _write_npz_document(
        target,
        ModelStateDocument(
            state=export_pytorch_state_dict(document.state, runtime_name=runtime_name),
            runtime_name=runtime_name,
        ),
        feature_name="PyTorch state-dict writing",
        schema_value=_PYTORCH_STATE_DICT_NPZ_SCHEMA_VALUE,
    )


def write_model_state(
    path: str | Path,
    state: ModelState,
    *,
    state_format: str | None = None,
    runtime_name: str | None = None,
) -> None:
    write_model_state_document(
        path,
        ModelStateDocument(state=state, runtime_name=runtime_name),
        state_format=state_format,
    )
