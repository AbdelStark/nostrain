from __future__ import annotations

from .model import ModelState, TensorState
from .runtime import (
    LINEAR_BIAS_PARAMETER,
    LINEAR_REGRESSION_RUNTIME,
    LINEAR_WEIGHT_PARAMETER,
    MLP_HIDDEN_BIAS_PARAMETER,
    MLP_HIDDEN_WEIGHT_PARAMETER,
    MLP_OUTPUT_BIAS_PARAMETER,
    MLP_OUTPUT_WEIGHT_PARAMETER,
    MLP_REGRESSION_RUNTIME,
    infer_training_runtime_from_state,
    resolve_training_runtime,
)

PYTORCH_LINEAR_BIAS_KEY = "bias"
PYTORCH_LINEAR_WEIGHT_KEY = "weight"
PYTORCH_MLP_HIDDEN_BIAS_KEY = "hidden.bias"
PYTORCH_MLP_HIDDEN_WEIGHT_KEY = "hidden.weight"
PYTORCH_MLP_OUTPUT_BIAS_KEY = "output.bias"
PYTORCH_MLP_OUTPUT_WEIGHT_KEY = "output.weight"

_LINEAR_EXPORT_MAP = {
    LINEAR_BIAS_PARAMETER: PYTORCH_LINEAR_BIAS_KEY,
    LINEAR_WEIGHT_PARAMETER: PYTORCH_LINEAR_WEIGHT_KEY,
}
_MLP_EXPORT_MAP = {
    MLP_HIDDEN_BIAS_PARAMETER: PYTORCH_MLP_HIDDEN_BIAS_KEY,
    MLP_HIDDEN_WEIGHT_PARAMETER: PYTORCH_MLP_HIDDEN_WEIGHT_KEY,
    MLP_OUTPUT_BIAS_PARAMETER: PYTORCH_MLP_OUTPUT_BIAS_KEY,
    MLP_OUTPUT_WEIGHT_PARAMETER: PYTORCH_MLP_OUTPUT_WEIGHT_KEY,
}
_PYTORCH_EXPORT_MAPS = {
    LINEAR_REGRESSION_RUNTIME: _LINEAR_EXPORT_MAP,
    MLP_REGRESSION_RUNTIME: _MLP_EXPORT_MAP,
}
_PYTORCH_IMPORT_MAPS = {
    runtime_name: {target_name: source_name for source_name, target_name in export_map.items()}
    for runtime_name, export_map in _PYTORCH_EXPORT_MAPS.items()
}


def _sorted_state(parameters: list[TensorState]) -> ModelState:
    return ModelState(parameters=tuple(sorted(parameters, key=lambda parameter: parameter.name)))


def _strip_shared_prefix(
    parameter_map: dict[str, TensorState],
) -> tuple[str, dict[str, TensorState]] | None:
    if not parameter_map:
        return None
    if any("." not in name for name in parameter_map):
        return None
    first_segments = {name.split(".", 1)[0] for name in parameter_map}
    if len(first_segments) != 1:
        return None
    prefix = next(iter(first_segments))
    prefix_with_separator = f"{prefix}."
    stripped_map: dict[str, TensorState] = {}
    for name, tensor in parameter_map.items():
        stripped_name = name[len(prefix_with_separator) :]
        if stripped_name in stripped_map:
            raise ValueError(
                "PyTorch state dict contains duplicate parameter names after "
                f"stripping shared prefix {prefix_with_separator!r}"
            )
        stripped_map[stripped_name] = tensor
    return prefix, stripped_map


def _normalize_pytorch_parameters(
    state_dict: ModelState,
    *,
    runtime_name: str | None = None,
) -> tuple[str, dict[str, TensorState]]:
    explicit_runtime = resolve_training_runtime(runtime_name) if runtime_name is not None else None
    current_map = state_dict.parameter_map()
    expected_key_sets = {
        candidate_runtime: frozenset(mapping)
        for candidate_runtime, mapping in _PYTORCH_IMPORT_MAPS.items()
    }

    while True:
        current_keys = frozenset(current_map)
        if explicit_runtime is not None:
            if current_keys == expected_key_sets[explicit_runtime]:
                return explicit_runtime, current_map
        else:
            for candidate_runtime, expected_keys in expected_key_sets.items():
                if current_keys == expected_keys:
                    return candidate_runtime, current_map

        stripped = _strip_shared_prefix(current_map)
        if stripped is None:
            break
        _, current_map = stripped

    if explicit_runtime is not None:
        expected_keys = ", ".join(sorted(expected_key_sets[explicit_runtime]))
        raise ValueError(
            "PyTorch state dict does not match the expected parameter layout for "
            f"{explicit_runtime!r}; expected keys: {expected_keys}"
        )
    raise ValueError(
        "could not infer runtime from PyTorch state dict parameter names: "
        + ", ".join(sorted(state_dict.parameter_map()))
    )


def infer_training_runtime_from_pytorch_state_dict(state_dict: ModelState) -> str:
    runtime_name, _ = _normalize_pytorch_parameters(state_dict)
    return runtime_name


def export_pytorch_state_dict(
    state: ModelState,
    *,
    runtime_name: str | None = None,
) -> ModelState:
    resolved_runtime = resolve_training_runtime(runtime_name, state=state)
    export_map = _PYTORCH_EXPORT_MAPS[resolved_runtime]
    parameter_map = state.parameter_map()
    exported_parameters = [
        TensorState(
            name=state_dict_name,
            shape=parameter_map[canonical_name].shape,
            values=parameter_map[canonical_name].values,
        )
        for canonical_name, state_dict_name in sorted(
            export_map.items(),
            key=lambda item: item[1],
        )
    ]
    return _sorted_state(exported_parameters)


def import_pytorch_state_dict(
    state_dict: ModelState,
    *,
    runtime_name: str | None = None,
) -> ModelState:
    resolved_runtime, normalized_parameters = _normalize_pytorch_parameters(
        state_dict,
        runtime_name=runtime_name,
    )
    import_map = _PYTORCH_IMPORT_MAPS[resolved_runtime]
    canonical_state = _sorted_state(
        [
            TensorState(
                name=import_map[state_dict_name],
                shape=tensor.shape,
                values=tensor.values,
            )
            for state_dict_name, tensor in normalized_parameters.items()
        ]
    )
    inferred_runtime = infer_training_runtime_from_state(canonical_state)
    if inferred_runtime != resolved_runtime:
        raise ValueError(
            "PyTorch state dict runtime layout resolved to "
            f"{resolved_runtime!r}, but tensor shapes matched {inferred_runtime!r}"
        )
    return canonical_state
