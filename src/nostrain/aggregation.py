from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model import ModelState, add_states, apply_delta, scale_state, zeros_like


@dataclass(frozen=True)
class OuterStepResult:
    aggregated_delta: ModelState
    momentum_state: ModelState
    update_delta: ModelState
    next_state: ModelState


def aggregate_deltas(deltas: Iterable[ModelState]) -> ModelState:
    delta_list = list(deltas)
    if not delta_list:
        raise ValueError("at least one delta is required for aggregation")

    running = delta_list[0]
    for delta in delta_list[1:]:
        running = add_states(running, delta)
    return scale_state(running, 1.0 / len(delta_list))


def nesterov_outer_step(
    base_state: ModelState,
    aggregated_delta: ModelState,
    *,
    learning_rate: float = 0.7,
    momentum: float = 0.9,
    previous_momentum: ModelState | None = None,
) -> OuterStepResult:
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be within [0, 1)")

    prior_velocity = zeros_like(aggregated_delta) if previous_momentum is None else previous_momentum
    velocity = add_states(scale_state(prior_velocity, momentum), aggregated_delta)
    nesterov_update = add_states(aggregated_delta, scale_state(velocity, momentum))
    update_delta = scale_state(nesterov_update, learning_rate)
    next_state = apply_delta(base_state, update_delta)
    return OuterStepResult(
        aggregated_delta=aggregated_delta,
        momentum_state=velocity,
        update_delta=update_delta,
        next_state=next_state,
    )
