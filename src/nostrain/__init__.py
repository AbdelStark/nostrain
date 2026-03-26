"""Core protocol tooling for nostrain."""

from .aggregation import OuterStepResult, aggregate_deltas, nesterov_outer_step
from .compression import (
    CompressedGradientPayload,
    CompressionCodec,
    compress_delta,
    decompress_payload,
    inspect_payload,
)
from .model import (
    ModelState,
    TensorLayout,
    TensorState,
    add_states,
    apply_delta,
    compute_delta,
    scale_state,
    state_digest,
    subtract_states,
    zeros_like,
)
from .protocol import (
    GradientEventMetadata,
    NostrainEvent,
    ParsedGradientEvent,
    build_gradient_event,
    parse_gradient_event,
)

__all__ = [
    "CompressedGradientPayload",
    "CompressionCodec",
    "GradientEventMetadata",
    "ModelState",
    "NostrainEvent",
    "OuterStepResult",
    "ParsedGradientEvent",
    "TensorLayout",
    "TensorState",
    "add_states",
    "aggregate_deltas",
    "apply_delta",
    "build_gradient_event",
    "compress_delta",
    "compute_delta",
    "decompress_payload",
    "inspect_payload",
    "nesterov_outer_step",
    "parse_gradient_event",
    "scale_state",
    "state_digest",
    "subtract_states",
    "zeros_like",
]

__version__ = "0.1.0"
