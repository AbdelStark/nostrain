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
from .relay import (
    CollectedGradientEvent,
    RelayCollectionResult,
    RelayPublishResult,
    collect_gradient_events,
    publish_gradient_event,
)

__all__ = [
    "CompressedGradientPayload",
    "CompressionCodec",
    "CollectedGradientEvent",
    "GradientEventMetadata",
    "ModelState",
    "NostrainEvent",
    "OuterStepResult",
    "ParsedGradientEvent",
    "RelayCollectionResult",
    "RelayPublishResult",
    "TensorLayout",
    "TensorState",
    "add_states",
    "aggregate_deltas",
    "apply_delta",
    "build_gradient_event",
    "compress_delta",
    "collect_gradient_events",
    "compute_delta",
    "decompress_payload",
    "inspect_payload",
    "nesterov_outer_step",
    "parse_gradient_event",
    "publish_gradient_event",
    "scale_state",
    "state_digest",
    "subtract_states",
    "zeros_like",
]

__version__ = "0.2.0"
