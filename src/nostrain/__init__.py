"""Core protocol tooling for nostrain."""

from .compression import (
    CompressedGradientPayload,
    CompressionCodec,
    compress_delta,
    decompress_payload,
    inspect_payload,
)
from .model import ModelState, TensorLayout, TensorState, apply_delta, compute_delta, state_digest
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
    "ParsedGradientEvent",
    "TensorLayout",
    "TensorState",
    "apply_delta",
    "build_gradient_event",
    "compress_delta",
    "compute_delta",
    "decompress_payload",
    "inspect_payload",
    "parse_gradient_event",
    "state_digest",
]

__version__ = "0.1.0"
