from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import base64
import heapq
import math
import struct
from typing import Any
import zlib

from .model import ModelState, TensorLayout

try:
    import zstandard  # type: ignore
except ImportError:  # pragma: no cover - exercised when optional extra is installed
    zstandard = None


_CONTAINER_MAGIC = b"NSCP"
_WIRE_MAGIC = b"NSTR"
_FORMAT_VERSION = 1


class CompressionCodec(str, Enum):
    ZLIB = "zlib"
    ZSTD = "zstd"


_CODEC_TO_ID = {
    CompressionCodec.ZLIB: 1,
    CompressionCodec.ZSTD: 2,
}
_ID_TO_CODEC = {value: key for key, value in _CODEC_TO_ID.items()}
_RAW_HEADER = struct.Struct("<4sBdIIHd")


@dataclass(frozen=True)
class CompressedGradientPayload:
    payload: str
    codec: CompressionCodec
    topk_ratio: float
    total_values: int
    selected_values: int
    parameter_count: int
    scale: float
    wire_bytes: int
    encoded_bytes: int
    dense_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / self.wire_bytes if self.wire_bytes else 0.0

    @property
    def density(self) -> float:
        return self.selected_values / self.total_values if self.total_values else 0.0

    @property
    def compression_label(self) -> str:
        return f"topk-{self.topk_ratio:.4f}-int8-{self.codec.value}"

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "stats": {
                "codec": self.codec.value,
                "topk_ratio": self.topk_ratio,
                "total_values": self.total_values,
                "selected_values": self.selected_values,
                "density": self.density,
                "parameter_count": self.parameter_count,
                "scale": self.scale,
                "wire_bytes": self.wire_bytes,
                "encoded_bytes": self.encoded_bytes,
                "dense_bytes": self.dense_bytes,
                "compression_ratio": self.compression_ratio,
                "compression_label": self.compression_label,
            },
        }


@dataclass(frozen=True)
class _ParsedPayload:
    topk_ratio: float
    total_values: int
    selected_values: int
    scale: float
    layouts: tuple[TensorLayout, ...]
    indices: tuple[int, ...]
    quantized_values: tuple[int, ...]


def _compress_bytes(raw: bytes, codec: CompressionCodec) -> bytes:
    if codec is CompressionCodec.ZLIB:
        return zlib.compress(raw, level=9)
    if codec is CompressionCodec.ZSTD:
        if zstandard is None:
            raise RuntimeError(
                "zstd compression requested but the optional 'zstandard' package is not installed"
            )
        compressor = zstandard.ZstdCompressor(level=9)
        return compressor.compress(raw)
    raise ValueError(f"unsupported codec: {codec}")


def _decompress_bytes(raw: bytes, codec: CompressionCodec) -> bytes:
    if codec is CompressionCodec.ZLIB:
        return zlib.decompress(raw)
    if codec is CompressionCodec.ZSTD:
        if zstandard is None:
            raise RuntimeError(
                "zstd payload received but the optional 'zstandard' package is not installed"
            )
        decompressor = zstandard.ZstdDecompressor()
        return decompressor.decompress(raw)
    raise ValueError(f"unsupported codec: {codec}")


def _select_topk(flat_values: tuple[float, ...], topk_ratio: float) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not 0 < topk_ratio <= 1:
        raise ValueError(f"topk ratio must be within (0, 1], got {topk_ratio}")
    if not flat_values:
        raise ValueError("cannot compress an empty tensor payload")

    selected_count = max(1, math.ceil(len(flat_values) * topk_ratio))
    if selected_count >= len(flat_values):
        indices = tuple(range(len(flat_values)))
    else:
        hottest = heapq.nlargest(
            selected_count,
            range(len(flat_values)),
            key=lambda index: (abs(flat_values[index]), -index),
        )
        indices = tuple(sorted(hottest))
    values = tuple(flat_values[index] for index in indices)
    return indices, values


def _quantize(values: tuple[float, ...]) -> tuple[float, tuple[int, ...]]:
    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs == 0:
        scale = 1.0
        return scale, tuple(0 for _ in values)

    scale = max_abs / 127.0
    quantized: list[int] = []
    for value in values:
        scaled = int(round(value / scale))
        quantized.append(max(-127, min(127, scaled)))
    return scale, tuple(quantized)


def _pack_raw(
    layouts: tuple[TensorLayout, ...],
    topk_ratio: float,
    total_values: int,
    indices: tuple[int, ...],
    quantized_values: tuple[int, ...],
    scale: float,
) -> bytes:
    raw = bytearray()
    raw.extend(
        _RAW_HEADER.pack(
            _WIRE_MAGIC,
            _FORMAT_VERSION,
            float(topk_ratio),
            total_values,
            len(indices),
            len(layouts),
            float(scale),
        )
    )
    for layout in layouts:
        name_bytes = layout.name.encode("utf-8")
        raw.extend(struct.pack("<H", len(name_bytes)))
        raw.extend(name_bytes)
        raw.extend(struct.pack("<H", len(layout.shape)))
        for dimension in layout.shape:
            raw.extend(struct.pack("<I", dimension))
    if indices:
        raw.extend(struct.pack(f"<{len(indices)}I", *indices))
        raw.extend(struct.pack(f"<{len(quantized_values)}b", *quantized_values))
    return bytes(raw)


def _parse_raw(raw: bytes) -> _ParsedPayload:
    if len(raw) < _RAW_HEADER.size:
        raise ValueError("compressed payload is truncated")

    magic, version, topk_ratio, total_values, selected_values, parameter_count, scale = _RAW_HEADER.unpack_from(
        raw, 0
    )
    if magic != _WIRE_MAGIC:
        raise ValueError("compressed payload has an invalid wire magic")
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported compressed payload version: {version}")

    offset = _RAW_HEADER.size
    layouts: list[TensorLayout] = []
    running_offset = 0
    for _ in range(parameter_count):
        if offset + 2 > len(raw):
            raise ValueError("payload truncated while reading parameter name length")
        (name_length,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        if offset + name_length > len(raw):
            raise ValueError("payload truncated while reading parameter name")
        name = raw[offset : offset + name_length].decode("utf-8")
        offset += name_length
        if offset + 2 > len(raw):
            raise ValueError("payload truncated while reading parameter rank")
        (rank,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        shape: list[int] = []
        for _ in range(rank):
            if offset + 4 > len(raw):
                raise ValueError("payload truncated while reading parameter shape")
            (dimension,) = struct.unpack_from("<I", raw, offset)
            offset += 4
            shape.append(dimension)
        length = 1
        for dimension in shape:
            length *= dimension
        layouts.append(
            TensorLayout(
                name=name,
                shape=tuple(shape),
                offset=running_offset,
                length=length,
            )
        )
        running_offset += length

    if running_offset != total_values:
        raise ValueError(
            f"payload tensor layout encodes {running_offset} values but header says {total_values}"
        )

    indices: tuple[int, ...] = ()
    quantized_values: tuple[int, ...] = ()
    if selected_values:
        indices_size = 4 * selected_values
        values_size = selected_values
        if offset + indices_size + values_size > len(raw):
            raise ValueError("payload truncated while reading sparse values")
        indices = struct.unpack_from(f"<{selected_values}I", raw, offset)
        offset += indices_size
        quantized_values = struct.unpack_from(f"<{selected_values}b", raw, offset)
        offset += values_size

    if offset != len(raw):
        raise ValueError("payload contains trailing bytes")

    return _ParsedPayload(
        topk_ratio=topk_ratio,
        total_values=total_values,
        selected_values=selected_values,
        scale=scale,
        layouts=tuple(layouts),
        indices=tuple(indices),
        quantized_values=tuple(quantized_values),
    )


def _decode_container(payload: str) -> tuple[CompressionCodec, bytes, int]:
    container = base64.b64decode(payload.encode("ascii"))
    if len(container) < 6:
        raise ValueError("compressed payload container is truncated")
    if container[:4] != _CONTAINER_MAGIC:
        raise ValueError("compressed payload container has an invalid magic")
    version = container[4]
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported container version: {version}")
    codec_id = container[5]
    codec = _ID_TO_CODEC.get(codec_id)
    if codec is None:
        raise ValueError(f"unsupported container codec id: {codec_id}")
    return codec, container[6:], len(container)


def inspect_payload(payload: str) -> CompressedGradientPayload:
    codec, compressed_bytes, wire_bytes = _decode_container(payload)
    raw = _decompress_bytes(compressed_bytes, codec)
    parsed = _parse_raw(raw)
    return CompressedGradientPayload(
        payload=payload,
        codec=codec,
        topk_ratio=parsed.topk_ratio,
        total_values=parsed.total_values,
        selected_values=parsed.selected_values,
        parameter_count=len(parsed.layouts),
        scale=parsed.scale,
        wire_bytes=wire_bytes,
        encoded_bytes=len(payload.encode("ascii")),
        dense_bytes=parsed.total_values * 4,
    )


def compress_delta(
    delta: ModelState,
    *,
    topk_ratio: float = 0.01,
    codec: CompressionCodec | str = CompressionCodec.ZLIB,
) -> CompressedGradientPayload:
    codec_enum = CompressionCodec(codec)
    flat_values, layouts = delta.flatten()
    indices, selected_values = _select_topk(flat_values, topk_ratio)
    scale, quantized_values = _quantize(selected_values)
    raw = _pack_raw(
        layouts=layouts,
        topk_ratio=topk_ratio,
        total_values=len(flat_values),
        indices=indices,
        quantized_values=quantized_values,
        scale=scale,
    )
    compressed = _compress_bytes(raw, codec_enum)
    container = _CONTAINER_MAGIC + bytes([_FORMAT_VERSION, _CODEC_TO_ID[codec_enum]]) + compressed
    payload = base64.b64encode(container).decode("ascii")
    return CompressedGradientPayload(
        payload=payload,
        codec=codec_enum,
        topk_ratio=topk_ratio,
        total_values=len(flat_values),
        selected_values=len(indices),
        parameter_count=delta.parameter_count,
        scale=scale,
        wire_bytes=len(container),
        encoded_bytes=len(payload.encode("ascii")),
        dense_bytes=len(flat_values) * 4,
    )


def decompress_payload(payload: str | CompressedGradientPayload) -> ModelState:
    raw_payload = payload.payload if isinstance(payload, CompressedGradientPayload) else payload
    codec, compressed_bytes, _wire_bytes = _decode_container(raw_payload)
    raw = _decompress_bytes(compressed_bytes, codec)
    parsed = _parse_raw(raw)
    flat_values = [0.0] * parsed.total_values
    for index, quantized_value in zip(parsed.indices, parsed.quantized_values):
        if index >= parsed.total_values:
            raise ValueError(
                f"payload index {index} is out of bounds for {parsed.total_values} values"
            )
        flat_values[index] = quantized_value * parsed.scale
    return ModelState.from_flat_values(parsed.layouts, tuple(flat_values))
