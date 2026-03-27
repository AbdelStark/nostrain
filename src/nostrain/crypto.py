from __future__ import annotations

import hashlib
import string

SECP256K1_FIELD_SIZE = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_GENERATOR = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
_HEX_DIGITS = set(string.hexdigits)

Point = tuple[int, int]


def normalize_hex_string(value: str, *, byte_length: int, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != byte_length * 2:
        raise ValueError(f"{label} must be {byte_length * 2} hex characters")
    if any(character not in _HEX_DIGITS for character in normalized):
        raise ValueError(f"{label} must be hex")
    return normalized


def normalize_secret_key_hex(secret_key_hex: str) -> str:
    normalized = normalize_hex_string(secret_key_hex, byte_length=32, label="secret key")
    scalar = int(normalized, 16)
    if not 1 <= scalar <= SECP256K1_ORDER - 1:
        raise ValueError("secret key must be within secp256k1 order range")
    return normalized


def normalize_public_key_hex(public_key_hex: str) -> str:
    normalized = normalize_hex_string(public_key_hex, byte_length=32, label="public key")
    if _lift_x(int(normalized, 16)) is None:
        raise ValueError("public key is not a valid secp256k1 x-only coordinate")
    return normalized


def normalize_signature_hex(signature_hex: str) -> str:
    return normalize_hex_string(signature_hex, byte_length=64, label="signature")


def secret_key_to_public_key(secret_key_hex: str) -> str:
    secret_key = int(normalize_secret_key_hex(secret_key_hex), 16)
    public_point = _point_mul(_GENERATOR, secret_key)
    assert public_point is not None
    return _bytes_from_int(public_point[0]).hex()


def schnorr_sign(
    message: bytes,
    secret_key_hex: str,
    *,
    aux_rand: bytes | None = None,
) -> str:
    secret_key = int(normalize_secret_key_hex(secret_key_hex), 16)
    if aux_rand is None:
        aux_rand = b"\x00" * 32
    if len(aux_rand) != 32:
        raise ValueError("aux_rand must be exactly 32 bytes")

    public_point = _point_mul(_GENERATOR, secret_key)
    assert public_point is not None
    adjusted_secret = secret_key if _has_even_y(public_point) else SECP256K1_ORDER - secret_key
    t = _xor_bytes(
        _bytes_from_int(adjusted_secret),
        _tagged_hash("BIP0340/aux", aux_rand),
    )
    nonce_scalar = int.from_bytes(
        _tagged_hash(
            "BIP0340/nonce",
            t + _bytes_from_int(public_point[0]) + message,
        ),
        byteorder="big",
    ) % SECP256K1_ORDER
    if nonce_scalar == 0:
        raise RuntimeError("derived nonce is invalid")

    nonce_point = _point_mul(_GENERATOR, nonce_scalar)
    assert nonce_point is not None
    adjusted_nonce = nonce_scalar if _has_even_y(nonce_point) else SECP256K1_ORDER - nonce_scalar
    if adjusted_nonce != nonce_scalar:
        nonce_point = _point_mul(_GENERATOR, adjusted_nonce)
        assert nonce_point is not None

    challenge = int.from_bytes(
        _tagged_hash(
            "BIP0340/challenge",
            _bytes_from_int(nonce_point[0]) + _bytes_from_int(public_point[0]) + message,
        ),
        byteorder="big",
    ) % SECP256K1_ORDER
    signature = (
        _bytes_from_int(nonce_point[0])
        + _bytes_from_int((adjusted_nonce + challenge * adjusted_secret) % SECP256K1_ORDER)
    )

    if not schnorr_verify(message, _bytes_from_int(public_point[0]).hex(), signature.hex()):
        raise RuntimeError("generated signature failed verification")
    return signature.hex()


def schnorr_verify(message: bytes, public_key_hex: str, signature_hex: str) -> bool:
    public_key = normalize_public_key_hex(public_key_hex)
    signature = bytes.fromhex(normalize_signature_hex(signature_hex))
    public_point = _lift_x(int(public_key, 16))
    assert public_point is not None

    r = int.from_bytes(signature[:32], byteorder="big")
    s = int.from_bytes(signature[32:], byteorder="big")
    if r >= SECP256K1_FIELD_SIZE or s >= SECP256K1_ORDER:
        return False

    challenge = int.from_bytes(
        _tagged_hash(
            "BIP0340/challenge",
            signature[:32] + bytes.fromhex(public_key) + message,
        ),
        byteorder="big",
    ) % SECP256K1_ORDER
    candidate = _point_add(
        _point_mul(_GENERATOR, s),
        _point_mul(public_point, SECP256K1_ORDER - challenge),
    )
    return (
        candidate is not None
        and _has_even_y(candidate)
        and candidate[0] == r
    )


def _tagged_hash(tag: str, payload: bytes) -> bytes:
    tag_hash = hashlib.sha256(tag.encode("utf-8")).digest()
    return hashlib.sha256(tag_hash + tag_hash + payload).digest()


def _bytes_from_int(value: int) -> bytes:
    return value.to_bytes(32, byteorder="big")


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(left_byte ^ right_byte for left_byte, right_byte in zip(left, right))


def _has_even_y(point: Point) -> bool:
    return point[1] % 2 == 0


def _lift_x(x_coordinate: int) -> Point | None:
    if x_coordinate >= SECP256K1_FIELD_SIZE:
        return None
    y_squared = (pow(x_coordinate, 3, SECP256K1_FIELD_SIZE) + 7) % SECP256K1_FIELD_SIZE
    y_coordinate = pow(
        y_squared,
        (SECP256K1_FIELD_SIZE + 1) // 4,
        SECP256K1_FIELD_SIZE,
    )
    if pow(y_coordinate, 2, SECP256K1_FIELD_SIZE) != y_squared:
        return None
    return (
        x_coordinate,
        y_coordinate if y_coordinate % 2 == 0 else SECP256K1_FIELD_SIZE - y_coordinate,
    )


def _point_add(left: Point | None, right: Point | None) -> Point | None:
    if left is None:
        return right
    if right is None:
        return left
    if left[0] == right[0] and left[1] != right[1]:
        return None
    if left == right:
        slope = (
            3
            * left[0]
            * left[0]
            * pow(2 * left[1], SECP256K1_FIELD_SIZE - 2, SECP256K1_FIELD_SIZE)
        ) % SECP256K1_FIELD_SIZE
    else:
        slope = (
            (right[1] - left[1])
            * pow(right[0] - left[0], SECP256K1_FIELD_SIZE - 2, SECP256K1_FIELD_SIZE)
        ) % SECP256K1_FIELD_SIZE
    x_coordinate = (slope * slope - left[0] - right[0]) % SECP256K1_FIELD_SIZE
    y_coordinate = (slope * (left[0] - x_coordinate) - left[1]) % SECP256K1_FIELD_SIZE
    return (x_coordinate, y_coordinate)


def _point_mul(point: Point | None, scalar: int) -> Point | None:
    result: Point | None = None
    addend = point
    for bit_index in range(256):
        if (scalar >> bit_index) & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
    return result
