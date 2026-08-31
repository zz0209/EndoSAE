"""Canonical byte fingerprint and scalar summary for cross-runtime tensors.

The format is deliberately independent of torch/numpy serialization. Runtime
adapters must emit contiguous bytes in the declared canonical dtype/order.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Iterable, Sequence


class TensorFingerprintError(ValueError):
    """Raised when tensor bytes or metadata violate the canonical format."""


DTYPE_FORMAT = {"uint8": ("B", 1), "float32-le": ("<f", 4)}


def fingerprint_tensor_payload(
    payload: bytes, *, dtype: str, shape: Sequence[int], layout: str
) -> str:
    """Hash canonical metadata plus raw C-contiguous tensor bytes."""
    normalized_shape = _validate(dtype, shape, layout, payload)
    header = json.dumps(
        {"dtype": dtype, "layout": layout, "order": "C", "shape": normalized_shape},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"endosae.tensor-bytes.v0\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def summarize_tensor_payload(
    payload: bytes, *, dtype: str, shape: Sequence[int], layout: str
) -> dict[str, float]:
    """Return deterministic population statistics using two-pass ``math.fsum``."""
    _validate(dtype, shape, layout, payload)
    values = list(_values(payload, dtype))
    if not values or any(not math.isfinite(value) for value in values):
        raise TensorFingerprintError("summary requires non-empty finite values")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {"min": min(values), "max": max(values), "mean": mean, "std": math.sqrt(variance)}


def _validate(dtype: str, shape: Sequence[int], layout: str, payload: bytes) -> list[int]:
    if dtype not in DTYPE_FORMAT:
        raise TensorFingerprintError("unsupported canonical dtype")
    if not isinstance(layout, str) or not layout:
        raise TensorFingerprintError("layout must be non-empty")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
        raise TensorFingerprintError("shape must be a non-empty sequence")
    normalized = list(shape)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in normalized):
        raise TensorFingerprintError("shape must contain positive integers")
    if not isinstance(payload, bytes):
        raise TensorFingerprintError("payload must be immutable bytes")
    # Python 3.7 is the pinned EndoFM reference runtime and predates math.prod.
    element_count = 1
    for dimension in normalized:
        element_count *= dimension
    expected = element_count * DTYPE_FORMAT[dtype][1]
    if len(payload) != expected:
        raise TensorFingerprintError("payload length disagrees with dtype and shape")
    return normalized


def _values(payload: bytes, dtype: str) -> Iterable[float]:
    fmt, width = DTYPE_FORMAT[dtype]
    if dtype == "uint8":
        return (float(value) for value in payload)
    return (item[0] for item in struct.iter_unpack(fmt, payload))
