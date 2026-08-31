"""Validate immutable legacy-preprocessed tensors before modern model-port use.

This contract isolates preprocessing from later model-port parity tests.  It does
not establish checkpoint compatibility, model-output parity, or G1 admission.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReferenceTensorExchangeError(ValueError):
    """Raised when a reference tensor exchange record is unsafe or inconsistent."""


HEX64 = frozenset("0123456789abcdefABCDEF")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise ReferenceTensorExchangeError(f"{field} must be a 64-character SHA-256")


def _require_positive_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ReferenceTensorExchangeError("tensor.shape must be a non-empty sequence")
    shape = tuple(value)
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
        raise ReferenceTensorExchangeError("tensor.shape entries must be positive integers")
    return shape


def _contained_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ReferenceTensorExchangeError("tensor.relative_path must be non-empty")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReferenceTensorExchangeError("tensor asset escapes project root") from exc
    return candidate


def validate_reference_tensor_exchange(
    record: Mapping[str, Any], project_root: Path, *, verify_asset: bool = True
) -> Path:
    """Validate provenance and optionally verify the exact on-disk tensor bytes."""

    if not isinstance(record, Mapping):
        raise ReferenceTensorExchangeError("record must be a mapping")
    if record.get("schema_version") != "endosae.reference-tensor-exchange.v0":
        raise ReferenceTensorExchangeError("unsupported schema_version")
    if record.get("purpose") != "legacy-preprocessed-input-for-model-port-parity":
        raise ReferenceTensorExchangeError("unexpected purpose")
    if record.get("g1_admission") is not False:
        raise ReferenceTensorExchangeError("v0 exchange record must not claim G1 admission")

    tensor = record.get("tensor")
    if not isinstance(tensor, Mapping):
        raise ReferenceTensorExchangeError("tensor must be a mapping")
    required_tensor = {
        "relative_path", "sha256", "bytes", "dtype", "byte_order", "layout", "shape"
    }
    missing = sorted(required_tensor.difference(tensor))
    if missing:
        raise ReferenceTensorExchangeError(f"missing tensor fields: {', '.join(missing)}")
    _require_sha256(tensor["sha256"], "tensor.sha256")
    if tensor["dtype"] != "float32" or tensor["byte_order"] != "little":
        raise ReferenceTensorExchangeError("v0 requires little-endian float32 bytes")
    if tensor["layout"] != "C,T,H,W":
        raise ReferenceTensorExchangeError("v0 requires C,T,H,W layout")
    shape = _require_positive_shape(tensor["shape"])
    if len(shape) != 4 or shape[0] != 3:
        raise ReferenceTensorExchangeError("v0 tensor must have shape [3,T,H,W]")
    expected_bytes = 4
    for dim in shape:
        expected_bytes *= dim
    if tensor["bytes"] != expected_bytes:
        raise ReferenceTensorExchangeError("tensor.bytes disagrees with float32 shape")

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ReferenceTensorExchangeError("provenance must be a mapping")
    for field in (
        "checkpoint_sha256", "fixture_manifest_sha256", "preprocessing_plan_sha256",
        "preprocessing_runner_sha256", "raw_asset_manifest_sha256",
    ):
        _require_sha256(provenance.get(field), f"provenance.{field}")
    if provenance.get("runtime_id") != "lr0-cpu":
        raise ReferenceTensorExchangeError("reference tensor must originate from lr0-cpu")
    if provenance.get("stage") not in {"resize", "normalized-crop-0", "normalized-crop-1", "normalized-crop-2"}:
        raise ReferenceTensorExchangeError("unsupported preprocessing stage")

    consumer = record.get("consumer_constraints")
    if not isinstance(consumer, Mapping):
        raise ReferenceTensorExchangeError("consumer_constraints must be a mapping")
    if consumer.get("must_bypass_image_preprocessing") is not True:
        raise ReferenceTensorExchangeError("consumer must bypass image preprocessing")
    if consumer.get("must_verify_hash_before_load") is not True:
        raise ReferenceTensorExchangeError("consumer must verify hash before load")
    if consumer.get("checkpoint_deserialization_authorized") is not False:
        raise ReferenceTensorExchangeError("exchange contract cannot authorize deserialization")

    asset_path = _contained_path(project_root, tensor["relative_path"])
    if verify_asset:
        if not asset_path.is_file():
            raise ReferenceTensorExchangeError("tensor asset is missing")
        if asset_path.stat().st_size != tensor["bytes"]:
            raise ReferenceTensorExchangeError("tensor asset byte size mismatch")
        if _sha256(asset_path).lower() != tensor["sha256"].lower():
            raise ReferenceTensorExchangeError("tensor asset SHA-256 mismatch")
    return asset_path
