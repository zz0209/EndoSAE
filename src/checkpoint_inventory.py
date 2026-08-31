"""Semantic validation for isolated checkpoint tensor inventory records."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping


class CheckpointInventoryError(ValueError):
    """Raised when an inventory or its behavior gate is inconsistent."""


def validate_checkpoint_inventory(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise CheckpointInventoryError("inventory must be a mapping")
    required = {
        "schema_version", "status", "checkpoint_sha256", "preflight_report_sha256",
        "isolated_execution", "network_disabled", "sensitive_mounts_absent",
        "loader_runtime", "loader_api", "weights_only_requested",
        "weights_only_succeeded", "top_level_keys", "selected_state_dict_path",
        "prefix_filter", "tensors", "missing_keys", "unexpected_keys",
        "allowlist", "behavior_testing_allowed",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise CheckpointInventoryError(f"missing inventory fields: {', '.join(missing)}")
    if record["schema_version"] != "endosae.checkpoint-inventory.v0":
        raise CheckpointInventoryError("unsupported inventory schema")
    if record["status"] not in {"pass", "fail", "invalid"}:
        raise CheckpointInventoryError("invalid inventory status")
    for key in ("checkpoint_sha256", "preflight_report_sha256"):
        _require_sha256(record[key], key)
    for key in (
        "isolated_execution", "network_disabled", "sensitive_mounts_absent",
        "weights_only_requested", "weights_only_succeeded", "behavior_testing_allowed",
    ):
        if not isinstance(record[key], bool):
            raise CheckpointInventoryError(f"{key} must be boolean")
    for key in ("loader_runtime", "loader_api", "selected_state_dict_path", "prefix_filter"):
        _require_string(record[key], key)
    top_level = _unique_string_list(record["top_level_keys"], "top_level_keys")
    missing_keys = _unique_string_list(record["missing_keys"], "missing_keys")
    unexpected_keys = _unique_string_list(record["unexpected_keys"], "unexpected_keys")

    tensors = record["tensors"]
    if not isinstance(tensors, list):
        raise CheckpointInventoryError("tensors must be a list")
    tensor_names = set()
    for tensor in tensors:
        if not isinstance(tensor, Mapping):
            raise CheckpointInventoryError("tensor inventory item must be a mapping")
        if set(tensor) != {"name", "shape", "dtype", "numel"}:
            raise CheckpointInventoryError("tensor item fields must be exact")
        _require_string(tensor["name"], "tensor.name")
        _require_string(tensor["dtype"], "tensor.dtype")
        if tensor["name"] in tensor_names:
            raise CheckpointInventoryError("tensor names must be unique")
        tensor_names.add(tensor["name"])
        shape = tensor["shape"]
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        ):
            raise CheckpointInventoryError("tensor shape must contain non-negative integers")
        expected_numel = math.prod(shape)
        if tensor["numel"] != expected_numel:
            raise CheckpointInventoryError("tensor numel disagrees with shape")

    allowlist = record["allowlist"]
    if not isinstance(allowlist, Mapping):
        raise CheckpointInventoryError("allowlist must be a mapping")
    required_allowlist = {
        "status", "frozen_before_behavior_testing", "missing_keys", "unexpected_keys"
    }
    if set(allowlist) != required_allowlist:
        raise CheckpointInventoryError("allowlist fields must be exact")
    if allowlist["status"] not in {"not-frozen", "frozen"}:
        raise CheckpointInventoryError("invalid allowlist status")
    if not isinstance(allowlist["frozen_before_behavior_testing"], bool):
        raise CheckpointInventoryError("allowlist freeze flag must be boolean")
    allowed_missing = _unique_string_list(allowlist["missing_keys"], "allowlist.missing_keys")
    allowed_unexpected = _unique_string_list(
        allowlist["unexpected_keys"], "allowlist.unexpected_keys"
    )

    if record["status"] == "pass":
        if not (
            record["isolated_execution"]
            and record["network_disabled"]
            and record["sensitive_mounts_absent"]
            and record["weights_only_requested"]
            and record["weights_only_succeeded"]
        ):
            raise CheckpointInventoryError("pass requires isolated successful weights-only inventory")
        if not top_level or not tensors:
            raise CheckpointInventoryError("pass requires non-empty key and tensor inventories")

    if record["behavior_testing_allowed"]:
        if record["status"] != "pass":
            raise CheckpointInventoryError("behavior testing requires passed inventory")
        if allowlist["status"] != "frozen" or not allowlist["frozen_before_behavior_testing"]:
            raise CheckpointInventoryError("behavior testing requires pre-frozen allowlist")
        if sorted(missing_keys) != sorted(allowed_missing):
            raise CheckpointInventoryError("missing keys disagree with frozen allowlist")
        if sorted(unexpected_keys) != sorted(allowed_unexpected):
            raise CheckpointInventoryError("unexpected keys disagree with frozen allowlist")


def _require_sha256(value: Any, key: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointInventoryError(f"{key} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CheckpointInventoryError(f"{key} must be hexadecimal") from exc


def _require_string(value: Any, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointInventoryError(f"{key} must be non-empty")


def _unique_string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CheckpointInventoryError(f"{key} must be a string list")
    if len(value) != len(set(value)):
        raise CheckpointInventoryError(f"{key} must be unique")
    return value


def compare_parameter_mapping(
    checkpoint_tensors: list[Mapping[str, Any]],
    model_tensors: list[Mapping[str, Any]],
    *,
    checkpoint_prefix: str,
    model_prefix: str = "",
) -> dict[str, Any]:
    """Compare a selected checkpoint branch with an instantiated model schema.

    This function consumes metadata only and never loads or copies tensor values.
    Prefix removal is explicit; no suffix matching or fallback renaming is allowed.
    """
    if not checkpoint_prefix:
        raise CheckpointInventoryError("checkpoint_prefix must be non-empty")
    checkpoint = _index_mapping_tensors(checkpoint_tensors, checkpoint_prefix, "checkpoint")
    model = _index_mapping_tensors(model_tensors, model_prefix, "model")
    checkpoint_names = set(checkpoint)
    model_names = set(model)
    missing = sorted(model_names - checkpoint_names)
    unexpected = sorted(checkpoint_names - model_names)
    shape_mismatches = []
    dtype_mismatches = []
    for name in sorted(checkpoint_names & model_names):
        if checkpoint[name]["shape"] != model[name]["shape"]:
            shape_mismatches.append({
                "name": name,
                "checkpoint_shape": checkpoint[name]["shape"],
                "model_shape": model[name]["shape"],
            })
        if _canonical_dtype(checkpoint[name]["dtype"]) != _canonical_dtype(model[name]["dtype"]):
            dtype_mismatches.append({
                "name": name,
                "checkpoint_dtype": checkpoint[name]["dtype"],
                "model_dtype": model[name]["dtype"],
            })
    passed = not (missing or unexpected or shape_mismatches or dtype_mismatches)
    return {
        "schema_version": "endosae.parameter-mapping-result.v0",
        "checkpoint_prefix": checkpoint_prefix,
        "model_prefix": model_prefix,
        "checkpoint_tensor_count": len(checkpoint),
        "model_tensor_count": len(model),
        "canonical_name_sha256": _canonical_name_digest(sorted(checkpoint)),
        "missing_in_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "exact_mapping_pass": passed,
        "fallback_used": False,
        "behavior_testing_allowed": False,
    }


def _index_mapping_tensors(
    tensors: list[Mapping[str, Any]], prefix: str, label: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(tensors, list):
        raise CheckpointInventoryError(f"{label} tensors must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for tensor in tensors:
        if not isinstance(tensor, Mapping) or set(tensor) != {"name", "shape", "dtype", "numel"}:
            raise CheckpointInventoryError(f"invalid {label} tensor metadata")
        _require_string(tensor["name"], f"{label}.name")
        _require_string(tensor["dtype"], f"{label}.dtype")
        shape = tensor["shape"]
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        ):
            raise CheckpointInventoryError(f"invalid {label} tensor shape")
        if tensor["numel"] != math.prod(shape):
            raise CheckpointInventoryError(f"invalid {label} tensor numel")
        name = tensor["name"]
        if not name.startswith(prefix):
            continue
        canonical = name[len(prefix):]
        if not canonical:
            raise CheckpointInventoryError(f"empty canonical {label} name")
        if canonical in indexed:
            raise CheckpointInventoryError(f"duplicate canonical {label} name: {canonical}")
        indexed[canonical] = tensor
    if not indexed:
        raise CheckpointInventoryError(f"no {label} tensors matched prefix")
    return indexed


def _canonical_dtype(value: str) -> str:
    return value.removeprefix("torch.")


def _canonical_name_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()
