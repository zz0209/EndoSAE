"""Admission rules for the future isolated checkpoint inventory report."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class CheckpointInventoryContractError(ValueError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "6fc7a64a044f1eff3b7f9eb233df37a3607735848d534364f12c2aebad1aea70"
REQUIRED_TOP_LEVEL = ["student", "teacher", "optimizer", "epoch", "args", "dino_loss", "fp16_scaler"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_link(link: Mapping[str, Any], label: str) -> None:
    path = (PROJECT_ROOT / str(link.get("path", ""))).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise CheckpointInventoryContractError(f"{label} escapes project") from exc
    if not path.is_file() or _sha256(path) != link.get("sha256"):
        raise CheckpointInventoryContractError(f"{label} hash mismatch")


def validate_inventory_contract(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != "endosae.isolated-checkpoint-inventory-contract.v0":
        raise CheckpointInventoryContractError("unsupported schema")
    if record.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise CheckpointInventoryContractError("checkpoint identity drift")
    _verify_link(record.get("safe_globals_policy", {}), "safe_globals_policy")
    _verify_link(record.get("static_parameter_registry", {}), "static_parameter_registry")
    loader = record.get("required_loader", {})
    if loader != {
        "api": "torch.load",
        "weights_only": True,
        "map_location": "cpu",
        "network_enabled": False,
        "sensitive_mounts_present": False,
        "safe_globals_scope": "exact-frozen-policy-only",
    }:
        raise CheckpointInventoryContractError("loader boundary drift")
    output = record.get("required_output", {})
    if output.get("top_level_key_order") != REQUIRED_TOP_LEVEL:
        raise CheckpointInventoryContractError("top-level expectation drift")
    required_fields = {
        "checkpoint_sha256", "loader_environment_sha256", "isolation_attestation_sha256",
        "top_level_keys", "recursive_tensor_count", "recursive_tensor_numel",
        "tensor_records", "non_tensor_type_summary", "student_teacher_mapping_summary",
        "unexpected_globals", "status",
    }
    if set(output.get("fields", [])) != required_fields:
        raise CheckpointInventoryContractError("inventory fields incomplete")
    tensor_fields = output.get("tensor_record_fields")
    if tensor_fields != ["path", "shape", "dtype", "numel", "requires_grad", "storage_reference"]:
        raise CheckpointInventoryContractError("tensor record schema drift")
    stops = record.get("hard_stops", {})
    if not all(stops.get(key) is True for key in (
        "unexpected_global_or_dynamic_type", "weights_only_failure", "top_level_key_mismatch",
        "nonfinite_or_invalid_tensor_metadata", "duplicate_tensor_path", "policy_or_checkpoint_hash_mismatch",
    )):
        raise CheckpointInventoryContractError("hard-stop policy incomplete")
    if record.get("checkpoint_deserialization_performed") is not False or record.get("g1_admission") is not False:
        raise CheckpointInventoryContractError("contract cannot claim execution or G1")

