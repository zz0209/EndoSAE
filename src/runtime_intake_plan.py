"""Validation for the checkpoint/runtime intake stop boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class RuntimeIntakePlanError(ValueError):
    """Raised when an unresolved intake plan accidentally grants execution."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_runtime_intake_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or plan.get("schema_version") != "endosae.runtime-intake-plan.v0":
        raise RuntimeIntakePlanError("invalid runtime intake plan schema")
    for section in ("host", "official_environment", "checkpoint", "security_boundary", "load_acceptance"):
        if not isinstance(plan.get(section), Mapping):
            raise RuntimeIntakePlanError(f"missing plan section: {section}")
    stages = plan.get("ordered_stages")
    if not isinstance(stages, list) or len(stages) < 7 or len(stages) != len(set(stages)):
        raise RuntimeIntakePlanError("ordered stages must be unique and complete")
    checkpoint = plan["checkpoint"]
    security = plan["security_boundary"]
    acceptance = plan["load_acceptance"]
    rights_status = checkpoint.get("rights_status")
    if rights_status not in {
        "blocked-unresolved",
        "upstream-unresolved-internal-use-authorized",
        "cleared",
    }:
        raise RuntimeIntakePlanError("invalid checkpoint rights status")
    rights_path = (PROJECT_ROOT / str(checkpoint.get("rights_evidence_path", ""))).resolve()
    try:
        rights_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeIntakePlanError("rights evidence path escapes project root") from exc
    if not rights_path.is_file() or _sha256(rights_path) != checkpoint.get("rights_evidence_sha256"):
        raise RuntimeIntakePlanError("rights evidence hash mismatch")
    static_audit_path = (PROJECT_ROOT / str(checkpoint.get("pickle_static_audit_path", ""))).resolve()
    try:
        static_audit_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeIntakePlanError("pickle static audit path escapes project root") from exc
    if not static_audit_path.is_file() or _sha256(static_audit_path) != checkpoint.get("pickle_static_audit_sha256"):
        raise RuntimeIntakePlanError("pickle static audit hash mismatch")
    no_isolation = not any(
        plan["host"].get(key) is True
        for key in ("docker_available", "podman_available", "wsl_usable_by_current_process")
    )
    capability_path = (PROJECT_ROOT / str(security.get("capability_record_path", ""))).resolve()
    try:
        capability_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeIntakePlanError("isolation capability path escapes project root") from exc
    if not capability_path.is_file() or _sha256(capability_path) != security.get("capability_record_sha256"):
        raise RuntimeIntakePlanError("isolation capability hash mismatch")
    safe_policy_path = (PROJECT_ROOT / str(security.get("safe_globals_policy_path", ""))).resolve()
    try:
        safe_policy_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeIntakePlanError("safe-globals policy path escapes project root") from exc
    if not safe_policy_path.is_file() or _sha256(safe_policy_path) != security.get("safe_globals_policy_sha256"):
        raise RuntimeIntakePlanError("safe-globals policy hash mismatch")
    if rights_status == "blocked-unresolved":
        if checkpoint.get("download_authorized") is not False:
            raise RuntimeIntakePlanError("unresolved rights must block download authorization")
        if checkpoint.get("deserialization_authorized") is not False:
            raise RuntimeIntakePlanError("unresolved rights must block deserialization")
    if rights_status == "upstream-unresolved-internal-use-authorized":
        if checkpoint.get("download_authorized") is not True:
            raise RuntimeIntakePlanError("user-authorized internal use must admit download")
        if checkpoint.get("deserialization_authorized") is not False:
            raise RuntimeIntakePlanError("internal-use authorization alone cannot admit deserialization")
    if no_isolation and security.get("host_deserialization_allowed") is not False:
        raise RuntimeIntakePlanError("absence of isolation must block host deserialization")
    if security.get("legacy_pickle_load_allowed") is not False:
        raise RuntimeIntakePlanError("legacy pickle load cannot be pre-authorized")
    if acceptance.get("strict_false_message_is_not_a_pass") is not True:
        raise RuntimeIntakePlanError("non-strict load output cannot establish acceptance")
    if acceptance.get("default_weights_only_expected_to_succeed") is not False:
        raise RuntimeIntakePlanError("static unsafe globals must block default weights-only expectation")
    if acceptance.get("unsafe_globals_require_frozen_review") != [
        "argparse.Namespace", "numpy.core.multiarray.scalar", "numpy.dtype", "numpy.dtypes.Float64DType"
    ]:
        raise RuntimeIntakePlanError("unsafe-global review set drift")
    if acceptance.get("safe_globals_policy_frozen") is not True:
        raise RuntimeIntakePlanError("safe-globals policy must be frozen before first attempt")
    if acceptance.get("formal_activation_cache_allowed") is not False:
        raise RuntimeIntakePlanError("formal cache must remain blocked before G1")
