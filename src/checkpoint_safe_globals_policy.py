"""Validate the exact, pre-execution weights-only allowlist and stop policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class CheckpointSafeGlobalsPolicyError(ValueError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SYMBOLS = [
    "argparse.Namespace",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "numpy.dtypes.Float64DType",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_safe_globals_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != "endosae.checkpoint-safe-globals-policy.v0":
        raise CheckpointSafeGlobalsPolicyError("unsupported schema")
    if policy.get("status") != "frozen-awaiting-isolation":
        raise CheckpointSafeGlobalsPolicyError("policy must remain frozen and blocked")
    for key in ("static_audit", "loader_environment"):
        link = policy.get(key)
        if not isinstance(link, Mapping):
            raise CheckpointSafeGlobalsPolicyError(f"missing link: {key}")
        path = (PROJECT_ROOT / str(link.get("path", ""))).resolve()
        try:
            path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise CheckpointSafeGlobalsPolicyError(f"{key} escapes project") from exc
        if not path.is_file() or _sha256(path) != link.get("sha256"):
            raise CheckpointSafeGlobalsPolicyError(f"{key} hash mismatch")
    call = policy.get("loader_call", {})
    expected_call = {"api": "torch.load", "weights_only": True, "map_location": "cpu", "safe_globals_context_only": True, "clear_safe_globals_before_and_after": True, "torch_force_weights_only_load": "1", "torch_force_no_weights_only_load_must_be_absent": True}
    if call != expected_call:
        raise CheckpointSafeGlobalsPolicyError("loader call controls drift")
    allowlist = policy.get("exact_allowlist")
    symbols = [item.get("symbol") for item in allowlist or [] if isinstance(item, Mapping)]
    if symbols != EXPECTED_SYMBOLS or len(set(symbols)) != len(symbols):
        raise CheckpointSafeGlobalsPolicyError("safe-global allowlist drift")
    context = policy.get("pickle_context_evidence", {})
    if context.get("dtype_code") != "f8" or context.get("byte_order") != "<":
        raise CheckpointSafeGlobalsPolicyError("dynamic dtype evidence drift")
    if context.get("candidate_keys_are_static_syntax_not_loaded_schema") is not True:
        raise CheckpointSafeGlobalsPolicyError("static top-level candidates overstated")
    failure = policy.get("failure_policy", {})
    if set(failure.values()) != {True, False}:
        raise CheckpointSafeGlobalsPolicyError("failure policy malformed")
    for key in ("stop_on_any_unexpected_global_or_dynamic_type", "stop_on_weights_only_failure"):
        if failure.get(key) is not True:
            raise CheckpointSafeGlobalsPolicyError("loader must stop on unexpected content")
    for key in ("allow_additional_globals_after_failure", "fallback_weights_only_false", "legacy_host_load"):
        if failure.get(key) is not False:
            raise CheckpointSafeGlobalsPolicyError("unsafe fallback enabled")
    if policy.get("isolation_required") is not True:
        raise CheckpointSafeGlobalsPolicyError("isolation cannot be weakened")
    for key in ("deserialization_performed", "deserialization_allowed_on_current_host", "g1_admission"):
        if policy.get(key) is not False:
            raise CheckpointSafeGlobalsPolicyError(f"{key} must remain false")
