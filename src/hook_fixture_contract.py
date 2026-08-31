"""Semantic validation for a future EndoFM observational-hook runtime fixture."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class HookFixtureError(ValueError):
    """Raised when a hook fixture result is internally inconsistent."""


def validate_hook_fixture(record: Mapping[str, Any]) -> None:
    """Reject a claimed pass unless the hook and input were observational only."""

    if not isinstance(record, Mapping):
        raise HookFixtureError("fixture must be a mapping")
    required = {
        "schema_version", "status", "fixture_id", "model_commit",
        "checkpoint_sha256", "preprocessing_id", "input_asset_sha256",
        "seed", "device", "dtype", "eval_mode", "inference_mode",
        "hook_target", "hook_kind", "hook_call_count", "hook_removed_after_run",
        "baseline_output_shape", "hooked_output_shape", "activation_shape",
        "baseline_output_sha256", "hooked_output_sha256",
        "input_before_sha256", "input_after_sha256", "atol", "rtol",
        "max_abs_diff", "allclose_passed",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise HookFixtureError(f"missing fixture fields: {', '.join(missing)}")
    if record["schema_version"] != "endosae.hook-fixture.v0":
        raise HookFixtureError("unsupported schema_version")
    if record["status"] not in {"pass", "fail", "invalid"}:
        raise HookFixtureError("invalid status")
    for key in ("fixture_id", "model_commit", "preprocessing_id", "device", "dtype", "hook_target"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise HookFixtureError(f"{key} must be non-empty")
    for key in ("checkpoint_sha256", "input_asset_sha256", "baseline_output_sha256", "hooked_output_sha256", "input_before_sha256", "input_after_sha256"):
        _require_sha256(record[key], key)
    for key in ("baseline_output_shape", "hooked_output_shape", "activation_shape"):
        _require_shape(record[key], key)
    if record["hook_kind"] != "read-only-forward-output":
        raise HookFixtureError("unexpected hook_kind")
    if not isinstance(record["hook_call_count"], int) or isinstance(record["hook_call_count"], bool) or record["hook_call_count"] < 0:
        raise HookFixtureError("hook_call_count must be a non-negative integer")
    for key in ("eval_mode", "inference_mode", "hook_removed_after_run", "allclose_passed"):
        if not isinstance(record[key], bool):
            raise HookFixtureError(f"{key} must be boolean")
    for key in ("atol", "rtol", "max_abs_diff"):
        value = record[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise HookFixtureError(f"{key} must be non-negative")

    if record["status"] == "pass":
        if not record["eval_mode"] or not record["inference_mode"]:
            raise HookFixtureError("pass requires eval and inference mode")
        if record["hook_call_count"] != 1:
            raise HookFixtureError("pass requires exactly one hook call")
        if not record["hook_removed_after_run"]:
            raise HookFixtureError("pass requires hook removal")
        if record["baseline_output_shape"] != record["hooked_output_shape"]:
            raise HookFixtureError("pass requires equal output shapes")
        if record["input_before_sha256"] != record["input_after_sha256"]:
            raise HookFixtureError("pass requires unchanged input")
        if not record["allclose_passed"]:
            raise HookFixtureError("pass requires numerical allclose")
        if record["max_abs_diff"] > record["atol"]:
            raise HookFixtureError("max_abs_diff exceeds recorded atol")


def validate_hook_fixture_v1(record: Mapping[str, Any]) -> None:
    """Validate v1, which adds post-removal, batch, and frame-order probes."""

    base = dict(record)
    if base.get("schema_version") != "endosae.hook-fixture.v1":
        raise HookFixtureError("unsupported v1 schema_version")
    base["schema_version"] = "endosae.hook-fixture.v0"
    validate_hook_fixture(base)
    extra = {
        "post_removal_output_shape", "post_removal_output_sha256",
        "post_removal_allclose_passed", "batch_single_checked",
        "batch_single_allclose_passed", "batch_single_max_abs_diff",
        "frame_order_probe_checked", "reversed_input_sha256",
        "reversed_output_sha256", "frame_order_output_changed",
        "frame_order_max_abs_diff",
    }
    missing = sorted(extra.difference(record))
    if missing:
        raise HookFixtureError(f"missing v1 fixture fields: {', '.join(missing)}")
    _require_shape(record["post_removal_output_shape"], "post_removal_output_shape")
    for key in ("post_removal_output_sha256", "reversed_input_sha256", "reversed_output_sha256"):
        _require_sha256(record[key], key)
    for key in (
        "post_removal_allclose_passed", "batch_single_checked",
        "batch_single_allclose_passed", "frame_order_probe_checked",
        "frame_order_output_changed",
    ):
        if not isinstance(record[key], bool):
            raise HookFixtureError(f"{key} must be boolean")
    for key in ("batch_single_max_abs_diff", "frame_order_max_abs_diff"):
        value = record[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise HookFixtureError(f"{key} must be non-negative")

    if record["status"] == "pass":
        if record["post_removal_output_shape"] != record["baseline_output_shape"]:
            raise HookFixtureError("pass requires post-removal output shape recovery")
        if not record["post_removal_allclose_passed"]:
            raise HookFixtureError("pass requires post-removal numerical recovery")
        if not record["batch_single_checked"] or not record["batch_single_allclose_passed"]:
            raise HookFixtureError("pass requires batch-vs-single equivalence")
        if record["batch_single_max_abs_diff"] > record["atol"]:
            raise HookFixtureError("batch-vs-single difference exceeds atol")
        if not record["frame_order_probe_checked"]:
            raise HookFixtureError("pass requires frame-order probe")
        if record["reversed_input_sha256"] == record["input_before_sha256"]:
            raise HookFixtureError("reversed input must differ from original input")
        if not record["frame_order_output_changed"] or record["frame_order_max_abs_diff"] <= record["atol"]:
            raise HookFixtureError("pass requires frame-order-sensitive output")


def _require_sha256(value: Any, key: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise HookFixtureError(f"{key} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HookFixtureError(f"{key} must be hexadecimal") from exc


def _require_shape(value: Any, key: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise HookFixtureError(f"{key} must be a non-empty shape")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in value):
        raise HookFixtureError(f"{key} must contain positive integers")
