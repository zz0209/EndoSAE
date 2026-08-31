"""Validate the preprocessing parity preregistration and admission boundary."""

from __future__ import annotations

from typing import Any, Mapping


class PreprocessingParityPlanError(ValueError):
    pass


def validate_preprocessing_parity_plan(plan: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "status", "frozen_at", "fixture_spec_sha256",
        "reference_role", "candidate_role", "comparison_unit", "metrics",
        "failure_policy", "reference_result_present", "candidate_result_present",
        "parity_passed", "formal_preprocessing_admission_allowed",
    }
    if not isinstance(plan, Mapping) or set(plan) != required:
        raise PreprocessingParityPlanError("parity plan fields must be exact")
    if plan["schema_version"] != "endosae.preprocessing-parity-plan.v0":
        raise PreprocessingParityPlanError("unsupported parity plan schema")
    if plan["status"] not in {"draft", "frozen-before-reference-run", "complete"}:
        raise PreprocessingParityPlanError("invalid parity plan status")
    _require_sha(plan["fixture_spec_sha256"])
    if plan["reference_role"] != "legacy-cpu-numerical-reference":
        raise PreprocessingParityPlanError("unexpected reference role")
    if plan["candidate_role"] != "modern-cpu-preprocessing-candidate":
        raise PreprocessingParityPlanError("unexpected candidate role")
    metrics = _mapping(plan["metrics"], "metrics")
    if set(metrics) != {"decoded_input", "normalized_cthw", "resized_and_crop_tensors", "geometry"}:
        raise PreprocessingParityPlanError("metric stages must be exact")
    if metrics["decoded_input"].get("exact_fingerprint_required") is not True:
        raise PreprocessingParityPlanError("decoded input must match exactly")
    if metrics["normalized_cthw"].get("exact_fingerprint_required") is not True:
        raise PreprocessingParityPlanError("normalization must match exactly")
    resized = _mapping(metrics["resized_and_crop_tensors"], "resized metrics")
    expected_resized = {
        "exact_fingerprint_preferred", "fallback_max_abs_diff",
        "fallback_mean_abs_diff", "fallback_fraction_over_1e_6",
    }
    if set(resized) != expected_resized or resized["exact_fingerprint_preferred"] is not True:
        raise PreprocessingParityPlanError("resize metric fields must be exact")
    if not (0 <= resized["fallback_mean_abs_diff"] <= resized["fallback_max_abs_diff"] <= 1e-6):
        raise PreprocessingParityPlanError("resize tolerances exceed frozen ceiling")
    if resized["fallback_fraction_over_1e_6"] != 0.0:
        raise PreprocessingParityPlanError("no elements may exceed 1e-6")
    policy = _mapping(plan["failure_policy"], "failure_policy")
    if not all(value is True for value in policy.values()):
        raise PreprocessingParityPlanError("all failure safeguards must be enabled")
    for key in (
        "reference_result_present", "candidate_result_present", "parity_passed",
        "formal_preprocessing_admission_allowed",
    ):
        if not isinstance(plan[key], bool):
            raise PreprocessingParityPlanError(f"{key} must be boolean")
    if plan["parity_passed"] and not (plan["reference_result_present"] and plan["candidate_result_present"]):
        raise PreprocessingParityPlanError("parity pass requires both results")
    if plan["formal_preprocessing_admission_allowed"] and not (
        plan["status"] == "complete" and plan["parity_passed"]
    ):
        raise PreprocessingParityPlanError("formal admission requires completed parity")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreprocessingParityPlanError(f"{name} must be a mapping")
    return value


def _require_sha(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise PreprocessingParityPlanError("fixture spec hash must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PreprocessingParityPlanError("fixture spec hash must be hexadecimal") from exc
