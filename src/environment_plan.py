"""Semantic stop-boundary checks for the two-environment EndoFM plan."""

from __future__ import annotations

from typing import Any, Mapping


class EnvironmentPlanError(ValueError):
    """Raised when an environment plan could bypass reference/parity gates."""


def validate_environment_plan(plan: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "status", "official_environment_source",
        "reference_environment", "modern_environment", "parity",
        "dependency_install_authorized", "formal_activation_cache_allowed",
    }
    if not isinstance(plan, Mapping) or set(plan) != required:
        raise EnvironmentPlanError("environment plan fields must be exact")
    if plan["schema_version"] != "endosae.environment-plan.v0":
        raise EnvironmentPlanError("unsupported environment plan schema")
    if plan["status"] not in {"blocked", "candidate", "pass"}:
        raise EnvironmentPlanError("invalid environment plan status")
    for key in ("dependency_install_authorized", "formal_activation_cache_allowed"):
        if not isinstance(plan[key], bool):
            raise EnvironmentPlanError(f"{key} must be boolean")

    reference = _mapping(plan["reference_environment"], "reference_environment")
    modern = _mapping(plan["modern_environment"], "modern_environment")
    parity = _mapping(plan["parity"], "parity")
    if reference.get("role") != "legacy-cpu-numerical-reference":
        raise EnvironmentPlanError("reference role must remain numerical reference")
    expected_pins = {
        "torch": "1.8.0", "torchvision": "0.9.0", "av": "10.0.0",
        "numpy": "1.21.5", "pillow": "6.2.2", "timm": "0.4.12",
        "fvcore": "0.1.5.post20221221", "einops": "0.6.1", "kornia": "0.5.8",
    }
    if reference.get("python") != "3.7.16" or reference.get("packages") != expected_pins:
        raise EnvironmentPlanError("reference pins must match the official environment")
    if modern.get("role") != "cuda-throughput-candidate":
        raise EnvironmentPlanError("modern role must remain throughput candidate")

    reference_pass = reference.get("status") == "pass"
    modern_pass = modern.get("status") == "pass"
    parity_pass = parity.get("status") == "pass"
    criteria_frozen = parity.get("criteria_frozen_before_comparison") is True
    if reference.get("checkpoint_loading_allowed") and not reference_pass:
        raise EnvironmentPlanError("reference checkpoint loading requires passed environment")
    if modern.get("checkpoint_loading_allowed") and not (reference_pass and modern_pass):
        raise EnvironmentPlanError("modern checkpoint loading requires passed reference and modern environments")
    if modern.get("formal_extraction_allowed") and not (
        reference_pass and modern_pass and parity_pass and criteria_frozen
    ):
        raise EnvironmentPlanError("formal extraction requires passed frozen parity chain")
    if plan["formal_activation_cache_allowed"] and not (
        plan["status"] == "pass"
        and reference_pass and modern_pass and parity_pass and criteria_frozen
        and modern.get("formal_extraction_allowed") is True
    ):
        raise EnvironmentPlanError("formal cache requires complete two-environment parity")


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentPlanError(f"{key} must be a mapping")
    return value
