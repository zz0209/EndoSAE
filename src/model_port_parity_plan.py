"""Validate pre-observation criteria for legacy-to-modern EndoFM model parity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class ModelPortParityPlanError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_port_parity_plan(plan: Mapping[str, Any], root: Path) -> None:
    if plan.get("schema_version") != "endosae.model-port-parity-plan.v0":
        raise ModelPortParityPlanError("unsupported schema")
    if plan.get("status") != "preregistered-blocked" or plan.get("g1_admission") is not False:
        raise ModelPortParityPlanError("current plan must remain blocked and non-admitting")
    if plan.get("checkpoint_deserialization_allowed") is not False or plan.get("formal_model_port_allowed") is not False:
        raise ModelPortParityPlanError("plan cannot authorize checkpoint/model-port execution")
    for key in ("reference_tensor_exchange", "modern_environment_lock"):
        link = plan.get(key)
        if not isinstance(link, Mapping):
            raise ModelPortParityPlanError(f"missing link: {key}")
        path = (root / link.get("path", "")).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ModelPortParityPlanError(f"{key} escapes project") from exc
        if not path.is_file():
            raise ModelPortParityPlanError(f"missing linked file: {key}")
        expected = link.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(path).lower() != expected.lower():
            raise ModelPortParityPlanError(f"hash mismatch: {key}")
    execution = plan.get("execution", {})
    expected_execution = {"dtype": "float32", "device": "cpu", "eval_mode": True, "inference_mode": True, "batch_size": 1, "input_layout": "C,T,H,W", "image_preprocessing_bypassed": True}
    if execution != expected_execution:
        raise ModelPortParityPlanError("execution controls drifted")
    names = [stage.get("name") for stage in plan.get("stages", []) if isinstance(stage, Mapping)]
    expected_names = ["patch_embed", "block_0_residual", "block_5_residual", "block_11_residual", "final_normalized_tokens", "behavior_output"]
    if names != expected_names or not all(stage.get("required") is True for stage in plan["stages"]):
        raise ModelPortParityPlanError("required parity stages drifted")
    criteria = plan.get("frozen_numeric_criteria", {})
    expected_criteria = {"max_abs_error": 1e-4, "mean_abs_error": 1e-6, "relative_l2_error": 1e-6, "minimum_cosine_similarity": 0.999999, "nonfinite_values_allowed": 0, "behavior_argmax_must_match": True}
    if criteria != expected_criteria:
        raise ModelPortParityPlanError("numeric criteria drifted")
    if plan.get("decision_rule") != "all-required-stages-pass-conjunctively":
        raise ModelPortParityPlanError("decision rule must be conjunctive")
    required = plan.get("required_before_run")
    if not isinstance(required, list) or len(required) < 5:
        raise ModelPortParityPlanError("pre-run dependencies are incomplete")
