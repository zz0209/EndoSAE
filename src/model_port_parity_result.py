"""File-backed admission rules for staged EndoFM model-port parity."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


class ModelPortParityResultError(ValueError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["patch_embed", "block_0_residual", "block_5_residual", "block_11_residual", "final_normalized_tokens", "behavior_output"]
CRITERIA = {"max_abs_error": 1e-4, "mean_abs_error": 1e-6, "relative_l2_error": 1e-6, "cosine_similarity": 0.999999, "nonfinite_values": 0}
INCOMPLETE_MISSING = ["isolated-checkpoint-inventory", "parameter-mapping-report", "legacy-reference-stage-assets", "modern-candidate-stage-assets"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_link(link: Mapping[str, Any], label: str) -> Path:
    path = (PROJECT_ROOT / str(link.get("path", ""))).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ModelPortParityResultError(f"{label} escapes project") from exc
    if not path.is_file() or _sha256(path) != link.get("sha256"):
        raise ModelPortParityResultError(f"{label} hash mismatch")
    return path


def validate_model_port_parity_result(record: Mapping[str, Any], *, verify_stage_assets: bool = True) -> None:
    if record.get("schema_version") != "endosae.model-port-parity-result.v0":
        raise ModelPortParityResultError("unsupported schema")
    _verify_link(record.get("plan", {}), "plan")
    _verify_link(record.get("safe_globals_policy", {}), "safe_globals_policy")
    if record.get("g1_admission") is not False:
        raise ModelPortParityResultError("model parity alone cannot admit G1")
    status = record.get("status")
    if status == "incomplete":
        if record.get("inventory_report") is not None or record.get("parameter_mapping_report") is not None:
            raise ModelPortParityResultError("incomplete template cannot claim unavailable reports")
        if record.get("stage_results") != [] or record.get("missing_requirements") != INCOMPLETE_MISSING:
            raise ModelPortParityResultError("incomplete requirements drift")
        if record.get("model_port_parity_pass") is not False or record.get("formal_model_port_allowed") is not False:
            raise ModelPortParityResultError("incomplete result cannot pass")
        return
    if status not in {"pass", "fail"}:
        raise ModelPortParityResultError("invalid status")
    _verify_link(record.get("inventory_report", {}), "inventory_report")
    _verify_link(record.get("parameter_mapping_report", {}), "parameter_mapping_report")
    if record.get("missing_requirements") != []:
        raise ModelPortParityResultError("complete result cannot retain missing requirements")
    stages = record.get("stage_results")
    if not isinstance(stages, list) or [x.get("name") for x in stages if isinstance(x, Mapping)] != STAGES:
        raise ModelPortParityResultError("stage set/order drift")
    all_pass = True
    for stage in stages:
        for side in ("reference", "candidate"):
            asset = stage.get(side)
            if not isinstance(asset, Mapping) or asset.get("dtype") != "float32-le" or not asset.get("shape"):
                raise ModelPortParityResultError(f"invalid {side} asset")
            if verify_stage_assets:
                _verify_link(asset, f"{stage['name']}.{side}")
        if stage["reference"].get("shape") != stage["candidate"].get("shape"):
            raise ModelPortParityResultError("stage shapes differ")
        for metric in ("max_abs_error", "mean_abs_error", "relative_l2_error", "cosine_similarity"):
            if not isinstance(stage.get(metric), (int, float)) or isinstance(stage.get(metric), bool) or not math.isfinite(stage[metric]):
                raise ModelPortParityResultError(f"invalid metric: {metric}")
        computed = (
            stage["max_abs_error"] <= CRITERIA["max_abs_error"]
            and stage["mean_abs_error"] <= CRITERIA["mean_abs_error"]
            and stage["relative_l2_error"] <= CRITERIA["relative_l2_error"]
            and stage["cosine_similarity"] >= CRITERIA["cosine_similarity"]
            and stage.get("nonfinite_values") == 0
            and (stage["name"] != "behavior_output" or stage.get("behavior_argmax_match") is True)
        )
        if stage.get("pass") is not computed:
            raise ModelPortParityResultError(f"stage pass flag disagrees: {stage['name']}")
        all_pass = all_pass and computed
    if record.get("model_port_parity_pass") is not all_pass or (status == "pass") is not all_pass:
        raise ModelPortParityResultError("overall status disagrees with conjunctive stages")
    if record.get("formal_model_port_allowed") is not False:
        raise ModelPortParityResultError("parity result alone cannot allow formal model port")
