"""File-backed validation and exact-first comparison for preprocessing parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.preprocessing_parity_plan import validate_preprocessing_parity_plan


class PreprocessingParityResultError(ValueError):
    pass


def validate_preprocessing_parity_result(record: Mapping[str, Any], root: Path) -> None:
    """Validate a parity result and every local artifact it claims to reference."""
    required = {
        "schema_version", "status", "parity_plan", "fixture_spec",
        "candidate_report", "reference_report", "raw_diff_report", "missing_requirements",
        "stage_results", "first_divergent_stage", "parity_passed",
        "formal_preprocessing_admission_allowed",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise PreprocessingParityResultError("parity result fields must be exact")
    if record["schema_version"] != "endosae.preprocessing-parity-result.v0":
        raise PreprocessingParityResultError("unsupported parity result schema")
    if record["status"] not in {"incomplete", "pass", "fail", "needs-raw-diff"}:
        raise PreprocessingParityResultError("invalid parity result status")

    plan = _load_ref(record["parity_plan"], root, "parity_plan")
    fixture = _load_ref(record["fixture_spec"], root, "fixture_spec")
    candidate = _load_ref(record["candidate_report"], root, "candidate_report")
    validate_preprocessing_parity_plan(plan)
    _validate_fixture(fixture)
    _validate_probe(candidate, "candidate")
    if plan["fixture_spec_sha256"] != record["fixture_spec"]["sha256"]:
        raise PreprocessingParityResultError("plan fixture hash does not match referenced fixture")
    if candidate["fixture_spec_sha256"] != record["fixture_spec"]["sha256"]:
        raise PreprocessingParityResultError("candidate fixture hash does not match referenced fixture")

    if not isinstance(record["missing_requirements"], list) or not all(
        isinstance(item, str) and item for item in record["missing_requirements"]
    ):
        raise PreprocessingParityResultError("missing_requirements must be a list of labels")
    for key in ("parity_passed", "formal_preprocessing_admission_allowed"):
        if not isinstance(record[key], bool):
            raise PreprocessingParityResultError(f"{key} must be boolean")

    if record["status"] == "incomplete":
        if record["reference_report"] is not None:
            raise PreprocessingParityResultError("incomplete result cannot reference a legacy report")
        if record["raw_diff_report"] is not None:
            raise PreprocessingParityResultError("incomplete result cannot reference a raw diff")
        if record["missing_requirements"] != ["legacy-reference-report"]:
            raise PreprocessingParityResultError("incomplete result must identify the legacy reference")
        if record["stage_results"] is not None or record["first_divergent_stage"] is not None:
            raise PreprocessingParityResultError("incomplete result cannot claim stage comparisons")
        if record["parity_passed"] or record["formal_preprocessing_admission_allowed"]:
            raise PreprocessingParityResultError("incomplete result cannot pass or admit preprocessing")
        return

    if record["reference_report"] is None:
        raise PreprocessingParityResultError("completed comparison requires a legacy reference report")
    reference = _load_ref(record["reference_report"], root, "reference_report")
    _validate_probe(reference, "reference")
    if reference["fixture_spec_sha256"] != record["fixture_spec"]["sha256"]:
        raise PreprocessingParityResultError("reference fixture hash does not match referenced fixture")
    expected = compare_preprocessing_reports(candidate, reference)
    if expected["status"] == "needs-raw-diff":
        if record["raw_diff_report"] is None:
            pass
        else:
            raw_diff = _load_ref(record["raw_diff_report"], root, "raw_diff_report")
            _validate_raw_diff(raw_diff, root)
            expected["status"] = raw_diff["status"]
            expected["parity_passed"] = raw_diff["parity_passed"]
    elif record["raw_diff_report"] is not None:
        raise PreprocessingParityResultError("raw diff is only valid after tensor fingerprint divergence")
    for field in ("status", "stage_results", "first_divergent_stage", "parity_passed"):
        if record[field] != expected[field]:
            raise PreprocessingParityResultError(f"recorded {field} does not match recomputed comparison")
    if record["missing_requirements"]:
        raise PreprocessingParityResultError("completed comparison cannot retain missing requirements")
    admitted = record["status"] == "pass" and record["parity_passed"]
    if record["formal_preprocessing_admission_allowed"] != admitted:
        raise PreprocessingParityResultError("formal admission must equal an exact parity pass")


def compare_preprocessing_reports(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Compare report metadata; tensor mismatches require raw assets for frozen numeric fallback."""
    _validate_probe(candidate, "candidate")
    _validate_probe(reference, "reference")
    stages = {
        "decoded_input": candidate["input_tensor_fingerprint"] == reference["input_tensor_fingerprint"],
        "normalized_cthw": _tensor_stage_equal(candidate["normalization"], reference["normalization"]),
        "resize_geometry": _geometry_equal(candidate["resize"], reference["resize"]),
        "resize_tensor": candidate["resize"]["tensor_fingerprint"] == reference["resize"]["tensor_fingerprint"],
        "crop_geometry": _views_geometry_equal(candidate["views"], reference["views"]),
        "crop_tensors": _views_tensors_equal(candidate["views"], reference["views"]),
    }
    order = ("decoded_input", "normalized_cthw", "resize_geometry", "resize_tensor", "crop_geometry", "crop_tensors")
    first = next((name for name in order if not stages[name]), None)
    if first is None:
        status = "pass"
    elif first in {"resize_tensor", "crop_tensors"} and all(
        stages[name] for name in ("decoded_input", "normalized_cthw", "resize_geometry", "crop_geometry")
    ):
        status = "needs-raw-diff"
    else:
        status = "fail"
    return {
        "status": status,
        "stage_results": stages,
        "first_divergent_stage": first,
        "parity_passed": status == "pass",
    }


def _load_ref(value: Any, root: Path, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PreprocessingParityResultError(f"{name} reference fields must be exact")
    _require_sha(value["sha256"], name)
    root = root.resolve()
    path = (root / value["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PreprocessingParityResultError(f"{name} path escapes project root") from exc
    if not path.is_file():
        raise PreprocessingParityResultError(f"{name} file does not exist")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != value["sha256"].lower():
        raise PreprocessingParityResultError(f"{name} hash mismatch")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreprocessingParityResultError(f"{name} must be UTF-8 JSON") from exc
    if not isinstance(loaded, Mapping):
        raise PreprocessingParityResultError(f"{name} must contain a JSON object")
    return loaded


def _validate_fixture(fixture: Mapping[str, Any]) -> None:
    if fixture.get("schema_version") != "endosae.synthetic-clip.v0":
        raise PreprocessingParityResultError("unsupported fixture schema")


def _validate_probe(report: Mapping[str, Any], role: str) -> None:
    if report.get("schema_version") != "endosae.synthetic-preprocessing-probe.v0":
        raise PreprocessingParityResultError(f"unsupported {role} probe schema")
    expected_status = "implementation-only" if role == "candidate" else "legacy-reference"
    if report.get("status") != expected_status:
        raise PreprocessingParityResultError(f"{role} probe must have status {expected_status}")
    _require_sha(report.get("fixture_spec_sha256"), f"{role} fixture")
    _require_sha(report.get("input_tensor_fingerprint"), f"{role} input")
    for stage in ("normalization", "resize"):
        if not isinstance(report.get(stage), Mapping):
            raise PreprocessingParityResultError(f"{role} {stage} stage missing")
        _require_sha(report[stage].get("tensor_fingerprint"), f"{role} {stage}")
    views = report.get("views")
    if not isinstance(views, list) or not views:
        raise PreprocessingParityResultError(f"{role} views must be non-empty")
    for view in views:
        if not isinstance(view, Mapping):
            raise PreprocessingParityResultError(f"{role} view must be a mapping")
        _require_sha(view.get("tensor_fingerprint"), f"{role} crop")


def _validate_raw_diff(report: Mapping[str, Any], root: Path) -> None:
    if report.get("schema_version") != "endosae.preprocessing-raw-diff.v0":
        raise PreprocessingParityResultError("unsupported raw diff schema")
    if report.get("status") not in {"pass", "fail"}:
        raise PreprocessingParityResultError("invalid raw diff status")
    if report.get("thresholds") != {
        "max_abs": 1e-6,
        "mean_abs": 1e-7,
        "fraction_over_1e_6": 0.0,
    }:
        raise PreprocessingParityResultError("raw diff thresholds drifted")
    manifest_path = (root.resolve() / str(report.get("manifest_path", ""))).resolve()
    try:
        manifest_path.relative_to(root.resolve())
    except ValueError as exc:
        raise PreprocessingParityResultError("raw manifest path escapes project root") from exc
    if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report.get("manifest_sha256"):
        raise PreprocessingParityResultError("raw manifest hash mismatch")
    stages = report.get("stages")
    if not isinstance(stages, list) or [item.get("stage") for item in stages] != [
        "resize", "crop_0", "crop_1", "crop_2"
    ]:
        raise PreprocessingParityResultError("raw diff stages must be exact and ordered")
    passed = all(item.get("passed") is True for item in stages)
    if report.get("parity_passed") is not passed:
        raise PreprocessingParityResultError("raw diff parity flag mismatch")
    if report.get("formal_preprocessing_admission_allowed") is not passed:
        raise PreprocessingParityResultError("raw diff admission flag mismatch")
    if report.get("status") != ("pass" if passed else "fail"):
        raise PreprocessingParityResultError("raw diff status mismatch")


def _tensor_stage_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("shape") == right.get("shape") and left.get("tensor_fingerprint") == right.get("tensor_fingerprint")


def _geometry_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("shape", "short_side", "mode", "align_corners")
    return all(left.get(key) == right.get(key) for key in keys)


def _views_geometry_equal(left: list[Any], right: list[Any]) -> bool:
    keys = ("spatial_sample_index", "crop_x", "crop_y", "shape")
    return len(left) == len(right) and all(
        all(a.get(key) == b.get(key) for key in keys) for a, b in zip(left, right)
    )


def _views_tensors_equal(left: list[Any], right: list[Any]) -> bool:
    return len(left) == len(right) and all(
        a.get("tensor_fingerprint") == b.get("tensor_fingerprint") for a, b in zip(left, right)
    )


def _require_sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise PreprocessingParityResultError(f"{name} must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PreprocessingParityResultError(f"{name} must be hexadecimal") from exc
