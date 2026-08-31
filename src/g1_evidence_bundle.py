"""Validate the file-backed hash chain required before G1 activation caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class G1EvidenceBundleError(ValueError):
    """Raised when a G1 evidence bundle is incomplete or internally inconsistent."""


def validate_g1_evidence_bundle(bundle: Mapping[str, Any], bundle_root: str | Path) -> None:
    required = {
        "schema_version", "status", "checkpoint_sha256", "preflight_report",
        "inventory_report", "preprocessing_contract", "preprocessing_runtime_fixture", "hook_fixtures",
        "temporal_coordinate_fixture", "sampling_index_asset",
        "behavior_testing_allowed", "formal_activation_cache_allowed",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise G1EvidenceBundleError("bundle fields must be exact")
    if bundle["schema_version"] != "endosae.g1-evidence-bundle.v0":
        raise G1EvidenceBundleError("unsupported bundle schema")
    if bundle["status"] not in {"pass", "fail", "invalid"}:
        raise G1EvidenceBundleError("invalid bundle status")
    _require_sha(bundle["checkpoint_sha256"], "checkpoint_sha256")
    for key in ("behavior_testing_allowed", "formal_activation_cache_allowed"):
        if not isinstance(bundle[key], bool):
            raise G1EvidenceBundleError(f"{key} must be boolean")

    root = Path(bundle_root).resolve()
    preflight, preflight_hash = _load_ref(bundle["preflight_report"], root)
    inventory, _ = _load_ref(bundle["inventory_report"], root)
    preprocessing_contract, preprocessing_contract_hash = _load_ref(bundle["preprocessing_contract"], root)
    preprocessing_runtime, _ = _load_ref(bundle["preprocessing_runtime_fixture"], root)
    temporal, _ = _load_ref(bundle["temporal_coordinate_fixture"], root)
    _, sampling_hash = _load_ref(bundle["sampling_index_asset"], root)

    if preflight.get("schema_version") != "endosae.checkpoint-preflight.v0" or preflight.get("status") != "pass":
        raise G1EvidenceBundleError("preflight report must pass")
    if preflight.get("file_sha256") != bundle["checkpoint_sha256"]:
        raise G1EvidenceBundleError("preflight checkpoint hash mismatch")
    if inventory.get("schema_version") != "endosae.checkpoint-inventory.v0" or inventory.get("status") != "pass":
        raise G1EvidenceBundleError("inventory report must pass")
    if inventory.get("checkpoint_sha256") != bundle["checkpoint_sha256"]:
        raise G1EvidenceBundleError("inventory checkpoint hash mismatch")
    if inventory.get("preflight_report_sha256") != preflight_hash:
        raise G1EvidenceBundleError("inventory does not link observed preflight file")
    if bool(inventory.get("behavior_testing_allowed")) != bundle["behavior_testing_allowed"]:
        raise G1EvidenceBundleError("behavior gate disagrees with inventory")
    if preprocessing_contract.get("schema_version") != "endosae.reference-input-contract.v0":
        raise G1EvidenceBundleError("unexpected preprocessing source contract")
    if (
        preprocessing_runtime.get("schema_version") != "endosae.preprocessing-runtime-fixture.v0"
        or preprocessing_runtime.get("status") != "pass"
    ):
        raise G1EvidenceBundleError("preprocessing runtime fixture must pass")
    if preprocessing_runtime.get("reference_contract_sha256") != preprocessing_contract_hash:
        raise G1EvidenceBundleError("preprocessing runtime fixture does not link observed contract")
    if preprocessing_runtime.get("sampling_index_asset_sha256") != sampling_hash:
        raise G1EvidenceBundleError("preprocessing runtime fixture does not link observed sampling asset")

    hooks = bundle["hook_fixtures"]
    if not isinstance(hooks, list) or len(hooks) < 3:
        raise G1EvidenceBundleError("at least three file-backed hook fixtures are required")
    fixture_ids = set()
    for reference in hooks:
        record, _ = _load_ref(reference, root)
        if record.get("schema_version") != "endosae.hook-fixture.v1" or record.get("status") != "pass":
            raise G1EvidenceBundleError("every hook fixture must be a v1 pass")
        if record.get("checkpoint_sha256") != bundle["checkpoint_sha256"]:
            raise G1EvidenceBundleError("hook fixture checkpoint hash mismatch")
        fixture_id = record.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id or fixture_id in fixture_ids:
            raise G1EvidenceBundleError("hook fixture IDs must be non-empty and unique")
        fixture_ids.add(fixture_id)

    if temporal.get("schema_version") != "endosae.temporal-coordinate-fixture.v0" or temporal.get("status") != "pass":
        raise G1EvidenceBundleError("temporal-coordinate fixture must pass")
    if temporal.get("checkpoint_sha256") != bundle["checkpoint_sha256"]:
        raise G1EvidenceBundleError("temporal fixture checkpoint hash mismatch")
    if temporal.get("sampling_index_asset_sha256") != sampling_hash:
        raise G1EvidenceBundleError("temporal fixture does not link observed sampling asset")

    if bundle["status"] == "pass" and not bundle["behavior_testing_allowed"]:
        raise G1EvidenceBundleError("passed bundle requires behavior testing authorization")
    if bundle["formal_activation_cache_allowed"]:
        if bundle["status"] != "pass" or not bundle["behavior_testing_allowed"]:
            raise G1EvidenceBundleError("formal cache requires a passed behavior-authorized bundle")


def _load_ref(reference: Any, root: Path) -> tuple[dict[str, Any], str]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise G1EvidenceBundleError("artifact reference fields must be path and sha256")
    _require_sha(reference["sha256"], "artifact sha256")
    path = (root / reference["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise G1EvidenceBundleError("artifact path escapes bundle root") from exc
    if not path.is_file():
        raise G1EvidenceBundleError("referenced artifact is missing")
    observed = _sha256(path)
    if observed != reference["sha256"].lower():
        raise G1EvidenceBundleError("referenced artifact hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G1EvidenceBundleError("referenced artifact must be JSON") from exc
    if not isinstance(payload, dict):
        raise G1EvidenceBundleError("referenced JSON artifact must be an object")
    return payload, observed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, key: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise G1EvidenceBundleError(f"{key} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise G1EvidenceBundleError(f"{key} must be hexadecimal") from exc
