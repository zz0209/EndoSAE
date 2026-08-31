"""Conservative validation for the EndoFM checkpoint rights evidence record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class CheckpointRightsError(ValueError):
    """Raised when an unresolved rights record overstates permission."""


REQUIRED_CLARIFICATION_FRAGMENTS = {
    "research-use permission",
    "Apache-2.0",
    "redistribution",
    "derived activations",
    "code commit",
    "citation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_rights_evidence(
    record: Mapping[str, Any], project_root: Path
) -> None:
    if not isinstance(record, Mapping):
        raise CheckpointRightsError("record must be a mapping")
    if record.get("schema_version") != "endosae.checkpoint-rights-evidence.v0":
        raise CheckpointRightsError("unsupported schema_version")
    if record.get("status") != "user-authorized-internal-use-upstream-scope-unresolved":
        raise CheckpointRightsError("unexpected rights status")

    root = project_root.resolve()
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise CheckpointRightsError("checkpoint section is required")
    if checkpoint.get("official_file_id") != "1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl":
        raise CheckpointRightsError("official checkpoint file id drift")
    local_path = (root / str(checkpoint.get("local_path", ""))).resolve()
    try:
        local_path.relative_to(root)
    except ValueError as exc:
        raise CheckpointRightsError("checkpoint path escapes project root") from exc
    if not local_path.is_file():
        raise CheckpointRightsError("downloaded checkpoint is missing")
    if local_path.stat().st_size != checkpoint.get("expected_size_bytes"):
        raise CheckpointRightsError("downloaded checkpoint size mismatch")
    if local_path.stat().st_size != checkpoint.get("local_size_bytes"):
        raise CheckpointRightsError("recorded local checkpoint size mismatch")
    if _sha256(local_path) != checkpoint.get("local_sha256"):
        raise CheckpointRightsError("downloaded checkpoint hash mismatch")
    if checkpoint.get("preflight_status") != "pass":
        raise CheckpointRightsError("checkpoint container preflight has not passed")
    preflight_path = (root / str(checkpoint.get("preflight_report_path", ""))).resolve()
    try:
        preflight_path.relative_to(root)
    except ValueError as exc:
        raise CheckpointRightsError("preflight report path escapes project root") from exc
    if not preflight_path.is_file():
        raise CheckpointRightsError("checkpoint preflight report is missing")
    if _sha256(preflight_path) != checkpoint.get("preflight_report_sha256"):
        raise CheckpointRightsError("checkpoint preflight report hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass":
        raise CheckpointRightsError("checkpoint preflight report did not pass")
    if preflight.get("file_sha256") != checkpoint.get("local_sha256"):
        raise CheckpointRightsError("checkpoint and preflight hashes disagree")
    if preflight.get("file_size_bytes") != checkpoint.get("local_size_bytes"):
        raise CheckpointRightsError("checkpoint and preflight sizes disagree")

    evidence = record.get("official_evidence")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise CheckpointRightsError("official evidence must include README and LICENSE")
    evidence_ids = set()
    explicit_weight_scope = False
    for item in evidence:
        if not isinstance(item, Mapping):
            raise CheckpointRightsError("evidence item must be a mapping")
        evidence_ids.add(item.get("source_id"))
        candidate = (root / str(item.get("path", ""))).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CheckpointRightsError("evidence path escapes project root") from exc
        if not candidate.is_file() or _sha256(candidate) != item.get("sha256"):
            raise CheckpointRightsError("evidence file hash mismatch")
        explicit_weight_scope |= item.get("explicitly_mentions_weight_license_scope") is True
    if evidence_ids != {"endofm-pinned-readme", "endofm-pinned-license"}:
        raise CheckpointRightsError("unexpected or missing official evidence sources")

    boundary = record.get("inference_boundary")
    if not isinstance(boundary, Mapping):
        raise CheckpointRightsError("inference boundary is required")
    for field in (
        "public_download_link_is_permission",
        "repository_code_license_automatically_covers_checkpoint",
        "same_page_placement_is_explicit_weight_scope",
        "legal_conclusion_claimed",
    ):
        if boundary.get(field) is not False:
            raise CheckpointRightsError(f"unsafe rights inference: {field}")

    authorization = record.get("project_authorization")
    if not isinstance(authorization, Mapping):
        raise CheckpointRightsError("project authorization is required")
    if authorization.get("authorized_by") != "user":
        raise CheckpointRightsError("internal use must be user-authorized")
    required_scope = {
        "download-official-checkpoint-to-project-quarantine",
        "internal-project-research-use",
        "project-local-environment-setup",
    }
    scope = authorization.get("scope")
    if not isinstance(scope, list) or set(scope) != required_scope:
        raise CheckpointRightsError("project authorization scope drift")
    for field in (
        "administrative_clearance_completed_or_in_progress",
        "does_not_establish_upstream_weight_license_scope",
        "does_not_authorize_checkpoint_or-derived-asset-redistribution",
    ):
        if authorization.get(field) is not True:
            raise CheckpointRightsError(f"unsafe project authorization boundary: {field}")

    questions = record.get("required_clarifications")
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise CheckpointRightsError("required clarifications must be a string list")
    joined = "\n".join(questions)
    missing = sorted(fragment for fragment in REQUIRED_CLARIFICATION_FRAGMENTS if fragment not in joined)
    if missing:
        raise CheckpointRightsError("missing clarification topics: " + ", ".join(missing))

    admission = record.get("admission")
    if not isinstance(admission, Mapping):
        raise CheckpointRightsError("admission section is required")
    if admission.get("download_authorized") is not True:
        raise CheckpointRightsError("user authorization must admit project-local download")
    if admission.get("checkpoint_use_authorized") is not True:
        raise CheckpointRightsError("user authorization must admit internal research use")
    if not explicit_weight_scope:
        for field in (
            "deserialization_authorized",
            "derived_asset_release_authorized",
            "g0_checkpoint_pass",
            "g1_evidence_allowed",
        ):
            if admission.get(field) is not False:
                raise CheckpointRightsError(f"unresolved upstream scope must block {field}")
