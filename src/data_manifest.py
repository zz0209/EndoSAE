"""Semantic validation for EndoSAE dataset manifests.

The JSON schema constrains shape.  These checks enforce research invariants that
JSON Schema cannot express compactly: temporal order, grouping identity, label
compatibility, rights gates, and split leakage across related records.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping, Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when a manifest violates a research invariant."""


def validate_record(record: Mapping[str, Any], *, require_authorized: bool = False) -> None:
    required = {
        "schema_version",
        "dataset_id",
        "dataset_version",
        "source_uri",
        "rights_status",
        "grouping_level",
        "group_id",
        "video_id",
        "asset_sha256",
        "temporal",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ManifestValidationError(f"missing required fields: {missing}")

    if record["schema_version"] != "0.1.0":
        raise ManifestValidationError("unsupported schema_version")
    if not _SHA256.fullmatch(str(record["asset_sha256"])):
        raise ManifestValidationError("asset_sha256 must be 64 lowercase hex characters")

    rights = record["rights_status"]
    if rights == "denied":
        raise ManifestValidationError("denied assets must not enter a runnable manifest")
    if require_authorized and rights not in {"verified_public", "verified_restricted"}:
        raise ManifestValidationError("formal use requires verified rights status")

    temporal = record["temporal"]
    if temporal.get("ordered") is not True:
        raise ManifestValidationError("temporal.ordered must be true")
    start = temporal.get("start_frame")
    end = temporal.get("end_frame")
    fps = temporal.get("fps")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ManifestValidationError("start_frame must be a non-negative integer")
    if not isinstance(end, int) or isinstance(end, bool) or end < start:
        raise ManifestValidationError("end_frame must be an integer >= start_frame")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ManifestValidationError("fps must be positive")

    grouping_level = record["grouping_level"]
    group_id = record["group_id"]
    if not isinstance(group_id, str) or not group_id:
        raise ManifestValidationError("group_id must be non-empty")
    direct_group_fields = {
        "patient": "patient_id",
        "procedure": "procedure_id",
        "source_case": "source_case_id",
    }
    if grouping_level in direct_group_fields:
        field = direct_group_fields[grouping_level]
        if record.get(field) != group_id:
            raise ManifestValidationError(f"{field} must equal group_id for {grouping_level} grouping")
    elif grouping_level == "phantom_segment":
        if not record.get("phantom_id") or not record.get("segment_id"):
            raise ManifestValidationError("phantom_segment grouping requires phantom_id and segment_id")
    elif grouping_level != "unknown":
        raise ManifestValidationError("unsupported grouping_level")

    labels = record.get("labels") or {}
    target = labels.get("underlying_target_status")
    visibility = labels.get("observability")
    if target == "absent_confirmed" and visibility != "not_applicable":
        raise ManifestValidationError("absent_confirmed requires not_applicable observability")
    if target == "present_supported" and visibility == "not_applicable":
        raise ManifestValidationError("present_supported cannot have not_applicable observability")


def validate_split_integrity(records: Iterable[Mapping[str, Any]]) -> None:
    """Reject leakage of a group or declared pair across analysis splits."""

    group_splits: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        validate_record(record)
        split = record.get("split", "unassigned")
        group_key = (record["dataset_id"], record["grouping_level"], record["group_id"])
        group_splits[group_key].add(split)
        pair_id = record.get("paired_asset_id")
        if pair_id:
            pair_splits[(record["dataset_id"], pair_id)].add(split)

    leaked_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
    if leaked_groups:
        raise ManifestValidationError(f"group split leakage: {leaked_groups}")
    leaked_pairs = {key: value for key, value in pair_splits.items() if len(value) > 1}
    if leaked_pairs:
        raise ManifestValidationError(f"paired asset split leakage: {leaked_pairs}")
