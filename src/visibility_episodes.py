"""Semantic contracts for Route A visible→censored→visible episodes."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = ("pre_visible", "censored", "post_visible")
_VISIBLE = {"visible_full", "visible_partial"}
_CENSORED = {"censored_in_view", "out_of_view"}
_MECHANISMS = {
    "instrument_occlusion", "fluid_or_bubble", "debris",
    "specularity_or_saturation", "blur_or_fast_motion",
    "lens_contamination", "frame_boundary", "other", "unknown",
}


class VisibilityEpisodeError(ValueError):
    """Raised when an episode cannot support the intended A0 estimand."""


def validate_visibility_episode(
    episode: Mapping[str, Any],
    manifest_records: Iterable[Mapping[str, Any]],
    *,
    confirmatory: bool = False,
) -> None:
    """Validate identity, order, labels, split, and manifest provenance.

    Exactly three anchor windows are required. This is an A0 contract, not a
    claim that all censoring events naturally have only three segments.
    """

    required = {
        "schema_version", "episode_id", "dataset_id", "dataset_version",
        "source_case_id", "target_id", "split", "identity_evidence", "segments",
    }
    missing = sorted(required - episode.keys())
    if missing:
        raise VisibilityEpisodeError(f"missing episode fields: {missing}")
    if episode["schema_version"] != "0.1.0":
        raise VisibilityEpisodeError("unsupported episode schema_version")
    for field in ("episode_id", "dataset_id", "dataset_version", "source_case_id", "target_id"):
        if not isinstance(episode[field], str) or not episode[field]:
            raise VisibilityEpisodeError(f"{field} must be non-empty")
    if episode["split"] not in {"development", "confirmation", "test", "external"}:
        raise VisibilityEpisodeError("invalid episode split")
    if episode["identity_evidence"] not in {
        "continuous_mask_track", "expert_confirmed", "source_ground_truth"
    }:
        raise VisibilityEpisodeError("identity evidence is insufficient")

    segments = episode["segments"]
    if not isinstance(segments, list) or len(segments) != 3:
        raise VisibilityEpisodeError("episode requires exactly three anchor segments")
    if tuple(segment.get("role") for segment in segments) != _ROLES:
        raise VisibilityEpisodeError("segments must be ordered pre_visible, censored, post_visible")
    asset_hashes = {segment.get("manifest_asset_sha256") for segment in segments}
    if len(asset_hashes) != 1:
        raise VisibilityEpisodeError(
            "v0 episode anchors must share one continuous manifest asset; cross-clip time is unverified"
        )

    records_by_hash = {record.get("asset_sha256"): record for record in manifest_records}
    previous_end = None
    for segment in segments:
        asset_hash = segment.get("manifest_asset_sha256")
        if not isinstance(asset_hash, str) or not _SHA256.fullmatch(asset_hash):
            raise VisibilityEpisodeError("segment has invalid manifest asset hash")
        record = records_by_hash.get(asset_hash)
        if record is None:
            raise VisibilityEpisodeError("segment does not resolve to a manifest record")
        for field in ("dataset_id", "dataset_version", "source_case_id", "split"):
            if record.get(field) != episode[field]:
                raise VisibilityEpisodeError(f"segment manifest disagrees on {field}")
        labels = record.get("labels") or {}
        if labels.get("underlying_target_status") != "present_supported":
            raise VisibilityEpisodeError("episode anchors require present_supported target status")

        start, end = segment.get("start_frame"), segment.get("end_frame")
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
            raise VisibilityEpisodeError("segment frame bounds must be integers")
        if end < start:
            raise VisibilityEpisodeError("segment end_frame precedes start_frame")
        bounds = record.get("temporal") or {}
        if start < bounds.get("start_frame", start) or end > bounds.get("end_frame", end):
            raise VisibilityEpisodeError("segment lies outside manifest asset bounds")
        if previous_end is not None and start <= previous_end:
            raise VisibilityEpisodeError("episode segments overlap or are out of temporal order")
        previous_end = end

        role = segment["role"]
        observability = segment.get("observability")
        mechanisms = segment.get("censoring_mechanisms")
        if not isinstance(mechanisms, list) or len(mechanisms) != len(set(mechanisms)):
            raise VisibilityEpisodeError("censoring mechanisms must be a unique list")
        if any(item not in _MECHANISMS for item in mechanisms):
            raise VisibilityEpisodeError("unsupported censoring mechanism")
        if role in {"pre_visible", "post_visible"}:
            if observability not in _VISIBLE or mechanisms:
                raise VisibilityEpisodeError("visible anchors require visible state and no censoring mechanism")
        else:
            if observability not in _CENSORED or not mechanisms:
                raise VisibilityEpisodeError("censored anchor requires censored state and a mechanism")
            if confirmatory and "unknown" in mechanisms:
                raise VisibilityEpisodeError("confirmatory censoring mechanism cannot be unknown")

        annotation = segment.get("annotation_status")
        if annotation not in {"source_ground_truth", "derived_rule", "expert_confirmed", "candidate"}:
            raise VisibilityEpisodeError("invalid annotation status")
        if confirmatory and annotation in {"derived_rule", "candidate"}:
            raise VisibilityEpisodeError("confirmatory anchors require source or expert confirmation")
