"""Semantic validation for blinded Route A annotation records."""

from __future__ import annotations

import re
from typing import Any, Mapping


_ITEM_ID = re.compile(r"^A0-[A-Z0-9]{8}$")
_CENSORED = {"censored_in_view", "out_of_view"}
_OBSERVABILITY = {"visible_full", "visible_partial", "censored_in_view", "out_of_view", "unknown"}
_MECHANISMS = {"instrument_occlusion", "fluid_or_bubble", "debris", "specularity_or_saturation", "blur_or_fast_motion", "lens_contamination", "frame_boundary", "other", "unknown"}
_EXCLUSIONS = {"none", "identity_unresolved", "no_true_censoring", "insufficient_context", "corrupt_or_missing_frames", "other"}


class VisibilityAnnotationError(ValueError):
    """Raised when an annotation record is internally inconsistent."""


def validate_annotation_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "blind_item_id", "annotator_id", "round_id",
        "same_target_pre_post", "middle_observability", "censoring_mechanisms",
        "boundary_start_frame", "boundary_end_frame", "confidence", "exclude_reason",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise VisibilityAnnotationError(f"missing annotation fields: {missing}")
    if record["schema_version"] != "0.1.0" or not _ITEM_ID.fullmatch(str(record["blind_item_id"])):
        raise VisibilityAnnotationError("invalid schema version or blind item id")
    if not record["annotator_id"]:
        raise VisibilityAnnotationError("annotator_id must be non-empty")
    if record["round_id"] not in {"calibration", "independent", "adjudication"}:
        raise VisibilityAnnotationError("invalid annotation round")
    if record["same_target_pre_post"] not in {"yes", "no", "uncertain"}:
        raise VisibilityAnnotationError("invalid target identity judgment")
    if record["same_target_pre_post"] != "yes" and record["exclude_reason"] != "identity_unresolved":
        raise VisibilityAnnotationError("unresolved target identity must be excluded explicitly")

    observability = record["middle_observability"]
    mechanisms = record["censoring_mechanisms"]
    if observability not in _OBSERVABILITY or record["exclude_reason"] not in _EXCLUSIONS:
        raise VisibilityAnnotationError("invalid observability or exclusion value")
    if not isinstance(mechanisms, list) or len(mechanisms) != len(set(mechanisms)):
        raise VisibilityAnnotationError("censoring_mechanisms must be a unique list")
    if any(mechanism not in _MECHANISMS for mechanism in mechanisms):
        raise VisibilityAnnotationError("unsupported censoring mechanism")
    if "unknown" in mechanisms and len(mechanisms) != 1:
        raise VisibilityAnnotationError("unknown mechanism cannot be combined with named mechanisms")
    start, end = record["boundary_start_frame"], record["boundary_end_frame"]
    if observability in _CENSORED:
        if not mechanisms:
            raise VisibilityAnnotationError("censored annotation requires a mechanism")
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or end < start:
            raise VisibilityAnnotationError("censored annotation requires ordered integer boundaries")
    else:
        if mechanisms or start is not None or end is not None:
            raise VisibilityAnnotationError("non-censored annotation cannot carry mechanism or boundaries")
        if observability in {"visible_full", "visible_partial"} and record["exclude_reason"] != "no_true_censoring":
            raise VisibilityAnnotationError("visible middle segment must be excluded as no_true_censoring")
    confidence = record["confidence"]
    if not isinstance(confidence, int) or isinstance(confidence, bool) or not 1 <= confidence <= 4:
        raise VisibilityAnnotationError("confidence must be an integer from 1 to 4")
