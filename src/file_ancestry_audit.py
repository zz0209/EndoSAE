"""Recompute hashes and line similarity for pinned upstream ancestry evidence."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping


class FileAncestryError(ValueError):
    """Raised when a recorded ancestry comparison no longer reproduces."""


def validate_file_ancestry(registry: Mapping[str, Any], project_root: Path) -> None:
    if registry.get("schema_version") != "endosae.file-ancestry.v0":
        raise FileAncestryError("unsupported schema_version")
    comparisons = registry.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise FileAncestryError("comparisons must be non-empty")
    _validate_history(registry.get("upstream_history"))
    seen = set()
    for comparison in comparisons:
        comparison_id = comparison.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id or comparison_id in seen:
            raise FileAncestryError("comparison_id must be non-empty and unique")
        seen.add(comparison_id)
        upstream = _checked_project_path(project_root, comparison.get("upstream_path"))
        endofm = _checked_project_path(project_root, comparison.get("endofm_path"))
        upstream_bytes = upstream.read_bytes()
        endofm_bytes = endofm.read_bytes()
        if _sha256(upstream_bytes) != comparison.get("upstream_sha256"):
            raise FileAncestryError(f"{comparison_id} upstream hash mismatch")
        if _sha256(endofm_bytes) != comparison.get("endofm_sha256"):
            raise FileAncestryError(f"{comparison_id} EndoFM hash mismatch")
        observed = SequenceMatcher(
            None, upstream_bytes.splitlines(), endofm_bytes.splitlines(), autojunk=False
        ).ratio()
        expected = comparison.get("line_similarity")
        if not isinstance(expected, (int, float)) or abs(observed - expected) > 1e-12:
            raise FileAncestryError(f"{comparison_id} line similarity mismatch")


def _validate_history(history: Any) -> None:
    if not isinstance(history, Mapping) or history.get("svt_history_complete") is not True:
        raise FileAncestryError("complete SVT history evidence is required")
    required = (
        "timesformer_best_matching_blob",
        "decoder_earliest_blob",
        "helpers_earliest_blob",
    )
    for key in required:
        item = history.get(key)
        if not isinstance(item, Mapping):
            raise FileAncestryError(f"missing history item: {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(item.get("first_commit", ""))):
            raise FileAncestryError(f"invalid history commit: {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(item.get("git_blob", ""))):
            raise FileAncestryError(f"invalid history blob: {key}")


def _checked_project_path(project_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FileAncestryError("path must be non-empty")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise FileAncestryError("path escapes project root")
    if not path.is_file():
        raise FileAncestryError(f"missing ancestry file: {relative}")
    return path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
