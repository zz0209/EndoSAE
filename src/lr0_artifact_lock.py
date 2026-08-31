"""Validate the immutable-input boundary for the proposed LR0 environment."""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import unquote, urlparse


class LR0ArtifactLockError(ValueError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED = {
    "bootstrap": ("micromamba", "2.9.0-0"),
    "python": ("python", "3.7.16"),
    "numpy": ("numpy", "1.21.5"),
    "torch": ("torch", "1.8.0+cpu"),
    "typing_extensions": ("typing_extensions", "4.3.0"),
}


def validate_lr0_artifact_lock(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "status", "audited_at", "target", "artifacts",
        "lock_complete", "dry_run_allowed", "download_allowed", "install_allowed",
        "checkpoint_allowed", "g1_evidence_allowed",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise LR0ArtifactLockError("lock fields must be exact")
    if record["schema_version"] != "endosae.lr0-artifact-lock.v0":
        raise LR0ArtifactLockError("unsupported lock schema")
    if record["status"] != "complete-metadata-quarantine-downloads-authorized":
        raise LR0ArtifactLockError("unexpected LR0 lock status")
    target = _mapping(record["target"], "target")
    if target != {
        "platform": "win-64",
        "python_abi": "cp37-cp37m",
        "purpose": "codec-independent synthetic tensor preprocessing parity only",
    }:
        raise LR0ArtifactLockError("LR0 target changed")

    artifacts = record["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 5:
        raise LR0ArtifactLockError("exactly five LR0 artifacts are required")
    by_key = {}
    for item in artifacts:
        item = _mapping(item, "artifact")
        if set(item) != {"role", "package", "version", "filename", "url", "sha256", "hash_source", "status", "boundary"}:
            raise LR0ArtifactLockError("artifact fields must be exact")
        key = "bootstrap" if item["role"] == "bootstrap" else item["package"]
        if key in by_key:
            raise LR0ArtifactLockError("artifact roles/packages must be unique")
        by_key[key] = item
        parsed = urlparse(item["url"])
        if parsed.scheme != "https" or not parsed.netloc or item["filename"] not in unquote(parsed.path):
            raise LR0ArtifactLockError("artifact URL must be HTTPS and name the exact file")
        if item["package"] == "micromamba" and f"/download/{item['version']}/" not in parsed.path:
            raise LR0ArtifactLockError("bootstrap URL must contain the exact versioned release")
        if item["status"] == "solver-metadata-hash-locked":
            if not str(item["hash_source"]).startswith("micromamba-2.9.0-dry-run-"):
                raise LR0ArtifactLockError("solver metadata must identify the frozen dry run")
        else:
            source = urlparse(item["hash_source"])
            if source.scheme != "https" or not source.netloc:
                raise LR0ArtifactLockError("hash source must be an HTTPS official metadata page")
        digest = item["sha256"]
        if item["status"] in {
            "official-metadata-hash-locked",
            "official-asset-quarantine-hash-locked",
            "solver-metadata-hash-locked",
        }:
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise LR0ArtifactLockError("hash-locked artifacts require a lowercase SHA-256")
        else:
            raise LR0ArtifactLockError("unsupported artifact status")

    if set(by_key) != set(_EXPECTED):
        raise LR0ArtifactLockError("LR0 artifact set must be exact")
    for key, (package, version) in _EXPECTED.items():
        if (by_key[key]["package"], by_key[key]["version"]) != (package, version):
            raise LR0ArtifactLockError(f"{key} pin changed")
    if any(item["sha256"] is None for item in artifacts):
        raise LR0ArtifactLockError("complete lock cannot retain unresolved hashes")
    for field in ("lock_complete", "dry_run_allowed", "download_allowed"):
        if record[field] is not True:
            raise LR0ArtifactLockError("complete lock must admit quarantine downloads and dry run")
    for field in ("install_allowed", "checkpoint_allowed", "g1_evidence_allowed"):
        if record[field] is not False:
            raise LR0ArtifactLockError("artifact lock alone cannot authorize install or G1 evidence")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LR0ArtifactLockError(f"{name} must be a mapping")
    return value
