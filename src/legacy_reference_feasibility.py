"""Validate the staged, no-install legacy reference feasibility record."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class LegacyReferenceFeasibilityError(ValueError):
    pass


def validate_legacy_reference_feasibility(record: Mapping[str, Any], root: Path) -> None:
    required = {
        "schema_version", "status", "audited_at", "official_environment", "host",
        "artifact_evidence", "stages", "authorization_packet", "formal_activation_cache_allowed",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise LegacyReferenceFeasibilityError("feasibility fields must be exact")
    if record["schema_version"] != "endosae.legacy-reference-feasibility.v0":
        raise LegacyReferenceFeasibilityError("unsupported feasibility schema")
    if record["status"] != "lr0-installed-preprocessing-parity-failed":
        raise LegacyReferenceFeasibilityError("unexpected legacy reference status")

    env = _mapping(record["official_environment"], "official_environment")
    env_required = {"path", "sha256", "declared_subdir", "python", "python_build", "torch", "torchvision", "numpy", "av"}
    if set(env) != env_required:
        raise LegacyReferenceFeasibilityError("official environment fields must be exact")
    path = _contained_file(root, env["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != env["sha256"]:
        raise LegacyReferenceFeasibilityError("official environment hash mismatch")
    text = path.read_text(encoding="utf-8")
    pins = {
        "python": f"python={env['python']}={env['python_build']}",
        "torch": f"torch=={env['torch']}",
        "torchvision": f"torchvision=={env['torchvision']}",
        "numpy": f"numpy={env['numpy']}=",
        "av": f"av=={env['av']}",
    }
    for name, pin in pins.items():
        if pin not in text:
            raise LegacyReferenceFeasibilityError(f"{name} pin is not present in the official environment")
    if env["declared_subdir"] != "linux-64" or "ld_impl_linux-64" not in text:
        raise LegacyReferenceFeasibilityError("official environment platform must remain Linux-specific")

    host = _mapping(record["host"], "host")
    if set(host) != {"native_subdir", "conda_or_micromamba_available", "linux_isolation_available", "dependency_install_authorized"}:
        raise LegacyReferenceFeasibilityError("host fields must be exact")
    if host["native_subdir"] != "win-64":
        raise LegacyReferenceFeasibilityError("audited host must be win-64")
    if host["conda_or_micromamba_available"] is not True or host["dependency_install_authorized"] is not True:
        raise LegacyReferenceFeasibilityError("installed LR0 requires available authorized project tooling")
    if host["linux_isolation_available"] is not False:
        raise LegacyReferenceFeasibilityError("LR0 installation cannot claim Linux isolation")

    evidence = record["artifact_evidence"]
    if not isinstance(evidence, list) or len(evidence) < 6:
        raise LegacyReferenceFeasibilityError("artifact evidence is incomplete")
    packages = {item.get("package"): item for item in evidence if isinstance(item, Mapping)}
    if set(packages) != {"python", "torch", "numpy", "pillow", "opencv-python", "av"}:
        raise LegacyReferenceFeasibilityError("artifact evidence package set must be exact")
    if packages["python"].get("version") != "3.7.16" or packages["python"].get("win_64_binary") != "verified-conda":
        raise LegacyReferenceFeasibilityError("exact Windows Python must be evidenced via conda")
    if packages["av"].get("version") != "10.0.0" or packages["av"].get("win_64_binary") != "not-present-on-pypi-release":
        raise LegacyReferenceFeasibilityError("PyAV binary boundary must remain explicit")

    stages = record["stages"]
    if not isinstance(stages, list) or [item.get("id") for item in stages] != ["LR0", "LR1", "LR2"]:
        raise LegacyReferenceFeasibilityError("legacy stages must be ordered LR0, LR1, LR2")
    lr0, lr1, lr2 = stages
    if lr0.get("required_packages") != ["python==3.7.16", "numpy==1.21.5", "torch==1.8.0+cpu", "typing_extensions==4.3.0"]:
        raise LegacyReferenceFeasibilityError("LR0 must remain the minimal tensor-only environment")
    if "av" not in lr0.get("excluded_as_unnecessary", []):
        raise LegacyReferenceFeasibilityError("LR0 must not be blocked on PyAV")
    if "einops==0.6.1" not in lr1.get("required_packages", []):
        raise LegacyReferenceFeasibilityError("LR1 must include the pinned model import dependency")
    if "av==10.0.0" not in lr2.get("required_packages", []) or lr2.get("preferred_platform") != "linux-64-isolated":
        raise LegacyReferenceFeasibilityError("LR2 must preserve the unresolved Linux decode boundary")
    for stage in stages:
        if stage.get("checkpoint_allowed") is not False or stage.get("g1_evidence_allowed") is not False:
            raise LegacyReferenceFeasibilityError("unexecuted stages cannot admit checkpoint or G1 evidence")

    if lr0.get("status") != "installed-reference-preprocessing-parity-failed":
        raise LegacyReferenceFeasibilityError("LR0 execution state is stale")
    packet = _mapping(record["authorization_packet"], "authorization_packet")
    if packet.get("status") != "authorized-and-executed":
        raise LegacyReferenceFeasibilityError("LR0 authorization state is stale")
    if packet.get("downloads_allowed") is not True or packet.get("install_allowed") is not True:
        raise LegacyReferenceFeasibilityError("authorized LR0 must record download and install admission")
    if packet.get("external_project_environment_mutation_allowed") is not False:
        raise LegacyReferenceFeasibilityError("authorization cannot expand to external environments")
    if record["formal_activation_cache_allowed"] is not False:
        raise LegacyReferenceFeasibilityError("feasibility evidence cannot allow formal activation cache")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyReferenceFeasibilityError(f"{name} must be a mapping")
    return value


def _contained_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise LegacyReferenceFeasibilityError("environment path must be a string")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LegacyReferenceFeasibilityError("environment path escapes project root") from exc
    if not path.is_file():
        raise LegacyReferenceFeasibilityError("environment file does not exist")
    return path
