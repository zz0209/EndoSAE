"""Semantic checks for the project-owned modern CUDA environment lock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class ModernEnvironmentLockError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def validate_modern_environment_lock(lock: Mapping[str, Any], root: Path) -> None:
    if lock.get("schema_version") != "endosae.modern-environment-lock.v0":
        raise ModernEnvironmentLockError("unsupported schema")
    if lock.get("status") != "installed-probed-preprocessing-not-parity":
        raise ModernEnvironmentLockError("current lock must preserve failed parity status")
    if lock.get("python") != "3.13.7" or lock.get("platform") != "win_amd64":
        raise ModernEnvironmentLockError("unexpected runtime identity")
    report = (root / lock.get("install_report_relative_path", "")).resolve()
    try:
        report.relative_to(root.resolve())
    except ValueError as exc:
        raise ModernEnvironmentLockError("install report escapes project") from exc
    if not report.is_file() or _sha256(report).lower() != lock.get("install_report_sha256", "").lower():
        raise ModernEnvironmentLockError("install report hash mismatch")
    expected = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "numpy": "2.5.2", "pillow": "12.3.0"}
    packages = lock.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(expected):
        raise ModernEnvironmentLockError("package lock is incomplete")
    for name, version in expected.items():
        if packages[name].get("version") != version or not _hex64(packages[name].get("wheel_sha256")):
            raise ModernEnvironmentLockError(f"invalid package lock: {name}")
    runtime = lock.get("runtime_probe", {})
    if runtime.get("torch_cuda_build") != "12.8" or runtime.get("cuda_available") is not True:
        raise ModernEnvironmentLockError("CUDA runtime probe not satisfied")
    if runtime.get("torch_torchvision_builds_coherent") is not True:
        raise ModernEnvironmentLockError("torch/torchvision builds are not coherent")
    smoke = runtime.get("gpu_smoke", {})
    if smoke.get("operation") != "float32-64x64-matmul" or not _hex64(smoke.get("output_sha256")):
        raise ModernEnvironmentLockError("GPU smoke probe is missing")
    if not isinstance(smoke.get("max_memory_allocated_bytes"), int) or smoke["max_memory_allocated_bytes"] <= 0:
        raise ModernEnvironmentLockError("GPU smoke memory observation is invalid")
    preprocessing = lock.get("preprocessing_probe", {})
    if preprocessing.get("checkpoint_accessed") is not False or preprocessing.get("g1_evidence") is not False:
        raise ModernEnvironmentLockError("preprocessing probe cannot claim checkpoint/G1 evidence")
    if preprocessing.get("matches_external_torch_2_8_raw_assets_exactly") is not True:
        raise ModernEnvironmentLockError("project/external reproducibility not established")
    if preprocessing.get("matches_lr0_under_frozen_tolerance") is not False:
        raise ModernEnvironmentLockError("failed LR0 parity must remain explicit")
    for key in ("checkpoint_loading_allowed", "formal_extraction_allowed", "formal_activation_cache_allowed"):
        if lock.get(key) is not False:
            raise ModernEnvironmentLockError(f"{key} must remain false")
