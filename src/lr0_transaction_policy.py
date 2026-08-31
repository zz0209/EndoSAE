"""Validate the no-side-effect transaction policy for LR0."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


class LR0TransactionPolicyError(ValueError):
    pass


def validate_lr0_transaction_policy(record: Mapping[str, Any], root: Path) -> None:
    required = {
        "schema_version", "status", "audited_at", "artifact_lock", "bootstrap_license",
        "transaction", "official_pytorch_conda_alternative", "unresolved_requirements",
        "dry_run_allowed", "download_allowed", "install_allowed", "checkpoint_allowed",
        "g1_evidence_allowed",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise LR0TransactionPolicyError("policy fields must be exact")
    if record["schema_version"] != "endosae.lr0-transaction-policy.v0":
        raise LR0TransactionPolicyError("unsupported transaction policy schema")
    if record["status"] != "transitive-lock-frozen-install-authorized":
        raise LR0TransactionPolicyError("unexpected LR0 transaction status")

    lock = _mapping(record["artifact_lock"], "artifact_lock")
    if set(lock) != {"path", "sha256"}:
        raise LR0TransactionPolicyError("artifact lock reference fields must be exact")
    path = _contained_file(root, lock["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock["sha256"]:
        raise LR0TransactionPolicyError("artifact lock hash mismatch")

    license_record = _mapping(record["bootstrap_license"], "bootstrap_license")
    if license_record != {
        "family": "BSD-3-Clause",
        "source": "https://raw.githubusercontent.com/mamba-org/mamba/main/LICENSE",
        "binary_redistribution_permitted_with_conditions": True,
        "notice_required": True,
        "project_acceptance_recorded": True,
    }:
        raise LR0TransactionPolicyError("bootstrap license boundary changed")

    tx = _mapping(record["transaction"], "transaction")
    if set(tx) != {"environment_prefix", "shell_initialization", "global_path_mutation", "user_condarc_read", "conda_phase", "pip_phase"}:
        raise LR0TransactionPolicyError("transaction fields must be exact")
    if tx["environment_prefix"] != "artifacts/environments/lr0":
        raise LR0TransactionPolicyError("LR0 prefix must remain project-owned")
    if any(tx[key] is not False for key in ("shell_initialization", "global_path_mutation", "user_condarc_read")):
        raise LR0TransactionPolicyError("transaction cannot mutate shell/global state or read user config")
    conda = _mapping(tx["conda_phase"], "conda_phase")
    if conda != {
        "channels": ["https://repo.anaconda.com/pkgs/main"],
        "override_channels": True,
        "channel_priority": "strict",
        "requested_specs": ["python=3.7.16=h6244533_0"],
        "explicit_lock_path": "configs/lr0_conda_explicit_win64.txt",
        "explicit_lock_sha256": hashlib.sha256(
            (root / "configs" / "lr0_conda_explicit_win64.txt").read_bytes()
        ).hexdigest(),
        "explicit_transitive_lock_required_before_install": True,
    }:
        raise LR0TransactionPolicyError("conda phase must be exact, isolated, and dry-run-only")
    pip = _mapping(tx["pip_phase"], "pip_phase")
    if pip != {
        "network_index_enabled": False,
        "require_hashes": True,
        "artifact_packages": ["numpy", "torch", "typing_extensions"],
        "exact_urls_from_artifact_lock_only": True,
    }:
        raise LR0TransactionPolicyError("pip phase must use only hash-locked direct artifacts")

    alternative = _mapping(record["official_pytorch_conda_alternative"], "official_pytorch_conda_alternative")
    if alternative.get("official_command_exists") is not True or alternative.get("win_64_python37_cpu_build_located") is not False:
        raise LR0TransactionPolicyError("PyTorch conda alternative boundary changed")
    if alternative.get("replacement_authorized") is not False:
        raise LR0TransactionPolicyError("an unverified conda build cannot replace the torch wheel")
    unresolved = record["unresolved_requirements"]
    if unresolved != []:
        raise LR0TransactionPolicyError("unresolved requirements must remain explicit and ordered")
    for field in ("dry_run_allowed", "download_allowed", "install_allowed"):
        if record[field] is not True:
            raise LR0TransactionPolicyError("frozen transaction must admit download, dry run, and install")
    for field in ("checkpoint_allowed", "g1_evidence_allowed"):
        if record[field] is not False:
            raise LR0TransactionPolicyError("LR0 installation alone cannot admit checkpoint or G1 evidence")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LR0TransactionPolicyError(f"{name} must be a mapping")
    return value


def _contained_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise LR0TransactionPolicyError("artifact lock path must be a string")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LR0TransactionPolicyError("artifact lock path escapes project root") from exc
    if not path.is_file():
        raise LR0TransactionPolicyError("artifact lock file does not exist")
    return path
