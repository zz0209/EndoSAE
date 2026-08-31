"""Fail-closed first-load inventory runner for an approved isolated host.

Importing this module does not load a checkpoint.  The CLI requires a signed-off
isolation attestation and emits metadata only; tensor values are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_SHA256 = "6fc7a64a044f1eff3b7f9eb233df37a3607735848d534364f12c2aebad1aea70"
POLICY_SHA256 = "b55ebbd879bb4b9a992585d50f39e984ba0ae7aa20a9574827a1d21a58156212"
EXPECTED_TOP_LEVEL = ["student", "teacher", "optimizer", "epoch", "args", "dino_loss", "fp16_scaler"]
APPROVED_BACKENDS = {"windows-sandbox", "wsl2-disposable", "docker", "podman", "disposable-offline-host"}


class IsolatedInventoryError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_attestation(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != "endosae.first-load-isolation-attestation.v0":
        raise IsolatedInventoryError("unsupported isolation attestation")
    if record.get("backend") not in APPROVED_BACKENDS:
        raise IsolatedInventoryError("unapproved isolation backend")
    if record.get("network_enabled") is not False:
        raise IsolatedInventoryError("network must be disabled")
    if record.get("sensitive_mounts_present") is not False:
        raise IsolatedInventoryError("sensitive mounts must be absent")
    if record.get("ephemeral_workspace") is not True:
        raise IsolatedInventoryError("workspace must be ephemeral")
    if record.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise IsolatedInventoryError("attested checkpoint mismatch")
    if record.get("safe_globals_policy_sha256") != POLICY_SHA256:
        raise IsolatedInventoryError("safe-globals policy mismatch")
    if not isinstance(record.get("attestation_id"), str) or not record["attestation_id"]:
        raise IsolatedInventoryError("missing attestation id")


def _canonical_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode("utf-8")).hexdigest()


def inventory_loaded_object(root: Any, torch_module: Any) -> dict[str, Any]:
    if not isinstance(root, Mapping):
        raise IsolatedInventoryError("checkpoint root is not a mapping")
    top_level = list(root.keys())
    if top_level != EXPECTED_TOP_LEVEL:
        raise IsolatedInventoryError(f"top-level key mismatch: {top_level!r}")

    tensors: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    seen_paths: set[str] = set()
    storage_aliases: dict[int, str] = {}

    def visit(value: Any, path: str) -> None:
        if torch_module.is_tensor(value):
            if path in seen_paths:
                raise IsolatedInventoryError(f"duplicate tensor path: {path}")
            seen_paths.add(path)
            shape = [int(item) for item in value.shape]
            numel = int(value.numel())
            expected = 1
            for item in shape:
                if item < 0:
                    raise IsolatedInventoryError(f"negative tensor dimension: {path}")
                expected *= item
            if expected != numel:
                raise IsolatedInventoryError(f"tensor numel mismatch: {path}")
            storage_identity = id(value.untyped_storage())
            if storage_identity not in storage_aliases:
                storage_aliases[storage_identity] = f"storage-{len(storage_aliases):06d}"
            tensors.append({
                "path": path,
                "shape": shape,
                "dtype": str(value.dtype),
                "numel": numel,
                "requires_grad": bool(value.requires_grad),
                "storage_reference": storage_aliases[storage_identity],
            })
            return
        type_counts[f"{type(value).__module__}.{type(value).__qualname__}"] += 1
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, (str, int)):
                    raise IsolatedInventoryError(f"unsupported mapping key type at {path}")
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(root, "")
    student = root["student"]
    teacher = root["teacher"]
    if not isinstance(student, Mapping) or not isinstance(teacher, Mapping):
        raise IsolatedInventoryError("student/teacher branches must be mappings")
    student_names = [str(key) for key in student]
    teacher_names = [str(key).removeprefix("module.") for key in teacher]
    mapping_summary = {
        "student_count": len(student_names),
        "teacher_count": len(teacher_names),
        "canonical_sets_equal": sorted(student_names) == sorted(teacher_names),
        "student_canonical_sha256": _canonical_digest(student_names),
        "teacher_canonical_sha256": _canonical_digest(teacher_names),
    }
    return {
        "top_level_keys": top_level,
        "recursive_tensor_count": len(tensors),
        "recursive_tensor_numel": sum(item["numel"] for item in tensors),
        "tensor_records": sorted(tensors, key=lambda item: item["path"]),
        "non_tensor_type_summary": dict(sorted(type_counts.items())),
        "student_teacher_mapping_summary": mapping_summary,
    }


def isolated_load_and_inventory(checkpoint: Path, attestation_path: Path) -> dict[str, Any]:
    attestation_bytes = attestation_path.read_bytes()
    attestation = json.loads(attestation_bytes.decode("utf-8"))
    validate_attestation(attestation)
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise IsolatedInventoryError("checkpoint hash mismatch")
    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is not None:
        raise IsolatedInventoryError("unsafe torch environment override is present")
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"

    import numpy as np
    import torch

    allowed = [argparse.Namespace, np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
    torch.serialization.clear_safe_globals()
    try:
        with torch.serialization.safe_globals(allowed):
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise IsolatedInventoryError("weights-only load failed; no fallback is permitted") from exc
    finally:
        torch.serialization.clear_safe_globals()

    result = inventory_loaded_object(loaded, torch)
    result.update({
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "loader_environment_sha256": attestation["loader_environment_sha256"],
        "isolation_attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
        "unexpected_globals": [],
        "status": "pass",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise IsolatedInventoryError("refusing to overwrite an existing inventory")
    result = isolated_load_and_inventory(args.checkpoint, args.attestation)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

