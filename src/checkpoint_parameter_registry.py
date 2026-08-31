"""Non-executing registry of checkpoint parameter-name strings.

This module scans pickle opcodes only.  It never invokes the pickle virtual
machine and therefore cannot establish a loaded checkpoint schema.
"""

from __future__ import annotations

import hashlib
import pickletools
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


class CheckpointParameterRegistryError(ValueError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "artifacts/quarantine/endofm-main/endo_fm.pth"
CHECKPOINT_SHA256 = "6fc7a64a044f1eff3b7f9eb233df37a3607735848d534364f12c2aebad1aea70"
PICKLE_MEMBER = "archive/data.pkl"
PARAMETER_RE = re.compile(r"^(?:module\.)?(?:backbone|head)(?:\.[A-Za-z0-9_]+)+$")
PARAMETER_SUFFIXES = (".weight", ".bias", ".weight_g", ".weight_v", "cls_token", "pos_embed", "time_embed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def scan_parameter_names(path: Path = CHECKPOINT) -> dict[str, Any]:
    if _sha256(path) != CHECKPOINT_SHA256:
        raise CheckpointParameterRegistryError("checkpoint hash mismatch")
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(PICKLE_MEMBER)
    strings = {
        argument
        for opcode, argument, _ in pickletools.genops(payload)
        if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}
        and isinstance(argument, str)
    }
    candidates = sorted(
        value for value in strings
        if PARAMETER_RE.fullmatch(value) and value.endswith(PARAMETER_SUFFIXES)
    )
    direct = sorted(value for value in candidates if not value.startswith("module."))
    module = sorted(value.removeprefix("module.") for value in candidates if value.startswith("module."))
    categories = Counter()
    for name in direct:
        if name.startswith("backbone.blocks."):
            categories["backbone_block"] += 1
        elif name in {"backbone.cls_token", "backbone.pos_embed", "backbone.time_embed"}:
            categories["backbone_embedding"] += 1
        elif name.startswith("backbone.patch_embed."):
            categories["backbone_patch_embed"] += 1
        elif name.startswith("backbone.norm."):
            categories["backbone_final_norm"] += 1
        elif name.startswith("head."):
            categories["head"] += 1
        else:
            categories["unclassified"] += 1
    return {
        "direct_count": len(direct),
        "module_prefixed_count": len(module),
        "canonical_sets_mirrored": direct == module,
        "canonical_name_sha256": _name_digest(direct),
        "categories": dict(sorted(categories.items())),
    }


def validate_registry_record(record: Mapping[str, Any], *, rescan: bool = True) -> None:
    if record.get("schema_version") != "endosae.checkpoint-parameter-name-registry.v0":
        raise CheckpointParameterRegistryError("unsupported schema")
    if record.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise CheckpointParameterRegistryError("checkpoint identity drift")
    if record.get("method") != "pickle-opcode-string-scan-no-deserialization":
        raise CheckpointParameterRegistryError("unsafe or unknown method")
    if record.get("loaded_schema_claimed") is not False or record.get("g1_admission") is not False:
        raise CheckpointParameterRegistryError("static registry cannot admit a loaded schema or G1")
    observed = record.get("observed", {})
    expected = {
        "direct_count": 255,
        "module_prefixed_count": 255,
        "canonical_sets_mirrored": True,
        "canonical_name_sha256": "e653bde17a190807dee28a0c37c687e0ac35599234f688875f4a42801c9848b1",
        "categories": {
            "backbone_block": 240,
            "backbone_embedding": 3,
            "backbone_final_norm": 2,
            "backbone_patch_embed": 2,
            "head": 8,
        },
    }
    if observed != expected:
        raise CheckpointParameterRegistryError("frozen static registry drift")
    if rescan and scan_parameter_names() != expected:
        raise CheckpointParameterRegistryError("checkpoint rescan disagrees with registry")
