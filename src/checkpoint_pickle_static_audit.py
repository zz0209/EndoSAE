"""Static pickle-opcode audit for a PyTorch ZIP checkpoint.

The implementation never calls pickle.load, torch.load, persistent_load, or a
pickle VM. It reads one named member and disassembles opcodes with pickletools.
Passing this audit is not authorization to deserialize the checkpoint.
"""

from __future__ import annotations

import hashlib
import pickletools
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


class CheckpointPickleStaticAuditError(ValueError):
    pass


EXPECTED_GLOBALS = [
    "collections.OrderedDict",
    "torch._utils._rebuild_tensor_v2",
    "torch.FloatStorage",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "_codecs.encode",
    "argparse.Namespace",
]
EXPECTED_UNSAFE_GLOBALS = [
    "argparse.Namespace",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
]
FORBIDDEN_DYNAMIC_OPCODES = frozenset({"STACK_GLOBAL", "EXT1", "EXT2", "EXT4", "OBJ", "INST", "NEWOBJ_EX"})


def audit_pickle_member(checkpoint: Path, member: str = "archive/data.pkl") -> dict[str, Any]:
    with zipfile.ZipFile(checkpoint) as archive:
        info = archive.getinfo(member)
        payload = archive.read(info)
        storage_members = sum(1 for name in archive.namelist() if name.startswith("archive/data/"))
    counts: Counter[str] = Counter()
    globals_seen: list[dict[str, Any]] = []
    last_position = -1
    for opcode, argument, position in pickletools.genops(payload):
        counts[opcode.name] += 1
        last_position = position
        if opcode.name == "GLOBAL":
            module, name = argument.split(" ", 1)
            globals_seen.append({"symbol": f"{module}.{name}", "position": position})
    return {
        "pickle_member": member,
        "pickle_member_size_bytes": len(payload),
        "pickle_member_sha256": hashlib.sha256(payload).hexdigest(),
        "opcode_count": sum(counts.values()),
        "opcode_counts": dict(sorted(counts.items())),
        "global_symbols": globals_seen,
        "forbidden_dynamic_opcodes_present": sorted(FORBIDDEN_DYNAMIC_OPCODES.intersection(counts)),
        "persistent_id_opcode_count": counts["BINPERSID"] + counts["PERSID"],
        "storage_member_count": storage_members,
        "complete_stop": counts["STOP"] == 1 and last_position >= 0,
    }


def validate_static_audit_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "endosae.checkpoint-pickle-static-audit.v0":
        raise CheckpointPickleStaticAuditError("unsupported schema")
    if report.get("status") != "pass-static-structure-only":
        raise CheckpointPickleStaticAuditError("static audit status must not imply load safety")
    if report.get("deserialization_performed") is not False or report.get("checkpoint_loading_allowed") is not False:
        raise CheckpointPickleStaticAuditError("static audit cannot authorize or perform loading")
    if report.get("checkpoint_sha256") != "6fc7a64a044f1eff3b7f9eb233df37a3607735848d534364f12c2aebad1aea70":
        raise CheckpointPickleStaticAuditError("checkpoint identity drift")
    if report.get("preflight_report_sha256") != "afced7ffb18bea6021aa0abc813c2b110d670fa38eb46b2ea93ba4b4420aa955":
        raise CheckpointPickleStaticAuditError("preflight report identity drift")
    if report.get("pickle_member") != "archive/data.pkl" or report.get("pickle_member_size_bytes") != 156239:
        raise CheckpointPickleStaticAuditError("pickle member identity drift")
    digest = report.get("pickle_member_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise CheckpointPickleStaticAuditError("invalid pickle member hash")
    symbols = [item.get("symbol") for item in report.get("global_symbols", []) if isinstance(item, Mapping)]
    if symbols != EXPECTED_GLOBALS:
        raise CheckpointPickleStaticAuditError("GLOBAL symbol set/order drift")
    if report.get("torch_static_unsafe_globals") != EXPECTED_UNSAFE_GLOBALS:
        raise CheckpointPickleStaticAuditError("torch static unsafe-global result drift")
    if report.get("forbidden_dynamic_opcodes_present") != []:
        raise CheckpointPickleStaticAuditError("dynamic global/extension opcode present")
    if report.get("complete_stop") is not True:
        raise CheckpointPickleStaticAuditError("pickle stream lacks a unique STOP")
    if report.get("persistent_id_opcode_count") != report.get("storage_member_count"):
        raise CheckpointPickleStaticAuditError("persistent IDs and storage members disagree")
    if report.get("weights_only_default_expected_to_succeed") is not False:
        raise CheckpointPickleStaticAuditError("unsafe globals must block default weights-only expectation")
    if report.get("safe_globals_allowlist_frozen") is not False:
        raise CheckpointPickleStaticAuditError("safe-globals allowlist is not yet frozen")
