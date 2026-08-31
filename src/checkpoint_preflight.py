"""Non-deserializing structural preflight for a quarantined checkpoint file.

The implementation reads archive metadata only. It never imports torch, calls
pickle, or extracts archive members. Passing preflight means only that the
container is structurally bounded; it does not make checkpoint contents safe.
"""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


class CheckpointPreflightError(ValueError):
    """Raised when a checkpoint container violates a frozen intake limit."""


@dataclass(frozen=True)
class PreflightLimits:
    max_file_bytes: int = 4 * 1024**3
    max_members: int = 100_000
    max_total_uncompressed_bytes: int = 16 * 1024**3
    max_member_uncompressed_bytes: int = 4 * 1024**3
    max_compression_ratio: float = 1_000.0


@dataclass(frozen=True)
class CheckpointPreflightResult:
    schema_version: str
    status: str
    file_sha256: str
    file_size_bytes: int
    container_format: str
    member_count: int
    total_uncompressed_bytes: int
    contains_pickle_named_member: bool
    encrypted_member_count: int
    link_or_special_member_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def preflight_checkpoint(
    checkpoint_path: str | Path, limits: PreflightLimits = PreflightLimits()
) -> CheckpointPreflightResult:
    """Inspect ZIP/TAR metadata without deserializing or extracting payloads."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise CheckpointPreflightError("checkpoint path must be an existing file")
    size = path.stat().st_size
    if size <= 0 or size > limits.max_file_bytes:
        raise CheckpointPreflightError("checkpoint file size exceeds intake limit")
    digest = _sha256_file(path)
    if zipfile.is_zipfile(path):
        return _preflight_zip(path, digest, size, limits)
    if tarfile.is_tarfile(path):
        return _preflight_tar(path, digest, size, limits)
    raise CheckpointPreflightError("unknown or raw-pickle container; stop before deserialization")


def _preflight_zip(
    path: Path, digest: str, size: int, limits: PreflightLimits
) -> CheckpointPreflightResult:
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        _validate_member_count(members, limits)
        names = [member.filename for member in members]
        _validate_names(names)
        total = 0
        encrypted = 0
        contains_pickle = False
        for member in members:
            if member.file_size < 0 or member.file_size > limits.max_member_uncompressed_bytes:
                raise CheckpointPreflightError("ZIP member exceeds uncompressed size limit")
            total += member.file_size
            if total > limits.max_total_uncompressed_bytes:
                raise CheckpointPreflightError("ZIP aggregate uncompressed size exceeds limit")
            compressed = max(member.compress_size, 1)
            if member.file_size / compressed > limits.max_compression_ratio:
                raise CheckpointPreflightError("ZIP member compression ratio exceeds limit")
            encrypted += int(bool(member.flag_bits & 0x1))
            contains_pickle |= PurePosixPath(member.filename.replace("\\", "/")).name.endswith(
                (".pkl", ".pickle")
            )
        if encrypted:
            raise CheckpointPreflightError("encrypted ZIP members are not accepted")
    return CheckpointPreflightResult(
        "endosae.checkpoint-preflight.v0", "pass", digest, size, "zip",
        len(members), total, contains_pickle, encrypted, 0,
    )


def _preflight_tar(
    path: Path, digest: str, size: int, limits: PreflightLimits
) -> CheckpointPreflightResult:
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        _validate_member_count(members, limits)
        names = [member.name for member in members]
        _validate_names(names)
        total = 0
        special = 0
        contains_pickle = False
        for member in members:
            if member.size < 0 or member.size > limits.max_member_uncompressed_bytes:
                raise CheckpointPreflightError("TAR member exceeds uncompressed size limit")
            total += member.size
            if total > limits.max_total_uncompressed_bytes:
                raise CheckpointPreflightError("TAR aggregate uncompressed size exceeds limit")
            special += int(member.issym() or member.islnk() or member.isdev() or member.isfifo())
            contains_pickle |= PurePosixPath(member.name.replace("\\", "/")).name.endswith(
                (".pkl", ".pickle")
            )
        if special:
            raise CheckpointPreflightError("TAR link/device/FIFO members are not accepted")
    return CheckpointPreflightResult(
        "endosae.checkpoint-preflight.v0", "pass", digest, size, "tar",
        len(members), total, contains_pickle, 0, special,
    )


def _validate_member_count(members: list[object], limits: PreflightLimits) -> None:
    if not members or len(members) > limits.max_members:
        raise CheckpointPreflightError("archive member count outside intake limit")


def _validate_names(names: list[str]) -> None:
    normalized = []
    for name in names:
        posix = PurePosixPath(name.replace("\\", "/"))
        if not name or posix.is_absolute() or ".." in posix.parts:
            raise CheckpointPreflightError("unsafe archive member path")
        if len(posix.parts) and ":" in posix.parts[0]:
            raise CheckpointPreflightError("drive-qualified archive member path")
        normalized.append(str(posix))
    if len(normalized) != len(set(normalized)):
        raise CheckpointPreflightError("duplicate archive member names")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
