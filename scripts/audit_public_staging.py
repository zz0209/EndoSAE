"""Fail closed when the staged public snapshot contains private research assets."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {
    "artifacts", "configs", "data", "docs", "figures", "literature",
    "results", "third_party", "tmp", ".agents", ".codex",
}
FORBIDDEN_NAMES = {"AGENTS.md", "RRD.md"}
ALLOWED_MARKDOWN = {"master_log.md"}
FORBIDDEN_SUFFIXES = {
    ".md", ".markdown", ".pth", ".pt", ".ckpt", ".safetensors", ".onnx",
    ".npy", ".npz", ".parquet", ".zip", ".tar", ".gz", ".7z", ".mp4",
    ".avi", ".mov", ".dcm", ".nii", ".pem", ".key", ".pfx", ".p12",
}
MAX_BYTES = 10 * 1024 * 1024
SECRET_PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "aws-access-key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


def staged_paths() -> list[Path]:
    result = subprocess.run(
        [
            "git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "--cached",
            "--name-only", "--diff-filter=ACMR", "-z",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def staged_payload(relative: Path) -> bytes:
    """Audit the index bytes that will be committed, including file mode."""
    normalized = relative.as_posix()
    entry = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z", "--", normalized], cwd=ROOT,
    ).split(b"\0")
    entries = [item for item in entry if item]
    if len(entries) != 1 or entries[0].split(b" ", 1)[0] not in {b"100644", b"100755"}:
        raise ValueError("index entry must be one regular file")
    return subprocess.check_output(["git", "show", ":" + normalized], cwd=ROOT)


def audit(paths: list[Path], payload_reader=None) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        normalized = relative.as_posix()
        if relative.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden governance file: {normalized}")
            continue
        if relative.parts and relative.parts[0] in FORBIDDEN_ROOTS:
            errors.append(f"forbidden research-asset root: {normalized}")
            continue
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES and normalized not in ALLOWED_MARKDOWN:
            errors.append(f"forbidden file type: {normalized}")
            continue
        try:
            if payload_reader is None:
                target = ROOT / relative
                if target.is_symlink() or not target.is_file():
                    raise ValueError("not a regular file")
                payload = target.read_bytes()
            else:
                payload = payload_reader(relative)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            errors.append(f"unreadable regular staged file: {normalized} ({error})")
            continue
        size = len(payload)
        if size > MAX_BYTES:
            errors.append(f"file exceeds {MAX_BYTES} bytes: {normalized} ({size})")
            continue
        try:
            payload.decode("utf-8")
            if b"\0" in payload:
                raise ValueError("NUL byte in text source")
        except (UnicodeDecodeError, ValueError):
            errors.append(f"non-text payload in public source snapshot: {normalized}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                errors.append(f"possible {label}: {normalized}")
    return errors


def main() -> int:
    paths = staged_paths()
    if not paths:
        print("public staging audit: no staged files")
        return 1
    errors = audit(paths, staged_payload)
    if errors:
        print("public staging audit: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(f"public staging audit: PASS ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
