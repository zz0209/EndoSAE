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
FORBIDDEN_NAMES = {"AGENTS.md", "master_log.md", "RRD.md"}
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


def audit(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        normalized = relative.as_posix()
        if relative.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden governance file: {normalized}")
            continue
        if relative.parts and relative.parts[0] in FORBIDDEN_ROOTS:
            errors.append(f"forbidden research-asset root: {normalized}")
            continue
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file type: {normalized}")
            continue
        target = ROOT / relative
        if not target.is_file():
            errors.append(f"staged path is not a regular file: {normalized}")
            continue
        size = target.stat().st_size
        if size > MAX_BYTES:
            errors.append(f"file exceeds {MAX_BYTES} bytes: {normalized} ({size})")
            continue
        payload = target.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                errors.append(f"possible {label}: {normalized}")
    return errors


def main() -> int:
    paths = staged_paths()
    if not paths:
        print("public staging audit: no staged files")
        return 1
    errors = audit(paths)
    if errors:
        print("public staging audit: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(f"public staging audit: PASS ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
