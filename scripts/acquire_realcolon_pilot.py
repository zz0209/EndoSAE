"""Acquire the frozen REAL-Colon pilot archives with resumable, audited transfer.

This script owns download/verification/promotion only. It does not extract data,
generate labels, or run experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_FRACTION = 0.0025
LOG_PATH = None
PROGRESS_PATH = None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # A concurrent Windows reader can briefly deny replacement of a status file.
    for attempt in range(10):
        try:
            os.replace(partial, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def emit_progress(phase: str, completed: int, total: int, detail: str) -> None:
    percent = 100.0 if total == 0 else 100.0 * completed / total
    line = f"PROGRESS {percent:6.2f}% | {phase} | {detail}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    if PROGRESS_PATH is not None:
        atomic_json(PROGRESS_PATH, {"phase": phase, "completed_bytes_or_files": completed,
                    "total_bytes_or_files": total, "percent": percent, "detail": detail,
                    "updated_at_unix": time.time(), "pid": os.getpid()})


def hash_file(path: Path, expected_md5: str) -> tuple[str, str]:
    total = path.stat().st_size
    completed = 0
    next_fraction = 0.0
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            md5.update(chunk)
            sha256.update(chunk)
            completed += len(chunk)
            fraction = 1.0 if total == 0 else completed / total
            if fraction >= next_fraction or completed == total:
                emit_progress("verify", completed, total, path.name)
                next_fraction = fraction + PROGRESS_FRACTION
    actual_md5 = md5.hexdigest()
    if actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(f"MD5 mismatch for {path}: {actual_md5} != {expected_md5}")
    return actual_md5, sha256.hexdigest()


def download(file_record: dict, target_dir: Path) -> dict:
    expected_size = int(file_record["bytes"])
    destination = target_dir / file_record["name"]
    partial = destination.with_suffix(destination.suffix + ".partial")

    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"Existing destination has wrong size: {destination}")
        md5, sha256 = hash_file(destination, file_record["md5"])
        return {**file_record, "path": str(destination), "md5": md5, "sha256": sha256, "status": "verified-existing"}

    target_dir.mkdir(parents=True, exist_ok=True)
    resume_at = partial.stat().st_size if partial.exists() else 0
    if resume_at > expected_size:
        raise RuntimeError(f"Partial file is larger than expected: {partial}")
    if resume_at == expected_size:
        # Verification/status interruption after transfer needs no HTTP request.
        md5, sha256 = hash_file(partial, file_record["md5"])
        os.replace(partial, destination)
        return {**file_record, "path": str(destination), "md5": md5, "sha256": sha256,
                "status": "verified-complete-partial-promoted"}

    request = urllib.request.Request(file_record["download_url"])
    if resume_at:
        request.add_header("Range", f"bytes={resume_at}-")
    response = urllib.request.urlopen(request, timeout=120)
    status = getattr(response, "status", response.getcode())
    if resume_at and status != 206:
        response.close()
        partial.unlink()
        resume_at = 0
        response = urllib.request.urlopen(file_record["download_url"], timeout=120)
        status = getattr(response, "status", response.getcode())
    if status not in (200, 206):
        response.close()
        raise RuntimeError(f"Unexpected HTTP status {status} for {file_record['name']}")
    if resume_at and not response.headers.get("Content-Range", "").startswith(f"bytes {resume_at}-"):
        response.close()
        raise RuntimeError("range response does not start at the requested resume offset")

    completed = resume_at
    next_fraction = 0.0 if expected_size == 0 else completed / expected_size
    mode = "ab" if resume_at else "wb"
    emit_progress("download", completed, expected_size, file_record["name"])
    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)
            completed += len(chunk)
            fraction = 1.0 if expected_size == 0 else completed / expected_size
            if fraction >= next_fraction or completed == expected_size:
                emit_progress("download", completed, expected_size, file_record["name"])
                next_fraction = fraction + PROGRESS_FRACTION
        handle.flush()
        os.fsync(handle.fileno())
    if completed != expected_size:
        raise RuntimeError(f"Size mismatch for {partial}: {completed} != {expected_size}")

    md5, sha256 = hash_file(partial, file_record["md5"])
    os.replace(partial, destination)
    return {**file_record, "path": str(destination), "md5": md5, "sha256": sha256, "status": "downloaded-verified-promoted"}


def main() -> int:
    global LOG_PATH, PROGRESS_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--option", default="minimum-four-study-pilot")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = [item for item in config["frame_acquisition_options"] if item["option"] == args.option]
    if len(options) != 1:
        raise RuntimeError(f"Expected one acquisition option named {args.option}")
    option = options[0]
    if option.get("status") != "admitted-download-contract-frozen-not-started":
        raise RuntimeError("Acquisition option is not in the frozen admitted state")
    files = option.get("files", [])
    expected_count = int(option.get("expected_file_count", 4))
    if not files or len(files) != expected_count:
        raise RuntimeError("acquisition files disagree with the frozen expected count")
    if len({row["name"] for row in files}) != len(files):
        raise RuntimeError("duplicate acquisition file")
    for row in files:
        if row["name"] != row["video_id"] + "_frames.tar.gz" or Path(row["name"]).name != row["name"]:
            raise RuntimeError("invalid frame archive filename")
        if len(row["md5"]) != 32 or any(c not in "0123456789abcdef" for c in row["md5"].lower()):
            raise RuntimeError("A complete official MD5 is required before transfer")
    target_dir = Path(option["target_directory"])
    identity = {"source": str(args.config), "option": args.option, "files": files,
                "target_directory": str(target_dir)}
    results = []
    if args.resume:
        previous = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError("resume configuration changed")
        metrics_path = args.run_dir / "metrics.json"
        if metrics_path.exists():
            for record in json.loads(metrics_path.read_text(encoding="utf-8"))["completed_files"]:
                path = Path(record["path"])
                if (path.is_file() and path.stat().st_size == record["bytes"]
                        and path.stat().st_mtime_ns == record.get("verified_mtime_ns")):
                    results.append(record)
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(args.run_dir / "config.json", identity)
    LOG_PATH = args.run_dir / "stdout.log"
    PROGRESS_PATH = args.run_dir / "progress.json"
    status = "failed"
    failure = None
    try:
        for file_record in files:
            if any(row["name"] == file_record["name"] for row in results):
                continue
            atomic_json(args.run_dir / "status.json", {"status": "running", "pid": os.getpid(),
                        "current_file": file_record["name"], "completed_file_count": len(results),
                        "started_at_unix": started})
            record = download(file_record, target_dir)
            record["verified_mtime_ns"] = Path(record["path"]).stat().st_mtime_ns
            results.append(record)
            atomic_json(args.run_dir / "metrics.json", {"status": "running", "completed_files": results,
                        "expected_file_count": len(files)})
        status = "pass"
        return 0
    except Exception as error:
        failure = repr(error)
        emit_progress("error", len(results), len(files), failure)
        raise
    finally:
        metrics = {
            "schema_version": "endosae.realcolon-acquisition.v0",
            "status": status,
            "formal_experiment": False,
            "scientific_claim_allowed": False,
            "fallback_used": False,
            "elapsed_seconds": time.time() - started,
            "target_directory": str(target_dir),
            "completed_files": results,
            "expected_file_count": len(files),
            "error": failure,
        }
        atomic_json(args.run_dir / "metrics.json", metrics)
        atomic_json(args.run_dir / "status.json", {"status": status, "completed_file_count": len(results), "error": failure})
        emit_progress("acquisition", len(results), len(files), status)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
