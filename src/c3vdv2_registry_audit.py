"""Audit the official C3VDv2 registered-video summary without loading images."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.c3vdv2_contracts import C3VDv2ContractError, parse_video_id


@dataclass(frozen=True)
class RegisteredSummary:
    videos: int
    frames: int
    physical_units: int
    matched_v2_v3_pairs: int
    missing_v2_v3: tuple[str, ...]
    frame_mismatched_pairs: tuple[str, ...]


def audit_registered_summary(path: str | Path) -> RegisteredSummary:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise C3VDv2ContractError("registered summary is empty")

    grouped: dict[str, dict[int, int]] = defaultdict(dict)
    total_frames = 0
    for row in rows:
        video_id = row.get("Video Name", "")
        identity = parse_video_id(video_id)
        expected = (row.get("Colon"), row.get("Segment"), row.get("Phantom Number"), row.get("Video Number"))
        observed = (identity.colon, identity.segment, f"t{identity.take}", f"v{identity.version}")
        if expected != observed:
            raise C3VDv2ContractError(f"row columns disagree with Video Name: {video_id}")
        try:
            frames = int(row["Num_frames"])
        except (KeyError, TypeError, ValueError) as error:
            raise C3VDv2ContractError(f"invalid Num_frames for {video_id}") from error
        if frames <= 0:
            raise C3VDv2ContractError(f"non-positive Num_frames for {video_id}")
        versions = grouped[identity.physical_unit_id]
        if identity.version in versions:
            raise C3VDv2ContractError(f"duplicate version for {identity.physical_unit_id}")
        versions[identity.version] = frames
        total_frames += frames

    missing = tuple(sorted(unit for unit, versions in grouped.items() if not {2, 3} <= versions.keys()))
    mismatched = tuple(
        sorted(unit for unit, versions in grouped.items() if {2, 3} <= versions.keys() and versions[2] != versions[3])
    )
    return RegisteredSummary(
        videos=len(rows),
        frames=total_frames,
        physical_units=len(grouped),
        matched_v2_v3_pairs=sum({2, 3} <= versions.keys() for versions in grouped.values()),
        missing_v2_v3=missing,
        frame_mismatched_pairs=mismatched,
    )
