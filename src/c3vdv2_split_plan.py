"""Deterministic metadata-only candidate planning for the C3VDv2 paired pilot."""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.c3vdv2_contracts import C3VDv2ContractError, parse_video_id


@dataclass(frozen=True)
class PairUnit:
    unit_id: str
    geometry_group: str
    colon: str
    segment: str
    frames: int


def endofm_deterministic_indices(
    video_frames: int,
    *,
    num_frames: int = 8,
    sampling_rate: int = 32,
    source_fps: float = 30.0,
    target_fps: float = 30.0,
) -> tuple[int, ...]:
    """Emulate pinned decoder test sampling, including end-frame clamping.

    The pinned loader defines clip_size as sampling_rate * num_frames /
    target_fps * source_fps, then samples num_frames evenly over
    [0, clip_size - 1], casts to long, and clamps to the decoded video end.
    """

    if video_frames <= 0 or num_frames < 2 or sampling_rate <= 0 or source_fps <= 0 or target_fps <= 0:
        raise C3VDv2ContractError("invalid temporal sampling parameters")
    clip_size = sampling_rate * num_frames / target_fps * source_fps
    end = clip_size - 1.0
    last = video_frames - 1
    return tuple(min(math.floor(index * end / (num_frames - 1)), last) for index in range(num_frames))


def endofm_required_unclamped_frames(
    *, num_frames: int = 8, sampling_rate: int = 32, source_fps: float = 30.0, target_fps: float = 30.0
) -> int:
    """Minimum decoded frames needed for the pinned clip span without clamping."""

    if num_frames < 2 or sampling_rate <= 0 or source_fps <= 0 or target_fps <= 0:
        raise C3VDv2ContractError("invalid temporal sampling parameters")
    return math.ceil(sampling_rate * num_frames / target_fps * source_fps)


def load_complete_pairs(path: str | Path) -> tuple[PairUnit, ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[int, tuple[int, str, str]]] = defaultdict(dict)
    for row in rows:
        identity = parse_video_id(row.get("Video Name", ""))
        frames = int(row["Num_frames"])
        grouped[identity.physical_unit_id][identity.version] = (frames, identity.colon, identity.segment)

    pairs = []
    for unit_id, versions in grouped.items():
        if not {2, 3} <= versions.keys():
            continue
        if versions[2][0] != versions[3][0]:
            raise C3VDv2ContractError(f"v2/v3 frame mismatch for {unit_id}")
        frames, colon, segment = versions[2]
        pairs.append(PairUnit(unit_id, f"{colon}_{segment}", colon, segment, frames))
    return tuple(sorted(pairs, key=lambda item: item.unit_id))


def plan_geometry_grouped_split(
    pairs: tuple[PairUnit, ...],
    *,
    seed: str = "endosae-c3vdv2-split-v0",
) -> dict[str, str]:
    """Return a candidate 9/3/3 geometry-group split balanced on pair counts.

    All textures/takes for the same colon+segment geometry remain together.
    The optimizer balances total pairs and fully unclamped default-rate pairs;
    the seed only breaks equally scored assignments.
    """

    groups: dict[str, list[PairUnit]] = defaultdict(list)
    for pair in pairs:
        groups[pair.geometry_group].append(pair)
    names = tuple(sorted(groups))
    if len(names) != 15:
        raise C3VDv2ContractError(f"expected 15 geometry groups, found {len(names)}")

    total_pairs = len(pairs)
    required_frames = endofm_required_unclamped_frames()
    clean_pairs = sum(pair.frames >= required_frames for pair in pairs)
    targets = {
        "development": (round(total_pairs * 0.6), round(clean_pairs * 0.6)),
        "confirmation": (round(total_pairs * 0.2), round(clean_pairs * 0.2)),
        "test": (total_pairs - round(total_pairs * 0.6) - round(total_pairs * 0.2), clean_pairs - round(clean_pairs * 0.6) - round(clean_pairs * 0.2)),
    }

    best = None
    for confirmation in itertools.combinations(names, 3):
        remainder = tuple(name for name in names if name not in confirmation)
        for test in itertools.combinations(remainder, 3):
            assignment = {
                name: ("confirmation" if name in confirmation else "test" if name in test else "development")
                for name in names
            }
            if any(len({groups[name][0].colon for name in names if assignment[name] == split}) < 2 for split in targets):
                continue
            score = 0
            for split, (target_total, target_clean) in targets.items():
                members = [pair for pair in pairs if assignment[pair.geometry_group] == split]
                observed_total = len(members)
                observed_clean = sum(pair.frames >= required_frames for pair in members)
                score += (observed_total - target_total) ** 2 + (observed_clean - target_clean) ** 2
            signature = "|".join(f"{name}:{assignment[name]}" for name in names)
            tie = hashlib.sha256(f"{seed}|{signature}".encode()).hexdigest()
            candidate = (score, tie, assignment)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        raise C3VDv2ContractError("no valid geometry-grouped split found")
    return best[2]


def summarize_plan(pairs: tuple[PairUnit, ...], assignment: dict[str, str]) -> dict[str, dict[str, int]]:
    summary = {}
    for split in ("development", "confirmation", "test"):
        members = [pair for pair in pairs if assignment[pair.geometry_group] == split]
        summary[split] = {
            "geometry_groups": len({pair.geometry_group for pair in members}),
            "pairs": len(members),
            "default_rate_unclamped_pairs": sum(
                pair.frames >= endofm_required_unclamped_frames() for pair in members
            ),
            "tail_clamped_unique_pairs": sum(
                pair.frames < endofm_required_unclamped_frames()
                and len(set(endofm_deterministic_indices(pair.frames))) == 8
                for pair in members
            ),
            "boundary_repeat_pairs": sum(
                len(set(endofm_deterministic_indices(pair.frames))) < 8 for pair in members
            ),
        }
    return summary
