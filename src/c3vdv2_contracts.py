"""Small, dependency-free contracts for the C3VDv2 B0 pilot.

These helpers deliberately stop short of loading images or warping activations.
They encode only conventions that can be checked against the official dataset
description before EndoFM or an SAE is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping


FLOW_LIMIT_PX = 20.0
UINT16_MAX = 65535
_VIDEO_ID = re.compile(
    r"^(?P<colon>c\d+)_(?P<segment>[a-z0-9]+)_t(?P<take>\d+)_v(?P<version>[123])$"
)


class C3VDv2ContractError(ValueError):
    """Raised when a C3VDv2 identifier or modality inventory is inconsistent."""


@dataclass(frozen=True)
class VideoIdentity:
    colon: str
    segment: str
    take: int
    version: int

    @property
    def physical_unit_id(self) -> str:
        """Unit shared by clean/debris repeats; it must never cross a split."""

        return f"{self.colon}_{self.segment}_t{self.take}"

    @property
    def clean_debris_pair_id(self) -> str:
        return f"{self.physical_unit_id}_v2v3"


def parse_video_id(video_id: str) -> VideoIdentity:
    match = _VIDEO_ID.fullmatch(video_id)
    if match is None:
        raise C3VDv2ContractError(f"unsupported C3VDv2 video id: {video_id!r}")
    return VideoIdentity(
        colon=match.group("colon"),
        segment=match.group("segment"),
        take=int(match.group("take")),
        version=int(match.group("version")),
    )


def validate_clean_debris_pair(clean_id: str, debris_id: str) -> str:
    """Return pair ID only for a v2-clean/v3-debris matched acquisition."""

    clean = parse_video_id(clean_id)
    debris = parse_video_id(debris_id)
    if clean.version != 2 or debris.version != 3:
        raise C3VDv2ContractError("a clean/debris pair must be ordered v2 then v3")
    if clean.physical_unit_id != debris.physical_unit_id:
        raise C3VDv2ContractError("v2/v3 records do not share a physical acquisition unit")
    return clean.clean_debris_pair_id


def decode_flow_uint16(value: int, *, limit_px: float = FLOW_LIMIT_PX) -> float:
    """Decode one official 16-bit flow channel from [0, 65535] to ±limit_px.

    C3VDv2 defines optical flow from the current frame to the previous frame.
    The direction is a dataset contract and must not be silently inverted by a
    downstream warp implementation.
    """

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT16_MAX:
        raise C3VDv2ContractError("flow value must be an integer in [0, 65535]")
    if limit_px <= 0:
        raise C3VDv2ContractError("flow limit must be positive")
    return (value / UINT16_MAX) * (2.0 * limit_px) - limit_px


def validate_frame_inventory(modalities: Mapping[str, Iterable[int]]) -> tuple[int, ...]:
    """Ensure dense GT modalities contain the same continuous frame IDs.

    The caller extracts IDs from archive member names.  Keeping archive parsing
    outside this function avoids freezing an unverified filename grammar.
    """

    required = {"rgb", "depth", "normals", "optical_flow", "occlusion", "diffuse"}
    missing = sorted(required - modalities.keys())
    if missing:
        raise C3VDv2ContractError(f"missing modalities: {missing}")

    normalized = {name: tuple(sorted(set(ids))) for name, ids in modalities.items() if name in required}
    rgb_ids = normalized["rgb"]
    if not rgb_ids:
        raise C3VDv2ContractError("empty frame inventory")
    expected = tuple(range(rgb_ids[0], rgb_ids[-1] + 1))
    if rgb_ids != expected:
        raise C3VDv2ContractError("RGB frame IDs are not continuous")
    mismatched = sorted(name for name, ids in normalized.items() if ids != rgb_ids)
    if mismatched:
        raise C3VDv2ContractError(f"frame IDs do not align for: {mismatched}")
    return rgb_ids
