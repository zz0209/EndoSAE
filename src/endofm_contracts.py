"""Pure-Python contracts for EndoFM activation extraction.

This module deliberately has no torch dependency. It encodes invariants inferred
from the pinned official EndoFM source so they can be tested before the legacy
runtime and checkpoint are available. Passing these checks does not establish
G1; runtime fixtures must still confirm the inferred contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """Raised when an activation or metadata contract is internally invalid."""


@dataclass(frozen=True)
class TokenLayout:
    """EndoFM TimeSformer token layout for one clip.

    Patch tokens are flattened with spatial position outside and time inside:
    ``((h * grid_width + w) * frames + t)``. Token zero is the global CLS.
    """

    frames: int
    image_height: int
    image_width: int
    patch_size: int
    hidden_size: int
    has_global_cls: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "frames": self.frames,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(f"{name} must be a positive integer, got {value!r}")
        if self.image_height % self.patch_size:
            raise ContractError("image_height must be divisible by patch_size")
        if self.image_width % self.patch_size:
            raise ContractError("image_width must be divisible by patch_size")

    @property
    def grid_height(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.patch_size

    @property
    def patch_tokens(self) -> int:
        return self.grid_height * self.grid_width * self.frames

    @property
    def token_count(self) -> int:
        return self.patch_tokens + int(self.has_global_cls)

    @property
    def activation_shape_per_clip(self) -> tuple[int, int]:
        return (self.token_count, self.hidden_size)

    def patch_token_index(self, h: int, w: int, t: int) -> int:
        """Return the absolute token index, including the optional CLS offset."""
        self._check_coordinate("h", h, self.grid_height)
        self._check_coordinate("w", w, self.grid_width)
        self._check_coordinate("t", t, self.frames)
        offset = int(self.has_global_cls)
        return offset + ((h * self.grid_width + w) * self.frames + t)

    def patch_coordinates(self, token_index: int) -> tuple[int, int, int]:
        """Invert :meth:`patch_token_index` for a non-CLS token."""
        if not isinstance(token_index, int) or isinstance(token_index, bool):
            raise ContractError("token_index must be an integer")
        offset = int(self.has_global_cls)
        flat = token_index - offset
        if flat < 0 or flat >= self.patch_tokens:
            raise ContractError(
                f"token_index {token_index} is not a patch token in ["
                f"{offset}, {self.token_count})"
            )
        spatial, t = divmod(flat, self.frames)
        h, w = divmod(spatial, self.grid_width)
        return h, w, t

    @staticmethod
    def _check_coordinate(name: str, value: int, upper: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ContractError(f"{name} must be an integer")
        if value < 0 or value >= upper:
            raise ContractError(f"{name}={value} outside [0, {upper})")


REQUIRED_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "model_repository",
        "model_commit",
        "checkpoint_sha256",
        "preprocessing_id",
        "layer",
        "hook_kind",
        "dtype",
        "shape",
        "token_layout",
        "sampling",
        "source_manifest_sha256",
        "extractor_commit",
    }
)


def validate_cache_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate required activation-cache provenance fields.

    This checks structural integrity only. It cannot verify that hashes point to
    the intended files or that the stated preprocessing was actually executed.
    """
    if not isinstance(metadata, Mapping):
        raise ContractError("metadata must be a mapping")
    missing = sorted(REQUIRED_CACHE_FIELDS.difference(metadata))
    if missing:
        raise ContractError(f"missing cache metadata fields: {', '.join(missing)}")

    if metadata["schema_version"] != "endosae.activation-cache.v0":
        raise ContractError("unsupported schema_version")

    for key in (
        "model_repository",
        "model_commit",
        "preprocessing_id",
        "layer",
        "hook_kind",
        "dtype",
        "extractor_commit",
    ):
        _require_nonempty_string(metadata, key)

    _require_sha256(metadata, "checkpoint_sha256")
    _require_sha256(metadata, "source_manifest_sha256")
    _require_positive_shape(metadata["shape"])

    token_layout = metadata["token_layout"]
    if not isinstance(token_layout, Mapping):
        raise ContractError("token_layout must be a mapping")
    required_layout = {
        "order",
        "has_global_cls",
        "frames",
        "grid_height",
        "grid_width",
        "hidden_size",
    }
    missing_layout = sorted(required_layout.difference(token_layout))
    if missing_layout:
        raise ContractError(
            f"missing token_layout fields: {', '.join(missing_layout)}"
        )
    if token_layout["order"] != "spatial-major,time-minor":
        raise ContractError("unexpected token layout order")

    sampling = metadata["sampling"]
    if not isinstance(sampling, Mapping) or not sampling:
        raise ContractError("sampling must be a non-empty mapping")
    required_sampling = {
        "num_frames", "sampling_rate", "target_fps", "index_space",
        "index_policy", "clamp_policy", "decoder_source_sha256",
        "indices_schema_version", "indices_asset_sha256", "clip_count",
        "clamped_clip_count", "duplicate_index_clip_count",
        "contains_clamped_clips", "contains_duplicate_indices",
    }
    missing_sampling = sorted(required_sampling.difference(sampling))
    if missing_sampling:
        raise ContractError(f"missing sampling fields: {', '.join(missing_sampling)}")
    if sampling["index_space"] != "decoded-frame-index":
        raise ContractError("unexpected sampling index space")
    if sampling["index_policy"] != "pinned-endofm-linspace-long":
        raise ContractError("unexpected sampling index policy")
    if sampling["clamp_policy"] != "clamp-to-last-decoded-frame":
        raise ContractError("unexpected sampling clamp policy")
    if sampling["indices_schema_version"] != "endosae.sampling-indices.v0":
        raise ContractError("unexpected sampling indices schema")
    _require_sha256(sampling, "decoder_source_sha256")
    _require_sha256(sampling, "indices_asset_sha256")
    for key in ("num_frames", "sampling_rate", "clip_count"):
        if not isinstance(sampling[key], int) or isinstance(sampling[key], bool) or sampling[key] <= 0:
            raise ContractError(f"sampling.{key} must be a positive integer")
    if not isinstance(sampling["target_fps"], (int, float)) or isinstance(sampling["target_fps"], bool) or sampling["target_fps"] <= 0:
        raise ContractError("sampling.target_fps must be positive")
    for count_key, flag_key in (
        ("clamped_clip_count", "contains_clamped_clips"),
        ("duplicate_index_clip_count", "contains_duplicate_indices"),
    ):
        count = sampling[count_key]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= sampling["clip_count"]:
            raise ContractError(f"sampling.{count_key} must be within clip_count")
        if sampling[flag_key] is not (count > 0):
            raise ContractError(f"sampling.{flag_key} disagrees with {count_key}")


def validate_sampling_index_asset(asset: Mapping[str, Any]) -> None:
    """Validate exact requested and actual frame indices for every cached clip."""

    if not isinstance(asset, Mapping) or asset.get("schema_version") != "endosae.sampling-indices.v0":
        raise ContractError("invalid sampling index asset schema")
    records = asset.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("sampling index asset requires records")
    seen_clip_ids = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ContractError("sampling index record must be a mapping")
        required = {
            "clip_id", "manifest_asset_sha256", "source_video_id",
            "source_frame_count", "source_fps", "target_fps", "num_frames",
            "sampling_rate", "requested_frame_indices", "actual_frame_indices",
            "was_clamped", "has_duplicate_indices",
        }
        missing = sorted(required.difference(record))
        if missing:
            raise ContractError(f"missing sampling index fields: {', '.join(missing)}")
        clip_id = record["clip_id"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen_clip_ids:
            raise ContractError("clip_id must be non-empty and unique")
        seen_clip_ids.add(clip_id)
        _require_sha256(record, "manifest_asset_sha256")
        _require_nonempty_string(record, "source_video_id")
        frame_count = record["source_frame_count"]
        num_frames = record["num_frames"]
        if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
            raise ContractError("source_frame_count must be positive")
        if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
            raise ContractError("num_frames must be positive")
        sampling_rate = record["sampling_rate"]
        if not isinstance(sampling_rate, int) or isinstance(sampling_rate, bool) or sampling_rate <= 0:
            raise ContractError("sampling_rate must be a positive integer")
        for fps_key in ("source_fps", "target_fps"):
            fps = record[fps_key]
            if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
                raise ContractError(f"{fps_key} must be positive")
        requested = record["requested_frame_indices"]
        actual = record["actual_frame_indices"]
        if not isinstance(requested, list) or not isinstance(actual, list) or len(requested) != num_frames or len(actual) != num_frames:
            raise ContractError("sampling index lengths must equal num_frames")
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in requested):
            raise ContractError("requested indices must be non-negative integers")
        if requested != sorted(requested):
            raise ContractError("requested indices must be nondecreasing")
        expected_actual = [min(index, frame_count - 1) for index in requested]
        if actual != expected_actual:
            raise ContractError("actual indices do not match clamp policy")
        was_clamped = any(index >= frame_count for index in requested)
        has_duplicates = len(set(actual)) < len(actual)
        if not isinstance(record["was_clamped"], bool):
            raise ContractError("was_clamped must be boolean")
        if not isinstance(record["has_duplicate_indices"], bool):
            raise ContractError("has_duplicate_indices must be boolean")
        if record["was_clamped"] is not was_clamped:
            raise ContractError("was_clamped flag disagrees with indices")
        if record["has_duplicate_indices"] is not has_duplicates:
            raise ContractError("has_duplicate_indices flag disagrees with indices")


def _require_nonempty_string(metadata: Mapping[str, Any], key: str) -> None:
    value = metadata[key]
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")


def _require_sha256(metadata: Mapping[str, Any], key: str) -> None:
    value = metadata[key]
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{key} must be a 64-character hexadecimal SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{key} must be hexadecimal") from exc


def _require_positive_shape(shape: Any) -> None:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ContractError("shape must be a sequence")
    if not shape:
        raise ContractError("shape cannot be empty")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape):
        raise ContractError("shape dimensions must be positive integers")


DEFAULT_ENDOFM_LAYOUT = TokenLayout(
    frames=8,
    image_height=224,
    image_width=224,
    patch_size=16,
    hidden_size=768,
)
