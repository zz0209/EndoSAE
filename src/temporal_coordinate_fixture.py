"""Semantic contract for EndoFM temporal-coordinate runtime evidence.

The fixture links exact decoder indices to frame-coded patch-embedding
fingerprints and to the inferred H-W-T token layout. It deliberately does not
import torch. A valid record is not a runtime result unless it was produced by
the pinned extractor and checkpoint named in the record.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.endofm_contracts import ContractError, TokenLayout, validate_sampling_index_asset


class TemporalCoordinateFixtureError(ValueError):
    """Raised when temporal-coordinate evidence is internally inconsistent."""


def validate_temporal_coordinate_fixture(
    record: Mapping[str, Any], sampling_asset: Mapping[str, Any], sampling_asset_sha256: str
) -> None:
    """Validate a coordinate fixture against exact decoded-frame provenance."""

    if not isinstance(record, Mapping):
        raise TemporalCoordinateFixtureError("fixture must be a mapping")
    required = {
        "schema_version", "status", "fixture_id", "model_commit",
        "checkpoint_sha256", "preprocessing_id", "sampling_index_asset_sha256",
        "clip_id", "device", "dtype", "hook_target", "input_layout",
        "image_height", "image_width", "patch_size", "hidden_size",
        "original_input_sha256", "permuted_input_sha256",
        "original_actual_frame_indices", "permutation",
        "permuted_actual_frame_indices", "activation_shape",
        "original_frame_fingerprints", "permuted_frame_fingerprints",
        "token_landmarks",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise TemporalCoordinateFixtureError(
            f"missing temporal-coordinate fields: {', '.join(missing)}"
        )
    if record["schema_version"] != "endosae.temporal-coordinate-fixture.v0":
        raise TemporalCoordinateFixtureError("unsupported schema_version")
    if record["model_commit"] != "206427ebfb77a937ef0cd60370331bcedd74e5a2":
        raise TemporalCoordinateFixtureError("fixture does not target pinned EndoFM commit")
    if record["status"] not in {"pass", "fail", "invalid"}:
        raise TemporalCoordinateFixtureError("invalid status")
    for key in (
        "fixture_id", "model_commit", "preprocessing_id", "clip_id", "device", "dtype"
    ):
        _require_nonempty_string(record[key], key)
    for key in (
        "checkpoint_sha256", "sampling_index_asset_sha256",
        "original_input_sha256", "permuted_input_sha256",
    ):
        _require_sha256(record[key], key)
    _require_sha256(sampling_asset_sha256, "observed sampling asset hash")
    if record["sampling_index_asset_sha256"].lower() != sampling_asset_sha256.lower():
        raise TemporalCoordinateFixtureError("sampling asset hash does not match fixture provenance")
    if record["hook_target"] != "patch_embed":
        raise TemporalCoordinateFixtureError("coordinate fixture must hook patch_embed")
    if record["input_layout"] != "B,C,T,H,W":
        raise TemporalCoordinateFixtureError("unexpected input layout")

    try:
        validate_sampling_index_asset(sampling_asset)
    except ContractError as exc:
        raise TemporalCoordinateFixtureError(f"invalid sampling asset: {exc}") from exc
    records = [item for item in sampling_asset["records"] if item["clip_id"] == record["clip_id"]]
    if len(records) != 1:
        raise TemporalCoordinateFixtureError("clip_id must resolve to exactly one sampling record")
    sampling = records[0]

    layout = _layout_from_record(record, sampling["num_frames"])
    original_indices = _integer_list(record["original_actual_frame_indices"], "original_actual_frame_indices")
    if original_indices != sampling["actual_frame_indices"]:
        raise TemporalCoordinateFixtureError("fixture indices disagree with sampling provenance")
    permutation = _integer_list(record["permutation"], "permutation")
    expected_positions = list(range(layout.frames))
    if sorted(permutation) != expected_positions or permutation == expected_positions:
        raise TemporalCoordinateFixtureError("permutation must be a nonidentity bijection")
    expected_permuted_indices = [original_indices[position] for position in permutation]
    if record["permuted_actual_frame_indices"] != expected_permuted_indices:
        raise TemporalCoordinateFixtureError("permuted indices do not follow permutation")

    _require_shape(record["activation_shape"], "activation_shape")
    expected_shape = [layout.frames, layout.grid_height * layout.grid_width, layout.hidden_size]
    if record["activation_shape"] != expected_shape:
        raise TemporalCoordinateFixtureError("unexpected patch_embed activation shape")

    original_fingerprints = _hash_list(
        record["original_frame_fingerprints"], layout.frames, "original_frame_fingerprints"
    )
    permuted_fingerprints = _hash_list(
        record["permuted_frame_fingerprints"], layout.frames, "permuted_frame_fingerprints"
    )
    expected_permuted_fingerprints = [original_fingerprints[position] for position in permutation]

    landmarks = record["token_landmarks"]
    if not isinstance(landmarks, list):
        raise TemporalCoordinateFixtureError("token_landmarks must be a list")
    observed_t = set()
    for landmark in landmarks:
        if not isinstance(landmark, Mapping):
            raise TemporalCoordinateFixtureError("token landmark must be a mapping")
        required_landmark = {"h", "w", "t", "token_index", "decoded_frame_index"}
        if required_landmark.difference(landmark):
            raise TemporalCoordinateFixtureError("incomplete token landmark")
        try:
            expected_token = layout.patch_token_index(landmark["h"], landmark["w"], landmark["t"])
        except ContractError as exc:
            raise TemporalCoordinateFixtureError(f"invalid token landmark: {exc}") from exc
        if landmark["token_index"] != expected_token:
            raise TemporalCoordinateFixtureError("token landmark index disagrees with H-W-T layout")
        if landmark["decoded_frame_index"] != original_indices[landmark["t"]]:
            raise TemporalCoordinateFixtureError("landmark decoded frame disagrees with sampling provenance")
        if landmark["h"] == 0 and landmark["w"] == 0:
            observed_t.add(landmark["t"])

    if record["status"] == "pass":
        if sampling["has_duplicate_indices"]:
            raise TemporalCoordinateFixtureError("pass requires duplicate-free decoded-frame indices")
        if len(set(original_fingerprints)) != layout.frames:
            raise TemporalCoordinateFixtureError("pass requires unique frame fingerprints")
        if record["original_input_sha256"] == record["permuted_input_sha256"]:
            raise TemporalCoordinateFixtureError("pass requires distinct permuted input")
        if permuted_fingerprints != expected_permuted_fingerprints:
            raise TemporalCoordinateFixtureError("frame fingerprints do not follow exact permutation")
        if observed_t != set(range(layout.frames)):
            raise TemporalCoordinateFixtureError("pass requires all temporal landmarks at spatial anchor (0,0)")


def _layout_from_record(record: Mapping[str, Any], frames: int) -> TokenLayout:
    values = {}
    for key in ("image_height", "image_width", "patch_size", "hidden_size"):
        value = record[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TemporalCoordinateFixtureError(f"{key} must be a positive integer")
        values[key] = value
    try:
        return TokenLayout(frames=frames, **values)
    except ContractError as exc:
        raise TemporalCoordinateFixtureError(f"invalid token layout: {exc}") from exc


def _integer_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value
    ):
        raise TemporalCoordinateFixtureError(f"{key} must contain non-negative integers")
    return value


def _hash_list(value: Any, length: int, key: str) -> list[str]:
    if not isinstance(value, list) or len(value) != length:
        raise TemporalCoordinateFixtureError(f"{key} length must equal frame count")
    for item in value:
        _require_sha256(item, key)
    return value


def _require_nonempty_string(value: Any, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TemporalCoordinateFixtureError(f"{key} must be non-empty")


def _require_sha256(value: Any, key: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise TemporalCoordinateFixtureError(f"{key} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise TemporalCoordinateFixtureError(f"{key} must be hexadecimal") from exc


def _require_shape(value: Any, key: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TemporalCoordinateFixtureError(f"{key} must be a non-empty shape")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value):
        raise TemporalCoordinateFixtureError(f"{key} must contain positive integers")
