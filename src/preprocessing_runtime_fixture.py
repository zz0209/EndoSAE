"""Runtime record contract for the EndoFM validation preprocessing path.

The protocol state machine is pure Python.  It reproduces how the pinned
dataset freezes view indices at construction and resolves them against the
possibly mutated shared config in ``__getitem__``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class PreprocessingFixtureError(ValueError):
    """Raised when preprocessing runtime evidence is not self-consistent."""


PROTOCOL_COUNTS = {
    "official-faithful": (1, 3),
    "corrected-center": (1, 1),
    "corrected-three-crop": (3, 3),
}


def evaluation_view_plan(
    construction_num_spatial_crops: int,
    runtime_num_spatial_crops: int,
    num_ensemble_views: int = 1,
) -> list[dict[str, int]]:
    """Return stored-view to temporal/spatial resolution under shared cfg state."""
    for name, value in {
        "construction_num_spatial_crops": construction_num_spatial_crops,
        "runtime_num_spatial_crops": runtime_num_spatial_crops,
        "num_ensemble_views": num_ensemble_views,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PreprocessingFixtureError(f"{name} must be a positive integer")
    if runtime_num_spatial_crops not in {1, 3}:
        raise PreprocessingFixtureError("runtime spatial crops must be 1 or 3")
    stored = construction_num_spatial_crops * num_ensemble_views
    return [
        {
            "stored_view_index": index,
            "temporal_sample_index": index // runtime_num_spatial_crops,
            "spatial_sample_index": (
                index % runtime_num_spatial_crops if runtime_num_spatial_crops > 1 else 1
            ),
        }
        for index in range(stored)
    ]


def validate_preprocessing_runtime_fixture(
    record: Mapping[str, Any],
    contract_path: str | Path,
    sampling_asset_path: str | Path,
) -> None:
    required = {
        "schema_version", "status", "fixture_id", "reference_contract_sha256",
        "sampling_index_asset_sha256", "clip_id", "protocol",
        "construction_num_spatial_crops", "runtime_num_spatial_crops",
        "num_ensemble_views", "decoder_backend", "decoded_layout",
        "decoded_dtype", "decoded_channel_order", "decoded_tensor_sha256",
        "normalized_layout", "normalized_dtype", "normalized_shape",
        "normalized_tensor_sha256", "normalization_summary", "views",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise PreprocessingFixtureError("runtime fixture fields must be exact")
    if record["schema_version"] != "endosae.preprocessing-runtime-fixture.v0":
        raise PreprocessingFixtureError("unsupported fixture schema")
    if record["status"] not in {"pass", "fail", "invalid"}:
        raise PreprocessingFixtureError("invalid fixture status")
    for key in (
        "fixture_id", "clip_id", "protocol", "decoder_backend", "decoded_layout",
        "decoded_dtype", "decoded_channel_order", "normalized_layout", "normalized_dtype",
    ):
        _require_string(record[key], key)
    for key in (
        "reference_contract_sha256", "sampling_index_asset_sha256",
        "decoded_tensor_sha256", "normalized_tensor_sha256",
    ):
        _require_sha(record[key], key)

    contract_path = Path(contract_path)
    sampling_path = Path(sampling_asset_path)
    if _sha256(contract_path) != record["reference_contract_sha256"].lower():
        raise PreprocessingFixtureError("reference contract file hash mismatch")
    if _sha256(sampling_path) != record["sampling_index_asset_sha256"].lower():
        raise PreprocessingFixtureError("sampling asset file hash mismatch")
    contract = _json_object(contract_path)
    sampling = _json_object(sampling_path)
    if contract.get("schema_version") != "endosae.reference-input-contract.v0":
        raise PreprocessingFixtureError("unexpected reference contract")
    records = sampling.get("records")
    if not isinstance(records, list) or not any(
        isinstance(item, Mapping) and item.get("clip_id") == record["clip_id"] for item in records
    ):
        raise PreprocessingFixtureError("clip_id missing from sampling asset")

    protocol = record["protocol"]
    if protocol not in PROTOCOL_COUNTS:
        raise PreprocessingFixtureError("unknown preprocessing protocol")
    expected_counts = PROTOCOL_COUNTS[protocol]
    observed_counts = (
        record["construction_num_spatial_crops"], record["runtime_num_spatial_crops"]
    )
    if observed_counts != expected_counts:
        raise PreprocessingFixtureError("protocol disagrees with construction/runtime crop state")
    expected_plan = evaluation_view_plan(*observed_counts, record["num_ensemble_views"])

    shape = record["normalized_shape"]
    if shape != [3, 8, 224, 224]:
        raise PreprocessingFixtureError("normalized tensor shape must be C,T,H,W = 3,8,224,224")
    summary = record["normalization_summary"]
    if not isinstance(summary, Mapping) or set(summary) != {"min", "max", "mean", "std"}:
        raise PreprocessingFixtureError("normalization summary fields must be exact")
    for key, value in summary.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise PreprocessingFixtureError(f"normalization summary {key} must be finite")
    if summary["min"] > summary["max"] or summary["std"] < 0:
        raise PreprocessingFixtureError("invalid normalization summary bounds")

    views = record["views"]
    if not isinstance(views, list) or len(views) != len(expected_plan):
        raise PreprocessingFixtureError("view count disagrees with frozen construction state")
    for view, expected in zip(views, expected_plan):
        required_view = {
            "stored_view_index", "temporal_sample_index", "spatial_sample_index",
            "resized_height", "resized_width", "crop_x", "crop_y", "output_tensor_sha256",
        }
        if not isinstance(view, Mapping) or set(view) != required_view:
            raise PreprocessingFixtureError("view fields must be exact")
        if any(view[key] != value for key, value in expected.items()):
            raise PreprocessingFixtureError("view indices disagree with protocol state machine")
        _require_sha(view["output_tensor_sha256"], "view.output_tensor_sha256")
        for key in ("resized_height", "resized_width", "crop_x", "crop_y"):
            if not isinstance(view[key], int) or isinstance(view[key], bool) or view[key] < 0:
                raise PreprocessingFixtureError(f"view.{key} must be a non-negative integer")
        h, w, idx = view["resized_height"], view["resized_width"], view["spatial_sample_index"]
        if h < 224 or w < 224:
            raise PreprocessingFixtureError("resized dimensions must contain a 224 crop")
        expected_y = math.ceil((h - 224) / 2)
        expected_x = math.ceil((w - 224) / 2)
        if h > w:
            expected_y = (0, math.ceil((h - 224) / 2), h - 224)[idx]
        else:
            expected_x = (0, math.ceil((w - 224) / 2), w - 224)[idx]
        if (view["crop_x"], view["crop_y"]) != (expected_x, expected_y):
            raise PreprocessingFixtureError("crop coordinates disagree with pinned uniform_crop")

    if record["status"] == "pass":
        expected_literals = {
            "decoder_backend": "pyav", "decoded_layout": "T,H,W,C",
            "decoded_dtype": "uint8", "decoded_channel_order": "RGB",
            "normalized_layout": "C,T,H,W", "normalized_dtype": "float32",
        }
        for key, value in expected_literals.items():
            if record[key] != value:
                raise PreprocessingFixtureError(f"pass requires {key}={value}")


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreprocessingFixtureError("linked JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, key: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise PreprocessingFixtureError(f"{key} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PreprocessingFixtureError(f"{key} must be hexadecimal") from exc


def _require_string(value: Any, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PreprocessingFixtureError(f"{key} must be non-empty")
