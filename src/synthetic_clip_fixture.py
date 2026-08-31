"""Deterministic project-owned THWC uint8 clip generator for runtime parity."""

from __future__ import annotations

from typing import Any, Mapping

from src.tensor_fingerprint import fingerprint_tensor_payload


class SyntheticClipError(ValueError):
    """Raised when a synthetic clip specification is inconsistent."""


def generate_synthetic_clip(spec: Mapping[str, Any]) -> bytes:
    required = {
        "schema_version", "fixture_id", "layout", "dtype", "shape",
        "generator", "anchor_expectations", "expected_tensor_fingerprint",
    }
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise SyntheticClipError("synthetic clip spec fields must be exact")
    if spec["schema_version"] != "endosae.synthetic-clip.v0":
        raise SyntheticClipError("unsupported synthetic clip schema")
    if spec["layout"] != "T,H,W,C" or spec["dtype"] != "uint8":
        raise SyntheticClipError("v0 requires THWC uint8")
    shape = spec["shape"]
    if shape != [8, 120, 200, 3]:
        raise SyntheticClipError("v0 shape must be 8,120,200,3")
    generator = spec["generator"]
    if not isinstance(generator, Mapping) or set(generator) != {
        "formula", "offset", "t_coefficient", "y_coefficient", "x_coefficient",
        "c_coefficient", "modulus",
    }:
        raise SyntheticClipError("generator fields must be exact")
    if generator["formula"] != "(offset+t*T+y*Y+x*X+c*C) mod modulus":
        raise SyntheticClipError("unexpected generator formula")
    for key in ("offset", "t_coefficient", "y_coefficient", "x_coefficient", "c_coefficient"):
        if not isinstance(generator[key], int) or isinstance(generator[key], bool):
            raise SyntheticClipError(f"generator {key} must be integer")
    if generator["modulus"] != 256:
        raise SyntheticClipError("uint8 generator modulus must be 256")

    t_size, h_size, w_size, c_size = shape
    payload = bytearray(t_size * h_size * w_size * c_size)
    cursor = 0
    for t in range(t_size):
        for y in range(h_size):
            for x in range(w_size):
                for c in range(c_size):
                    payload[cursor] = _value(generator, t, y, x, c)
                    cursor += 1

    anchors = spec["anchor_expectations"]
    if not isinstance(anchors, list) or not anchors:
        raise SyntheticClipError("anchor expectations are required")
    for anchor in anchors:
        if not isinstance(anchor, Mapping) or set(anchor) != {"t", "y", "x", "c", "value"}:
            raise SyntheticClipError("anchor fields must be exact")
        coordinates = [anchor[key] for key in ("t", "y", "x", "c")]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in coordinates):
            raise SyntheticClipError("anchor coordinates must be integers")
        if any(value < 0 or value >= upper for value, upper in zip(coordinates, shape)):
            raise SyntheticClipError("anchor coordinate out of range")
        if anchor["value"] != _value(generator, *coordinates):
            raise SyntheticClipError("anchor value disagrees with generator")

    result = bytes(payload)
    observed = fingerprint_tensor_payload(result, dtype="uint8", shape=shape, layout="T,H,W,C")
    if observed != spec["expected_tensor_fingerprint"]:
        raise SyntheticClipError("generated tensor fingerprint disagrees with frozen spec")
    return result


def _value(generator: Mapping[str, Any], t: int, y: int, x: int, c: int) -> int:
    return (
        generator["offset"]
        + t * generator["t_coefficient"]
        + y * generator["y_coefficient"]
        + x * generator["x_coefficient"]
        + c * generator["c_coefficient"]
    ) % generator["modulus"]
