"""Hash-verified numeric comparison of canonical preprocessing tensor assets."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


class PreprocessingRawDiffError(ValueError):
    pass


def compare_raw_assets(manifest_path: str | Path, root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest_relative = manifest_file.relative_to(root_path)
    except ValueError as exc:
        raise PreprocessingRawDiffError("manifest path escapes project root") from exc
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "endosae.preprocessing-raw-assets.v0":
        raise PreprocessingRawDiffError("unsupported raw asset manifest")
    if manifest.get("dtype") != "float32-le" or manifest.get("layout") != "C,T,H,W":
        raise PreprocessingRawDiffError("raw asset dtype/layout drift")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or [item.get("stage") for item in stages] != [
        "resize", "crop_0", "crop_1", "crop_2"
    ]:
        raise PreprocessingRawDiffError("raw stages must be exact and ordered")

    thresholds = {"max_abs": 1e-6, "mean_abs": 1e-7, "fraction_over_1e_6": 0.0}
    results = []
    for stage in stages:
        shape = stage.get("shape")
        if not isinstance(shape, list) or any(not isinstance(x, int) or x <= 0 for x in shape):
            raise PreprocessingRawDiffError("invalid stage shape")
        expected_count = 1
        for dimension in shape:
            expected_count *= dimension
        pair = []
        for role in ("reference", "candidate"):
            asset = stage.get(role)
            if not isinstance(asset, Mapping):
                raise PreprocessingRawDiffError("asset entry missing")
            path = (root_path / str(asset.get("path", ""))).resolve()
            try:
                path.relative_to(root_path)
            except ValueError as exc:
                raise PreprocessingRawDiffError("asset path escapes project root") from exc
            payload = path.read_bytes()
            if len(payload) != expected_count * 4 or len(payload) != asset.get("size_bytes"):
                raise PreprocessingRawDiffError("asset size disagrees with shape")
            if hashlib.sha256(payload).hexdigest() != asset.get("sha256"):
                raise PreprocessingRawDiffError("asset hash mismatch")
            values = array.array("f")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            pair.append(values)
        differences = [abs(float(left) - float(right)) for left, right in zip(pair[0], pair[1])]
        maximum = max(differences)
        over_threshold = sum(value > 1e-6 for value in differences)
        metrics = {
            "max_abs": maximum,
            "mean_abs": math.fsum(differences) / len(differences),
            "fraction_over_1e_6": over_threshold / len(differences),
            "different_element_count": sum(value != 0.0 for value in differences),
            "element_count": len(differences),
        }
        passed = (
            metrics["max_abs"] <= thresholds["max_abs"]
            and metrics["mean_abs"] <= thresholds["mean_abs"]
            and metrics["fraction_over_1e_6"] == thresholds["fraction_over_1e_6"]
        )
        results.append({"stage": stage["stage"], "metrics": metrics, "passed": passed})
    all_passed = all(item["passed"] for item in results)
    return {
        "schema_version": "endosae.preprocessing-raw-diff.v0",
        "status": "pass" if all_passed else "fail",
        "manifest_path": manifest_relative.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "thresholds": thresholds,
        "stages": results,
        "parity_passed": all_passed,
        "formal_preprocessing_admission_allowed": all_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_raw_assets(args.manifest, args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
