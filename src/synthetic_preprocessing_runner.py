"""Torch runner for the project-owned synthetic EndoFM preprocessing clip.

Outputs are implementation-only until the same specification is run in the
legacy reference environment and compared under the frozen parity plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.synthetic_clip_fixture import generate_synthetic_clip
from src.tensor_fingerprint import fingerprint_tensor_payload, summarize_tensor_payload


def run_synthetic_preprocessing(
    spec_path: str | Path,
    environment_label: str,
    report_status: str = "implementation-only",
    asset_dir: str | Path | None = None,
) -> dict[str, Any]:
    if report_status not in {"implementation-only", "legacy-reference"}:
        raise ValueError("unsupported synthetic preprocessing report status")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    path = Path(spec_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    source_payload = generate_synthetic_clip(spec)
    source = np.frombuffer(source_payload, dtype=np.uint8).reshape(spec["shape"]).copy()
    frames = torch.from_numpy(source)
    normalized_thwc = frames.float() / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45])
    std = torch.tensor([0.225, 0.225, 0.225])
    normalized_thwc = (normalized_thwc - mean) / std
    normalized_cthw = normalized_thwc.permute(3, 0, 1, 2).contiguous()

    height, width = normalized_cthw.shape[2:]
    short_side = 224
    if width < height:
        resized_height = math.floor(height / width * short_side)
        resized_width = short_side
    else:
        resized_height = short_side
        resized_width = math.floor(width / height * short_side)
    resized = torch.nn.functional.interpolate(
        normalized_cthw,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).contiguous()

    output_dir = Path(asset_dir) if asset_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    views = []
    raw_assets = []
    for spatial_index in (0, 1, 2):
        y_offset = math.ceil((resized_height - 224) / 2)
        x_offset = math.ceil((resized_width - 224) / 2)
        if resized_height > resized_width:
            y_offset = (0, y_offset, resized_height - 224)[spatial_index]
        else:
            x_offset = (0, x_offset, resized_width - 224)[spatial_index]
        crop = resized[:, :, y_offset:y_offset + 224, x_offset:x_offset + 224].contiguous()
        crop_payload = _float32_payload(crop, np)
        if output_dir is not None:
            raw_assets.append(_write_raw_asset(
                output_dir, f"crop_{spatial_index}.f32le", crop_payload,
                list(crop.shape), "C,T,H,W",
            ))
        views.append({
            "spatial_sample_index": spatial_index,
            "crop_x": x_offset,
            "crop_y": y_offset,
            "shape": list(crop.shape),
            "tensor_fingerprint": fingerprint_tensor_payload(
                crop_payload, dtype="float32-le", shape=list(crop.shape), layout="C,T,H,W"
            ),
        })

    normalized_payload = _float32_payload(normalized_cthw, np)
    resized_payload = _float32_payload(resized, np)
    if output_dir is not None:
        raw_assets.insert(0, _write_raw_asset(
            output_dir, "resize.f32le", resized_payload,
            list(resized.shape), "C,T,H,W",
        ))
    report = {
        "schema_version": "endosae.synthetic-preprocessing-probe.v0",
        "status": report_status,
        "environment_label": environment_label,
        "fixture_spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_tensor_fingerprint": spec["expected_tensor_fingerprint"],
        "runtime": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": "cpu",
            "threads": 1,
        },
        "normalization": {
            "mean": [0.45, 0.45, 0.45],
            "std": [0.225, 0.225, 0.225],
            "shape": list(normalized_cthw.shape),
            "tensor_fingerprint": fingerprint_tensor_payload(
                normalized_payload, dtype="float32-le", shape=list(normalized_cthw.shape), layout="C,T,H,W"
            ),
            "summary": summarize_tensor_payload(
                normalized_payload, dtype="float32-le", shape=list(normalized_cthw.shape), layout="C,T,H,W"
            ),
        },
        "resize": {
            "mode": "bilinear",
            "align_corners": False,
            "short_side": 224,
            "shape": list(resized.shape),
            "tensor_fingerprint": fingerprint_tensor_payload(
                resized_payload, dtype="float32-le", shape=list(resized.shape), layout="C,T,H,W"
            ),
        },
        "views": views,
        "checkpoint_accessed": False,
        "g1_evidence": False,
    }
    if output_dir is not None:
        report["raw_assets"] = raw_assets
    return report


def _float32_payload(tensor: Any, np: Any) -> bytes:
    return tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")


def _write_raw_asset(
    output_dir: Path, filename: str, payload: bytes, shape: list[int], layout: str
) -> dict[str, Any]:
    path = output_dir / filename
    path.write_bytes(payload)
    return {
        "filename": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dtype": "float32-le",
        "shape": shape,
        "layout": layout,
        "size_bytes": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--environment-label", required=True)
    parser.add_argument(
        "--status",
        choices=("implementation-only", "legacy-reference"),
        default="implementation-only",
    )
    parser.add_argument("--asset-dir")
    args = parser.parse_args()
    print(json.dumps(
        run_synthetic_preprocessing(
            args.spec, args.environment_label, args.status, args.asset_dir
        ),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
