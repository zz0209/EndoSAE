"""Read-only target-Python capability probe for EndoFM environment candidates."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from typing import Any

from src.tensor_fingerprint import fingerprint_tensor_payload


PACKAGES = ("numpy", "torch", "torchvision", "av", "Pillow", "timm", "fvcore", "einops", "kornia")


def collect_runtime_capability(label: str) -> dict[str, Any]:
    packages = {}
    for package in PACKAGES:
        module = "PIL" if package == "Pillow" else package
        present = importlib.util.find_spec(module) is not None
        version = None
        if present:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "present-version-unresolved"
        packages[package] = {"present": present, "version": version}
    report: dict[str, Any] = {
        "schema_version": "endosae.runtime-capability-probe.v0",
        "status": "probe-only",
        "label": label,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": None,
        "synthetic_interpolate": None,
        "checkpoint_accessed": False,
    }
    if packages["torch"]["present"]:
        import torch

        cuda_available = torch.cuda.is_available()
        report["torch_runtime"] = {
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "device_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        }
        source = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        output = torch.nn.functional.interpolate(
            source, size=(4, 5), mode="bilinear", align_corners=False
        ).contiguous().cpu()
        payload = output.numpy().astype("<f4", copy=False).tobytes(order="C")
        report["synthetic_interpolate"] = {
            "input_shape": [1, 1, 2, 3],
            "output_shape": list(output.shape),
            "mode": "bilinear",
            "align_corners": False,
            "output_fingerprint": fingerprint_tensor_payload(
                payload, dtype="float32-le", shape=list(output.shape), layout="N,C,H,W"
            ),
            "first_values": [float(value) for value in output.flatten()[:5]],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    print(json.dumps(collect_runtime_capability(args.label), sort_keys=True))


if __name__ == "__main__":
    main()
