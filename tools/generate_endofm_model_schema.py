"""Generate an auditable EndoFM backbone schema without loading checkpoint tensors.

This is a narrow G1 probe. It instantiates the pinned TimeSformer definition, scans
checkpoint pickle opcodes for names only, and records whether the two namespaces
match. It deliberately does not deserialize or load checkpoint tensors.
"""

from __future__ import annotations

import hashlib
import json
import pickletools
import platform
import sys
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT / "third_party" / "Endo-FM"
MODERN_SITE = PROJECT / "artifacts" / "environments" / "modern"
CHECKPOINT = PROJECT / "artifacts" / "quarantine" / "endofm-main" / "endo_fm.pth"
OUTPUT = PROJECT / "configs" / "probes" / "endofm_modern_random_model_schema_20260831.json"

sys.path[:0] = [str(MODERN_SITE), str(SOURCE_ROOT)]

import einops  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from models.timesformer import VisionTransformer  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def name_digest(names: list[str]) -> str:
    payload = "\n".join(sorted(names)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    torch.manual_seed(0)
    model_kwargs = {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "num_classes": 0,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "num_frames": 8,
        "attention_type": "divided_space_time",
    }
    model = VisionTransformer(**model_kwargs)
    records = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "numel": tensor.numel(),
        }
        for name, tensor in sorted(model.state_dict().items())
    ]
    model_names = [record["name"] for record in records]
    model_name_set = set(model_names)

    with zipfile.ZipFile(CHECKPOINT) as archive:
        member = next(
            name for name in archive.namelist() if name.endswith("/data.pkl") or name == "data.pkl"
        )
        payload = archive.read(member)
    strings = {
        argument
        for opcode, argument, _ in pickletools.genops(payload)
        if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}
        and isinstance(argument, str)
    }
    suffixes = (".weight", ".bias", "cls_token", "pos_embed", "time_embed")

    def canonical(prefix: str) -> list[str]:
        return sorted(
            {
                value[len(prefix) :]
                for value in strings
                if value.startswith(prefix) and value.endswith(suffixes)
            }
        )

    def compare(candidate: list[str]) -> dict[str, object]:
        candidate_set = set(candidate)
        return {
            "candidate_count": len(candidate),
            "name_sha256": name_digest(candidate),
            "set_equals_model": candidate_set == model_name_set,
            "missing_from_checkpoint": sorted(model_name_set - candidate_set),
            "unexpected_in_checkpoint": sorted(candidate_set - model_name_set),
        }

    source_files = {}
    for relative in [
        "models/timesformer.py",
        "models/helpers.py",
        "models/vit_utils.py",
        "models/__init__.py",
    ]:
        path = SOURCE_ROOT / relative
        source_files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    wheels = []
    for path in sorted((PROJECT / "artifacts" / "dependency-wheels" / "modern").glob("*.whl")):
        if path.name.startswith(("einops-0.6.1-", "timm-0.4.12-")):
            wheels.append(
                {"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )

    report = {
        "schema_version": "endosae.modern-random-model-schema.v0",
        "generated_at": "2026-08-31",
        "status": "pass-schema-only-runtime-behavior-unverified",
        "purpose": (
            "Record the exact random-initialized EndoFM backbone state schema and compare it "
            "with non-deserialized checkpoint key names."
        ),
        "source": {
            "repository": "third_party/Endo-FM",
            "pinned_commit": "a0e4f523f1777b57986e77c16688727ca0ba22cd",
            "files": source_files,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "einops": einops.__version__,
            "timm": timm.__version__,
            "supplementary_wheels": wheels,
            "scope_note": (
                "Modern environment used only for schema construction; it is not accepted as "
                "the EndoFM behavior-parity environment."
            ),
        },
        "construction": {
            "seed": 0,
            "class": "models.timesformer.VisionTransformer",
            "kwargs": model_kwargs,
        },
        "state_dict": {
            "tensor_count": len(records),
            "total_numel": sum(record["numel"] for record in records),
            "parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
            "name_sha256": name_digest(model_names),
            "tensors": records,
        },
        "checkpoint_static_comparison": {
            "checkpoint_path": "artifacts/quarantine/endofm-main/endo_fm.pth",
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "method": "ZIP data.pkl opcode string scan only; checkpoint was not deserialized.",
            "direct_backbone": compare(canonical("backbone.")),
            "module_backbone": compare(canonical("module.backbone.")),
        },
        "limitations": {
            "checkpoint_deserialized": False,
            "checkpoint_tensor_shapes_verified": False,
            "checkpoint_weights_loaded": False,
            "forward_executed": False,
            "behavior_parity_verified": False,
            "g1_admission": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "tensor_count": len(records),
                "total_numel": report["state_dict"]["total_numel"],
                "direct": report["checkpoint_static_comparison"]["direct_backbone"],
                "module": report["checkpoint_static_comparison"]["module_backbone"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
