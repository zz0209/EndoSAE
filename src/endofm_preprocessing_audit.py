"""Static, non-executing audit of the pinned EndoFM evaluation input path.

This module intentionally does not import EndoFM, torch, PyAV, or fvcore.  It
turns a small set of source-level preprocessing assumptions into regressions;
runtime tensor equivalence remains a separate G1 requirement.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingSourceAudit:
    source_sha256: dict[str, str]
    num_frames: int
    sampling_rate: int
    test_crop_size: int
    target_fps: int
    decoding_backend: str
    mean: list[float]
    std: list[float]
    decoded_layout: str
    model_input_layout: str
    uint8_scaled_by_255: bool
    normalize_before_permute: bool
    deterministic_uniform_crop: bool
    validation_config_mutation_hazard: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_preprocessing_source(
    eval_path: str | Path,
    dataset_path: str | Path,
    data_utils_path: str | Path,
    decoder_path: str | Path,
    defaults_path: str | Path,
    config_path: str | Path,
) -> PreprocessingSourceAudit:
    paths = {
        "eval_finetune.py": Path(eval_path),
        "datasets/ucf101.py": Path(dataset_path),
        "datasets/data_utils.py": Path(data_utils_path),
        "datasets/decoder.py": Path(decoder_path),
        "utils/defaults.py": Path(defaults_path),
        "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml": Path(config_path),
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    eval_text = text["eval_finetune.py"]
    dataset_text = text["datasets/ucf101.py"]
    utils_text = text["datasets/data_utils.py"]
    decoder_text = text["datasets/decoder.py"]
    defaults_text = text["utils/defaults.py"]
    yaml_text = text["models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"]

    set_one = eval_text.find("config.TEST.NUM_SPATIAL_CROPS = 1")
    build_val = eval_text.find('dataset_val = UCF101(cfg=config, mode="val"')
    set_three = eval_text.find("config.TEST.NUM_SPATIAL_CROPS = 3")
    mutation_hazard = (
        -1 not in (set_one, build_val, set_three)
        and set_one < build_val < set_three
        and "self.cfg = cfg" in dataset_text
        and "cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS" in dataset_text
        and "self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS" in dataset_text
    )

    normalize_at = dataset_text.find("frames = tensor_normalize(")
    permute_at = dataset_text.find("frames = frames.permute(3, 0, 1, 2)")
    crop_at = dataset_text.find("frames = spatial_sampling(")
    deterministic_crop = (
        "assert spatial_idx in [-1, 0, 1, 2]" in utils_text
        and "frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)" in utils_text
        and 0 <= normalize_at < permute_at < crop_at
    )

    return PreprocessingSourceAudit(
        source_sha256={name: _sha256(path) for name, path in paths.items()},
        num_frames=_yaml_int(yaml_text, "NUM_FRAMES"),
        sampling_rate=_yaml_int(yaml_text, "SAMPLING_RATE"),
        test_crop_size=_yaml_int(yaml_text, "TEST_CROP_SIZE"),
        target_fps=_python_int(defaults_text, "_C.DATA.TARGET_FPS"),
        decoding_backend=_python_string(defaults_text, "_C.DATA.DECODING_BACKEND"),
        mean=_python_float_list(defaults_text, "_C.DATA.MEAN"),
        std=_python_float_list(defaults_text, "_C.DATA.STD"),
        decoded_layout="T,H,W,C",
        model_input_layout="C,T,H,W",
        uint8_scaled_by_255=(
            "if tensor.dtype == torch.uint8:" in utils_text
            and "tensor = tensor / 255.0" in utils_text
        ),
        normalize_before_permute=0 <= normalize_at < permute_at,
        deterministic_uniform_crop=deterministic_crop,
        validation_config_mutation_hazard=mutation_hazard,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_int(text: str, key: str) -> int:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing YAML integer {key}")
    return int(match.group(1))


def _python_int(text: str, key: str) -> int:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing Python integer {key}")
    return int(match.group(1))


def _python_string(text: str, key: str) -> str:
    match = re.search(rf'^\s*{re.escape(key)}\s*=\s*["\']([^"\']+)["\']\s*$', text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing Python string {key}")
    return match.group(1)


def _python_float_list(text: str, key: str) -> list[float]:
    match = re.search(rf"^{re.escape(key)}\s*=\s*\[([^\]]+)\]\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing Python list {key}")
    return [float(value.strip()) for value in match.group(1).split(",")]
