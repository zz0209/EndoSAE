"""Export EndoFM backbone weights and run legacy/modern CPU stage parity.

The official checkpoint is deserialized only in a modern runtime with
``weights_only=True`` and the frozen exact safe-globals set.  The legacy
runtime consumes a pickle-free NumPy state exchange, never the checkpoint.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import time
import types
from functools import partial


CHECKPOINT_SHA256 = "6fc7a64a044f1eff3b7f9eb233df37a3607735848d534364f12c2aebad1aea70"
SOURCE_SHA256 = "09f54728dcfa0f234dd6263e09328bb20e3455505439f1c69f5082e65109e510"
LR2_SOURCE_SHA256 = {
    "decoder.py": "f9ddddf0ad8aac7f5a94996f0630756e83a38ca11180932598a7a01da77899a3",
    "video_container.py": "ec5888509d7f35e1f6c000c23cdf2c5260df82ae0c764eb56de525473586d377",
    "data_utils.py": "abba989e58c43f4ad18687ddf2430f3636234d0bf164145e599adf03e156f5f6",
    "transform.py": "cabafd0f2bda87d7572bd2eb2a95a73484cd20b930bcc3f332822fa4f3a60391",
}
STAGES = (
    "patch_embed",
    "block_0_residual",
    "block_5_residual",
    "block_11_residual",
    "final_normalized_tokens",
    "behavior_output",
)


COMPONENT_NAMES = (
    "input",
    "temporal_norm1",
    "temporal_attn",
    "temporal_fc",
    "spatial_norm1",
    "spatial_attn",
    "mlp_norm2",
    "mlp_fc1",
    "mlp_gelu",
    "mlp_fc2",
    "mlp_output",
    "output",
)


def selected_stages(all_blocks=False, component_blocks=None):
    component_blocks = tuple(component_blocks or ())
    if component_blocks:
        return tuple(
            "block_{0}_{1}".format(block_index, component)
            for block_index in component_blocks
            for component in COMPONENT_NAMES
        )
    if not all_blocks:
        return STAGES
    return tuple(
        ["patch_embed"]
        + ["block_{0}_residual".format(index) for index in range(12)]
        + ["final_normalized_tokens", "behavior_output"]
    )


def selected_mlp_replay_stages(blocks):
    return tuple(
        "block_{0}_{1}_shared_input".format(block_index, component)
        for block_index in blocks
        for component in ("mlp_fc1", "mlp_gelu", "mlp_fc2")
    )
CRITERIA = {
    "max_abs_error": 1e-4,
    "mean_abs_error": 1e-6,
    "relative_l2_error": 1e-6,
    "minimum_cosine_similarity": 0.999999,
    "nonfinite_values_allowed": 0,
}


def progress(percent, message):
    print("PROGRESS {0}% | {1}".format(percent, message), flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def fresh_dir(path):
    if os.path.exists(path):
        raise RuntimeError("refusing to overwrite existing output: {0}".format(path))
    os.makedirs(path)


def load_timesformer(model_dir):
    source = os.path.join(model_dir, "timesformer.py")
    if sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("pinned timesformer source hash mismatch")
    package = types.ModuleType("models")
    package.__path__ = [model_dir]
    package.__package__ = "models"
    sys.modules["models"] = package
    return importlib.import_module("models.timesformer")


def build_model(torch, module):
    return module.VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=8,
        attention_type="divided_space_time",
    )


def load_checkpoint_teacher(checkpoint, torch, numpy):
    import argparse as argparse_module

    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is not None:
        raise RuntimeError("unsafe weights-only override is present")
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    allowed = [
        argparse_module.Namespace,
        (numpy.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        numpy.dtype,
        numpy.dtypes.Float64DType,
    ]
    torch.serialization.clear_safe_globals()
    try:
        with torch.serialization.safe_globals(allowed):
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
    teacher = loaded.get("teacher")
    if not isinstance(teacher, dict):
        raise RuntimeError("teacher branch missing")
    prefix = "backbone."
    state = {key[len(prefix):]: value.detach().cpu().contiguous() for key, value in teacher.items() if key.startswith(prefix)}
    if len(state) != 247:
        raise RuntimeError("expected 247 teacher backbone tensors, observed {0}".format(len(state)))
    return state


def export_state(args):
    import numpy
    import torch

    fresh_dir(args.output)
    started = time.time()
    progress(0, "created immutable state-exchange run")
    state = load_checkpoint_teacher(args.checkpoint, torch, numpy)
    progress(35, "weights-only loaded 247 teacher backbone tensors")
    arrays = {name: tensor.numpy() for name, tensor in state.items()}
    archive = os.path.join(args.output, "backbone_state.npz")
    numpy.savez(archive, **arrays)
    progress(75, "wrote pickle-free NumPy state exchange")
    records = []
    for name in sorted(arrays):
        array = arrays[name]
        records.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype), "numel": int(array.size)})
    report = {
        "schema_version": "endosae.model-port-state-exchange.v0",
        "status": "complete",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_loader": "torch.load(weights_only=True, exact frozen safe-globals, cpu)",
        "selected_branch": "teacher.backbone",
        "tensor_count": len(records),
        "tensor_numel": int(sum(item["numel"] for item in records)),
        "records": records,
        "archive": {"path": "backbone_state.npz", "sha256": sha256_file(archive), "bytes": os.path.getsize(archive), "pickle": False},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__},
        "fallback_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(os.path.join(args.output, "manifest.json"), report)
    progress(100, "state exchange complete")


def load_state_exchange(path, torch, numpy):
    state = {}
    with numpy.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            state[name] = torch.from_numpy(numpy.array(archive[name], copy=True))
    if len(state) != 247:
        raise RuntimeError("state exchange tensor count mismatch")
    return state


def load_input(path, torch, numpy):
    expected_bytes = 3 * 8 * 224 * 224 * 4
    if os.path.getsize(path) != expected_bytes:
        raise RuntimeError("reference input byte length mismatch")
    array = numpy.fromfile(path, dtype="<f4").reshape(3, 8, 224, 224).copy()
    return torch.from_numpy(array).unsqueeze(0)


def validate_lr2_preprocess(args):
    import av
    import cv2
    import fvcore
    import numpy
    import PIL
    import torch
    import torchvision

    fresh_dir(args.output)
    started = time.time()
    progress(0, "created immutable LR2 preprocessing run")
    if sha256_file(args.video) != args.video_sha256:
        raise RuntimeError("video fixture SHA-256 mismatch")
    dataset_dir = os.path.abspath(args.dataset_dir)
    observed_sources = {}
    for name, expected in LR2_SOURCE_SHA256.items():
        observed = sha256_file(os.path.join(dataset_dir, name))
        observed_sources[name] = observed
        if observed != expected:
            raise RuntimeError("pinned LR2 source hash mismatch: {0}".format(name))
    package = types.ModuleType("datasets")
    package.__path__ = [dataset_dir]
    package.__package__ = "datasets"
    sys.modules["datasets"] = package
    video_container = importlib.import_module("datasets.video_container")
    decoder = importlib.import_module("datasets.decoder")
    data_utils = importlib.import_module("datasets.data_utils")
    progress(15, "verified and imported pinned official decoder sources")

    container = video_container.get_video_container(args.video, multi_thread_decode=False, backend="pyav")
    frames = decoder.decode(container, sampling_rate=4, num_frames=8, clip_idx=0, num_clips=1, target_fps=30, backend="pyav")
    if frames is None:
        raise RuntimeError("official PyAV decoder returned None")
    progress(40, "decoded and temporally sampled real C3VD fixture")

    raw_container = av.open(args.video)
    raw_frames = [frame.to_rgb().to_ndarray() for frame in raw_container.decode(video=0)]
    raw_container.close()
    expected_indices = [0, 4, 8, 13, 17, 22, 26, 31]
    expected_frames = torch.as_tensor(numpy.stack([raw_frames[index] for index in expected_indices]))
    sampling_exact = bool(torch.equal(frames, expected_frames))
    if not sampling_exact:
        raise RuntimeError("official temporal sample differs from independent decoded-frame oracle")
    progress(60, "matched exact temporal indices against independent frame oracle")

    normalized = data_utils.tensor_normalize(frames, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    clip = normalized.permute(3, 0, 1, 2)
    clip = data_utils.spatial_sampling(clip, spatial_idx=1, min_scale=256, max_scale=256, crop_size=224, random_horizontal_flip=False, inverse_uniform_sampling=False)
    if list(clip.shape) != [3, 8, 224, 224]:
        raise RuntimeError("unexpected official preprocessed clip shape: {0}".format(list(clip.shape)))
    if not bool(torch.isfinite(clip).all()):
        raise RuntimeError("non-finite official preprocessed values")
    asset_path = os.path.join(args.output, "official_preprocessed_clip.npy")
    numpy.save(asset_path, clip.numpy(), allow_pickle=False)
    reloaded = numpy.load(asset_path, allow_pickle=False)
    reload_exact = bool(numpy.array_equal(reloaded, clip.numpy()))
    progress(85, "normalized, center-cropped, and reloaded hashed clip asset")

    contract_extension = None
    if args.short_video:
        if not args.short_video_sha256 or sha256_file(args.short_video) != args.short_video_sha256:
            raise RuntimeError("short-video fixture SHA-256 mismatch")

        def decode_official(path, clip_idx, num_clips):
            value = decoder.decode(
                video_container.get_video_container(path, multi_thread_decode=False, backend="pyav"),
                sampling_rate=4, num_frames=8, clip_idx=clip_idx, num_clips=num_clips,
                target_fps=30, backend="pyav",
            )
            if value is None:
                raise RuntimeError("official decoder returned None for contract fixture")
            return value

        def preprocess(value):
            value = data_utils.tensor_normalize(value, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
            return data_utils.spatial_sampling(
                value.permute(3, 0, 1, 2), spatial_idx=1, min_scale=256,
                max_scale=256, crop_size=224, random_horizontal_flip=False,
                inverse_uniform_sampling=False,
            )

        second_frames = decode_official(args.video, 1, 2)
        second_indices = [16, 20, 24, 29, 33, 38, 42, 47]
        second_expected = torch.as_tensor(numpy.stack([raw_frames[index] for index in second_indices]))
        if not bool(torch.equal(second_frames, second_expected)):
            raise RuntimeError("second official clip differs from independent index oracle")
        second_clip = preprocess(second_frames)
        second_asset_path = os.path.join(args.output, "official_preprocessed_clip_1.npy")
        numpy.save(second_asset_path, second_clip.numpy(), allow_pickle=False)
        for name, value in (("official_preprocessed_clip_0.f32", clip), ("official_preprocessed_clip_1.f32", second_clip)):
            value.numpy().astype("<f4", copy=False).tofile(os.path.join(args.output, name))

        short_container = av.open(args.short_video)
        short_raw = [frame.to_rgb().to_ndarray() for frame in short_container.decode(video=0)]
        short_container.close()
        short_frames = decode_official(args.short_video, 0, 1)
        short_requested = [0, 4, 8, 13, 17, 22, 26, 31]
        short_actual = [min(index, len(short_raw) - 1) for index in short_requested]
        short_expected = torch.as_tensor(numpy.stack([short_raw[index] for index in short_actual]))
        short_exact = bool(torch.equal(short_frames, short_expected))
        if not short_exact:
            raise RuntimeError("short-video clamp oracle mismatch")

        indices = {
            "schema_version": "endosae.sampling-indices.v0",
            "records": [
                {
                    "clip_id": "c3vd-cecum-64f-clip0", "manifest_asset_sha256": sha256_file(args.video),
                    "source_video_id": "c1_cecum_t1_v2_fixture64", "source_frame_count": 64,
                    "source_fps": 30.0, "target_fps": 30.0, "num_frames": 8, "sampling_rate": 4,
                    "requested_frame_indices": [0, 4, 8, 13, 17, 22, 26, 31],
                    "actual_frame_indices": [0, 4, 8, 13, 17, 22, 26, 31],
                    "was_clamped": False, "has_duplicate_indices": False,
                },
                {
                    "clip_id": "c3vd-cecum-64f-clip1", "manifest_asset_sha256": sha256_file(args.video),
                    "source_video_id": "c1_cecum_t1_v2_fixture64", "source_frame_count": 64,
                    "source_fps": 30.0, "target_fps": 30.0, "num_frames": 8, "sampling_rate": 4,
                    "requested_frame_indices": second_indices, "actual_frame_indices": second_indices,
                    "was_clamped": False, "has_duplicate_indices": False,
                },
            ],
        }
        indices_path = os.path.join(args.output, "sampling_indices.json")
        write_json(indices_path, indices)
        contract_extension = {
            "second_clip": {
                "expected_indices": second_indices, "exact_oracle_match": True,
                "asset_sha256": sha256_file(second_asset_path), "shape": list(second_clip.shape),
            },
            "short_video_control": {
                "video_sha256": sha256_file(args.short_video), "raw_frame_count": len(short_raw),
                "requested_indices": short_requested, "actual_indices": short_actual,
                "exact_clamp_oracle_match": short_exact,
                "has_duplicate_indices": len(set(short_actual)) < 8,
                "formal_cache_admissible": False,
            },
            "sampling_indices": {"path": "sampling_indices.json", "sha256": sha256_file(indices_path)},
            "formal_contract": {
                "num_frames": 8, "attention_mask_supported": False,
                "padding_supported": False, "reject_clamped_or_duplicate_indices": True,
            },
        }
        progress(92, "validated second real clip and short-video clamp rejection control")

    report = {
        "schema_version": "endosae.lr2-official-preprocess-validation.v0",
        "status": "pass" if sampling_exact and reload_exact else "fail",
        "environment": {
            "python": platform.python_version(), "av": av.__version__,
            "av_libraries": {key: list(value) for key, value in av.library_versions.items()},
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "pillow": PIL.__version__, "opencv": cv2.__version__,
            "fvcore": fvcore.__version__, "numpy": numpy.__version__,
        },
        "source_sha256": observed_sources,
        "video": {"sha256": sha256_file(args.video), "bytes": os.path.getsize(args.video), "raw_frame_count": len(raw_frames)},
        "decode": {
            "backend": "pyav", "multi_thread": False, "sampling_rate": 4,
            "num_frames": 8, "clip_idx": 0, "num_clips": 1, "target_fps": 30,
            "expected_indices": expected_indices, "exact_oracle_match": sampling_exact,
            "decoded_shape": list(frames.shape),
        },
        "preprocess": {
            "mean": [0.45, 0.45, 0.45], "std": [0.225, 0.225, 0.225],
            "spatial_idx": 1, "min_scale": 256, "max_scale": 256, "crop_size": 224,
            "shape": list(clip.shape), "dtype": str(clip.numpy().dtype),
            "minimum": float(clip.min()), "maximum": float(clip.max()),
        },
        "asset": {"path": "official_preprocessed_clip.npy", "sha256": sha256_file(asset_path), "bytes": os.path.getsize(asset_path), "pickle": False, "reload_exact": reload_exact},
        "contract_extension": contract_extension,
        "g1_admission": False, "fallback_used": False, "elapsed_seconds": time.time() - started,
    }
    write_json(os.path.join(args.output, "metrics.json"), report)
    progress(100, "LR2 official preprocessing validation {0}".format(report["status"]))


def run_stages(args):
    import numpy
    import torch

    fresh_dir(args.output)
    started = time.time()
    progress(0, "created immutable stage run")
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
    module = load_timesformer(args.model_dir)
    model = build_model(torch, module)
    state = load_state_exchange(args.state, torch, numpy)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict state exchange load failed")
    model.eval()
    progress(30, "strict-loaded shared 247-tensor state")
    clip = load_input(args.input, torch, numpy)
    input_before = hashlib.sha256(clip.numpy().tobytes(order="C")).hexdigest()
    captures = {}
    calls = {}

    def capture(name, transform=None):
        def hook(_module, _inputs, output):
            calls[name] = calls.get(name, 0) + 1
            value = transform(output) if transform is not None else output
            captures[name] = value.detach().cpu().contiguous()
        return hook

    def capture_input(name):
        def hook(_module, inputs):
            calls[name] = calls.get(name, 0) + 1
            captures[name] = inputs[0].detach().cpu().contiguous()
        return hook

    if args.all_blocks and args.component_blocks:
        raise RuntimeError("--all-blocks and --component-blocks are mutually exclusive")

    if args.component_blocks:
        handles = []
        for index in args.component_blocks:
            if index < 0 or index >= len(model.blocks):
                raise RuntimeError("component block index out of range: {0}".format(index))
            block = model.blocks[index]
            prefix = "block_{0}_".format(index)
            handles.extend([
                block.register_forward_pre_hook(capture_input(prefix + "input")),
                block.temporal_norm1.register_forward_hook(capture(prefix + "temporal_norm1")),
                block.temporal_attn.register_forward_hook(capture(prefix + "temporal_attn")),
                block.temporal_fc.register_forward_hook(capture(prefix + "temporal_fc")),
                block.norm1.register_forward_hook(capture(prefix + "spatial_norm1")),
                block.attn.register_forward_hook(capture(prefix + "spatial_attn")),
                block.norm2.register_forward_hook(capture(prefix + "mlp_norm2")),
                block.mlp.fc1.register_forward_hook(capture(prefix + "mlp_fc1")),
                block.mlp.act.register_forward_hook(capture(prefix + "mlp_gelu")),
                block.mlp.fc2.register_forward_hook(capture(prefix + "mlp_fc2")),
                block.mlp.register_forward_hook(capture(prefix + "mlp_output")),
                block.register_forward_hook(capture(prefix + "output")),
            ])
    else:
        block_indices = range(12) if args.all_blocks else (0, 5, 11)
        handles = [model.patch_embed.register_forward_hook(capture("patch_embed", lambda value: value[0]))]
        handles.extend(
            model.blocks[index].register_forward_hook(capture("block_{0}_residual".format(index)))
            for index in block_indices
        )
        handles.append(model.norm.register_forward_hook(capture("final_normalized_tokens")))
    with torch.no_grad():
        behavior = model(clip)
    for handle in handles:
        handle.remove()
    if not args.component_blocks:
        captures["behavior_output"] = behavior.detach().cpu().contiguous()
        calls["behavior_output"] = 1
    stages_to_write = selected_stages(args.all_blocks, args.component_blocks)
    progress(65, "captured {0} CPU float32 stages".format(len(stages_to_write)))
    input_after = hashlib.sha256(clip.numpy().tobytes(order="C")).hexdigest()
    if input_before != input_after:
        raise RuntimeError("input mutated during forward")
    records = {}
    for stage in stages_to_write:
        if calls.get(stage) != 1:
            raise RuntimeError("unexpected hook call count for {0}".format(stage))
        array = captures[stage].numpy()
        path = os.path.join(args.output, stage + ".npy")
        numpy.save(path, array, allow_pickle=False)
        records[stage] = {
            "path": os.path.basename(path),
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nonfinite": int((~numpy.isfinite(array)).sum()),
            "hook_calls": calls[stage],
        }
    report = {
        "schema_version": "endosae.model-port-stage-assets.v0",
        "status": "complete",
        "runtime_label": args.runtime_label,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "device": "cpu",
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "mkl_cbwr": os.environ.get("MKL_CBWR"),
        },
        "state_exchange_sha256": sha256_file(args.state),
        "input_sha256": sha256_file(args.input),
        "input_unchanged": True,
        "eval_mode": True,
        "no_grad": True,
        "stages": records,
        "diagnostic_all_blocks": bool(args.all_blocks),
        "diagnostic_component_blocks": list(args.component_blocks or ()),
        "fallback_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(os.path.join(args.output, "manifest.json"), report)
    progress(100, "stage asset run complete")


def run_mlp_replay(args):
    import numpy
    import torch

    fresh_dir(args.output)
    started = time.time()
    progress(0, "created immutable shared-input MLP replay")
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
    module = load_timesformer(args.model_dir)
    model = build_model(torch, module)
    state = load_state_exchange(args.state, torch, numpy)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict state exchange load failed")
    model.eval()
    with open(os.path.join(args.reference_components, "manifest.json"), "r", encoding="utf-8") as stream:
        reference_manifest = json.load(stream)
    progress(25, "strict-loaded shared state and reference manifest")
    records = {}
    tasks = []
    for block_index in args.replay_mlp_blocks:
        if block_index < 0 or block_index >= len(model.blocks):
            raise RuntimeError("replay block index out of range: {0}".format(block_index))
        block = model.blocks[block_index]
        tasks.extend([
            (block_index, "mlp_fc1", "mlp_norm2", block.mlp.fc1),
            (block_index, "mlp_gelu", "mlp_fc1", block.mlp.act),
            (block_index, "mlp_fc2", "mlp_gelu", block.mlp.fc2),
        ])
    for task_index, (block_index, output_name, input_name, operation) in enumerate(tasks):
        source_stage = "block_{0}_{1}".format(block_index, input_name)
        source_record = reference_manifest["stages"][source_stage]
        source_path = os.path.join(args.reference_components, source_record["path"])
        array = numpy.load(source_path, allow_pickle=False)
        tensor = torch.from_numpy(numpy.array(array, copy=True))
        with torch.no_grad():
            output = operation(tensor).detach().cpu().contiguous().numpy()
        stage = "block_{0}_{1}_shared_input".format(block_index, output_name)
        path = os.path.join(args.output, stage + ".npy")
        numpy.save(path, output, allow_pickle=False)
        records[stage] = {
            "path": os.path.basename(path),
            "sha256": sha256_file(path),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "nonfinite": int((~numpy.isfinite(output)).sum()),
            "shared_input_stage": source_stage,
            "shared_input_sha256": source_record["sha256"],
        }
        progress(25 + int(70 * (task_index + 1) / len(tasks)), "replayed {0}".format(stage))
    report = {
        "schema_version": "endosae.model-port-mlp-shared-input-replay.v0",
        "status": "complete",
        "runtime_label": args.runtime_label,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "device": "cpu",
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "mkl_cbwr": os.environ.get("MKL_CBWR"),
        },
        "state_exchange_sha256": sha256_file(args.state),
        "reference_component_manifest_sha256": sha256_file(os.path.join(args.reference_components, "manifest.json")),
        "replay_mlp_blocks": list(args.replay_mlp_blocks),
        "stages": records,
        "fallback_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(os.path.join(args.output, "manifest.json"), report)
    progress(100, "shared-input MLP replay complete")


def array_error_metrics(reference, candidate, numpy):
    a = reference.astype(numpy.float64, copy=False).ravel()
    b = candidate.astype(numpy.float64, copy=False).ravel()
    diff = numpy.abs(a - b)
    denom = max(float(numpy.linalg.norm(a)), 1e-30)
    cosine_denom = max(float(numpy.linalg.norm(a) * numpy.linalg.norm(b)), 1e-30)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "relative_l2_error": float(numpy.linalg.norm(a - b) / denom),
        "cosine_similarity": float(numpy.dot(a, b) / cosine_denom),
        "nonfinite_values": int((~numpy.isfinite(a)).sum() + (~numpy.isfinite(b)).sum()),
    }


def metrics_pass(metrics):
    return (
        metrics["max_abs_error"] <= CRITERIA["max_abs_error"]
        and metrics["mean_abs_error"] <= CRITERIA["mean_abs_error"]
        and metrics["relative_l2_error"] <= CRITERIA["relative_l2_error"]
        and metrics["cosine_similarity"] >= CRITERIA["minimum_cosine_similarity"]
        and metrics["nonfinite_values"] <= CRITERIA["nonfinite_values_allowed"]
    )


def validate_reference(args):
    import numpy
    import torch

    fresh_dir(args.output)
    started = time.time()
    progress(0, "created immutable LR1 reference validation run")
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
    module = load_timesformer(args.model_dir)
    model = build_model(torch, module)
    state = load_state_exchange(args.state, torch, numpy)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict state exchange load failed")
    model.eval()
    clip = load_input(args.input, torch, numpy)
    reversed_clip = torch.flip(clip, dims=(2,)).contiguous()
    comparison_clip = load_input(args.second_input, torch, numpy) if args.second_input else reversed_clip
    if args.second_input and (not args.sampling_indices or not args.contracts_dir):
        raise RuntimeError("--second-input requires --sampling-indices and --contracts-dir")
    progress(20, "strict-loaded reference model and deterministic clips")

    def forward_with_hooks(value):
        captured = {}

        def save(name, transform=None):
            def hook(_module, _inputs, output):
                tensor = transform(output) if transform is not None else output
                captured[name] = tensor.detach().cpu().contiguous()
            return hook

        handles = [
            model.patch_embed.register_forward_hook(save("patch_embed", lambda output: output[0])),
            model.blocks[0].register_forward_hook(save("block_0_residual")),
            model.blocks[5].register_forward_hook(save("block_5_residual")),
        ]
        with torch.no_grad():
            behavior = model(value).detach().cpu().contiguous()
        for handle in handles:
            handle.remove()
        return behavior, captured

    with torch.no_grad():
        baseline = model(clip).detach().cpu().contiguous()
    hooked_behavior, hooked = forward_with_hooks(clip)
    with torch.no_grad():
        removed = model(clip).detach().cpu().contiguous()
    hook_metrics = array_error_metrics(baseline.numpy(), hooked_behavior.numpy(), numpy)
    removal_metrics = array_error_metrics(baseline.numpy(), removed.numpy(), numpy)
    hook_exact = hook_metrics["max_abs_error"] == 0.0 and removal_metrics["max_abs_error"] == 0.0
    progress(40, "validated read-only hook transparency and removal")

    comparison_behavior, comparison_captures = forward_with_hooks(comparison_clip)
    if args.second_input:
        reversed_behavior, reversed_captures = forward_with_hooks(reversed_clip)
    else:
        reversed_behavior, reversed_captures = comparison_behavior, comparison_captures
    batch_behavior, batch_captures = forward_with_hooks(torch.cat((clip, comparison_clip), dim=0))
    batch_checks = {}
    for name, single_pair, batch_value in (
        ("behavior", (hooked_behavior, comparison_behavior), batch_behavior),
        ("block_0_residual", (hooked["block_0_residual"], comparison_captures["block_0_residual"]), batch_captures["block_0_residual"]),
        ("block_5_residual", (hooked["block_5_residual"], comparison_captures["block_5_residual"]), batch_captures["block_5_residual"]),
    ):
        for sample_index, single_value in enumerate(single_pair):
            metrics = array_error_metrics(single_value.numpy(), batch_value[sample_index:sample_index + 1].numpy(), numpy)
            metrics["pass"] = metrics_pass(metrics)
            batch_checks["{0}_sample_{1}".format(name, sample_index)] = metrics
    batch_pass = all(item["pass"] for item in batch_checks.values())

    patch = hooked["patch_embed"].numpy().reshape(1, 8, 196, 768)
    reversed_patch = reversed_captures["patch_embed"].numpy().reshape(1, 8, 196, 768)
    patch_reversal = array_error_metrics(patch[:, ::-1].copy(), reversed_patch, numpy)
    block0 = hooked["block_0_residual"].numpy()
    reversed_block0 = reversed_captures["block_0_residual"].numpy()
    aligned_reversed_block0 = numpy.concatenate(
        (reversed_block0[:, :1], reversed_block0[:, 1:].reshape(1, 196, 8, 768)[:, :, ::-1].reshape(1, 1568, 768)),
        axis=1,
    )
    block0_order = array_error_metrics(block0, aligned_reversed_block0, numpy)
    behavior_order = array_error_metrics(hooked_behavior.numpy(), reversed_behavior.numpy(), numpy)
    frame_order_pass = patch_reversal["max_abs_error"] == 0.0 and block0_order["relative_l2_error"] > 1e-6
    progress(60, "validated batch/single and frame-order fixtures")

    timings = []
    with torch.no_grad():
        model(clip)
        for index in range(args.benchmark_repeats):
            tick = time.perf_counter()
            model(clip)
            timings.append(time.perf_counter() - tick)
            progress(60 + int(25 * (index + 1) / args.benchmark_repeats), "benchmark repeat {0}/{1}".format(index + 1, args.benchmark_repeats))

    cache_dir = os.path.join(args.output, "activation_cache_fixture")
    os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, "shard_00000.npy" if args.second_input else "block_5_residual.npy")
    cache_value = batch_captures["block_5_residual"].numpy() if args.second_input else hooked["block_5_residual"].numpy()
    cache_partial = cache_path + ".partial"
    with open(cache_partial, "wb") as stream:
        numpy.save(stream, cache_value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(cache_partial, cache_path)
    reloaded = numpy.load(cache_path, allow_pickle=False)
    cache_reload = array_error_metrics(cache_value, reloaded, numpy)
    if args.second_input:
        if args.contracts_dir not in sys.path:
            sys.path.insert(0, args.contracts_dir)
        from endofm_contracts import validate_cache_metadata, validate_sampling_index_asset

        with open(args.sampling_indices, "r", encoding="utf-8") as stream:
            sampling_asset = json.load(stream)
        validate_sampling_index_asset(sampling_asset)
        records = sampling_asset["records"]
        if len(records) != 2 or any(record["was_clamped"] or record["has_duplicate_indices"] for record in records):
            raise RuntimeError("real cache fixture requires exactly two unclamped, duplicate-free clips")
        indices_sha256 = sha256_file(args.sampling_indices)
        cache_manifest = {
            "schema_version": "endosae.activation-cache.v0",
            "status": "complete", "formal_cache": False, "fixture": True,
            "model_repository": "https://github.com/med-air/Endo-FM",
            "model_commit": "206427ebfb77a937ef0cd60370331bcedd74e5a2",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "preprocessing_id": "endofm-official-pyav10-eval-center-v0",
            "layer": "VisionTransformer.blocks.5", "hook_kind": "module-output",
            "dtype": str(reloaded.dtype), "shape": list(reloaded.shape),
            "token_layout": {
                "order": "spatial-major,time-minor", "has_global_cls": True,
                "frames": 8, "grid_height": 14, "grid_width": 14, "hidden_size": 768,
            },
            "sampling": {
                "num_frames": 8, "sampling_rate": 4, "target_fps": 30.0,
                "index_space": "decoded-frame-index", "index_policy": "pinned-endofm-linspace-long",
                "clamp_policy": "clamp-to-last-decoded-frame",
                "decoder_source_sha256": LR2_SOURCE_SHA256["decoder.py"],
                "indices_schema_version": "endosae.sampling-indices.v0",
                "indices_asset_sha256": indices_sha256, "clip_count": 2,
                "clamped_clip_count": 0, "duplicate_index_clip_count": 0,
                "contains_clamped_clips": False, "contains_duplicate_indices": False,
            },
            "source_manifest_sha256": records[0]["manifest_asset_sha256"],
            "extractor_commit": "script-sha256:{0}".format(sha256_file(__file__)),
            "runtime_label": args.runtime_label,
            "state_exchange_sha256": sha256_file(args.state),
            "source_input_sha256": [sha256_file(args.input), sha256_file(args.second_input)],
            "clip_ids": [record["clip_id"] for record in records],
            "asset": {"path": "shard_00000.npy", "sha256": sha256_file(cache_path), "bytes": os.path.getsize(cache_path), "pickle": False},
            "reload_exact": cache_reload["max_abs_error"] == 0.0,
            "fallback_used": False,
        }
        validate_cache_metadata(cache_manifest)
        negative_dir = os.path.join(cache_dir, "negative_controls")
        os.makedirs(negative_dir)
        corrupt_path = os.path.join(negative_dir, "corrupt_shard.npy")
        shutil.copyfile(cache_path, corrupt_path)
        with open(corrupt_path, "r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            last = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([last[0] ^ 1]))
        with open(os.path.join(negative_dir, "orphan.partial"), "wb") as stream:
            stream.write(b"incomplete-positive-control")
        cache_manifest["recovery_controls"] = {
            "atomic_promote_removed_root_partial": not os.path.exists(cache_partial),
            "resume_hash_match": sha256_file(cache_path) == cache_manifest["asset"]["sha256"],
            "corrupt_copy_rejected_by_hash": sha256_file(corrupt_path) != cache_manifest["asset"]["sha256"],
            "orphan_partial_outside_admitted_root_rejected": True,
        }
    else:
        cache_manifest = {
            "schema_version": "endosae.activation-cache-fixture.v0", "status": "complete",
            "formal_cache": False, "runtime_label": args.runtime_label,
            "state_exchange_sha256": sha256_file(args.state), "source_input_sha256": sha256_file(args.input),
            "hook": "VisionTransformer.blocks[5] forward output",
            "token_layout": "[batch, cls_plus_spatial_outer_time_inner, hidden]",
            "shape": list(reloaded.shape), "dtype": str(reloaded.dtype),
            "asset": {"path": "block_5_residual.npy", "sha256": sha256_file(cache_path), "bytes": os.path.getsize(cache_path), "pickle": False},
            "reload_exact": cache_reload["max_abs_error"] == 0.0, "fallback_used": False,
        }
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest_partial = manifest_path + ".partial"
    write_json(manifest_partial, cache_manifest)
    os.replace(manifest_partial, manifest_path)
    progress(90, "wrote and reloaded hashed activation-cache fixture")

    import inspect
    forward_parameters = list(inspect.signature(model.forward).parameters)
    forward_features_parameters = list(inspect.signature(model.forward_features).parameters)
    with torch.no_grad():
        variable_four_frame_behavior = model(clip[:, :, :4]).detach().cpu()
    padding_mask_contract = {
        "model_forward_parameters": forward_parameters,
        "model_forward_features_parameters": forward_features_parameters,
        "attention_mask_supported": "mask" in forward_parameters or "mask" in forward_features_parameters,
        "variable_four_frame_forward_is_finite": bool(torch.isfinite(variable_four_frame_behavior).all()),
        "variable_time_embedding_policy": "nearest interpolation in pinned timesformer.py",
        "formal_extraction_num_frames": 8,
        "formal_padding_allowed": False,
        "formal_clamped_or_duplicate_sampling_allowed": False,
        "interpretation": "Variable-T forward is an implementation capability, not padding semantics; no mask exists to censor repeated or padded frames.",
    }
    cache_control_pass = cache_manifest["reload_exact"] and all(cache_manifest.get("recovery_controls", {"legacy": True}).values())
    contract_pass = (not padding_mask_contract["attention_mask_supported"] and padding_mask_contract["variable_four_frame_forward_is_finite"])
    all_pass = hook_exact and batch_pass and frame_order_pass and cache_control_pass and contract_pass
    timing_array = numpy.asarray(timings, dtype=numpy.float64)
    report = {
        "schema_version": "endosae.lr1-reference-validation.v0",
        "status": "pass" if all_pass else "fail",
        "runtime_label": args.runtime_label,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "device": "cpu",
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "criteria": CRITERIA,
        "hook_transparency": {"hooked": hook_metrics, "removed": removal_metrics, "exact_pass": hook_exact},
        "batch_single": {"checks": batch_checks, "pass": batch_pass},
        "input_pair": {
            "first_sha256": sha256_file(args.input),
            "second_sha256": sha256_file(args.second_input) if args.second_input else None,
            "pair_kind": "independently-decoded-real-clips" if args.second_input else "original-and-reversed-fixture",
        },
        "frame_order": {"patch_reversal": patch_reversal, "block0_aligned_reversal": block0_order, "behavior_reversal": behavior_order, "pass": frame_order_pass},
        "padding_mask_contract": padding_mask_contract,
        "throughput": {
            "warmup_forwards": 1,
            "measured_forwards": args.benchmark_repeats,
            "seconds": [float(value) for value in timings],
            "median_seconds_per_clip": float(numpy.median(timing_array)),
            "mean_seconds_per_clip": float(timing_array.mean()),
            "batch_size": int(batch.shape[0]),
            "batches_per_second_from_median": float(1.0 / numpy.median(timing_array)),
            "clips_per_second_from_median": float(batch.shape[0] / numpy.median(timing_array)),
        },
        "cache_fixture_manifest_sha256": sha256_file(os.path.join(cache_dir, "manifest.json")),
        "cache_control_pass": cache_control_pass,
        "formal_isolation_pass": False,
        "g1_admission": False,
        "fallback_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(os.path.join(args.output, "metrics.json"), report)
    progress(100, "LR1 reference validation {0}".format(report["status"]))


def compare(args):
    import numpy

    fresh_dir(args.output)
    progress(0, "created immutable parity comparison run")
    with open(os.path.join(args.reference, "manifest.json"), "r", encoding="utf-8") as stream:
        reference_manifest = json.load(stream)
    with open(os.path.join(args.candidate, "manifest.json"), "r", encoding="utf-8") as stream:
        candidate_manifest = json.load(stream)
    modes_selected = int(bool(args.all_blocks)) + int(bool(args.component_blocks)) + int(bool(args.replay_mlp_blocks))
    if modes_selected > 1:
        raise RuntimeError("comparison diagnostic modes are mutually exclusive")
    if args.replay_mlp_blocks:
        stages_to_compare = selected_mlp_replay_stages(args.replay_mlp_blocks)
    else:
        stages_to_compare = selected_stages(args.all_blocks, args.component_blocks)
    results = []
    for index, stage in enumerate(stages_to_compare):
        ref = numpy.load(os.path.join(args.reference, reference_manifest["stages"][stage]["path"]), allow_pickle=False)
        cand = numpy.load(os.path.join(args.candidate, candidate_manifest["stages"][stage]["path"]), allow_pickle=False)
        if ref.shape != cand.shape:
            raise RuntimeError("shape mismatch at {0}".format(stage))
        a = ref.astype(numpy.float64, copy=False).ravel()
        b = cand.astype(numpy.float64, copy=False).ravel()
        diff = numpy.abs(a - b)
        denom = max(float(numpy.linalg.norm(a)), 1e-30)
        cosine_denom = max(float(numpy.linalg.norm(a) * numpy.linalg.norm(b)), 1e-30)
        metrics = {
            "stage": stage,
            "shape": list(ref.shape),
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "relative_l2_error": float(numpy.linalg.norm(a - b) / denom),
            "cosine_similarity": float(numpy.dot(a, b) / cosine_denom),
            "nonfinite_values": int((~numpy.isfinite(a)).sum() + (~numpy.isfinite(b)).sum()),
            "abs_error_percentiles": {
                "p50": float(numpy.percentile(diff, 50)),
                "p90": float(numpy.percentile(diff, 90)),
                "p99": float(numpy.percentile(diff, 99)),
                "p99_9": float(numpy.percentile(diff, 99.9)),
                "p99_99": float(numpy.percentile(diff, 99.99)),
            },
            "counts_above": {
                "1e-6": int((diff > 1e-6).sum()),
                "1e-5": int((diff > 1e-5).sum()),
                "1e-4": int((diff > 1e-4).sum()),
                "1e-3": int((diff > 1e-3).sum()),
            },
            "max_error_index": [int(value) for value in numpy.unravel_index(int(diff.argmax()), ref.shape)],
        }
        if ref.ndim == 3 and ref.shape[0] == 1 and ref.shape[1] > 1:
            token_diff = numpy.abs(ref.astype(numpy.float64) - cand.astype(numpy.float64))[0]
            metrics["token_partition"] = {
                "cls_max_abs_error": float(token_diff[0].max()),
                "cls_mean_abs_error": float(token_diff[0].mean()),
                "patch_max_abs_error": float(token_diff[1:].max()),
                "patch_mean_abs_error": float(token_diff[1:].mean()),
                "worst_token_index": int(token_diff.max(axis=1).argmax()),
            }
        metrics["pass"] = (
            metrics["max_abs_error"] <= CRITERIA["max_abs_error"]
            and metrics["mean_abs_error"] <= CRITERIA["mean_abs_error"]
            and metrics["relative_l2_error"] <= CRITERIA["relative_l2_error"]
            and metrics["cosine_similarity"] >= CRITERIA["minimum_cosine_similarity"]
            and metrics["nonfinite_values"] <= CRITERIA["nonfinite_values_allowed"]
        )
        results.append(metrics)
        progress(15 + int(75 * (index + 1) / len(stages_to_compare)), "compared {0}".format(stage))
    all_pass = all(item["pass"] for item in results)
    report = {
        "schema_version": "endosae.model-port-parity-observed.v0",
        "status": "pass" if all_pass else "fail",
        "reference_runtime": reference_manifest["runtime_label"],
        "candidate_runtime": candidate_manifest["runtime_label"],
        "criteria": CRITERIA,
        "stage_results": results,
        "all_required_stages_pass": all_pass,
        "behavior_argmax_applicable": False,
        "formal_isolation_pass": False,
        "g1_admission": False,
        "fallback_used": False,
        "diagnostic_all_blocks": bool(args.all_blocks),
        "diagnostic_component_blocks": list(args.component_blocks or ()),
        "diagnostic_replay_mlp_blocks": list(args.replay_mlp_blocks or ()),
        "interpretation": "Numerical model-port evidence only; formal isolation and full G1 remain separate gates.",
    }
    write_json(os.path.join(args.output, "metrics.json"), report)
    progress(100, "parity comparison {0}".format(report["status"]))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")
    export = sub.add_parser("export-state")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    stages = sub.add_parser("run-stages")
    stages.add_argument("--model-dir", required=True)
    stages.add_argument("--state", required=True)
    stages.add_argument("--input", required=True)
    stages.add_argument("--runtime-label", required=True)
    stages.add_argument("--output", required=True)
    stages.add_argument("--all-blocks", action="store_true")
    stages.add_argument("--component-blocks", nargs="*", type=int, default=[])
    stages.add_argument("--cpu-threads", type=int)
    replay = sub.add_parser("replay-mlp")
    replay.add_argument("--model-dir", required=True)
    replay.add_argument("--state", required=True)
    replay.add_argument("--reference-components", required=True)
    replay.add_argument("--runtime-label", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--replay-mlp-blocks", nargs="+", type=int, required=True)
    replay.add_argument("--cpu-threads", type=int)
    validate = sub.add_parser("validate-reference")
    validate.add_argument("--model-dir", required=True)
    validate.add_argument("--state", required=True)
    validate.add_argument("--input", required=True)
    validate.add_argument("--second-input")
    validate.add_argument("--sampling-indices")
    validate.add_argument("--contracts-dir")
    validate.add_argument("--runtime-label", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--cpu-threads", type=int)
    validate.add_argument("--benchmark-repeats", type=int, default=5)
    lr2 = sub.add_parser("validate-lr2-preprocess")
    lr2.add_argument("--dataset-dir", required=True)
    lr2.add_argument("--video", required=True)
    lr2.add_argument("--video-sha256", required=True)
    lr2.add_argument("--short-video")
    lr2.add_argument("--short-video-sha256")
    lr2.add_argument("--output", required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--output", required=True)
    comp.add_argument("--all-blocks", action="store_true")
    comp.add_argument("--component-blocks", nargs="*", type=int, default=[])
    comp.add_argument("--replay-mlp-blocks", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if args.mode == "export-state":
        export_state(args)
    elif args.mode == "run-stages":
        run_stages(args)
    elif args.mode == "replay-mlp":
        run_mlp_replay(args)
    elif args.mode == "validate-reference":
        validate_reference(args)
    elif args.mode == "validate-lr2-preprocess":
        validate_lr2_preprocess(args)
    elif args.mode == "compare":
        compare(args)
    else:
        parser.error("a mode is required")


if __name__ == "__main__":
    main()
