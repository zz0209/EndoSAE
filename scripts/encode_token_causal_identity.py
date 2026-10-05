import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import encode_causal_detection_identity as native_api
import train_acknowledgement_sae as shared
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import tokens_to_frames
from src.token_identity_sae import TokenIdentitySAE
from verify_token_identity_sae import numpy_forward


def model_specs(config):
    return [(f"{space}_{method}_seed{seed}", Path(training) / "fit" / method / f"seed{seed}")
            for space, training in config["training_runs"].items()
            for method in config["methods"] for seed in config["seeds"]]


def load_models(config, device):
    result = {}
    for key, folder in model_specs(config):
        definition = read_json(folder / "model_config.json")
        model = TokenIdentitySAE(definition, definition["method"])
        with np.load(folder / "model.npz", allow_pickle=False) as archive:
            model.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files})
        with np.load(folder / "normalization.npz", allow_pickle=False) as archive:
            mean, scale = archive["mean"].copy(), archive["scale"].copy()
        result[key] = (model.to(device).eval().requires_grad_(False), mean, scale)
    return result


def video_inputs(config, video):
    phase = next(name for name in ("development", "extension") if video in config[name + "_videos"])
    base = Path(config[phase + "_base"])
    settings = read_json(base / "config.json")
    directory = Path(settings["descriptors"]) / video
    receipt = read_json(directory / "complete.json")
    if receipt["status"] != "COMPLETE":
        raise ValueError("Incomplete causal descriptor reference")
    definition = read_json(directory / "config.json")
    prediction = Path(settings["predictions"]) / video / "detections.jsonl"
    with prediction.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    with np.load(directory / "indices.npz", allow_pickle=False) as data:
        tracks = {key: data[key].copy() for key in data.files}
    available = np.load(directory / "available.npy", allow_pickle=False)
    status = np.load(directory / "status_code.npy", allow_pickle=False)
    np.testing.assert_array_equal(tracks["frame_indices"], [r["frame_index"] for r in records])
    np.testing.assert_array_equal(np.diff(tracks["offsets"]), [len(r["detections"]) for r in records])
    np.testing.assert_array_equal(available, status == 1)
    if len(available) != tracks["offsets"][-1] or np.any(status == 0):
        raise ValueError("Incomplete reference candidates")
    targets = np.unique(np.searchsorted(tracks["offsets"][1:], np.flatnonzero(available), side="right"))
    sources = read_json(base / video / "summary.json")["episodes"]
    by_output = {}
    for episode in sources:
        if episode["click"]["available"]:
            by_output.setdefault(episode["click"]["output_index"], []).append(episode)
    files = [prediction, directory / "identity.json", directory / "indices.npz", directory / "config.json",
             directory / "complete.json", base / video / "summary.json"]
    return definition, directory, records, tracks, available, targets, by_output, files


def targets_for(targets, sources, smoke):
    if not smoke:
        return targets
    sampled = targets[np.linspace(0, len(targets) - 1, min(12, len(targets)), dtype=int)]
    return np.array(sorted(set(sampled.tolist()) | (set(sources) & set(targets.tolist()))), dtype=int)


def encode(run, smoke, resume):
    config = read_json(run / "config.json")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config["device"])
    root = run / "smoke_encoding" if smoke else Path(config["embedding_root"])
    root.mkdir(parents=True, exist_ok=True)
    videos = [config["development_videos"][0], config["extension_videos"][0]] if smoke else config["development_videos"] + config["extension_videos"]
    counts = {}
    for video in videos:
        _, _, _, _, _, targets, sources, _ = video_inputs(config, video)
        counts[video] = len(targets_for(targets, sources, smoke))
    total = sum(counts.values())
    models = load_models(config, device)
    paths = [run / "config.json", Path(__file__), ROOT / "src/token_identity_sae.py",
             Path(native_api.__file__), ROOT / "scripts/encode_rc27_cohort.py"]
    paths += [folder / name for _, folder in model_specs(config)
              for name in ("model.npz", "model_config.json", "normalization.npz")]
    code_identity = {str(path): shared.file_sha256(path) for path in paths}
    backbone = None
    completed, summaries = 0, []
    started = time.perf_counter()
    for video in videos:
        definition, reference, records, tracks, available, targets, sources, inputs = video_inputs(config, video)
        targets = targets_for(targets, sources, smoke)
        destination = root / video
        destination.mkdir(parents=True, exist_ok=True)
        identity = dict(files=code_identity, inputs={str(p): shared.file_sha256(p) for p in inputs},
                        video=video, smoke=smoke, targets=targets.tolist(), torch=str(torch.__version__),
                        numpy=np.__version__, device=str(device))
        if (destination / "identity.json").exists():
            if not resume or read_json(destination / "identity.json") != identity:
                raise ValueError("Causal encoding identity differs")
        else:
            atomic_write_json(destination / "identity.json", identity)
        if (destination / "complete.json").exists():
            summaries.append(read_json(destination / "complete.json"))
            completed += len(targets)
            continue
        if backbone is None:
            backbone = native_api.Encoder(read_json(definition["encoder_config"]), device)
        prepare = native_api.Inputs(definition["frame_root"], video)
        original = np.load(reference / "raw_mean.npy", mmap_mode="r", allow_pickle=False)
        progress_path = destination / "progress.json"
        previous = read_json(progress_path) if progress_path.exists() else dict(processed=0, max_mean_error=0., checks=[], seconds=0.)
        arrays = {}
        for key, (model, _, _) in models.items():
            dimension = model.latent_dim if model.identity_space == "code" else model.readout.out_features
            path = destination / (key + ".npy")
            arrays[key] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                                  shape=(len(available), dimension), dtype=np.float32)
            if previous["processed"] == 0:
                arrays[key][:] = np.nan
        status_path = destination / "encoded.npy"
        encoded = np.lib.format.open_memmap(status_path, mode="r+" if status_path.exists() else "w+",
                                           shape=available.shape, dtype=np.bool_)
        if previous["processed"] == 0:
            encoded[:] = False
        begin = time.perf_counter()
        max_error, checks = previous["max_mean_error"], previous["checks"]
        for number in range(previous["processed"], len(targets)):
            output = int(targets[number])
            native, _ = backbone(prepare(records, output))
            tokens = tokens_to_frames(native[None])
            start, end = tracks["offsets"][output:output + 2]
            for index in range(int(end - start)):
                position = int(start) + index
                mask, support, reason = native_api.support(records, tracks, output, index)
                if (reason == "available") != bool(available[position]):
                    raise ValueError("Causal support availability changed")
                if not available[position]:
                    continue
                raw = tokens[mask]
                error = float(np.max(np.abs(raw.mean(0, dtype=np.float64) - original[position])))
                max_error = max(error, max_error)
                if error > 1e-5:
                    raise ValueError(f"Native mean mismatch at {video}/{output}: {error}")
                for key, (model, mean, scale) in models.items():
                    values = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
                    with torch.no_grad():
                        code = model.encode(torch.from_numpy(values).to(device)).mean(0)
                        vector = F.normalize(model.readout(code), dim=0).cpu().numpy()
                    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
                        raise ValueError("Invalid causal identity vector")
                    arrays[key][position] = vector
                    if smoke and not any(c["model"] == key for c in checks):
                        reference_model = TokenIdentitySAE(read_json(dict(model_specs(config))[key] / "model_config.json"), model.method).double().eval()
                        reference_model.load_state_dict({k: v.detach().cpu().double() for k, v in model.state_dict().items()})
                        expected, _ = numpy_forward(reference_model, ((raw.astype(float) - mean) / scale)[None])
                        np.testing.assert_allclose(vector, expected[0], atol=2e-5, rtol=1e-4)
                        checks.append(dict(model=key, position=position, numpy_max_error=float(np.max(np.abs(vector - expected[0])))))
                encoded[position] = True
            if output in sources:
                for episode in sources[output]:
                    target = destination / "sources" / episode["episode_id"]
                    target.mkdir(parents=True, exist_ok=True)
                    index = episode["click"]["detection_index"]
                    mask, support, reason = native_api.support(records, tracks, output, index)
                    shared.save_npz(target / "observed_tokens.npz", tokens=tokens[mask], positions=np.column_stack(np.where(mask)))
                    atomic_write_json(target / "source.json", dict(click=episode["click"], support=support,
                        status=reason, position=int(start) + index, future_frames_used=False, ground_truth_regions_used=False))
            if (number + 1) % config["checkpoint_outputs"] == 0 or number + 1 == len(targets):
                for value in arrays.values():
                    value.flush()
                encoded.flush()
                atomic_write_json(progress_path, dict(processed=number + 1, total=len(targets), max_mean_error=max_error,
                    checks=checks, seconds=previous["seconds"] + time.perf_counter() - begin))
                atomic_write_json(run / ("smoke_encoding_progress.json" if smoke else "encoding_progress.json"),
                    dict(status="RUNNING", completed=completed + number + 1, total=total, video=video,
                         seconds=time.perf_counter() - started, updated_at=shared.now()))
                print("CAUSAL_TOKEN", video, number + 1, "/", len(targets), "total", completed + number + 1, "/", total, flush=True)
                pause_after_checkpoint(progress_path)
        if not smoke:
            np.testing.assert_array_equal(encoded, available)
        if any(not np.isfinite(value[encoded]).all() for value in arrays.values()):
            raise ValueError("Missing encoded vector")
        result = dict(status="COMPLETE", video=video, smoke=smoke, native_outputs=len(targets),
            detections=int(encoded.sum()), max_mean_error=max_error, checks=checks,
            seconds=previous["seconds"] + time.perf_counter() - begin, completed_at=shared.now(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), identity_sha256=shared.file_sha256(destination / "identity.json"))
        atomic_write_json(destination / "complete.json", result)
        summaries.append(result)
        completed += len(targets)
        del arrays, encoded, original
    if code_identity != {str(path): shared.file_sha256(path) for path in paths}:
        raise ValueError("Encoder inputs changed during execution")
    atomic_write_json(run / ("smoke_encoding_summary.json" if smoke else "encoding_summary.json"),
        dict(status="COMPLETE", smoke=smoke, root=str(root), videos=summaries, total_native_outputs=total, models=list(models)))
    atomic_write_json(run / ("smoke_encoding_progress.json" if smoke else "encoding_progress.json"),
        dict(status="COMPLETE", completed=completed, total=total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    encode(args.run, args.smoke, args.resume)
