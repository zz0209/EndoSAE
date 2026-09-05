"""Two fixed PCA controls and inference-only confirmation of existing models."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import digest, summarize, write_json
from src.evaluation.realcolon_task_supervised import SupportSampler, feature_stats, fit_one


def arrays(path, torch, device):
    with np.load(path, allow_pickle=False) as state:
        return {name: torch.tensor(state[name], device=device) for name in state.files}


def predict(raw, method, config, torch):
    """Load frozen native NPZ assets; never estimate statistics from these inputs."""
    record = config["models"][method]
    head = arrays(Path(record["head"]), torch, raw.device)
    if method == "raw_previous":
        mean, scale = head["mean"], head["scale"]
        norm = arrays(Path(record["normalization"]), torch, raw.device)
        x = (raw - norm["mean"]) / norm["rms"]
    else:
        norm = arrays(Path(config["normalization"]), torch, raw.device)
        x = (raw - norm["mean"]) / norm["rms"]
        mean, scale = head["feature_mean"], head["feature_scale"]
    if method.startswith("pca"):
        basis = norm["basis"][:, :record["dimension"]]
        transform = lambda batch: batch @ basis
    elif method == "topk_frozen_balanced":
        from src.sae.baselines import TopKAutoencoder
        model = TopKAutoencoder(768, 1536, 32).to(raw.device)
        model.load_state_dict(arrays(Path(record["dictionary"]), torch, raw.device), strict=True)
        model.eval()
        transform = model.encode_inference
    else:
        transform = lambda batch: batch
    predictions = []
    with torch.no_grad():
        for batch in x.split(4096):
            features = (transform(batch) - mean) / scale
            logits = torch.nn.functional.linear(features, head["weight"], head["bias"])
            predictions.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
    return np.concatenate(predictions)


def validate_assets(config):
    for path, expected in config["asset_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Frozen model/source changed: " + path)


def tokens_cuda(tokens, indices, torch):
    """Bound the host temporary to 16 clips on a memory-constrained desktop."""
    result = torch.empty((len(indices), 8, 196, 768), device="cuda", dtype=torch.float32)
    for start in range(0, len(indices), 16):
        selected = indices[start:start + 16]
        result[start:start + len(selected)].copy_(torch.from_numpy(np.array(tokens[selected], copy=True)))
    return result.reshape(-1, 768)


def training_inputs(config, torch):
    source = Path(config["source_run"])
    rows = [json.loads(line) for line in (source / "clip_manifest.jsonl").read_text().splitlines()]
    indices = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    cache = Path(config["training_cache"])
    for name, expected in config["cache_file_identity"].items():
        stat = (cache / name).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Previously verified training cache changed")
    tokens = np.load(cache / "tokens.npy", mmap_mode="r", allow_pickle=False)
    raw = tokens_cuda(tokens, indices, torch)
    masks = np.load(cache / "masks.npy", mmap_mode="r", allow_pickle=False)[indices]
    return raw, masks, [rows[i] for i in indices], indices


def fit_controls(run, config, torch):
    output = run / "baselines"
    output.mkdir(exist_ok=False)
    raw, masks, rows, source_indices = training_inputs(config, torch)
    norm = arrays(Path(config["normalization"]), torch, raw.device)
    x = (raw - norm["mean"]) / norm["rms"]
    indices = torch.arange(len(x), device=raw.device)
    sampler = SupportSampler(rows, masks, np.ones(len(rows), dtype=bool), raw.device, torch)
    task_config = json.loads(Path(config["task_config"]).read_text())
    write_json(output / "fit_scope.json", {"fit_videos": sampler.videos, "all_rows": rows,
               "source_clip_indices": source_indices.tolist(), "validation_videos": []})
    replay = {}
    for method in ("raw_balanced", "raw_previous", "topk_frozen_balanced", "pca64_balanced"):
        result = predict(raw, method, config, torch).reshape(masks.shape)
        old = np.load(config["models"][method]["training_predictions"], mmap_mode="r", allow_pickle=False)[source_indices]
        error = float(np.max(np.abs(result - old)))
        if error > 1e-6:
            raise ValueError("Frozen predictor failed training replay: " + method + " " + str(error))
        replay[method] = {"max_absolute_error": error, "n_values": result.size}
        print(json.dumps({"replay": method, **replay[method]}), flush=True)
    write_json(output / "replay.json", replay)
    for dimension in (32, 48):
        method = "pca" + str(dimension) + "_balanced"
        destination = output / method
        destination.mkdir()
        references = {"pca": norm["basis"][:, :dimension]}
        stats = {"pca": feature_stats(x, indices, lambda value: value @ references["pca"], torch)}
        # The existing generic PCA head backend accepts an explicit basis width.
        result = fit_one("pca64_balanced", {"head_weight_decay": 10.}, x, indices, sampler,
                         references, stats, task_config, destination, torch).reshape(masks.shape)
        training = json.loads((destination / "training.json").read_text())
        training.update(method=method, projection_dimension=dimension, backend="existing PCA head, explicit basis width")
        write_json(destination / "training.json", training)
        np.save(destination / "predictions.npy", result, allow_pickle=False)
        # Fitting output must be reproduced by the inference-only consumer.
        again = predict(raw, method, config, torch).reshape(masks.shape)
        if not np.allclose(result, again, atol=1e-6, rtol=0):
            raise ValueError("New PCA head inference failed replay")
        print(json.dumps({"fitted": method, "fit_videos": sampler.videos,
                          "inference_replay_error": float(np.max(np.abs(result - again)))}), flush=True)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "completion.json", {"status": "TWO_FIXED_CONTROLS_FITTED", "no_development_inputs": True})


def confirmation(run, config, torch):
    freeze = json.loads((run / "model_freeze.json").read_text())
    for path, expected in freeze["sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Final fixed asset changed: " + path)
    rows = [json.loads(line) for line in (run / "clip_manifest.jsonl").read_text().splitlines()]
    expected_videos = set(config["confirmation_videos"])
    if {r["video_id"] for r in rows} != expected_videos or any(r["split"] != "confirmation" for r in rows):
        raise ValueError("Confirmation cohort changed")
    source = Path(config["source_run"])
    all_old = [json.loads(line) for line in (source / "clip_manifest.jsonl").read_text().splitlines()]
    fit_indices = [i for i, r in enumerate(all_old) if r["split"] == "train"]
    fit_rows = [all_old[i] for i in fit_indices]
    if {r["video_id"] for r in fit_rows} & expected_videos:
        raise ValueError("Training/confirmation overlap")
    cache = Path(config["confirmation_cache"])
    encoded = json.loads((run / "encoded.json").read_text())
    if encoded["manifest_sha256"] != digest(run / "clip_manifest.jsonl"):
        raise ValueError("Encoded cohort mismatch")
    if digest(cache / "tokens.npy") != encoded["tokens_sha256"]:
        raise ValueError("Encoded tokens changed")
    tokens = np.load(cache / "tokens.npy", mmap_mode="r", allow_pickle=False)
    raw = tokens_cuda(tokens, list(range(len(rows))), torch)
    masks = np.load(cache / "masks.npy", allow_pickle=False)
    fit_masks = np.load(Path(config["training_cache"]) / "masks.npy", mmap_mode="r", allow_pickle=False)[fit_indices]
    combined_masks = np.concatenate([fit_masks, masks])
    combined_rows = fit_rows + rows
    output = run / "confirmation"
    output.mkdir(exist_ok=False)
    write_json(output / "fit_scope.json", {"fit_videos": sorted({r["video_id"] for r in fit_rows}),
               "validation_videos": sorted(expected_videos), "all_rows": combined_rows,
               "source_clip_indices": fit_indices})
    records = []
    for method in config["models"]:
        new_predictions = predict(raw, method, config, torch).reshape(masks.shape)
        record = config["models"][method]
        old = np.load(record["training_predictions"], mmap_mode="r", allow_pickle=False)
        train_predictions = old if method in ("pca32_balanced", "pca48_balanced") else old[fit_indices]
        predictions = np.concatenate([train_predictions, new_predictions])
        destination = output / method
        destination.mkdir()
        np.save(destination / "predictions.npy", predictions, allow_pickle=False)
        metrics = summarize(predictions, combined_masks, combined_rows)
        write_json(destination / "metrics.json", metrics)
        records.append({"method": method, "path": destination.as_posix(),
                        "validation": {v: metrics["per_video"][v] for v in sorted(expected_videos)}})
        print(json.dumps({"method": method, "confirmation_auroc": {v: metrics["per_video"][v]["clip_detection"]["auroc"]
                          for v in sorted(expected_videos)}}), flush=True)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "summary.json", {"status": "FIXED_MODEL_CONFIRMATION_COMPLETE", "records": records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit-controls", "confirm"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / "model_config.json").read_text())
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    validate_assets(config)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    write_json(run / "model_status.json", {"phase": args.phase, "status": "RUNNING", "pid": os.getpid()})
    try:
        (fit_controls if args.phase == "fit-controls" else confirmation)(run, config, torch)
    except Exception as error:
        write_json(run / "model_status.json", {"phase": args.phase, "status": "FAILED", "error": repr(error)})
        raise
    write_json(run / "model_status.json", {"phase": args.phase, "status": "COMPLETED",
               "elapsed_seconds": time.time() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


if __name__ == "__main__":
    main()
