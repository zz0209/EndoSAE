"""Match joint-model head regularization, reusing the exact training-fold fits.

This bounded consumer changes only head weight decay. A complete saved-prediction
replay at the old setting must pass before the new four-fold comparison starts.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import digest, summarize, write_json
from src.evaluation.realcolon_task_supervised import SupportSampler, feature_stats, fit_one, fold_masks


FIXED_KEYS = ("seed", "head_steps", "head_batch_size", "head_learning_rate", "finetune_learning_rate",
              "topk_batch_size", "topk_width", "topk_k", "heldout_folds", "source_manifest_sha256",
              "source_encoded_sha256", "source_token_sha256", "source_token_shape", "cache_file_identity")


def validate_config(config, reference):
    if any(config[key] != reference[key] for key in FIXED_KEYS):
        raise ValueError("A setting other than head regularization changed")
    if config["mse_weight_candidates"] != [.1]:
        raise ValueError("This revision preserves the previously selected MSE coefficient")


def validate_scope(scope, rows, fit_clips):
    expected_rows = [dict(row, split="train" if use else "validation") for row, use in zip(rows, fit_clips)]
    expected_fit = sorted({row["video_id"] for row, use in zip(rows, fit_clips) if use})
    expected_held = sorted({row["video_id"] for row, use in zip(rows, fit_clips) if not use})
    if scope["all_rows"] != expected_rows or scope["fit_videos"] != expected_fit or scope["validation_videos"] != expected_held:
        raise ValueError("Reference fold has different fitting/held-out videos or labels")
    if scope["fit_clip_indices"] != np.flatnonzero(fit_clips).tolist():
        raise ValueError("Reference fitting indices differ")


def load_references(source, expected_hashes, rows, fit_clips, config, device, torch):
    from src.sae.baselines import TopKAutoencoder
    for name, expected in expected_hashes.items():
        if digest(source / name) != expected:
            raise ValueError("Reference asset changed: " + name)
    scope = json.loads((source / "fit_scope.json").read_text())
    validate_scope(scope, rows, fit_clips)
    references = {}
    for label, k in (("topk", config["topk_k"]), ("dense", config["topk_width"])):
        model = TopKAutoencoder(config["source_token_shape"][-1], config["topk_width"], k).to(device)
        with np.load(source / (label + "_initial.npz"), allow_pickle=False) as state:
            tensors = {name: torch.tensor(state[name], device=device) for name in state.files}
        model.load_state_dict(tensors, strict=True)
        references[label] = model
    with np.load(source / "input_normalization_pca.npz", allow_pickle=False) as arrays:
        mean = torch.tensor(arrays["mean"], device=device)
        rms = torch.tensor(arrays["rms"], device=device)
    return references, mean, rms, scope


def execute(run, config, phase):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    reference = Path(config["reference_run"])
    if digest(reference / "config.json") != config["reference_config_sha256"]:
        raise ValueError("Reference configuration changed")
    base_config = json.loads((reference / "config.json").read_text())
    validate_config(config, base_config)
    core = Path(__file__).with_name("realcolon_task_supervised.py")
    if digest(core) != config["task_implementation_sha256"]:
        raise ValueError("Task implementation changed since the original comparison")
    if phase in ("cv", "fit"):
        proof = json.loads((run / "replay/verification.json").read_text())
        if (proof["status"] != "EXACT_REPLAY" or proof["reference_config_sha256"] != config["reference_config_sha256"]
                or proof["revision_config_sha256"] != digest(run / "config.json")):
            raise ValueError("Exact old-setting replay required before comparison")
    development = None
    if phase == "fit":
        development = json.loads((run / "development_config.json").read_text())
        if development["revision_config_sha256"] != digest(run / "config.json"):
            raise ValueError("Development plan refers to a different CV configuration")
        if development["verified_cv_sha256"] != digest(run / "cv/independent_verification.json"):
            raise ValueError("Development plan refers to different training results")
        if development["runner_sha256"] != digest(Path(__file__)):
            raise ValueError("Development runner changed after the plan was fixed")
    source = Path(config["source_run"])
    if digest(source / "clip_manifest.jsonl") != config["source_manifest_sha256"] or digest(source / "encoded.json") != config["source_encoded_sha256"]:
        raise ValueError("Source provenance changed")
    rows = [json.loads(line) for line in (source / "clip_manifest.jsonl").read_text().splitlines()]
    selected = [i for i, row in enumerate(rows) if phase == "fit" or row["split"] == "train"]
    rows = [rows[i] for i in selected]
    cache = Path(config["cache_dir"])
    for name, expected in config["cache_file_identity"].items():
        stat = (cache / name).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Previously verified cache changed")
    tokens = np.load(cache / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != tuple(config["source_token_shape"]):
        raise ValueError("Token cache shape mismatch")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    raw = torch.tensor(tokens[selected].reshape(-1, tokens.shape[-1]), device="cuda")
    masks = np.load(cache / "masks.npy", mmap_mode="r", allow_pickle=False)[selected]
    output = run / phase
    output.mkdir(exist_ok=False)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "task_helper_snapshot.py").write_bytes(core.read_bytes())
    write_json(output / "config.json", base_config if phase == "replay" else config)
    records = []
    folds = ([np.array([row["split"] == "train" for row in rows])] if phase == "fit"
             else list(fold_masks(rows, config["heldout_folds"])))
    if phase == "replay":
        folds = folds[:1]
    for index, fit_clips in enumerate(folds):
        fold = output / ("fold" + str(index))
        fold.mkdir()
        source_fold = reference / ("fit" if phase == "fit" else "cv") / fold.name
        asset_hashes = (development["reference_assets"] if phase == "fit" else config["reference_assets"])[fold.name]
        references, mean, rms, scope = load_references(source_fold, asset_hashes, rows, fit_clips, config, raw.device, torch)
        x = (raw - mean) / rms
        fit_indices = torch.tensor(np.flatnonzero(np.repeat(fit_clips, 8 * 196)), device=raw.device)
        sampler = SupportSampler(rows, masks, fit_clips, raw.device, torch)
        stats = {label: feature_stats(x, fit_indices, model.encode_inference, torch) for label, model in references.items()}
        write_json(fold / "fit_scope.json", scope)
        write_json(fold / "reference.json", {"source": source_fold.as_posix(), "sha256": asset_hashes})
        methods = ("topk_task",) if phase == "replay" else ("topk_task", "dense_task")
        active_config = base_config if phase == "replay" else config
        for method in methods:
            path = fold / (method + "_candidate0")
            path.mkdir()
            hyper = {"mse_weight": .1}
            predictions = fit_one(method, hyper, x, fit_indices, sampler, references, stats, active_config, path, torch).reshape(masks.shape)
            np.save(path / "predictions.npy", predictions, allow_pickle=False)
            metrics = summarize(predictions, masks, scope["all_rows"])
            write_json(path / "metrics.json", metrics)
            records.append({"method": method, "hyperparameters": hyper, "path": path.as_posix(),
                "validation": {v: metrics["per_video"][v] for v in scope["validation_videos"]}})
            print(json.dumps({"fold": index, "method": method, "head_weight_decay": active_config["joint_head_weight_decay"],
                "validation_auroc": {v: metrics["per_video"][v]["clip_detection"]["auroc"] for v in scope["validation_videos"]}}), flush=True)
            if phase == "replay":
                baseline_path = source_fold / (method + "_candidate0/predictions.npy")
                original = np.load(baseline_path, allow_pickle=False)
                if not np.array_equal(predictions, original):
                    raise ValueError("Old-setting predictions differ: max=" + str(float(np.max(np.abs(predictions - original)))))
                write_json(output / "verification.json", {"status": "EXACT_REPLAY", "n_prediction_values": predictions.size,
                    "maximum_absolute_difference": 0., "reference_config_sha256": config["reference_config_sha256"],
                    "revision_config_sha256": digest(run / "config.json"),
                    "reference_prediction_sha256": digest(baseline_path), "replay_prediction_sha256": digest(path / "predictions.npy")})
        del x, references, sampler, stats
    aggregate = {}
    for method in sorted({r["method"] for r in records}):
        values = [v for r in records if r["method"] == method for video, v in r["validation"].items()
                  if phase != "fit" or video in config["primary_development_videos"]]
        aggregate[method] = [{"hyperparameters": {"mse_weight": .1},
            "macro_video_auroc": float(np.mean([v["clip_detection"]["auroc"] for v in values])),
            "macro_positive_frame_patch_ap": float(np.mean([v["positive_frame_mean_patch_ap"] for v in values]))}]
    write_json(output / "summary.json", {"status": "COMPLETED_" + phase.upper(), "records": records,
        "candidates": aggregate, "selected": {m: v[0] for m, v in aggregate.items()},
        "elapsed_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "limitations": config["limitations"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("replay", "cv", "fit"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / "config.json").read_text())
    write_json(run / "status.json", {"status": "RUNNING", "phase": args.phase, "pid": os.getpid(), "started_at_unix": time.time()})
    try:
        execute(run, config, args.phase)
    except Exception as error:
        write_json(run / "status.json", {"status": "FAILED", "phase": args.phase, "error": repr(error), "ended_at_unix": time.time()})
        raise
    write_json(run / "status.json", {"status": "COMPLETED", "phase": args.phase, "ended_at_unix": time.time()})


if __name__ == "__main__":
    main()
