"""Training-video-only readout selection for the REAL-Colon task pilot.

Each fold refits representation statistics, PCA and the reconstruction SAE.
Uses no exposed-development pixels, tokens, labels or model outcomes.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import binary_metrics, digest, fit_head, write_json


def training_folds(rows):
    selected = [i for i, row in enumerate(rows) if row["split"] == "train"]
    videos = sorted({rows[i]["video_id"] for i in selected})
    if len(videos) < 3:
        raise ValueError("Need at least three training videos")
    for video in videos:
        fit = np.array([rows[i]["video_id"] != video for i in selected])
        if not fit.any() or fit.all():
            raise ValueError("Empty fit or held-out group")
        yield video, selected, fit


def train_dictionary(train_x, config, torch):
    from src.sae.baselines import TopKAutoencoder
    torch.manual_seed(config["seed"])
    sae = TopKAutoencoder(train_x.shape[1], config["topk_width"], config["topk_k"]).to(train_x.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config["topk_learning_rate"])
    generator = torch.Generator(device=train_x.device).manual_seed(config["seed"])
    history = []
    for step in range(config["topk_steps"]):
        index = torch.randint(len(train_x), (config["topk_batch_size"],), generator=generator, device=train_x.device)
        batch = train_x[index]
        reconstruction, _ = sae(batch, inference=False)
        loss = (reconstruction - batch).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_gradient_()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.)
        optimizer.step()
        sae.normalize_decoder_()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite fold SAE loss")
        if step % 250 == 0 or step + 1 == config["topk_steps"]:
            history.append({"step": step + 1, "mse": float(loss.detach())})
    return sae, history


def run_cv(run, config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source = Path(config["source_run"])
    manifest = source / "clip_manifest.jsonl"
    if digest(manifest) != config["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    folds = list(training_folds(rows))
    selected = folds[0][1]
    training_rows = [rows[i] for i in selected]
    if sorted({row["video_id"] for row in training_rows}) != config["training_videos"]:
        raise RuntimeError("Training cohort changed")
    cache = Path(config["cache_dir"])
    # Slice before materialization: development arrays are never loaded.
    raw = torch.tensor(np.load(cache / "tokens.npy", mmap_mode="r", allow_pickle=False)[selected].reshape(-1, 768), device=device)
    masks = np.load(cache / "masks.npy", mmap_mode="r", allow_pickle=False)[selected]
    labels = torch.tensor(masks.reshape(-1).astype(np.float32), device=device)
    clip_labels = np.array([row["class"] for row in training_rows], dtype=bool)
    output = run / "comparison"
    output.mkdir(exist_ok=False)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "task_helper_snapshot.py").write_bytes(Path(__file__).with_name("realcolon_task.py").read_bytes())
    write_json(output / "training_manifest.json", training_rows)
    records = []
    for held_out, _, fit_clips in folds:
        fold = output / held_out
        fold.mkdir()
        train_mask = torch.tensor(np.repeat(fit_clips, 8 * 196), device=device)
        mean = raw[train_mask].mean(0)
        rms = ((raw[train_mask] - mean).square().mean()).sqrt().clamp_min(1e-6)
        x = (raw - mean) / rms
        train_x = x[train_mask]
        with torch.no_grad():
            _, vectors = torch.linalg.eigh(train_x.T @ train_x / len(train_x))
            basis = vectors[:, -64:].flip(1)
        np.savez(fold / "pca.npz", mean=mean.cpu().numpy(), rms=rms.cpu().numpy(), basis=basis.cpu().numpy())
        write_json(fold / "fit_scope.json", {"held_out_video": held_out, "fit_videos": sorted({row["video_id"] for row, use in zip(training_rows, fit_clips) if use}),
                   "n_fit_tokens": len(train_x), "n_validation_clips": int((~fit_clips).sum())})

        def compare(name, features):
            for decay in config["weight_decay_candidates"]:
                head_config = dict(config, head_weight_decay=decay)
                label = name + "_wd" + str(decay).replace(".", "p")
                predictions = fit_head(features, train_mask, labels, head_config, fold, label, torch).reshape(len(selected), 8, 196)
                held_predictions = predictions[~fit_clips]
                scores = np.sort(held_predictions, axis=-1)[..., -4:].mean(-1).mean(-1)
                metrics = binary_metrics(clip_labels[~fit_clips], scores)
                record = {"held_out_video": held_out, "method": name, "weight_decay": decay,
                          "n_validation_clips": len(scores), "clip_detection": metrics}
                records.append(record)
                np.savez(fold / (label + "_predictions.npz"), patch_probabilities=held_predictions,
                         clip_scores=scores, clip_labels=clip_labels[~fit_clips])
                write_json(fold / (label + "_metrics.json"), record)
                print(json.dumps(record), flush=True)

        compare("raw_endofm", x)
        compare("pca64", x @ basis)
        sae, history = train_dictionary(train_x, config, torch)
        codes = torch.empty((len(x), config["topk_width"]), device=device)
        with torch.no_grad():
            for offset in range(0, len(x), 4096):
                _, code = sae(x[offset:offset + 4096], inference=True)
                codes[offset:offset + len(code)] = code
        np.savez(fold / "topk_sae.npz", **{key: value.detach().cpu().numpy() for key, value in sae.state_dict().items()},
                 input_mean=mean.cpu().numpy(), input_rms=rms.cpu().numpy())
        write_json(fold / "topk_training.json", {"history": history, "fit_scope": "fold fitting videos only"})
        compare("topk32_target_domain", codes)
        del sae, codes, x, train_x, basis, train_mask
    candidates, winners = {}, {}
    for method in ("raw_endofm", "pca64", "topk32_target_domain"):
        candidates[method] = []
        for decay in config["weight_decay_candidates"]:
            results = [r for r in records if r["method"] == method and r["weight_decay"] == decay]
            candidates[method].append({"weight_decay": decay,
                "macro_video_auroc": float(np.mean([r["clip_detection"]["auroc"] for r in results])),
                "macro_video_ap": float(np.mean([r["clip_detection"]["average_precision"] for r in results]))})
        winners[method] = max(candidates[method], key=lambda c: (c["macro_video_auroc"], c["weight_decay"]))
    write_json(output / "summary.json", {"status": "COMPLETED_TRAINING_SIDE_SELECTION", "records": records,
        "candidates": candidates, "selected": winners, "selection_rule": config["selection_rule"],
        "elapsed_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "limitations": ["three training videos only; each fold fits two videos", "selection scores are not an unbiased estimate of selected-model performance",
                        "no exposed development or expanded development evaluation", "biased balanced clip sample and bounding-box support labels",
                        "shared training videos across folds; no confirmatory significance", "head regularization only, not a new SAE method"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    write_json(run / "status.json", {"status": "RUNNING", "pid": os.getpid(), "started_at_unix": time.time()})
    try:
        run_cv(run, config)
    except Exception as error:
        write_json(run / "status.json", {"status": "FAILED", "error": repr(error), "ended_at_unix": time.time()})
        raise
    write_json(run / "status.json", {"status": "COMPLETED", "ended_at_unix": time.time()})


if __name__ == "__main__":
    main()
