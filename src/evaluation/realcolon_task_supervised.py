"""Grouped task-supervision comparison on the existing REAL-Colon token cache.

The current consumer is the task-supervision run config. All representations,
normalizers and task heads are fitted inside each training fold. Joint BCE and
reconstruction is a baseline training recipe, not an algorithmic novelty claim.
"""
import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from src.evaluation.realcolon_task import digest, summarize, write_json
from src.evaluation.realcolon_task_cv import train_dictionary


METHODS = ("raw_balanced", "pca64_balanced", "topk_frozen_balanced",
           "topk_reconstruction_continued", "topk_task", "dense_task")


def fold_masks(rows, heldout_folds):
    videos = {r["video_id"] for r in rows}
    if any(r["split"] != "train" for r in rows):
        raise ValueError("CV accepts training rows only")
    flattened = [v for fold in heldout_folds for v in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != videos:
        raise ValueError("Each training video must be held out exactly once")
    for held in heldout_folds:
        mask = np.array([r["video_id"] not in held for r in rows])
        if not mask.any() or mask.all():
            raise ValueError("Empty fit or validation fold")
        yield mask


class SupportSampler:
    """Uniform videos, then clips, then eligible tokens; 50/25/25 percent.

    Half the task batch is positive-clip box support, one quarter positive-clip
    outside support, and one quarter negative-clip background. No evaluation
    clip can enter the sampling tables. Selection is with replacement.
    """
    def __init__(self, rows, masks, fit_clips, device, torch):
        self.torch = torch
        self.device = device
        self.fit_clip_indices = np.flatnonzero(fit_clips).tolist()
        if not self.fit_clip_indices:
            raise ValueError("No fitting clips")
        self.videos = sorted({rows[i]["video_id"] for i in self.fit_clip_indices})
        n, frames, patches = masks.shape
        self.tokens_per_clip = frames * patches
        groups = [[[i for i in self.fit_clip_indices if rows[i]["video_id"] == v and bool(rows[i]["class"]) == label]
                   for label in (False, True)] for v in self.videos]
        if any(not group for video in groups for group in video):
            raise ValueError("Each fitting video needs positive and negative clips")
        maximum = max(len(g) for video in groups for g in video)
        clips = np.zeros((len(groups), 2, maximum), dtype=np.int64)
        clip_counts = np.zeros((len(groups), 2), dtype=np.int64)
        eligible = np.zeros((n, 2, self.tokens_per_clip), dtype=np.int64)
        counts = np.zeros((n, 2), dtype=np.int64)
        for v, video in enumerate(groups):
            for label, group in enumerate(video):
                clips[v, label, :len(group)] = group
                clip_counts[v, label] = len(group)
        for i in self.fit_clip_indices:
            for category, selection in enumerate((masks[i].reshape(-1), ~masks[i].reshape(-1))):
                indices = np.flatnonzero(selection)
                if not len(indices) and (category == 1 or rows[i]["class"]):
                    raise ValueError("Requested fitting clip has empty token support")
                eligible[i, category, :len(indices)] = indices + i * self.tokens_per_clip
                counts[i, category] = len(indices)
        self.clips = torch.tensor(clips, device=device)
        self.clip_counts = torch.tensor(clip_counts, device=device)
        self.eligible = torch.tensor(eligible, device=device)
        self.counts = torch.tensor(counts, device=device)

    def sample(self, batch_size, generator):
        torch = self.torch
        if batch_size % 4:
            raise ValueError("Task batch size must be divisible by four")
        category = torch.tensor([0, 0, 1, 2], device=self.device).repeat(batch_size // 4)
        video = torch.randint(len(self.videos), (batch_size,), device=self.device, generator=generator)
        positive_clip = (category < 2).long()
        offset = (torch.rand(batch_size, device=self.device, generator=generator) * self.clip_counts[video, positive_clip]).long()
        clip = self.clips[video, positive_clip, offset]
        outside = (category > 0).long()
        token = (torch.rand(batch_size, device=self.device, generator=generator) * self.counts[clip, outside]).long()
        return self.eligible[clip, outside, token], (category == 0).float()


def feature_stats(x, fit_indices, transform, torch):
    """Exact training-token moments, streamed so dense codes are never cached."""
    total, square, n = None, None, 0
    with torch.no_grad():
        for indices in fit_indices.split(4096):
            features = transform(x[indices]).double()
            if total is None:
                total, square = features.sum(0), features.square().sum(0)
            else:
                total += features.sum(0)
                square += features.square().sum(0)
            n += len(features)
    mean = total / n
    scale = (square / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    return mean.float(), scale.float()


def save_state(path, module, **extra):
    np.savez(path, **{k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}, **extra)


def candidates(method, config):
    key = "mse_weight" if method in ("topk_task", "dense_task") else "head_weight_decay"
    values = config["mse_weight_candidates"] if key == "mse_weight" else config["weight_decay_candidates"]
    return [{key: value} for value in values]


def fit_one(method, hyper, x, fit_indices, sampler, references, stats, config, output, torch):
    joint = method in ("topk_task", "dense_task", "topk_reconstruction_continued")
    model = None
    if method == "raw_balanced":
        transform = lambda value: value
        stat_name = "raw"
    elif method == "pca64_balanced":
        transform = lambda value: value @ references["pca"]
        stat_name = "pca"
    else:
        stat_name = "dense" if method == "dense_task" else "topk"
        model = copy.deepcopy(references[stat_name]) if joint else references[stat_name]
        transform = model.encode_inference
    mean, scale = stats[stat_name]
    torch.manual_seed(config["seed"])
    head = torch.nn.Linear(len(mean), 1).to(x.device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    parameters = [{"params": head.parameters(), "lr": config["head_learning_rate"],
                   "weight_decay": hyper.get("head_weight_decay", config["joint_head_weight_decay"])}]
    if joint:
        parameters.append({"params": model.parameters(), "lr": config["finetune_learning_rate"], "weight_decay": 0.})
    optimizer = torch.optim.AdamW(parameters)
    generator = torch.Generator(device=x.device).manual_seed(config["seed"])
    reconstruction_generator = torch.Generator(device=x.device).manual_seed(config["seed"] + 1)
    history = []
    for step in range(config["head_steps"]):
        indices, labels = sampler.sample(config["head_batch_size"], generator)
        if joint and method != "topk_reconstruction_continued":
            features = transform(x[indices])
        else:
            with torch.no_grad():
                features = transform(x[indices])
        logits = head((features - mean) / scale).squeeze(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss = bce
        mse = x.new_tensor(0.)
        if joint:
            selected = torch.randint(len(fit_indices), (config["topk_batch_size"],), device=x.device, generator=reconstruction_generator)
            batch = x[fit_indices[selected]]
            reconstruction, _ = model(batch, inference=False)
            mse = (reconstruction - batch).square().mean()
            loss = loss + hyper.get("mse_weight", 1.) * mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if joint:
            model.project_decoder_gradient_()
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.)
        optimizer.step()
        if joint:
            model.normalize_decoder_()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite task loss")
        if step % 200 == 0 or step + 1 == config["head_steps"]:
            history.append({"step": step + 1, "bce": float(bce.detach()), "mse": float(mse.detach())})
    predictions, l0_sum, error_sum, norm_sum = [], 0., 0., 0.
    with torch.no_grad():
        for batch in x.split(4096):
            features = transform(batch)
            predictions.append(torch.sigmoid(head((features - mean) / scale)).squeeze(-1).cpu().numpy())
            if model is not None:
                l0_sum += float((features != 0).sum())
                error_sum += float((model.decode(features) - batch).square().sum())
                norm_sum += float(batch.square().sum())
    save_state(output / "head.npz", head, feature_mean=mean.cpu().numpy(), feature_scale=scale.cpu().numpy())
    if joint:
        save_state(output / "dictionary.npz", model)
    write_json(output / "training.json", {"method": method, "hyperparameters": hyper, "history": history,
        "sampling": "video then clip then token; 50% positive support / 25% positive outside / 25% negative clip",
        "feature_normalization": "fixed initial representation statistics over fitting tokens only",
        "dictionary_receives_bce_gradient": method in ("topk_task", "dense_task"),
        "reconstruction_sampling": "uniform fitting tokens, independent of supervised sampler",
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "dictionary_parameters": sum(p.numel() for p in model.parameters()) if model is not None else 0,
        "all_included_tokens_l0": l0_sum / len(x) if model is not None else None,
        "all_included_tokens_nmse": error_sum / norm_sum if norm_sum else None})
    return np.concatenate(predictions)


def run_fold(output, raw, masks, rows, fit_clips, config, selections, torch):
    output.mkdir(exist_ok=False)
    fit_indices = torch.tensor(np.flatnonzero(np.repeat(fit_clips, masks.shape[1] * masks.shape[2])), device=raw.device)
    fit_raw = raw[fit_indices]
    mean = fit_raw.mean(0)
    rms = ((fit_raw - mean).square().mean()).sqrt().clamp_min(1e-6)
    del fit_raw
    x = (raw - mean) / rms
    train_x = x[fit_indices]
    with torch.no_grad():
        _, vectors = torch.linalg.eigh(train_x.T @ train_x / len(train_x))
        basis = vectors[:, -64:].flip(1)
    references = {"pca": basis}
    for label, k in (("topk", config["topk_k"]), ("dense", config["topk_width"])):
        dictionary, history = train_dictionary(train_x, dict(config, topk_k=k), torch)
        references[label] = dictionary
        save_state(output / (label + "_initial.npz"), dictionary)
        write_json(output / (label + "_pretraining.json"), {"history": history, "k": k})
    del train_x
    np.savez(output / "input_normalization_pca.npz", mean=mean.cpu().numpy(), rms=rms.cpu().numpy(), basis=basis.cpu().numpy())
    sampler = SupportSampler(rows, masks, fit_clips, x.device, torch)
    stats = {"raw": feature_stats(x, fit_indices, lambda v: v, torch),
             "pca": feature_stats(x, fit_indices, lambda v: v @ basis, torch)}
    for label in ("topk", "dense"):
        stats[label] = feature_stats(x, fit_indices, references[label].encode_inference, torch)
    evaluation_rows = [dict(row, split="train" if use else "validation") for row, use in zip(rows, fit_clips)]
    held = sorted({r["video_id"] for r, use in zip(rows, fit_clips) if not use})
    write_json(output / "fit_scope.json", {"fit_videos": sampler.videos, "validation_videos": held,
        "fit_clip_indices": sampler.fit_clip_indices, "all_rows": evaluation_rows})
    records = []
    for method in METHODS:
        for number, hyper in enumerate(selections[method]):
            path = output / (method + "_candidate" + str(number))
            path.mkdir()
            predictions = fit_one(method, hyper, x, fit_indices, sampler, references, stats, config, path, torch).reshape(masks.shape)
            metrics = summarize(predictions, masks, evaluation_rows)
            np.save(path / "predictions.npy", predictions, allow_pickle=False)
            write_json(path / "metrics.json", metrics)
            record = {"method": method, "hyperparameters": hyper, "path": path.as_posix(),
                      "validation": {v: metrics["per_video"][v] for v in held}}
            records.append(record)
            print(json.dumps({"method": method, "hyperparameters": hyper,
                "video_auroc": {v: metrics["per_video"][v]["clip_detection"]["auroc"] for v in held}}), flush=True)
    return records


def execute(run, config, phase):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if phase == "smoke":
        config = dict(config, head_steps=12, topk_steps=12)
    source = Path(config["source_run"])
    manifest = source / "clip_manifest.jsonl"
    if digest(manifest) != config["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    if digest(source / "encoded.json") != config["source_encoded_sha256"]:
        raise RuntimeError("Source encoding provenance changed")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    selected = [i for i, row in enumerate(rows) if phase == "fit" or row["split"] == "train"]
    rows = [rows[i] for i in selected]
    cache = Path(config["cache_dir"])
    for name, identity in config["cache_file_identity"].items():
        stat = (cache / name).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise RuntimeError("Previously verified cache file changed: " + name)
    # Do not materialize any development array during training-only selection.
    tokens = np.load(cache / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != tuple(config["source_token_shape"]):
        raise RuntimeError("Token cache shape changed")
    raw = torch.tensor(tokens[selected].reshape(-1, tokens.shape[-1]), device="cuda")
    masks = np.load(cache / "masks.npy", mmap_mode="r", allow_pickle=False)[selected]
    output = run / phase
    output.mkdir(exist_ok=False)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name in ("realcolon_task.py", "realcolon_task_cv.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(output / "config.json", config)
    write_json(output / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__})
    if phase in ("cv", "smoke"):
        selections = {m: candidates(m, config) for m in METHODS}
        folds = list(fold_masks(rows, config["heldout_folds"]))
        if phase == "smoke":
            selections = {m: value[:1] for m, value in selections.items()}
            folds = folds[:1]
    else:
        selection = json.loads((run / "cv/summary.json").read_text())
        selections = {m: [selection["selected"][m]["hyperparameters"]] for m in METHODS}
        folds = [np.array([r["split"] == "train" for r in rows])]
    records = []
    for index, fit_clips in enumerate(folds):
        print("FOLD {}/{}".format(index + 1, len(folds)), flush=True)
        records.extend(run_fold(output / ("fold" + str(index)), raw, masks, rows, fit_clips, config, selections, torch))
    aggregate, winners = {}, {}
    for method in METHODS:
        aggregate[method] = []
        for hyper in selections[method]:
            videos = [v for r in records if r["method"] == method and r["hyperparameters"] == hyper
                      for video, v in r["validation"].items() if phase != "fit" or video in config["primary_development_videos"]]
            item = {"hyperparameters": hyper, "macro_video_auroc": float(np.mean([v["clip_detection"]["auroc"] for v in videos])),
                    "macro_positive_frame_patch_ap": float(np.mean([v["positive_frame_mean_patch_ap"] for v in videos]))}
            aggregate[method].append(item)
        # First-listed candidate is the predeclared deterministic tie breaker.
        winners[method] = max(aggregate[method], key=lambda a: a["macro_video_auroc"])
    write_json(output / "summary.json", {"status": "COMPLETED_" + phase.upper(), "records": records,
        "candidates": aggregate, "selected": winners, "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "limitations": config["limitations"],
        "primary_aggregate_scope": config["primary_development_videos"] if phase == "fit" else "held-out training videos; smoke is non-research execution only"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "cv", "fit"))
    parser.add_argument("--run-dir", required=True, type=Path)
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
