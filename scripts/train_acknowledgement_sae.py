import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.acknowledgement_sae import (
    METHODS, AcknowledgementPredictor, AcknowledgementSAE, file_sha256,
    load_predictor, supervised_contrastive_loss,
)
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json, sharing_retry


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    sharing_retry(lambda: os.replace(temporary, path))


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    sharing_retry(lambda: os.replace(temporary, path))


def load_inputs(config):
    train_videos = list(config["train_video_ids"])
    validation_videos = list(config["validation_video_ids"])
    if len(train_videos) != len(set(train_videos)) or len(validation_videos) != len(set(validation_videos)):
        raise ValueError("Duplicate procedure IDs")
    if set(train_videos) & set(validation_videos):
        raise ValueError("Training and validation procedures overlap")
    records, arrays, receipts = [], [], []
    descriptor_root = Path(config["descriptor_root"])
    for video in train_videos + validation_videos:
        directory = descriptor_root / video
        receipt = read_json(directory / "complete.json")
        if receipt["status"] != "COMPLETE" or receipt["video_id"] != video:
            raise ValueError(f"Incomplete descriptor export: {video}")
        descriptor_hash = file_sha256(directory / "descriptors.npz")
        record_hash = file_sha256(directory / "records.json")
        if descriptor_hash != receipt["descriptor_sha256"] or record_hash != receipt["record_sha256"]:
            raise ValueError(f"Descriptor identity mismatch: {video}")
        local = read_json(directory / "records.json")
        expected_split = "train" if video in train_videos else "val"
        if not local or any(row["video_id"] != video or row["split"] != expected_split for row in local):
            raise ValueError(f"Unexpected or exposed test split in descriptor records: {video}")
        with np.load(directory / "descriptors.npz", allow_pickle=False) as archive:
            values = archive["raw"].copy()
        if values.shape != (len(local), int(config["input_dim"])) or not np.isfinite(values).all():
            raise ValueError(f"Invalid descriptor matrix: {video}")
        for row in local:
            if row["start_frame"] > row["end_frame"]:
                raise ValueError(f"Invalid chronological interval: {video}")
            row = dict(row)
            row["global_index"] = len(records)
            row["partition"] = expected_split
            records.append(row)
        arrays.append(values)
        receipts.append(dict(video_id=video, split=expected_split, descriptors=len(local),
                             descriptor_sha256=descriptor_hash, records_sha256=record_hash,
                             directory=str(directory.resolve())))
    identity_sets = [{(r["video_id"], r["lesion_id"]) for r in records if r["partition"] == split}
                     for split in ("train", "val")]
    if identity_sets[0] & identity_sets[1]:
        raise ValueError("Lesion identity leakage")
    return np.concatenate(arrays), records, receipts


def chronological_pairs(records, selected_videos):
    selected_videos = set(selected_videos)
    pairs = []
    for source_index, source in enumerate(records):
        if source["video_id"] not in selected_videos:
            continue
        for query_index, query in enumerate(records):
            if query["video_id"] != source["video_id"] or query["start_frame"] <= source["end_frame"]:
                continue
            same = source["lesion_id"] == query["lesion_id"]
            pairs.append(dict(
                source_index=source_index, query_index=query_index,
                video_id=source["video_id"], source_lesion_id=source["lesion_id"],
                query_lesion_id=query["lesion_id"], source_clip_id=source["clip_id"],
                query_clip_id=query["clip_id"], same_identity=same,
                source_end_frame=int(source["end_frame"]), query_start_frame=int(query["start_frame"]),
                previously_seen_other=bool(not same and query["lesion_first_frame"] <= source["end_frame"]),
                same_annotation_interval=bool(same and source["annotation_observation"]["interval_id"]
                                              == query["annotation_observation"]["interval_id"]),
            ))
    return pairs


def episode_bank(records, videos):
    groups = {}
    for pair in chronological_pairs(records, videos):
        source = pair["source_index"]
        group = groups.setdefault(source, dict(source=source, video_id=pair["video_id"], positive=[], negative=[]))
        group["positive" if pair["same_identity"] else "negative"].append(pair["query_index"])
    bank = [value for value in groups.values() if value["positive"] and value["negative"]]
    if not bank:
        raise ValueError("No training source has both a future same-identity and different-identity observation")
    return bank


def make_folds(records, train_videos, count, seed):
    if count < 2 or count > len(train_videos):
        raise ValueError("Grouped inner fold count must be between two and the training procedure count")
    rng = np.random.default_rng(seed)
    ordering = list(rng.permutation(sorted(train_videos)))
    ordering.sort(key=lambda v: -len({r["lesion_id"] for r in records if r["video_id"] == v}))
    folds = [[] for _ in range(count)]
    sizes = [0] * count
    for video in ordering:
        destination = min(range(count), key=lambda index: (sizes[index], len(folds[index]), index))
        folds[destination].append(str(video))
        sizes[destination] += sum(r["video_id"] == video for r in records)
    for fold in folds:
        pairs = chronological_pairs(records, fold)
        usable = [v for v in fold if len({p["same_identity"] for p in pairs if p["video_id"] == v}) == 2]
        if not usable:
            raise ValueError("An inner fold has no evaluable within-procedure identity comparison")
        episode_bank(records, set(train_videos) - set(fold))
    return folds


def sample_batch(bank, reconstruction_indices, rng, config):
    by_video = {}
    for episode in bank:
        by_video.setdefault(episode["video_id"], []).append(episode)
    video_ids = sorted(by_video)
    source, positive, negative = [], [], []
    for _ in range(int(config["batch_episodes"])):
        video = video_ids[int(rng.integers(len(video_ids)))]
        episode = by_video[video][int(rng.integers(len(by_video[video])))]
        source.append(episode["source"])
        positive.append(int(rng.choice(episode["positive"])))
        negative.append(int(rng.choice(episode["negative"])))
    reconstruction = rng.choice(reconstruction_indices, size=min(int(config["reconstruction_batch"]),
                                                               len(reconstruction_indices)), replace=False)
    indices = np.unique(np.concatenate([source, positive, negative, reconstruction])).astype(np.int64)
    lookup = {int(value): i for i, value in enumerate(indices)}
    return dict(indices=indices, source=np.array([lookup[i] for i in source]),
                positive=np.array([lookup[i] for i in positive]),
                negative=np.array([lookup[i] for i in negative]),
                reconstruction=np.array([lookup[int(i)] for i in reconstruction]))


def pair_statistics(pairs, scores):
    by_video = {}
    for video in sorted({pair["video_id"] for pair in pairs}):
        indices = [i for i, pair in enumerate(pairs) if pair["video_id"] == video]
        labels = np.array([pairs[i]["same_identity"] for i in indices], dtype=np.int64)
        values = scores[indices]
        valid = len(np.unique(labels)) == 2
        by_video[video] = dict(pairs=len(indices), positives=int(labels.sum()),
                              negatives=int((1 - labels).sum()),
                              previously_seen_negative_pairs=sum(pairs[i]["previously_seen_other"] for i in indices),
                              auroc=float(roc_auc_score(labels, values)) if valid else None,
                              average_precision=float(average_precision_score(labels, values)) if valid else None)
    aucs = [row["auroc"] for row in by_video.values() if row["auroc"] is not None]
    return dict(macro_procedure_auroc=float(np.mean(aucs)) if aucs else None,
                evaluable_procedures=len(aucs), by_video=by_video)


@torch.no_grad()
def evaluate_pairs(model, features, pairs):
    codes = model.encode(features)
    if not torch.isfinite(codes).all() or torch.any(torch.linalg.vector_norm(codes, dim=-1) <= 0):
        raise ValueError("Evaluation found an invalid latent code")
    source = torch.tensor([pair["source_index"] for pair in pairs], device=features.device)
    query = torch.tensor([pair["query_index"] for pair in pairs], device=features.device)
    scores = model.pair_scores(codes[source], codes[query]).cpu().numpy()
    report = pair_statistics(pairs, scores)
    report["pairs"] = [dict(pair, score=float(score)) for pair, score in zip(pairs, scores)]
    return report


def model_config(config, method):
    return {**{key: config[key] for key in ("input_dim", "latent_dim", "top_k", "memory_k",
                                          "gate_hidden", "temperature")}, "method": method}


def training_job(config, method, seed, raw, records, fit_videos, held_videos, directory,
                 steps, checkpoints, identity, context, resume, stop_after_step):
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    fit_identity = dict(configuration_sha256=json_digest(config), method=method, seed=seed,
                        fit_video_ids=sorted(fit_videos), held_video_ids=sorted(held_videos),
                        steps=steps, checkpoints=checkpoints, **identity)
    identity_hash = json_digest(fit_identity)
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary["identity_sha256"] != identity_hash or summary["status"] != "COMPLETE":
            raise ValueError(f"Existing job identity differs: {directory}")
        if not resume:
            raise FileExistsError(f"Completed job exists; use --resume to reuse it: {directory}")
        return summary
    identity_path = directory / "identity.json"
    if identity_path.exists() and read_json(identity_path) != fit_identity:
        raise ValueError(f"Refusing to reuse a different job: {directory}")
    atomic_write_json(identity_path, fit_identity)
    spec = model_config(config, method)
    atomic_write_json(directory / "model_config.json", spec)
    train_indices = np.array([i for i, row in enumerate(records) if row["video_id"] in fit_videos])
    if len(train_indices) == 0:
        raise ValueError("No training descriptors")
    scaler = StandardScaler().fit(raw[train_indices])
    save_npz(directory / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_,
             n_samples_seen=scaler.n_samples_seen_)
    device = torch.device(config["device"])
    features = torch.from_numpy(scaler.transform(raw).astype(np.float32)).to(device)
    labels_by_identity = {key: i for i, key in enumerate(sorted({(r["video_id"], r["lesion_id"]) for r in records}))}
    labels = torch.tensor([labels_by_identity[(r["video_id"], r["lesion_id"])] for r in records], device=device)
    bank = episode_bank(records, fit_videos)
    held_pairs = chronological_pairs(records, held_videos)
    fit_pairs = chronological_pairs(records, fit_videos)
    atomic_write_json(directory / "training_episodes.json", bank)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AcknowledgementSAE(spec).to(device)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    history, evaluations, sequence = [], [], []
    checkpoint_path = directory / "checkpoint.pt"
    first_step = 1
    elapsed_before = 0.0
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"Checkpoint exists; use --resume: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload["identity_sha256"] != identity_hash:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        rng.bit_generator.state = json.loads(payload["numpy_rng_json"])
        history = payload["history"]
        evaluations = payload["evaluations"]
        sequence = payload["sequence"]
        first_step = int(payload["step"]) + 1
        elapsed_before = float(payload["elapsed_seconds"])
    started = time.perf_counter()
    for step in range(first_step, steps + 1):
        model.train()
        batch = sample_batch(bank, train_indices, rng, config)
        indices = batch["indices"]
        sequence.append({key: value.tolist() for key, value in batch.items()})
        codes = model.encode(features[indices])
        reconstruction = F.mse_loss(model.decoder(codes[batch["reconstruction"]]),
                                    features[indices[batch["reconstruction"]]])
        identity_loss = supervised_contrastive_loss(codes, labels[indices], float(config["temperature"]))
        positive_logits = model.pair_logits(codes[batch["source"]], codes[batch["positive"]])
        negative_logits = model.pair_logits(codes[batch["source"]], codes[batch["negative"]])
        episode_loss = 0.5 * (F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean())
        loss = float(config["reconstruction_weight"]) * reconstruction + float(config["identity_weight"]) * identity_loss
        if method != "sae_identity":
            loss = loss + float(config["episode_weight"]) * episode_loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                raise ValueError(f"Missing or invalid gradient: {name}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("Training has no finite nonzero gradient")
        optimizer.step()
        model.normalize_decoder()
        row = dict(step=step, total_loss=float(loss.detach()), reconstruction_loss=float(reconstruction.detach()),
                   identity_loss=float(identity_loss.detach()), episode_loss=float(episode_loss.detach()),
                   gradient_norm=gradient_norm, unique_descriptors=len(indices),
                   active_components=float((codes > 0).sum(dim=-1).float().mean().detach()),
                   source_components=float((model.memory(codes[batch["source"]]) > 0).sum(dim=-1).float().mean().detach()))
        history.append(row)
        if step in checkpoints or step == steps:
            model.eval()
            evaluation = evaluate_pairs(model, features, held_pairs)
            atomic_write_json(directory / f"held_step{step:04d}.json", evaluation)
            evaluations.append(dict(step=step, macro_procedure_auroc=evaluation["macro_procedure_auroc"],
                                    evaluable_procedures=evaluation["evaluable_procedures"]))
        should_save = (step % int(config["checkpoint_every"]) == 0 or step in checkpoints
                       or step == steps or (stop_after_step is not None and step >= stop_after_step))
        if should_save:
            elapsed = elapsed_before + time.perf_counter() - started
            save_torch(checkpoint_path, dict(identity_sha256=identity_hash, step=step,
                                            model=model.state_dict(), optimizer=optimizer.state_dict(),
                                            torch_rng=torch.get_rng_state(), numpy_rng_json=json.dumps(rng.bit_generator.state),
                                            history=history, evaluations=evaluations, sequence=sequence,
                                            elapsed_seconds=elapsed))
            progress = dict(context, status="RUNNING", method=method, seed=seed, step=step, steps=steps,
                            elapsed_seconds=elapsed, losses=row, updated_at=now(), checkpoint=str(checkpoint_path.resolve()))
            atomic_write_json(directory / "progress.json", progress)
            atomic_write_json(Path(context["progress_path"]), progress)
            print(json.dumps({key: progress[key] for key in ("status", "method", "seed", "fold", "step", "steps", "elapsed_seconds", "losses")}), flush=True)
            pause_after_checkpoint(checkpoint_path)
            if stop_after_step is not None and step >= stop_after_step:
                progress["status"] = "PAUSED"
                atomic_write_json(directory / "progress.json", progress)
                atomic_write_json(Path(context["progress_path"]), progress)
                raise SystemExit(75)
    model.eval()
    save_npz(directory / "model.npz", **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    atomic_write_json(directory / "history.json", history)
    atomic_write_json(directory / "sequence.json", sequence)
    train_report = evaluate_pairs(model, features, fit_pairs)
    held_report = evaluate_pairs(model, features, held_pairs)
    atomic_write_json(directory / "selected_train_pairs.json", train_report)
    atomic_write_json(directory / "selected_held_pairs.json", held_report)
    with torch.no_grad():
        codes = model.encode(features[train_indices])
        nmse = float(F.mse_loss(model.decoder(codes), features[train_indices]) /
                     features[train_indices].square().mean())
        component_changes = {prefix: float(sum((value - initial[name]).square().sum()
                                               for name, value in model.state_dict().items()
                                               if name.startswith(prefix)).sqrt())
                             for prefix in (["encoder", "decoder", "gate"] if model.gate is not None else ["encoder", "decoder"])}
    if any(value <= 0 for value in component_changes.values()):
        raise ValueError("An intended trainable model component did not change")
    predictor = load_predictor(directory, config["device"])
    direct = AcknowledgementPredictor(model, scaler.mean_, scaler.scale_, config["device"])
    example_source = raw[bank[0]["source"]]
    expected = direct.score(example_source, raw[train_indices])
    actual = predictor.score(example_source, raw[train_indices])
    if not np.array_equal(expected, actual):
        raise ValueError("Exported model does not exactly reproduce the in-memory predictor")
    contributions = predictor.component_contributions(example_source, raw[train_indices])
    if not np.allclose(contributions.sum(axis=1), actual, rtol=1e-5, atol=1e-6):
        raise ValueError("Component contributions do not reproduce matching scores")
    summary = dict(status="COMPLETE", identity_sha256=identity_hash, method=method, seed=seed,
                   steps=steps, evaluations=evaluations, elapsed_seconds=elapsed_before + time.perf_counter() - started,
                   training_procedures=len(fit_videos), training_descriptors=len(train_indices),
                   training_episode_sources=len(bank), held_procedures=len(held_videos),
                   reconstruction_nmse=nmse, parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   component_parameter_change_l2=component_changes, model_reload_exact=True,
                   contribution_sum_verified=True, model_sha256=file_sha256(directory / "model.npz"),
                   normalization_sha256=file_sha256(directory / "normalization.npz"),
                   sequence_sha256=file_sha256(directory / "sequence.json"),
                   final_held_macro_procedure_auroc=held_report["macro_procedure_auroc"],
                   runtime=dict(torch=str(torch.__version__), numpy=np.__version__, sklearn=sklearn.__version__,
                                device=config["device"], threads=int(config["threads"])), completed_at=now())
    atomic_write_json(summary_path, summary)
    atomic_write_json(directory / "progress.json", dict(context, status="COMPLETE", method=method, seed=seed,
                                                       step=steps, steps=steps, updated_at=now()))
    return summary


def run_training(config, run, phase, selected_method, resume, stop_after_step):
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    if config["device"] != "cpu" and not str(config["device"]).startswith("cuda"):
        raise ValueError("Training device must be cpu or cuda")
    raw, records, receipts = load_inputs(config)
    methods = list(config["methods"]) if selected_method == "all" else [selected_method]
    if any(method not in METHODS for method in methods):
        raise ValueError("Unknown configured method")
    seeds = list(config["seeds"])
    output = run / "smoke" if phase == "smoke" else run
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "training_progress.json"
    identity = dict(input_identity_sha256=json_digest(receipts), source_hashes={
        "trainer": file_sha256(__file__), "model": file_sha256(ROOT / "src" / "acknowledgement_sae.py"),
        "checkpoint_io": file_sha256(ROOT / "src" / "checkpoint_io.py")})
    atomic_write_json(output / "training_config.json", config)
    atomic_write_json(output / "descriptor_inputs.json", receipts)
    atomic_write_json(output / "descriptor_records.json", records)
    train_videos, validation_videos = config["train_video_ids"], config["validation_video_ids"]
    if phase == "smoke":
        seeds = seeds[:1]
        folds = []
        steps = int(config.get("smoke_steps", 6))
        checkpoints = [steps]
    else:
        folds = make_folds(records, train_videos, int(config["inner_folds"]), int(config.get("fold_seed", 20260928)))
        steps = int(config["steps"])
        checkpoints = [int(step) for step in config["checkpoints"]]
        if checkpoints != sorted(set(checkpoints)) or not checkpoints or checkpoints[-1] != steps:
            raise ValueError("Checkpoints must be increasing, unique, and finish at configured steps")
    atomic_write_json(output / "grouped_folds.json", dict(
        training_procedures=train_videos, validation_procedures=validation_videos, inner_held_folds=folds,
        selection="Mean inner-fold procedure AUROC; earliest checkpoint wins ties. Official validation excluded.",
        test_access="No test inputs accepted by this trainer."))
    total_jobs = len(methods) * len(seeds) * (len(folds) + 1)
    completed_jobs = 0
    summaries, selection = [], []
    for seed in seeds:
        for method in methods:
            fold_summaries = []
            for fold_index, held_videos in enumerate(folds):
                fit_videos = sorted(set(train_videos) - set(held_videos))
                directory = output / "inner_folds" / method / f"seed{seed}" / f"fold{fold_index}"
                context = dict(phase=phase, fold=fold_index, progress_path=str(progress_path.resolve()),
                               completed_jobs=completed_jobs, total_jobs=total_jobs)
                summary = training_job(config, method, seed, raw, records, fit_videos, held_videos, directory,
                                       steps, checkpoints, identity, context, resume, stop_after_step)
                fold_summaries.append(summary)
                completed_jobs += 1
            if fold_summaries:
                candidates = []
                for checkpoint_step in checkpoints:
                    values = [next(row["macro_procedure_auroc"] for row in summary["evaluations"]
                                   if row["step"] == checkpoint_step) for summary in fold_summaries]
                    if any(value is None for value in values):
                        raise ValueError("Inner model selection received an undefined procedure AUROC")
                    candidates.append(dict(step=checkpoint_step, mean_inner_fold_auroc=float(np.mean(values)),
                                           fold_aurocs=values))
                selected = max(candidates, key=lambda row: (row["mean_inner_fold_auroc"], -row["step"]))
                selected_steps = selected["step"]
            else:
                candidates, selected_steps = [], steps
            selection.append(dict(method=method, seed=seed, selected_steps=selected_steps, candidates=candidates))
            atomic_write_json(output / "checkpoint_selection.json", selection)
            atomic_write_json(output / f"checkpoint_selection_{method}.json",
                              [item for item in selection if item["method"] == method])
            directory = output / "fit" / method / f"seed{seed}"
            context = dict(phase=phase, fold="full_training", progress_path=str(progress_path.resolve()),
                           completed_jobs=completed_jobs, total_jobs=total_jobs)
            summary = training_job(config, method, seed, raw, records, train_videos, validation_videos, directory,
                                   selected_steps, [selected_steps], identity, context, resume, stop_after_step)
            summaries.append(dict(method=method, seed=seed, fit_directory=str(directory.resolve()), summary=summary))
            completed_jobs += 1
            atomic_write_json(progress_path, dict(context, status="RUNNING", method=method, seed=seed,
                                                  completed_jobs=completed_jobs, step=selected_steps,
                                                  steps=selected_steps, updated_at=now()))
    for seed in seeds:
        matching = [row["summary"] for row in summaries if row["seed"] == seed]
        by_steps = {}
        for summary in matching:
            by_steps.setdefault(summary["steps"], []).append(summary["sequence_sha256"])
        if any(len(set(hashes)) != 1 for hashes in by_steps.values()):
            raise ValueError("Matched methods with the same seed and steps did not receive identical batches")
    result = dict(status="COMPLETE", phase=phase, completed_jobs=completed_jobs, total_jobs=total_jobs,
                  outputs=summaries, checkpoint_selection=selection, completed_at=now(), **identity)
    for method in methods:
        atomic_write_json(output / f"training_summary_{method}.json", dict(
            result, outputs=[item for item in summaries if item["method"] == method],
            checkpoint_selection=[item for item in selection if item["method"] == method]))
    atomic_write_json(output / "training_summary.json", result)
    atomic_write_json(progress_path, dict(status="COMPLETE", phase=phase, completed_jobs=completed_jobs,
                                          total_jobs=total_jobs, updated_at=now()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "fit", "status"))
    parser.add_argument("--method", default="all", choices=("all",) + METHODS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    run = Path(config.get("run_dir", config_path.parent))
    if args.phase == "status":
        print(json.dumps(read_json(run / "training_progress.json"), indent=2))
        return
    result = run_training(config, run, args.phase, args.method, args.resume, args.stop_after_step)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs")}))


if __name__ == "__main__":
    main()
