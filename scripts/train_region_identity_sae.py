import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_acknowledgement_sae as shared
from src.acknowledgement_sae import file_sha256, supervised_contrastive_loss
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.region_identity_sae import METHODS, RegionIdentityPredictor, RegionIdentitySAE, load_predictor


def load_inputs(config):
    videos = list(config["train_video_ids"]) + list(config["validation_video_ids"])
    if len(set(videos)) != len(videos):
        raise ValueError("Procedure partitions contain duplicates or overlap")
    root = Path(config["views_root"])
    arrays, validity, records, receipts = [], [], [], []
    for video in videos:
        folder = root / video
        receipt = read_json(folder / "complete.json")
        if receipt["status"] != "COMPLETE":
            raise ValueError(f"Incomplete region views: {video}")
        local = read_json(folder / "records.json")
        values = np.load(folder / "views.npy", allow_pickle=False)
        valid = np.load(folder / "valid.npy", allow_pickle=False)
        if values.dtype != np.float64 or values.shape != (len(local), 7, int(config["input_dim"])):
            raise ValueError(f"Invalid region view dimensions or dtype: {video}")
        if valid.dtype != np.bool_ or valid.shape != values.shape[:2] or not valid[:, 0].all():
            raise ValueError(f"Invalid canonical region availability: {video}")
        if not np.isfinite(values[valid]).all() or not valid[:, 1:].any(axis=1).all():
            raise ValueError(f"Missing finite noncanonical views: {video}")
        partition = "train" if video in config["train_video_ids"] else "val"
        if not local or any(row["video_id"] != video or row["split"] != partition for row in local):
            raise ValueError(f"Unexpected record partition: {video}")
        for row in local:
            records.append(dict(row, global_index=len(records), partition=partition))
        arrays.append(values)
        validity.append(valid)
        hashes = {name: file_sha256(folder / name) for name in
                  ("views.npy", "valid.npy", "records.json", "identity.json", "complete.json")}
        if receipt["video"] != video or receipt["split"] != partition:
            raise ValueError(f"Region receipt partition differs: {video}")
        if any(hashes[name] != receipt["assets"][name] for name in ("views.npy", "valid.npy", "records.json")):
            raise ValueError(f"Region input hash differs from export receipt: {video}")
        if hashes["identity.json"] != receipt["identity_sha256"]:
            raise ValueError(f"Region export identity differs: {video}")
        receipts.append(dict(video_id=video, records=len(local), valid_views=int(valid.sum()),
                             directory=str(folder.resolve()), sha256=hashes))
    return np.concatenate(arrays), np.concatenate(validity), records, receipts


@torch.no_grad()
def evaluate(model, features, valid, records, videos, directory, stem):
    indices = np.array([i for i, record in enumerate(records) if record["video_id"] in videos])
    base, variant = np.where(valid[indices, 1:])
    base, variant = indices[base], variant + 1
    codes = model.encode(features[base, variant])
    prediction = model.decoder(codes)
    own = (prediction - features[base, variant]).square().mean(dim=-1).cpu().numpy()
    canonical = (prediction - features[base, 0]).square().mean(dim=-1).cpu().numpy()
    baseline = (features[base, variant] - features[base, 0]).square().mean(dim=-1).cpu().numpy()
    support_jaccard = None
    if model.method.startswith("sparse_"):
        reference_support = model.encode(features[base, 0]) > 0
        view_support = codes > 0
        union = (reference_support | view_support).sum(dim=-1)
        if torch.any(union <= 0):
            raise ValueError("Empty support in region stability evaluation")
        support_jaccard = ((reference_support & view_support).sum(dim=-1) / union).cpu().numpy()
    by_video = {}
    for video in sorted(videos):
        clip_indices = [i for i in indices if records[i]["video_id"] == video]
        clip_self = [float(own[base == i].mean()) for i in clip_indices]
        clip_canonical = [float(canonical[base == i].mean()) for i in clip_indices]
        by_video[video] = dict(base_clips=len(clip_indices), evaluated_views=int(np.isin(base, clip_indices).sum()),
                              self_mse=float(np.mean(clip_self)), canonical_mse=float(np.mean(clip_canonical)),
                              input_to_canonical_mse=float(np.mean([baseline[base == i].mean() for i in clip_indices])),
                              sparse_support_jaccard=float(np.mean([support_jaccard[base == i].mean() for i in clip_indices]))
                              if support_jaccard is not None else None)
    report = dict(macro_procedure_self_mse=float(np.mean([row["self_mse"] for row in by_video.values()])),
                  macro_procedure_canonical_mse=float(np.mean([row["canonical_mse"] for row in by_video.values()])),
                  macro_procedure_input_to_canonical_mse=float(np.mean([row["input_to_canonical_mse"] for row in by_video.values()])),
                  macro_procedure_sparse_support_jaccard=float(np.mean([row["sparse_support_jaccard"] for row in by_video.values()]))
                  if support_jaccard is not None else None,
                  by_video=by_video, aggregation="Mean valid noncanonical views within clip; mean clips within procedure; mean procedures.")
    identity = shared.evaluate_pairs(model, features[:, 0], shared.chronological_pairs(records, videos))
    report["clean_identity"] = identity
    error_arrays = dict(base_indices=base, view_indices=variant, self_mse=own, canonical_mse=canonical,
                        input_to_canonical_mse=baseline)
    if support_jaccard is not None:
        error_arrays["sparse_support_jaccard"] = support_jaccard
    shared.save_npz(directory / f"{stem}_errors.npz", **error_arrays)
    atomic_write_json(directory / f"{stem}.json", report)
    return report


def training_job(config, method, seed, views, valid, records, fit_videos, held_videos,
                 directory, steps, checkpoints, identity, context, resume, stop_after_step):
    directory.mkdir(parents=True, exist_ok=True)
    job_identity = dict(configuration_sha256=shared.json_digest(config), method=method, seed=seed,
                        fit_video_ids=sorted(fit_videos), held_video_ids=sorted(held_videos),
                        steps=steps, checkpoints=checkpoints, **identity)
    identity_hash = shared.json_digest(job_identity)
    if (directory / "summary.json").exists():
        summary = read_json(directory / "summary.json")
        if not resume or summary["status"] != "COMPLETE" or summary["identity_sha256"] != identity_hash:
            raise ValueError(f"Existing completed job cannot be reused: {directory}")
        return summary
    if (directory / "identity.json").exists() and read_json(directory / "identity.json") != job_identity:
        raise ValueError(f"Existing job identity differs: {directory}")
    atomic_write_json(directory / "identity.json", job_identity)
    spec = dict(method=method, **{key: config[key] for key in ("input_dim", "latent_dim", "top_k")})
    atomic_write_json(directory / "model_config.json", spec)
    train_indices = np.array([i for i, row in enumerate(records) if row["video_id"] in fit_videos])
    scaler = StandardScaler().fit(views[train_indices, 0])
    shared.save_npz(directory / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_,
                    var=scaler.var_, n_samples_seen=scaler.n_samples_seen_)
    features = torch.from_numpy(((views - scaler.mean_) / scaler.scale_).astype(np.float32))
    label_map = {key: i for i, key in enumerate(sorted({(r["video_id"], r["lesion_id"]) for r in records}))}
    labels = torch.tensor([label_map[(r["video_id"], r["lesion_id"])] for r in records])
    bank = shared.episode_bank(records, fit_videos)
    atomic_write_json(directory / "training_episodes.json", bank)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    view_rng = np.random.default_rng(np.random.SeedSequence([seed, 72491]))
    available_views = [np.flatnonzero(row) for row in valid]
    model = RegionIdentitySAE(spec)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    history, evaluations, sequence = [], [], []
    checkpoint = directory / "checkpoint.pt"
    first_step, elapsed_before = 1, 0.0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Use --resume for {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity_sha256"] != identity_hash:
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        rng.bit_generator.state = json.loads(saved["numpy_rng_json"])
        view_rng.bit_generator.state = json.loads(saved["view_rng_json"])
        history, evaluations, sequence = saved["history"], saved["evaluations"], saved["sequence"]
        first_step, elapsed_before = int(saved["step"]) + 1, float(saved["elapsed_seconds"])
    started = time.perf_counter()
    for step in range(first_step, steps + 1):
        model.train()
        batch = shared.sample_batch(bank, train_indices, rng, config)
        indices = batch["indices"]
        chosen_views = np.array([view_rng.choice(available_views[i]) for i in indices], dtype=np.int64)
        sequence.append(dict(**{key: value.tolist() for key, value in batch.items()}, views=chosen_views.tolist()))
        inputs = features[indices, chosen_views]
        targets = features[indices, 0] if method.endswith("_canonical") else inputs
        codes = model.encode(inputs)
        reconstruction = F.mse_loss(model.decoder(codes), targets)
        identity_loss = supervised_contrastive_loss(codes, labels[indices], float(config["temperature"]))
        loss = float(config["reconstruction_weight"]) * reconstruction + float(config["identity_weight"]) * identity_loss
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Missing or nonfinite model gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("No finite nonzero gradient")
        optimizer.step()
        model.normalize_decoder()
        row = dict(step=step, total_loss=float(loss.detach()), reconstruction_loss=float(reconstruction.detach()),
                   identity_loss=float(identity_loss.detach()), gradient_norm=gradient_norm,
                   base_clips=len(indices), canonical_inputs=int((chosen_views == 0).sum()),
                   active_components=float((codes > 0).sum(dim=-1).float().mean().detach()))
        history.append(row)
        if step in checkpoints or step == steps:
            model.eval()
            report = evaluate(model, features, valid, records, held_videos, directory, f"held_step{step:04d}")
            evaluations.append(dict(step=step, macro_procedure_canonical_mse=report["macro_procedure_canonical_mse"],
                                    macro_procedure_self_mse=report["macro_procedure_self_mse"],
                                    by_video=report["by_video"],
                                    clean_macro_procedure_auroc=report["clean_identity"]["macro_procedure_auroc"]))
        if (step % int(config["checkpoint_every"]) == 0 or step in checkpoints or step == steps
                or (stop_after_step is not None and step >= stop_after_step)):
            elapsed = elapsed_before + time.perf_counter() - started
            shared.save_torch(checkpoint, dict(identity_sha256=identity_hash, step=step, model=model.state_dict(),
                              optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                              numpy_rng_json=json.dumps(rng.bit_generator.state), view_rng_json=json.dumps(view_rng.bit_generator.state),
                              history=history, evaluations=evaluations, sequence=sequence, elapsed_seconds=elapsed))
            progress = dict(context, status="RUNNING", method=method, seed=seed, step=step, steps=steps,
                            elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / step * (steps - step),
                            losses=row, updated_at=shared.now(), checkpoint=str(checkpoint.resolve()))
            atomic_write_json(directory / "progress.json", progress)
            atomic_write_json(Path(context["progress_path"]), progress)
            print(json.dumps(progress), flush=True)
            pause_after_checkpoint(checkpoint)
            if stop_after_step is not None and step >= stop_after_step:
                progress["status"] = "PAUSED"
                atomic_write_json(directory / "progress.json", progress)
                atomic_write_json(Path(context["progress_path"]), progress)
                raise SystemExit(75)
    model.eval()
    shared.save_npz(directory / "model.npz", **{key: value.detach().numpy() for key, value in model.state_dict().items()})
    atomic_write_json(directory / "history.json", history)
    atomic_write_json(directory / "sequence.json", sequence)
    train_report = evaluate(model, features, valid, records, fit_videos, directory, "selected_train")
    held_report = evaluate(model, features, valid, records, held_videos, directory, "selected_held")
    changes = {prefix: float(sum((value - initial[name]).square().sum() for name, value in model.state_dict().items()
                                  if name.startswith(prefix)).sqrt()) for prefix in ("encoder", "decoder")}
    if any(change <= 0 for change in changes.values()):
        raise ValueError("An intended trainable component did not change")
    predictor = load_predictor(directory)
    direct = RegionIdentityPredictor(model, scaler.mean_, scaler.scale_)
    example = views[train_indices[:16], 0]
    if not np.array_equal(predictor.score(example[0], example), direct.score(example[0], example)):
        raise ValueError("Model export changed scores")
    if not np.array_equal(predictor.decode_raw(example), direct.decode_raw(example)):
        raise ValueError("Model export changed decoded raw descriptors")
    summary = dict(status="COMPLETE", identity_sha256=identity_hash, method=method, seed=seed, steps=steps,
                   evaluations=evaluations, elapsed_seconds=elapsed_before + time.perf_counter() - started,
                   training_procedures=len(fit_videos), training_base_clips=len(train_indices),
                   valid_training_views=int(valid[train_indices].sum()), held_procedures=len(held_videos),
                   training_episode_sources=len(bank), parameter_count=sum(p.numel() for p in model.parameters()),
                   component_parameter_change_l2=changes, model_reload_exact=True, decoded_reload_exact=True,
                   final_train_canonical_mse=train_report["macro_procedure_canonical_mse"],
                   final_held_canonical_mse=held_report["macro_procedure_canonical_mse"],
                   model_sha256=file_sha256(directory / "model.npz"),
                   normalization_sha256=file_sha256(directory / "normalization.npz"),
                   sequence_sha256=file_sha256(directory / "sequence.json"),
                   runtime=dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                                torch=str(torch.__version__), device="cpu", threads=1), completed_at=shared.now())
    atomic_write_json(directory / "summary.json", summary)
    atomic_write_json(directory / "progress.json", dict(context, status="COMPLETE", method=method, seed=seed,
                                                       step=steps, steps=steps, updated_at=shared.now()))
    return summary


def source_hashes():
    return {"trainer": file_sha256(__file__), "model": file_sha256(ROOT / "src" / "region_identity_sae.py"),
            "shared_trainer": file_sha256(shared.__file__),
            "predictor_and_loss": file_sha256(ROOT / "src" / "acknowledgement_sae.py"),
            "checkpoint_io": file_sha256(ROOT / "src" / "checkpoint_io.py")}


def run_training(config, run, phase, selected_method, resume, stop_after_step):
    if config["device"] != "cpu" or int(config["threads"]) != 1:
        raise ValueError("This paired experiment requires CPU with one thread")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    views, valid, records, receipts = load_inputs(config)
    methods = list(config["methods"]) if selected_method == "all" else [selected_method]
    if any(method not in METHODS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("Unknown or repeated method")
    seeds = list(config["seeds"])
    output = run / "smoke" if phase == "smoke" else run
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "training_progress.json"
    identity = dict(input_identity_sha256=shared.json_digest(receipts), source_hashes=source_hashes())
    atomic_write_json(output / "training_config.json", config)
    atomic_write_json(output / "view_inputs.json", receipts)
    atomic_write_json(output / "descriptor_records.json", records)
    train_videos, held_videos = config["train_video_ids"], config["validation_video_ids"]
    if phase == "smoke":
        seeds, folds = seeds[:1], []
        steps = int(config.get("smoke_steps", 6))
        checkpoints = [steps]
    else:
        folds = shared.make_folds(records, train_videos, int(config["inner_folds"]), int(config.get("fold_seed", 20260928)))
        steps, checkpoints = int(config["steps"]), list(config["checkpoints"])
        if not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != steps:
            raise ValueError("Checkpoints must be increasing, unique, and finish at configured steps")
    atomic_write_json(output / "grouped_folds.json", dict(training_procedures=train_videos,
        validation_procedures=held_videos, inner_held_folds=folds,
        selection="Minimum canonical MSE, equally weighted held procedures over all inner folds; earliest exact tie.",
        normalization="Canonical means from fit procedures only; valid noncanonical views used for selection."))
    total_jobs = len(methods) * len(seeds) * (len(folds) + 1)
    completed_jobs, summaries, selections = 0, [], []
    for seed in seeds:
        for method in methods:
            fold_summaries = []
            for fold, held in enumerate(folds):
                directory = output / "inner_folds" / method / f"seed{seed}" / f"fold{fold}"
                context = dict(phase=phase, fold=fold, progress_path=str(progress_path.resolve()),
                               completed_jobs=completed_jobs, total_jobs=total_jobs)
                fold_summaries.append(training_job(config, method, seed, views, valid, records,
                    sorted(set(train_videos) - set(held)), held, directory, steps, checkpoints,
                    identity, context, resume, stop_after_step))
                completed_jobs += 1
            candidates = []
            for checkpoint_step in checkpoints if fold_summaries else []:
                values = [row["canonical_mse"] for summary in fold_summaries
                          for evaluation in summary["evaluations"] if evaluation["step"] == checkpoint_step
                          for row in evaluation["by_video"].values()]
                if len(values) != len(train_videos) or not np.isfinite(values).all():
                    raise ValueError("Incomplete held-procedure selection scores")
                candidates.append(dict(step=checkpoint_step, mean_procedure_canonical_mse=float(np.mean(values)),
                                       procedure_mses=values))
            selected_steps = min(candidates, key=lambda row: (row["mean_procedure_canonical_mse"], row["step"]))["step"] if candidates else steps
            selections.append(dict(method=method, seed=seed, selected_steps=selected_steps, candidates=candidates))
            atomic_write_json(output / f"checkpoint_selection_{method}.json", [row for row in selections if row["method"] == method])
            directory = output / "fit" / method / f"seed{seed}"
            context = dict(phase=phase, fold="full_training", progress_path=str(progress_path.resolve()),
                           completed_jobs=completed_jobs, total_jobs=total_jobs)
            summary = training_job(config, method, seed, views, valid, records, train_videos, held_videos,
                directory, selected_steps, [selected_steps], identity, context, resume, stop_after_step)
            summaries.append(dict(method=method, seed=seed, fit_directory=str(directory.resolve()), summary=summary))
            completed_jobs += 1
    for seed in seeds:
        sequences = [read_json(Path(row["fit_directory"]) / "sequence.json") for row in summaries if row["seed"] == seed]
        common_length = min(map(len, sequences))
        if any(sequence[:common_length] != sequences[0][:common_length] for sequence in sequences):
            raise ValueError("Paired methods received different batch or view sequences")
    if source_hashes() != identity["source_hashes"]:
        raise ValueError("Training source files changed during execution")
    result = dict(status="COMPLETE", phase=phase, completed_jobs=completed_jobs, total_jobs=total_jobs,
                  outputs=summaries, checkpoint_selection=selections, completed_at=shared.now(), **identity)
    for method in methods:
        atomic_write_json(output / f"training_summary_{method}.json", dict(result,
            outputs=[row for row in summaries if row["method"] == method],
            checkpoint_selection=[row for row in selections if row["method"] == method]))
    atomic_write_json(output / "training_summary.json", result)
    atomic_write_json(output / "checkpoint_selection.json", selections)
    atomic_write_json(progress_path, dict(status="COMPLETE", phase=phase, method=selected_method, completed_jobs=completed_jobs,
                                         total_jobs=total_jobs, updated_at=shared.now()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="all", choices=("all",) + METHODS)
    parser.add_argument("--phase", required=True, choices=("smoke", "fit"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    result = run_training(config, Path(config.get("run_dir", config_path.parent)), args.phase,
                          args.method, args.resume, args.stop_after_step)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs")}))


if __name__ == "__main__":
    main()
