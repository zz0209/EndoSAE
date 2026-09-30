import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
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
from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.token_memory_edit import (
    METHODS, BoundedGains, FrozenSupCon, TokenDictionary, TokenMemoryPredictor,
    load_predictor, residual_edit,
)


def load_inputs(config):
    videos = list(config["train_video_ids"]) + list(config["validation_video_ids"])
    if len(videos) != len(set(videos)):
        raise ValueError("Procedure partitions overlap or contain duplicates")
    inputs, records = [], []
    for video in videos:
        folder = Path(config["token_data_root"]) / "videos" / video
        receipt = read_json(folder / "complete.json")
        partition = "train" if video in config["train_video_ids"] else "val"
        if receipt["status"] != "COMPLETE" or receipt["video"] != video or receipt["split"] != partition:
            raise ValueError(f"Incomplete or mismatched token data: {video}")
        local = read_json(folder / "records.json")
        if len(local) != receipt["records"] or any(r["video_id"] != video or r["split"] != partition for r in local):
            raise ValueError(f"Token record partition differs: {video}")
        for name in ("records.json", "valid.npy", "record_token_index.npy", "canonical_raw.npy"):
            if file_sha256(folder / name) != receipt["assets"][name]:
                raise ValueError(f"Token metadata identity differs: {video}/{name}")
        if file_sha256(folder / "identity.json") != receipt["identity_sha256"]:
            raise ValueError(f"Token data identity differs: {video}")
        start = len(records)
        records.extend(dict(row, global_index=start + i, partition=partition) for i, row in enumerate(local))
        inputs.append(dict(video_id=video, partition=partition, directory=str(folder.resolve()),
                           start=start, count=len(local), receipt=receipt,
                           receipt_sha256=file_sha256(folder / "complete.json")))
    return inputs, records


def progress(context, directory, stage, step, steps, elapsed, losses=None, status="RUNNING"):
    row = dict(context, status=status, stage=stage, step=step, steps=steps,
               elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / max(step, 1) * (steps - step),
               updated_at=shared.now())
    if losses is not None:
        row["losses"] = losses
    atomic_write_json(directory / "progress.json", row)
    atomic_write_json(Path(context["progress_path"]), row)
    print(json.dumps(row), flush=True)


def save_checkpoint(path, model, optimizer, rng, step, history, sequence, identity, elapsed, **extra):
    shared.save_torch(path, dict(identity_sha256=identity, step=step, model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng_json=json.dumps(rng.bit_generator.state),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        history=history, sequence=sequence, elapsed_seconds=elapsed, **extra))
    pause_after_checkpoint(path)


def restore_checkpoint(path, model, optimizer, rng, identity, device, resume):
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"Use --resume for {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if state["identity_sha256"] != identity:
        raise ValueError("Checkpoint identity differs")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = json.loads(state["numpy_rng_json"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return state


def dictionary_fit(config, method, seed, inputs, fit_videos, directory, steps, identity, context,
                   resume, stop_stage, stop_after_step):
    spec = dict(method=method, reference_fit=str(Path(config["reference_fit"]).resolve()),
                **{key: config[key] for key in ("input_dim", "latent_dim", "top_k", "gain_bound")})
    atomic_write_json(directory / "model_config.json", spec)
    torch.manual_seed(seed)
    model = TokenDictionary(spec).to(config["device"])
    summary_path = directory / "dictionary_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if not resume or summary["identity_sha256"] != identity or summary["status"] != "COMPLETE":
            raise ValueError("Existing dictionary cannot be reused")
        if file_sha256(directory / "model.npz") != summary["model_sha256"]:
            raise ValueError("Saved dictionary weights changed")
        with np.load(directory / "model.npz", allow_pickle=False) as data:
            model.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files}, strict=True)
        with np.load(directory / "normalization.npz", allow_pickle=False) as data:
            return model.eval().requires_grad_(False), data["mean"].copy(), data["scale"].copy(), summary
    selected = [row for row in inputs if row["video_id"] in fit_videos]
    scaler = StandardScaler()
    chunks = []
    for row in selected:
        bank = np.load(Path(row["directory"]) / "bank.npy", mmap_mode="r", allow_pickle=False)
        if bank.dtype != np.float32 or bank.shape != (row["receipt"]["bank_tokens"], int(config["input_dim"])):
            raise ValueError("Invalid unsupervised token bank")
        if not np.isfinite(bank).all():
            raise ValueError("Nonfinite token bank")
        scaler.partial_fit(bank)
        chunks.append(torch.from_numpy(np.array(bank)).to(config["device"]))
    bank = torch.cat(chunks)
    del chunks
    shared.save_npz(directory / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_,
                    var=scaler.var_, n_samples_seen=scaler.n_samples_seen_)
    mean = torch.from_numpy(scaler.mean_).to(config["device"])
    scale = torch.from_numpy(scaler.scale_).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["dictionary_learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(seed)
    checkpoint = directory / "dictionary_checkpoint.pt"
    restored = restore_checkpoint(checkpoint, model, optimizer, rng, identity, config["device"], resume)
    first, elapsed_before = (int(restored["step"]) + 1, float(restored["elapsed_seconds"])) if restored else (1, 0.)
    history, sequence = (restored["history"], restored["sequence"]) if restored else ([], [])
    started = time.perf_counter()
    for step in range(first, steps + 1):
        indices = rng.integers(len(bank), size=int(config["dictionary_batch_size"]), dtype=np.int64)
        sequence.append(hashlib.sha256(indices.tobytes()).hexdigest())
        batch = ((bank[torch.from_numpy(indices).to(config["device"])].double() - mean) / scale).float()
        codes = model.encode(batch)
        loss = F.mse_loss(model.decoder(codes), batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite dictionary loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Invalid dictionary gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Dictionary has no nonzero finite gradient")
        optimizer.step()
        model.normalize_decoder()
        row = dict(step=step, reconstruction_mse=float(loss.detach()), gradient_norm=norm,
                   active_components=float((codes > 0).sum(-1).float().mean().detach()))
        history.append(row)
        stopping = stop_stage == "dictionary" and stop_after_step is not None and step >= stop_after_step
        if step % int(config["checkpoint_every"]) == 0 or step == steps or stopping:
            elapsed = elapsed_before + time.perf_counter() - started
            save_checkpoint(checkpoint, model, optimizer, rng, step, history, sequence, identity, elapsed)
            progress(context, directory, "dictionary", step, steps, elapsed, row, "PAUSED" if stopping else "RUNNING")
            if stopping:
                raise SystemExit(75)
    shared.save_npz(directory / "model.npz", **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    atomic_write_json(directory / "dictionary_history.json", history)
    atomic_write_json(directory / "dictionary_sequence.json", sequence)
    summary = dict(status="COMPLETE", identity_sha256=identity, steps=steps, fit_video_ids=fit_videos,
                   tokens=len(bank), token_labels_used=False, batch_size=int(config["dictionary_batch_size"]),
                   elapsed_seconds=elapsed_before + time.perf_counter() - started,
                   final_reconstruction_mse=history[-1]["reconstruction_mse"],
                   model_sha256=file_sha256(directory / "model.npz"),
                   normalization_sha256=file_sha256(directory / "normalization.npz"),
                   sequence_sha256=file_sha256(directory / "dictionary_sequence.json"))
    atomic_write_json(summary_path, summary)
    return model.eval().requires_grad_(False), scaler.mean_, scaler.scale_, summary


@torch.no_grad()
def pool_views(config, model, mean, scale, inputs, records, directory, identity, context, resume):
    path, receipt_path = directory / "pooled_views.npz", directory / "pool_summary.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if not resume or receipt["identity_sha256"] != identity or file_sha256(path) != receipt["sha256"]:
            raise ValueError("Existing pooled token codes cannot be reused")
        with np.load(path, allow_pickle=False) as data:
            return data["raw"].copy(), data["mean_codes"].copy(), data["valid"].copy()
    raw = np.full((len(records), 7, int(config["input_dim"])), np.nan, dtype=np.float64)
    pooled = np.full((len(records), 7, int(config["latent_dim"])), np.nan, dtype=np.float32)
    valid = np.zeros((len(records), 7), dtype=bool)
    mean_tensor = torch.from_numpy(mean).to(config["device"])
    scale_tensor = torch.from_numpy(scale).to(config["device"])
    reconstruction = {}
    started = time.perf_counter()
    for number, row in enumerate(inputs):
        folder = Path(row["directory"])
        tokens = np.load(folder / "tokens.npy", mmap_mode="r", allow_pickle=False)
        mapping = np.load(folder / "record_token_index.npy", allow_pickle=False)
        local_valid = np.load(folder / "valid.npy", allow_pickle=False)
        canonical = np.load(folder / "canonical_raw.npy", allow_pickle=False)
        with np.load(folder / "masks.npz", allow_pickle=False) as data:
            masks = data["masks"].copy()
        if (tokens.shape[1:] != (8, 196, 768) or masks.shape != (row["count"], 7, 8, 196)
                or local_valid.shape != (row["count"], 7) or not local_valid[:, 0].all()):
            raise ValueError("Token clip or mask dimensions differ")
        errors = []
        for token_index in np.unique(mapping):
            values = np.asarray(tokens[token_index])
            tensor = ((torch.from_numpy(values.copy()).to(config["device"]).double() - mean_tensor) / scale_tensor).float()
            code_tensor = model.encode(tensor.reshape(-1, 768))
            errors.append(float(F.mse_loss(model.decoder(code_tensor), tensor.reshape(-1, 768))))
            codes = code_tensor.cpu().numpy().reshape(8, 196, -1)
            for local in np.flatnonzero(mapping == token_index):
                index = row["start"] + int(local)
                valid[index] = local_valid[local]
                for view in np.flatnonzero(local_valid[local]):
                    mask = masks[local, view]
                    if not mask.any():
                        raise ValueError("A valid region has no token support")
                    raw[index, view] = values[mask].mean(axis=0, dtype=np.float64)
                    pooled[index, view] = codes[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
                if not np.allclose(raw[index, 0], canonical[local], atol=1e-8, rtol=0):
                    raise ValueError("Canonical token pooling changed the original descriptor")
        reconstruction[row["video_id"]] = float(np.mean(errors))
        progress(context, directory, "pool_tokens", number + 1, len(inputs), time.perf_counter() - started)
    if not np.isfinite(raw[valid]).all() or not np.isfinite(pooled[valid]).all() or not valid[:, 1:].any(axis=1).all():
        raise ValueError("Invalid pooled region views")
    shared.save_npz(path, raw=raw, mean_codes=pooled, valid=valid)
    atomic_write_json(receipt_path, dict(status="COMPLETE", identity_sha256=identity, sha256=file_sha256(path),
                                       token_reconstruction_mse_by_video=reconstruction, elapsed_seconds=time.perf_counter() - started))
    return raw, pooled, valid


def protection_point(report, floor):
    groups, excluded = {}, {}
    for video, counts in report["by_video"].items():
        pairs = [p for p in report["pairs"] if p["video_id"] == video]
        positive = np.sort([p["score"] for p in pairs if p["same_identity"]])
        negative = np.sort([p["score"] for p in pairs if not p["same_identity"]])
        if len(positive) and len(negative):
            groups[video] = (positive, negative)
        else:
            excluded[video] = dict(positives=len(positive), negatives=len(negative))
    if not groups:
        raise ValueError("No procedure supports the protection selection endpoint")
    observed = np.unique(np.concatenate([np.concatenate(pair) for pair in groups.values()]))
    thresholds = np.r_[observed, np.nextafter(observed[-1], np.inf)]
    recalls = np.array([1 - np.searchsorted(p, thresholds, side="left") / len(p) for p, n in groups.values()])
    retained = np.array([np.searchsorted(n, thresholds, side="left") / len(n) for p, n in groups.values()])
    eligible = np.flatnonzero(retained.mean(axis=0) >= floor - 1e-12)
    means = recalls.mean(axis=0)
    best = eligible[np.flatnonzero(means[eligible] >= means[eligible].max() - 1e-12)[-1]]
    return dict(threshold=float(thresholds[best]), recall=float(means[best]), retention=float(retained[:, best].mean()),
                procedure_rows=[dict(video=video, positives=len(pair[0]), negatives=len(pair[1]),
                                     recall=float(recalls[i, best]), retention=float(retained[i, best]))
                                for i, (video, pair) in enumerate(groups.items())], excluded_procedures=excluded)


@torch.no_grad()
def evaluate_gains(config, model, gains, reference, raw, codes, valid, scale, records, videos, directory, stem):
    selected = np.array([i for i, row in enumerate(records) if row["video_id"] in videos])
    local, views = np.where(valid[selected, 1:])
    indices, views = selected[local], views + 1
    corrected = residual_edit(raw[indices, views], codes[indices, views], gains(), model, scale)
    errors = ((corrected - raw[indices, 0]) / scale).square().mean(-1).cpu().numpy()
    baseline = ((raw[indices, views] - raw[indices, 0]) / scale).square().mean(-1).cpu().numpy()
    memories = reference(corrected)
    queries = reference(raw[:, 0])
    lookup = {(int(index), int(view)): position for position, (index, view) in enumerate(zip(indices, views))}
    pairs, values = [], []
    for pair in shared.chronological_pairs(records, videos):
        for view in np.flatnonzero(valid[pair["source_index"], 1:]) + 1:
            position = lookup[(pair["source_index"], int(view))]
            values.append((memories[position] * queries[pair["query_index"]]).sum())
            pairs.append(dict(pair, source_view=int(view)))
    scores = torch.stack(values).cpu().numpy()
    report = shared.pair_statistics(pairs, scores)
    report["pairs"] = [dict(pair, score=float(score)) for pair, score in zip(pairs, scores)]
    report["protection_point"] = protection_point(report, float(config["retention_floor"]))
    report["region_by_video"] = {video: dict(
        canonical_mse=float(np.mean([errors[indices == i].mean() for i in selected if records[i]["video_id"] == video])),
        input_to_canonical_mse=float(np.mean([baseline[indices == i].mean() for i in selected if records[i]["video_id"] == video])))
        for video in sorted(videos)}
    atomic_write_json(directory / f"{stem}.json", report)
    shared.save_npz(directory / f"{stem}_errors.npz", indices=indices, views=views, canonical_mse=errors, baseline_mse=baseline)
    return dict(step=int(stem.split("step")[-1]) if "step" in stem else None,
                protection_point=report["protection_point"], macro_procedure_auroc=report["macro_procedure_auroc"],
                canonical_mse=float(np.mean([r["canonical_mse"] for r in report["region_by_video"].values()])),
                input_to_canonical_mse=float(np.mean([r["input_to_canonical_mse"] for r in report["region_by_video"].values()])))


def gain_fit(config, model, mean, scale, raw_array, code_array, valid, records, fit_videos, held_videos,
             directory, steps, checkpoints, identity, seed, context, resume, stop_stage, stop_after_step):
    device = config["device"]
    raw, codes = torch.from_numpy(raw_array).to(device), torch.from_numpy(code_array).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    reference = FrozenSupCon(config["reference_fit"]).to(device)
    with torch.no_grad():
        queries = reference(raw[:, 0])
    gains = BoundedGains(model.latent_dim, config["gain_bound"]).to(device)
    optimizer = torch.optim.Adam(gains.parameters(), lr=float(config["gain_learning_rate"]))
    rng = np.random.default_rng(seed)
    view_rng = np.random.default_rng(np.random.SeedSequence([seed, 91841]))
    bank = shared.episode_bank(records, fit_videos)
    train_indices = np.array([i for i, row in enumerate(records) if row["video_id"] in fit_videos])
    available = [np.flatnonzero(row) for row in valid]
    checkpoint = directory / "gain_checkpoint.pt"
    state = restore_checkpoint(checkpoint, gains, optimizer, rng, identity, device, resume)
    first, elapsed_before = (int(state["step"]) + 1, float(state["elapsed_seconds"])) if state else (1, 0.)
    history, sequence, evaluations = (state["history"], state["sequence"], state["evaluations"]) if state else ([], [], [])
    if state:
        view_rng.bit_generator.state = json.loads(state["view_rng_json"])
    else:
        zero = residual_edit(raw[train_indices, 0], codes[train_indices, 0], gains(), model, scale_tensor)
        if not torch.equal(zero, raw[train_indices, 0]):
            raise ValueError("Zero gain changed the raw source")
        evaluations.append(evaluate_gains(config, model, gains, reference, raw, codes, valid, scale_tensor,
                                          records, held_videos, directory, "held_step0000"))
    started = time.perf_counter()
    for step in range(first, steps + 1):
        batch = shared.sample_batch(bank, train_indices, rng, config)
        indices = batch["indices"]
        views = np.array([view_rng.choice(available[i]) for i in indices])
        sequence.append(dict(**{key: value.tolist() for key, value in batch.items()}, views=views.tolist()))
        edited = residual_edit(raw[indices, views], codes[indices, views], gains(), model, scale_tensor)
        recovery = ((edited - raw[indices, 0]) / scale_tensor).square().mean()
        memories = reference(edited[batch["source"]])
        positive = (memories * queries[indices[batch["positive"]]]).sum(-1)
        negative = (memories * queries[indices[batch["negative"]]]).sum(-1)
        identity_loss = F.softplus((negative - positive) / float(config["temperature"])).mean()
        loss = recovery + float(config["identity_weight"]) * identity_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite gain objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gains.theta.grad is None or not torch.isfinite(gains.theta.grad).all():
            raise ValueError("Invalid gain gradient")
        gradient = float(torch.linalg.vector_norm(gains.theta.grad))
        if gradient <= 0:
            raise ValueError("Gain objective has no gradient")
        torch.nn.utils.clip_grad_norm_(gains.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        history.append(dict(step=step, total_loss=float(loss.detach()), canonical_loss=float(recovery.detach()),
                            identity_loss=float(identity_loss.detach()), gradient_norm=gradient,
                            gain_l2=float(torch.linalg.vector_norm(gains()).detach()),
                            gain_max_abs=float(gains().abs().max().detach())))
        if step in checkpoints or step == steps:
            evaluations.append(evaluate_gains(config, model, gains, reference, raw, codes, valid, scale_tensor,
                                              records, held_videos, directory, f"held_step{step:04d}"))
        stopping = stop_stage == "gain" and stop_after_step is not None and step >= stop_after_step
        if step % int(config["gain_checkpoint_every"]) == 0 or step in checkpoints or step == steps or stopping:
            elapsed = elapsed_before + time.perf_counter() - started
            save_checkpoint(checkpoint, gains, optimizer, rng, step, history, sequence, identity, elapsed,
                            evaluations=evaluations, view_rng_json=json.dumps(view_rng.bit_generator.state))
            progress(context, directory, "gain", step, steps, elapsed, history[-1], "PAUSED" if stopping else "RUNNING")
            if stopping:
                raise SystemExit(75)
    if steps == 0:
        save_checkpoint(checkpoint, gains, optimizer, rng, 0, history, sequence, identity, 0.,
                        evaluations=evaluations, view_rng_json=json.dumps(view_rng.bit_generator.state))
    shared.save_npz(directory / "gains.npz", theta=gains.theta.detach().cpu().numpy(), gains=gains().detach().cpu().numpy())
    atomic_write_json(directory / "gain_history.json", history)
    atomic_write_json(directory / "gain_sequence.json", sequence)
    atomic_write_json(directory / "gain_evaluations.json", evaluations)
    predictor = load_predictor(directory, device)
    direct = TokenMemoryPredictor(model, mean, scale, gains().detach().cpu().numpy(), reference, device)
    indices = train_indices[:16]
    actual = predictor.edit_raw(raw_array[indices, 0], code_array[indices, 0])
    expected = direct.edit_raw(raw_array[indices, 0], code_array[indices, 0])
    if not np.array_equal(actual, expected):
        raise ValueError("Gain export changed edited descriptors")
    zeros = np.zeros(model.latent_dim, dtype=np.float32)
    if not np.array_equal(predictor.edit_raw(raw_array[indices, 0], code_array[indices, 0], zeros), raw_array[indices, 0]):
        raise ValueError("Exported zero edit changed original descriptors")
    effective = ((code_array[train_indices, 0] * predictor.source_gains) != 0).sum(-1)
    return dict(gain_steps=steps, evaluations=evaluations, no_edit=bool(not np.any(predictor.source_gains)),
                zero_edit_exact=True, model_reload_exact=True, gain_parameters=model.latent_dim,
                effective_components_mean=float(effective.mean()), effective_components_max=int(effective.max()),
                elapsed_seconds=elapsed_before + time.perf_counter() - started,
                gains_sha256=file_sha256(directory / "gains.npz"), sequence_sha256=file_sha256(directory / "gain_sequence.json"))


def training_job(config, method, seed, inputs, records, fit_videos, held_videos, directory,
                 dictionary_steps, gain_steps, checkpoints, identity, context, resume, stop_stage, stop_after_step):
    directory.mkdir(parents=True, exist_ok=True)
    job_identity = dict(configuration_sha256=shared.json_digest(config), method=method, seed=seed,
                        fit_videos=fit_videos, held_videos=held_videos, dictionary_steps=dictionary_steps,
                        gain_steps=gain_steps, gain_checkpoints=checkpoints, **identity)
    key = shared.json_digest(job_identity)
    if (directory / "summary.json").exists():
        summary = read_json(directory / "summary.json")
        if not resume or summary["status"] != "COMPLETE" or summary["identity_sha256"] != key:
            raise ValueError("Existing token edit fit cannot be reused")
        return summary
    if (directory / "identity.json").exists() and read_json(directory / "identity.json") != job_identity:
        raise ValueError("Token edit job identity differs")
    atomic_write_json(directory / "identity.json", job_identity)
    model, mean, scale, dictionary = dictionary_fit(config, method, seed, inputs, fit_videos, directory,
        dictionary_steps, key, context, resume, stop_stage, stop_after_step)
    raw, codes, valid = pool_views(config, model, mean, scale, inputs, records, directory, key, context, resume)
    gate = gain_fit(config, model, mean, scale, raw, codes, valid, records, fit_videos, held_videos,
                    directory, gain_steps, checkpoints, key, seed, context, resume, stop_stage, stop_after_step)
    summary = dict(status="COMPLETE", identity_sha256=key, method=method, seed=seed,
                   dictionary=dictionary, **gate, completed_at=shared.now(),
                   runtime=dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                                torch=str(torch.__version__), device=config["device"], threads=int(config["threads"])))
    atomic_write_json(directory / "summary.json", summary)
    progress(context, directory, "complete", gain_steps, gain_steps, gate["elapsed_seconds"], status="COMPLETE")
    return summary


def source_hashes():
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "src/token_memory_edit.py", ROOT / "src/region_identity_sae.py",
        ROOT / "scripts/train_acknowledgement_sae.py", ROOT / "src/acknowledgement_sae.py", ROOT / "src/checkpoint_io.py")}


def run_training(config, run, phase, selected_method, resume, stop_stage=None, stop_after_step=None):
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs, records = load_inputs(config)
    methods = list(config["methods"]) if selected_method == "all" else [selected_method]
    if any(method not in METHODS for method in methods):
        raise ValueError("Unknown token edit method")
    seeds = config["seeds"][:1] if phase == "smoke" else config["seeds"]
    output = run / "smoke" if phase == "smoke" else run
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "training_config.json", config)
    atomic_write_json(output / "token_inputs.json", inputs)
    atomic_write_json(output / "descriptor_records.json", records)
    train_videos, validation_videos = config["train_video_ids"], config["validation_video_ids"]
    folds = [] if phase == "smoke" else shared.make_folds(records, train_videos, int(config["inner_folds"]), int(config["fold_seed"]))
    dict_steps = int(config["smoke_dictionary_steps"] if phase == "smoke" else config["dictionary_steps"])
    gain_steps = int(config["smoke_gain_steps"] if phase == "smoke" else config["gain_steps"])
    checkpoints = [0, gain_steps] if phase == "smoke" else list(config["gain_checkpoints"])
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0 or checkpoints[-1] != gain_steps:
        raise ValueError("Gain checkpoints must include zero and the final gain step")
    identity = dict(input_identity_sha256=shared.json_digest(inputs), source_hashes=source_hashes(),
                    reference_assets={name: file_sha256(Path(config["reference_fit"]) / name)
                                      for name in ("model.npz", "normalization.npz")})
    atomic_write_json(output / "grouped_folds.json", dict(training_procedures=train_videos,
        validation_procedures=validation_videos, inner_held_folds=folds,
        selection="Per-fold common threshold at 99% mean other-pair retention, then equal mean of all eligible held procedures; earliest step tie; zero is a candidate.",
        reference_scope="Dictionary and gains exclude held procedures. Existing fixed SupCon was fitted on all 19 training procedures."))
    total_jobs, completed_jobs = len(methods) * len(seeds) * (len(folds) + 1), 0
    summaries, selections = [], []
    for seed in seeds:
        for method in methods:
            fold_results = []
            for fold, held in enumerate(folds):
                fit = [v for v in train_videos if v not in held]
                directory = output / "inner_folds" / method / f"seed{seed}" / f"fold{fold}"
                context = dict(phase=phase, method=method, seed=seed, fold=fold,
                               progress_path=str((output / "training_progress.json").resolve()),
                               completed_jobs=completed_jobs, total_jobs=total_jobs)
                fold_results.append(training_job(config, method, seed, inputs, records, fit, held, directory,
                    dict_steps, gain_steps, checkpoints, identity, context, resume, stop_stage, stop_after_step))
                completed_jobs += 1
            candidates = []
            for step in checkpoints if folds else []:
                reports = [next(r for r in summary["evaluations"] if r["step"] == step) for summary in fold_results]
                rows = [row for report in reports for row in report["protection_point"]["procedure_rows"]]
                if len({row["video"] for row in rows}) != len(rows):
                    raise ValueError("A procedure appears in multiple held folds")
                candidates.append(dict(step=step, recall=float(np.mean([r["recall"] for r in rows])),
                                       procedure_rows=rows, fold_points=[r["protection_point"] for r in reports]))
            selected = max(candidates, key=lambda row: (row["recall"], -row["step"]))["step"] if candidates else gain_steps
            selections.append(dict(method=method, seed=seed, selected_gain_steps=selected, candidates=candidates))
            atomic_write_json(output / f"checkpoint_selection_{method}.json", [s for s in selections if s["method"] == method])
            directory = output / "fit" / method / f"seed{seed}"
            context = dict(phase=phase, method=method, seed=seed, fold="full_training",
                           progress_path=str((output / "training_progress.json").resolve()),
                           completed_jobs=completed_jobs, total_jobs=total_jobs)
            summary = training_job(config, method, seed, inputs, records, train_videos, validation_videos, directory,
                dict_steps, selected, [0] if selected == 0 else [0, selected], identity, context, resume, stop_stage, stop_after_step)
            summaries.append(dict(method=method, seed=seed, fit_directory=str(directory.resolve()), summary=summary))
            completed_jobs += 1
    for seed in seeds:
        matched = [Path(row["fit_directory"]) for row in summaries if row["seed"] == seed]
        for name in ("dictionary_sequence.json", "gain_sequence.json"):
            sequences = [read_json(folder / name) for folder in matched]
            common = min(map(len, sequences))
            if any(sequence[:common] != sequences[0][:common] for sequence in sequences):
                raise ValueError("Matched methods received different training sequences")
    if source_hashes() != identity["source_hashes"]:
        raise ValueError("Training source changed during execution")
    result = dict(status="COMPLETE", phase=phase, outputs=summaries, checkpoint_selection=selections,
                  completed_jobs=completed_jobs, total_jobs=total_jobs, **identity)
    atomic_write_json(output / "training_summary.json", result)
    for method in methods:
        atomic_write_json(output / f"training_summary_{method}.json", dict(result,
            outputs=[r for r in summaries if r["method"] == method],
            checkpoint_selection=[r for r in selections if r["method"] == method]))
    atomic_write_json(output / "training_progress.json", dict(status="COMPLETE", phase=phase, method=selected_method,
        completed_jobs=completed_jobs, total_jobs=total_jobs, updated_at=shared.now()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("smoke", "fit"), required=True)
    parser.add_argument("--method", choices=("all",) + METHODS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-stage", choices=("dictionary", "gain"))
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    if (args.stop_stage is None) != (args.stop_after_step is None):
        raise ValueError("Specify both --stop-stage and --stop-after-step")
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    result = run_training(config, Path(config.get("run_dir", config_path.parent)), args.phase, args.method,
                          args.resume, args.stop_stage, args.stop_after_step)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs")}))


if __name__ == "__main__":
    main()
