import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_acknowledgement_sae as shared
from train_frozen_identity_supcon import sample_batch, supcon
from train_temporal_shared_sae import protected_statistics
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.evaluation.realcolon_task import tokens_to_frames
from src.token_identity_sae import METHODS, TokenIdentitySAE, local_identity_loss, procedure_identity_loss, symmetric_maxsim


def source_identity():
    names = ["scripts/train_token_identity_sae.py", "src/token_identity_sae.py",
             "scripts/train_acknowledgement_sae.py", "scripts/train_frozen_identity_supcon.py",
             "scripts/train_temporal_shared_sae.py", "src/checkpoint_io.py",
             "src/evaluation/realcolon_task.py"]
    return {name: shared.file_sha256(ROOT / name) for name in names}


def prepare(run, video_filter, resume):
    config = read_json(run / "config.json")
    cohort = read_json(config["cohort_config"])
    videos = cohort["fit_video_ids"]["train"] + cohort["fit_video_ids"]["val"]
    selected = [video_filter] if video_filter else videos
    if any(video not in videos for video in selected):
        raise ValueError("Unspecified preparation procedure")
    started = time.perf_counter()
    for index, video in enumerate(selected):
        original = Path(cohort["input_descriptor_root"]) / video
        source = Path(config["views_root"]) / video
        receipt = read_json(original / "complete.json")
        cache = Path(cohort["cache_root"]) / video
        cache_receipt = read_json(cache / "complete.json")
        if receipt["status"] != "COMPLETE" or cache_receipt["status"] != "COMPLETE":
            raise ValueError("Incomplete native input")
        if shared.file_sha256(cache / "identity.json") != receipt["cache_identity_sha256"]:
            raise ValueError("Native cache identity differs")
        target = run / "tokens" / video
        target.mkdir(parents=True, exist_ok=True)
        identity = dict(video=video, cache_identity=receipt["cache_identity_sha256"],
            records_sha256=shared.file_sha256(original / "records.json"),
            masks_sha256=shared.file_sha256(source / "masks.npz"),
            descriptors_sha256=shared.file_sha256(original / "descriptors.npz"),
            config_sha256=shared.file_sha256(run / "config.json"), source_hashes=source_identity())
        if identity["records_sha256"] != receipt["record_sha256"] or identity["descriptors_sha256"] != receipt["descriptor_sha256"]:
            raise ValueError("Descriptor source identity differs")
        if (target / "complete.json").exists():
            saved = read_json(target / "complete.json")
            if not resume or saved["identity"] != identity:
                raise ValueError("Prepared input identity differs")
            if shared.file_sha256(target / "tokens.npz") != saved["tokens_sha256"]:
                raise ValueError("Prepared token asset changed")
            print("TOKEN_REUSE", video, saved["tokens"], flush=True)
            continue
        records = read_json(original / "records.json")
        if records != read_json(source / "records.json"):
            raise ValueError("Region and descriptor records differ")
        with np.load(source / "masks.npz", allow_pickle=False) as archive:
            masks = archive["masks"][:, 0].copy()
        with np.load(original / "descriptors.npz", allow_pickle=False) as archive:
            expected = archive["raw"].copy()
        native = np.load(cache / "block10.npy", mmap_mode="r", allow_pickle=False)
        tokens, offsets, positions, errors = [], [0], [], []
        for row, mask, reference in zip(records, masks, expected, strict=True):
            if mask.sum(axis=1).tolist() != row["roi_tokens_per_frame"]:
                raise ValueError("Canonical token support differs")
            value = tokens_to_frames(np.asarray(native[row["cache_clip_index"]])[None])[mask].astype(np.float32)
            if value.shape != (int(mask.sum()), config["input_dim"]) or not np.isfinite(value).all():
                raise ValueError("Invalid actual ROI tokens")
            error = float(np.max(np.abs(value.mean(axis=0, dtype=np.float64) - reference)))
            if error > 1e-8:
                raise ValueError(f"Canonical mean did not reproduce: {video}, {error}")
            errors.append(error)
            tokens.append(value)
            positions.append(np.column_stack(np.where(mask)).astype(np.int16))
            offsets.append(offsets[-1] + len(value))
        del native
        shared.save_npz(target / "tokens.npz", tokens=np.concatenate(tokens), offsets=np.array(offsets),
                        positions=np.concatenate(positions))
        atomic_write_json(target / "records.json", records)
        saved = dict(status="COMPLETE", identity=identity, tokens=offsets[-1], clips=len(records),
            max_mean_error=max(errors), tokens_sha256=shared.file_sha256(target / "tokens.npz"),
            records_sha256=shared.file_sha256(target / "records.json"), completed_at=shared.now())
        atomic_write_json(target / "complete.json", saved)
        atomic_write_json(run / "preparation_progress.json", dict(status="RUNNING", completed=index + 1,
            total=len(selected), video=video, seconds=time.perf_counter() - started))
        print("TOKEN_PREPARED", video, index + 1, "/", len(selected), "tokens", offsets[-1], flush=True)
        pause_after_checkpoint(target / "complete.json")
    if not video_filter:
        receipts = [read_json(run / "tokens" / video / "complete.json") for video in videos]
        atomic_write_json(run / "preparation_summary.json", dict(status="COMPLETE", receipts=receipts,
            tokens=sum(row["tokens"] for row in receipts), clips=sum(row["clips"] for row in receipts),
            seconds=time.perf_counter() - started))
        atomic_write_json(run / "preparation_progress.json", dict(status="COMPLETE", completed=len(videos), total=len(videos)))


def load_inputs(run):
    config = read_json(run / "config.json")
    prepared = Path(config.get("prepared_run", run))
    cohort = read_json(config["cohort_config"])
    records, values, offsets, receipts = [], [], [0], []
    for partition in ["train", "val"]:
        for video in cohort["fit_video_ids"][partition]:
            directory = prepared / "tokens" / video
            receipt = read_json(directory / "complete.json")
            if receipt["status"] != "COMPLETE" or shared.file_sha256(directory / "tokens.npz") != receipt["tokens_sha256"]:
                raise ValueError("Prepared tokens are incomplete or changed")
            local = read_json(directory / "records.json")
            if shared.file_sha256(directory / "records.json") != receipt["records_sha256"]:
                raise ValueError("Prepared record identity differs")
            if any(r["video_id"] != video or r["split"] != partition for r in local):
                raise ValueError("Unexpected data partition")
            with np.load(directory / "tokens.npz", allow_pickle=False) as archive:
                value, local_offsets = archive["tokens"].copy(), archive["offsets"].copy()
            if value.dtype != np.float32 or local_offsets.shape != (len(local) + 1,) or not np.all(np.diff(local_offsets) > 0):
                raise ValueError("Invalid prepared token dimensions")
            offsets.extend((local_offsets[1:] + offsets[-1]).tolist())
            values.append(value)
            base = len(records)
            records.extend(dict(row, global_index=base + i, partition=partition) for i, row in enumerate(local))
            receipts.append(receipt)
    if set(cohort["fit_video_ids"]["train"]) & set(cohort["fit_video_ids"]["val"]):
        raise ValueError("Procedure leakage")
    raw = np.concatenate(values)
    if offsets[-1] != len(raw) or len(records) != 336:
        raise ValueError("Unexpected cohort size")
    return config, cohort, raw, np.array(offsets), records, receipts


def normalize(raw, offsets, records, videos):
    selected = [i for i, row in enumerate(records) if row["video_id"] in videos]
    means, seconds = [], []
    for index in selected:
        value = raw[offsets[index]:offsets[index + 1]].astype(np.float64)
        means.append(value.mean(axis=0))
        seconds.append((value * value).mean(axis=0))
    mean = np.mean(means, axis=0)
    variance = np.mean(seconds, axis=0) - mean * mean
    if np.any(variance < -1e-10):
        raise ValueError("Invalid training variance")
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale == 0] = 1
    return mean, scale


@torch.no_grad()
def evaluate(model, values, offsets, records, videos, folder, stem, config):
    model.eval()
    indices = [i for i, row in enumerate(records) if row["video_id"] in videos]
    embeddings, metadata, codes, local_vectors = {}, [], [], {}
    interaction = config.get("identity_interaction", "pooled_cosine")
    for index in indices:
        tokens = values[offsets[index]:offsets[index + 1]][None]
        projected, decoded, local = model(tokens)
        if not torch.isfinite(projected).all() or torch.any(projected.norm(dim=-1) <= 0):
            raise ValueError("Invalid held identity embedding")
        embeddings[index] = projected[0].cpu().numpy()
        if interaction == "symmetric_maxsim":
            vectors = model.readout(local)[0]
            if not torch.isfinite(vectors).all() or torch.any(vectors.norm(dim=-1) <= 0):
                raise ValueError("Invalid held local identity vector")
            local_vectors[index] = F.normalize(vectors, dim=-1)
        row = dict(index=index, video_id=records[index]["video_id"], tokens=tokens.shape[1])
        if local is not None:
            token_code = local.mean(dim=1)
            mean_code = model.encode(tokens.mean(dim=1))
            pooled = model.pool(local) if model.method.startswith("token_") else mean_code
            gap = (token_code - mean_code).norm() / token_code.norm().clamp_min(1e-12)
            row.update(reconstruction_nmse=float((decoded - tokens).square().mean() / tokens.square().mean()),
                local_active=float((local > 0).sum(-1).float().mean()),
                pooled_active=int((pooled > 0).sum()), noncommutation=float(gap))
            codes.append(pooled[0].cpu().numpy())
        metadata.append(row)
    pairs = shared.chronological_pairs(records, videos)
    if interaction == "symmetric_maxsim":
        scores = np.array([float(symmetric_maxsim(local_vectors[row["source_index"]],
            local_vectors[row["query_index"]])) for row in pairs])
    else:
        scores = np.array([float(embeddings[row["source_index"]] @ embeddings[row["query_index"]]) for row in pairs])
    report = protected_statistics(pairs, scores, config["negative_quantile"])
    labels = np.array([row["same_identity"] for row in pairs])
    cross = labels & ~np.array([row["same_annotation_interval"] for row in pairs])
    for video, row in report["by_video"].items():
        mask = np.array([pair["video_id"] == video for pair in pairs])
        row.update(auroc=float(roc_auc_score(labels[mask], scores[mask])),
            positive_pairs=int(np.sum(mask & labels)), negative_pairs=int(np.sum(mask & ~labels)),
            cross_interval_pairs=int(np.sum(mask & cross)),
            cross_interval_recall=float(np.mean(scores[mask & cross] > report["threshold"])) if np.any(mask & cross) else None)
    report["macro_auroc"] = float(np.mean([row["auroc"] for row in report["by_video"].values()]))
    report["by_clip"] = metadata
    shared.save_npz(folder / (stem + ".npz"), indices=np.array(indices),
        embeddings=np.stack([embeddings[i] for i in indices]), scores=scores, same_identity=labels,
        cross_interval=cross, source=np.array([row["source_index"] for row in pairs]),
        query=np.array([row["query_index"] for row in pairs]),
        pooled_codes=np.stack(codes) if codes else np.empty((len(indices), 0)))
    atomic_write_json(folder / (stem + ".json"), report)
    return report


def fit(config, method, seed, raw, offsets, records, fit_videos, held_videos, folder,
        steps, checkpoints, identity, context, resume, stop_after):
    folder.mkdir(parents=True, exist_ok=True)
    job = dict(config=config, method=method, seed=seed, fit_videos=sorted(fit_videos), held_videos=sorted(held_videos),
               steps=steps, checkpoints=checkpoints, **identity)
    signature = shared.json_digest(job)
    if (folder / "summary.json").exists():
        result = read_json(folder / "summary.json")
        if not resume or result["identity_sha256"] != signature:
            raise ValueError("Completed fit identity differs")
        return result
    if 'initial_model_folder' in config:
        with np.load(Path(config['initial_model_folder']) / 'normalization.npz') as saved:
            mean, scale = saved['mean'], saved['scale']
    else:
        mean, scale = normalize(raw, offsets, records, fit_videos)
    shared.save_npz(folder / "normalization.npz", mean=mean, scale=scale)
    values = torch.from_numpy(((raw - mean) / scale).astype(np.float32)).to(config["device"])
    train_records = [dict(row, split="train" if row["video_id"] in fit_videos else "excluded") for row in records]
    label_map = {key: i for i, key in enumerate(sorted({(r["video_id"], r["lesion_id"]) for r in records}))}
    labels = torch.tensor([label_map[(r["video_id"], r["lesion_id"])] for r in records], device=config["device"])
    negative_scope = config.get("identity_negative_scope", "global")
    if negative_scope not in ("global", "global_and_procedure"):
        raise ValueError("Unrecognized identity negative scope")
    if negative_scope != "global" and config.get("identity_interaction", "pooled_cosine") != "pooled_cosine":
        raise ValueError("Procedure objective requires pooled cosine")
    procedure_map = {video: i for i, video in enumerate(sorted({r["video_id"] for r in records}))}
    procedures = torch.tensor([procedure_map[r["video_id"]] for r in records], device=config["device"])
    torch.manual_seed(seed)
    model = TokenIdentitySAE(config, method).to(config["device"])
    if 'initial_model_folder' in config:
        with np.load(Path(config['initial_model_folder']) / 'model.npz') as saved:
            model.load_state_dict({key: torch.from_numpy(saved[key]).to(config['device']) for key in saved.files})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    token_rng = np.random.default_rng(np.random.SeedSequence([seed, 71005]))
    history, evaluations, sequence = [], [], []
    initial, previous = 1, 0.
    checkpoint = folder / "checkpoint.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(checkpoint)
        saved = torch.load(checkpoint, map_location=config["device"], weights_only=True)
        if saved["identity_sha256"] != signature:
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = json.loads(saved["rng_json"])
        token_rng.bit_generator.state = json.loads(saved["token_rng_json"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if config["device"] == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        history, evaluations, sequence = saved["history"], saved["evaluations"], saved["sequence"]
        initial, previous = saved["step"] + 1, saved["seconds"]
    started = time.perf_counter()
    for step in range(initial, steps + 1):
        model.train()
        chosen = np.array(sample_batch(train_records, rng, config))
        if config.get("all_roi_tokens", False):
            if config.get("identity_interaction", "pooled_cosine") != "pooled_cosine":
                raise ValueError("All-token fitting requires pooled cosine")
            token_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in chosen])
            projected_rows, reconstruction_losses = [], []
            for index in chosen:
                samples = values[offsets[index]:offsets[index + 1]][None]
                projected, reconstruction, local = model(samples)
                projected_rows.append(projected)
                reconstruction_losses.append((reconstruction - samples).square().mean())
            projected = torch.cat(projected_rows)
            reconstruction_loss = torch.stack(reconstruction_losses).mean()
        else:
            token_indices = np.stack([token_rng.integers(offsets[i], offsets[i + 1], size=config["tokens_per_clip"]) for i in chosen])
            samples = values[torch.from_numpy(token_indices).to(config["device"])]
            projected, reconstruction, local = model(samples)
            reconstruction_loss = (reconstruction - samples).square().mean() if reconstruction is not None else projected.new_zeros(())
        sequence.append(dict(indices=chosen.tolist(), tokens_sha256=hashlib.sha256(token_indices.tobytes()).hexdigest()))
        if config.get("identity_interaction", "pooled_cosine") == "symmetric_maxsim":
            identity_loss = local_identity_loss(model, local, labels[chosen], config["temperature"])
        else:
            identity_loss = supcon(projected, labels[chosen], config["temperature"])
        global_identity_loss = identity_loss
        procedure_loss = None
        if negative_scope == "global_and_procedure":
            procedure_loss = procedure_identity_loss(projected, labels[chosen], procedures[chosen], config["temperature"])
            identity_loss = .5 * (identity_loss + procedure_loss)
        loss = identity_loss + config["reconstruction_weight"] * reconstruction_loss
        gradient_diagnostics = {}
        if config.get('reconstruction_gradient_interval'):
            assert not config.get('all_roi_tokens', False) and method != 'raw_supcon'
            assert 'training_budgets' not in config
            if step == 1 or step % config['reconstruction_gradient_interval'] == 0:
                parameters = tuple(model.encoder.parameters())
                task_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(
                    identity_loss, parameters, retain_graph=True)])
                reconstruction_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(
                    config['reconstruction_weight'] * reconstruction_loss, parameters, retain_graph=True)])
                task_norm, reconstruction_norm = task_gradient.norm(), reconstruction_gradient.norm()
                assert task_norm > 0 and reconstruction_norm > 0
                gradient_diagnostics = dict(encoder_gradient_cosine=float(
                    torch.dot(task_gradient, reconstruction_gradient) / (task_norm * reconstruction_norm)),
                    encoder_task_gradient_norm=float(task_norm),
                    encoder_reconstruction_gradient_norm=float(reconstruction_norm))
            if not config['reconstruction_encoder_gradient']:
                decoder_loss = (model.decoder(local.detach()) - samples).square().mean()
                loss = identity_loss + config['reconstruction_weight'] * decoder_loss
        if 'training_budgets' in config:
            assert not config.get('all_roi_tokens', False) and negative_scope == 'global'
            assert model.identity_space == 'code' and model.pooling == 'mean'
            assert config['training_budgets'] == [model.top_k, model.latent_dim]
            full = F.relu(model.encoder(samples))
            full_identity = supcon(F.normalize(full.mean(1), dim=-1), labels[chosen], config['temperature'])
            full_reconstruction = (model.decoder(full) - samples).square().mean()
            loss = .5 * (loss + full_identity + config['reconstruction_weight'] * full_reconstruction)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
        if any(p.grad is None for p in model.parameters()) or norm <= 0:
            raise ValueError("Missing training gradient")
        optimizer.step()
        model.normalize_decoder()
        history.append(dict(step=step, loss=float(loss.detach()), identity=float(identity_loss.detach()),
                            reconstruction=float(reconstruction_loss.detach()), gradient_norm=float(norm)))
        history[-1].update(gradient_diagnostics)
        if 'training_budgets' in config:
            history[-1].update(full_identity=float(full_identity.detach()), full_reconstruction=float(full_reconstruction.detach()))
        if procedure_loss is not None:
            same_procedure = procedures[chosen, None] == procedures[None, chosen]
            different_identity = labels[chosen, None] != labels[None, chosen]
            history[-1].update(global_identity=float(global_identity_loss.detach()),
                procedure_identity=float(procedure_loss.detach()),
                within_procedure_negatives=int((same_procedure & different_identity).sum()),
                cross_procedure_negatives=int((~same_procedure).sum()),
                anchors_with_procedure_negatives=int((same_procedure & different_identity).any(dim=1).sum()),
                anchors=len(chosen))
        if step in checkpoints:
            result = evaluate(model, values, offsets, records, held_videos, folder, f"held_{step:04d}", config)
            evaluations.append(dict(step=step, **result))
            if config.get("save_evaluation_models", False):
                shared.save_npz(folder / f"model_{step:04d}.npz",
                    **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        if step % config["checkpoint_every"] == 0 or step in checkpoints or step == steps or step == stop_after:
            elapsed = previous + time.perf_counter() - started
            shared.save_torch(checkpoint, dict(identity_sha256=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng_json=json.dumps(rng.bit_generator.state), token_rng_json=json.dumps(token_rng.bit_generator.state),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if config["device"] == "cuda" else None,
                step=step, history=history, evaluations=evaluations, sequence=sequence, seconds=elapsed))
            progress = dict(context, status="RUNNING", phase=f"{method} seed{seed}", step=step, steps=steps,
                            seconds=elapsed, losses=history[-1], updated_at=shared.now())
            atomic_write_json(Path(context["progress_path"]), progress)
            print(json.dumps(progress), flush=True)
            pause_after_checkpoint(checkpoint)
            if step == stop_after:
                raise SystemExit(75)
    shared.save_npz(folder / "model.npz", **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    atomic_write_json(folder / "model_config.json", dict(config, method=method))
    atomic_write_json(folder / "history.json", history)
    atomic_write_json(folder / "sequence.json", sequence)
    result = dict(status="COMPLETE", identity_sha256=signature, method=method, seed=seed, steps=steps,
        seconds=previous + time.perf_counter() - started, evaluations=evaluations,
        parameters=sum(p.numel() for p in model.parameters()), sequence_sha256=shared.file_sha256(folder / "sequence.json"),
        model_sha256=shared.file_sha256(folder / "model.npz"), completed_at=shared.now(),
        runtime=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=config["device"],
                     peak_cuda_bytes=torch.cuda.max_memory_allocated() if config["device"] == "cuda" else 0))
    atomic_write_json(folder / "summary.json", result)
    return result


def train(run, smoke, resume, output, stop_after):
    config, cohort, raw, offsets, records, receipts = load_inputs(run)
    methods = tuple(config["methods"])
    interaction = config.get("identity_interaction", "pooled_cosine")
    if interaction not in ("pooled_cosine", "symmetric_maxsim"):
        raise ValueError("Unrecognized identity interaction")
    expected = ("token_sparse", "token_dense") if interaction == "symmetric_maxsim" else METHODS
    if methods != expected:
        raise ValueError("Unrecognized formal method roster")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = output if output else run / "smoke" if smoke else run
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(input_sha256=shared.json_digest(receipts), source_hashes=source_identity())
    atomic_write_json(root / "records.json", records)
    atomic_write_json(root / "input_identity.json", identity)
    folds = shared.make_folds(records, cohort["fit_video_ids"]["train"], config["inner_folds"], config["fold_seed"])
    atomic_write_json(root / "folds.json", folds)
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    active_folds = folds[:1] if smoke else folds
    steps = config["smoke_steps"] if smoke else config["steps"]
    checkpoints = [steps] if smoke else config["checkpoints"]
    total = len(seeds) * len(methods) * (len(active_folds) + 1)
    outputs, selections = [], []
    for seed in seeds:
        for method in methods:
            inner = []
            for fold, held in enumerate(active_folds):
                folder = root / "inner" / method / f"seed{seed}" / f"fold{fold}"
                context = dict(completed_jobs=len(outputs), total_jobs=total, progress_path=str(root / "training_progress.json"))
                fitted = sorted(set(cohort["fit_video_ids"]["train"]) - set(held))
                result = fit(config, method, seed, raw, offsets, records, fitted, held, folder,
                             steps, checkpoints, identity, context, resume, stop_after)
                inner.append(result)
                outputs.append(dict(method=method, seed=seed, fold=fold, directory=str(folder), summary=result))
            candidates = []
            for step in checkpoints:
                rows = [row for item in inner for ev in item["evaluations"] if ev["step"] == step for row in ev["by_video"].values()]
                candidates.append(dict(step=step, recall=float(np.mean([row["recall"] for row in rows])), procedures=len(rows)))
            selected = min(candidates, key=lambda row: (-row["recall"], row["step"]))["step"]
            selections.append(dict(method=method, seed=seed, step=selected, candidates=candidates))
            atomic_write_json(root / "checkpoint_selection.json", selections)
            folder = root / "fit" / method / f"seed{seed}"
            context = dict(completed_jobs=len(outputs), total_jobs=total, progress_path=str(root / "training_progress.json"))
            result = fit(config, method, seed, raw, offsets, records, cohort["fit_video_ids"]["train"],
                cohort["fit_video_ids"]["val"], folder, selected, [selected], identity, context, resume, stop_after)
            outputs.append(dict(method=method, seed=seed, fold="full", directory=str(folder), summary=result))
    for seed in seeds:
        for fold in list(range(len(active_folds))) + ["full"]:
            sequences = [read_json(Path(row["directory"]) / "sequence.json") for row in outputs if row["seed"] == seed and row["fold"] == fold]
            length = min(map(len, sequences))
            if any(seq[:length] != sequences[0][:length] for seq in sequences):
                raise ValueError("Methods received different real training observations")
    if source_identity() != identity["source_hashes"]:
        raise ValueError("Source changed during training")
    atomic_write_json(root / "training_summary.json", dict(status="COMPLETE", outputs=outputs, selection=selections, **identity))
    atomic_write_json(root / "training_progress.json", dict(status="COMPLETE", completed_jobs=total, total_jobs=total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "train"], required=True)
    parser.add_argument("--video")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.video, args.resume)
    else:
        train(args.run, args.smoke, args.resume, args.output, args.stop_after)
