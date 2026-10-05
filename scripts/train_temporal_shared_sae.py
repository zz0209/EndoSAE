import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_acknowledgement_sae as shared_api
from train_region_identity_sae import load_inputs
from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.temporal_shared_sae import METHODS, TemporalSharedSAE, load_predictor, training_loss


def cross_observation_bank(records, videos):
    bank = shared_api.episode_bank(records, videos)
    result = []
    for item in bank:
        interval = records[item["source"]]["annotation_observation"]["interval_id"]
        positives = [i for i in item["positive"] if records[i]["annotation_observation"]["interval_id"] != interval]
        if positives:
            result.append(dict(item, positive=positives))
    if not result:
        raise ValueError("No genuine cross-interval identity episodes")
    return result


def protected_statistics(pairs, scores, quantile):
    videos = sorted({row["video_id"] for row in pairs})
    labels = np.asarray([row["same_identity"] for row in pairs])
    weights = np.zeros(len(pairs), dtype=np.float64)
    eligible = []
    for video in videos:
        mask = np.asarray([row["video_id"] == video for row in pairs])
        if np.any(mask & labels) and np.any(mask & ~labels):
            eligible.append(video)
            weights[mask & ~labels] = 1 / np.sum(mask & ~labels)
    negative = weights > 0
    if not eligible:
        raise ValueError("No held procedure contains both identity classes")
    threshold = float(np.quantile(scores[negative], quantile, method="inverted_cdf", weights=weights[negative]))
    rows = {}
    for video in eligible:
        mask = np.asarray([row["video_id"] == video for row in pairs])
        rows[video] = dict(recall=float(np.mean(scores[mask & labels] > threshold)),
                           negative_retention=float(np.mean(scores[mask & ~labels] <= threshold)))
    return dict(threshold=threshold, by_video=rows, macro_recall=float(np.mean([r["recall"] for r in rows.values()])),
                excluded_without_both_classes=sorted(set(videos) - set(eligible)))


@torch.no_grad()
def evaluate(model, features, records, videos, output, stem, config, interventions=False):
    model.eval()
    pairs = shared_api.chronological_pairs(records, videos)
    source = np.asarray([row["source_index"] for row in pairs])
    query = np.asarray([row["query_index"] for row in pairs])
    labels = np.asarray([row["same_identity"] for row in pairs])
    stable, private = model.encode_parts(features[:, 0])
    projected = F.normalize(model.readout(stable), dim=-1)
    if not torch.isfinite(projected).all() or torch.any(projected.norm(dim=-1) <= 0):
        raise ValueError("Invalid projected identity code")
    scores = (projected[source] * projected[query]).sum(-1).numpy()
    stable_unit, private_unit = F.normalize(stable, dim=-1), F.normalize(private, dim=-1)
    same_interval = np.asarray([row["same_annotation_interval"] for row in pairs])
    shared_cosine = (stable_unit[source] * stable_unit[query]).sum(-1).numpy()
    private_cosine = (private_unit[source] * private_unit[query]).sum(-1).numpy()
    support = stable > 0
    intersection = (support[source] & support[query]).sum(-1)
    union = (support[source] | support[query]).sum(-1)
    if torch.any(union <= 0):
        raise ValueError("Empty shared support")
    jaccard = (intersection / union).numpy()
    positive = np.flatnonzero(labels & ~same_interval)
    swap_error = np.full(len(pairs), np.nan)
    for start in range(0, len(positive), 256):
        selected = positive[start:start + 256]
        a, b = source[selected], query[selected]
        swap_error[selected] = .5 * ((model.reconstruct(stable[b], private[a]) - features[a, 0]).square().mean(-1)
                                    + (model.reconstruct(stable[a], private[b]) - features[b, 0]).square().mean(-1)).numpy()
    self_error = (model.reconstruct(stable, private) - features[:, 0]).square().mean(-1).numpy()
    report = protected_statistics(pairs, scores, float(config["negative_quantile"]))
    report["identity"] = shared_api.pair_statistics(pairs, scores)
    report["mechanism_by_video"] = {}
    for video in videos:
        mask = np.asarray([r["video_id"] == video for r in pairs])
        cross = mask & labels & ~same_interval
        negative = mask & ~labels
        own = np.asarray([r["video_id"] == video for r in records])
        row = dict(clips=int(own.sum()), self_mse=float(self_error[own].mean()), cross_pairs=int(cross.sum()),
                   negative_pairs=int(negative.sum()), shared_active=float(support[own].sum(-1).float().mean()),
                   private_active=float((private[own] > 0).sum(-1).float().mean()))
        for name, values in (("shared_cosine", shared_cosine), ("private_cosine", private_cosine), ("jaccard", jaccard)):
            row[name + "_same"] = float(values[cross].mean()) if cross.any() else None
            row[name + "_other"] = float(values[negative].mean()) if negative.any() else None
        row["swap_mse"] = float(swap_error[cross].mean()) if cross.any() else None
        report["mechanism_by_video"][video] = row
    arrays = dict(source=source, query=query, same_identity=labels, same_interval=same_interval, scores=scores,
                  shared_cosine=shared_cosine, private_cosine=private_cosine, shared_jaccard=jaccard,
                  swap_mse=swap_error, clip_self_mse=self_error)
    if interventions:
        altered, details = component_interventions(model, stable, projected, source, query, config)
        arrays.update(altered)
        atomic_write_json(output / (stem + "_interventions.json"), details)
    shared_api.save_npz(output / (stem + "_pairs.npz"), **arrays)
    atomic_write_json(output / (stem + ".json"), report)
    return report


@torch.no_grad()
def component_interventions(model, stable, projected, source, query, config):
    arrays = {name: np.full(len(source), np.nan) for name in ["selected"] + [f"random{i}" for i in range(config["random_controls"]) ]}
    details = []
    column_norm = model.readout.weight.norm(dim=0).numpy()
    for index in np.unique(source):
        code = stable[index].numpy()
        active = np.flatnonzero(code > 0)
        count = min(int(config["intervention_components"]), len(active) - 1)
        if count < 1:
            details.append(dict(source_index=int(index), status="UNDEFINED", active=len(active)))
            continue
        ranking = np.argsort(-(code[active] * column_norm[active]), kind="stable")
        groups = {"selected": active[ranking[:count]]}
        for control in range(config["random_controls"]):
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], int(index), control, 81917]))
            groups[f"random{control}"] = rng.choice(active, size=count, replace=False)
        norms = {name: float(np.linalg.norm(code[indices])) for name, indices in groups.items()}
        dose = min(norms.values())
        positions = np.flatnonzero(source == index)
        for name, indices in groups.items():
            altered = stable[index].clone()
            altered[indices] -= stable[index, indices] * (dose / norms[name])
            memory = F.normalize(model.readout(altered), dim=-1)
            if memory.norm() <= 0 or not torch.isfinite(memory).all():
                raise ValueError("Component removal produced an empty identity memory")
            arrays[name][positions] = (projected[query[positions]] @ memory).numpy()
        details.append(dict(source_index=int(index), status="DEFINED", active=len(active), removed_components=count,
                            latent_norm=dose, groups={name: values.tolist() for name, values in groups.items()},
                            unscaled_norms=norms, renormalized=True))
    return arrays, details


def source_hashes():
    paths = [Path(__file__), ROOT / "src/temporal_shared_sae.py", ROOT / "src/acknowledgement_sae.py",
             ROOT / "scripts/train_acknowledgement_sae.py", ROOT / "scripts/train_region_identity_sae.py",
             ROOT / "src/checkpoint_io.py"]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def training_job(config, method, seed, views, valid, records, fit_videos, held_videos, output,
                 steps, checkpoints, identity, context, resume, stop_after_step=None):
    output.mkdir(parents=True, exist_ok=True)
    job = dict(config_sha256=shared_api.json_digest(config), method=method, seed=seed,
               fit_video_ids=sorted(fit_videos), held_video_ids=sorted(held_videos), steps=steps,
               checkpoints=checkpoints, **identity)
    signature = shared_api.json_digest(job)
    if (output / "summary.json").exists():
        saved = read_json(output / "summary.json")
        if not resume or saved["identity_sha256"] != signature or saved["status"] != "COMPLETE":
            raise ValueError("Existing training job cannot be reused")
        return saved
    if (output / "identity.json").exists() and read_json(output / "identity.json") != job:
        raise ValueError("Existing job identity changed")
    atomic_write_json(output / "identity.json", job)
    indices = np.asarray([i for i, r in enumerate(records) if r["video_id"] in fit_videos])
    scaler = StandardScaler().fit(views[indices, 0])
    features = torch.from_numpy(((views - scaler.mean_) / scaler.scale_).astype(np.float32))
    shared_api.save_npz(output / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_)
    labels_map = {key: i for i, key in enumerate(sorted({(r["video_id"], r["lesion_id"]) for r in records}))}
    labels = torch.tensor([labels_map[(r["video_id"], r["lesion_id"])] for r in records])
    bank = cross_observation_bank(records, fit_videos)
    atomic_write_json(output / "training_episodes.json", bank)
    spec = dict(method=method, **{key: config[key] for key in ("input_dim", "shared_dim", "private_dim", "readout_dim", "top_k")})
    atomic_write_json(output / "model_config.json", spec)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    view_rng = np.random.default_rng(np.random.SeedSequence([seed, 71103]))
    model = TemporalSharedSAE(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    views_by_row = [np.flatnonzero(row) for row in valid]
    history, evaluations, sequence = [], [], []
    initial_step, elapsed_before = 1, 0.
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity_sha256"] != signature:
            raise ValueError("Checkpoint identity changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        rng.bit_generator.state = json.loads(saved["numpy_rng_json"])
        view_rng.bit_generator.state = json.loads(saved["view_rng_json"])
        history, evaluations, sequence = saved["history"], saved["evaluations"], saved["sequence"]
        initial_step, elapsed_before = saved["step"] + 1, saved["elapsed_seconds"]
    started = time.perf_counter()
    for step in range(initial_step, steps + 1):
        model.train()
        batch = shared_api.sample_batch(bank, indices, rng, config)
        selected = batch["indices"]
        chosen_views = np.asarray([view_rng.choice(views_by_row[i]) for i in selected])
        sequence.append(dict(**{key: value.tolist() for key, value in batch.items()}, views=chosen_views.tolist()))
        loss, losses, stable, private = training_loss(model, features[selected, chosen_views], labels[selected], batch, config)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Missing or nonfinite gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"]))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("No finite nonzero gradient")
        optimizer.step()
        model.normalize_decoder()
        row = dict(step=step, loss=float(loss.detach()), **{key: float(value.detach()) for key, value in losses.items()},
                   gradient_norm=gradient_norm, shared_active=float((stable > 0).sum(-1).float().mean()),
                   private_active=float((private > 0).sum(-1).float().mean()))
        history.append(row)
        if step in checkpoints:
            report = evaluate(model, features, records, held_videos, output, f"held_step{step:04d}", config)
            evaluations.append(dict(step=step, **report))
        if step % config["checkpoint_every"] == 0 or step in checkpoints or step == steps or step == stop_after_step:
            elapsed = elapsed_before + time.perf_counter() - started
            shared_api.save_torch(checkpoint, dict(identity_sha256=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(), numpy_rng_json=json.dumps(rng.bit_generator.state),
                view_rng_json=json.dumps(view_rng.bit_generator.state), history=history, evaluations=evaluations,
                sequence=sequence, step=step, elapsed_seconds=elapsed))
            progress = dict(context, status="RUNNING", method=method, seed=seed, step=step, steps=steps,
                            elapsed_seconds=elapsed, losses=row, updated_at=shared_api.now())
            atomic_write_json(output / "progress.json", progress)
            atomic_write_json(Path(context["progress_path"]), progress)
            print(json.dumps(progress), flush=True)
            pause_after_checkpoint(checkpoint)
            if step == stop_after_step:
                raise SystemExit(75)
    shared_api.save_npz(output / "model.npz", **{key: value.detach().numpy() for key, value in model.state_dict().items()})
    atomic_write_json(output / "history.json", history)
    atomic_write_json(output / "sequence.json", sequence)
    final = evaluate(model, features, records, held_videos, output, "held_final", dict(config, seed=seed), interventions=True)
    predictor = load_predictor(output)
    selected = np.asarray([i for i, r in enumerate(records) if r["video_id"] in held_videos])
    actual = predictor.encode(views[selected, 0])
    with torch.no_grad():
        expected = F.normalize(model.encode(features[selected, 0]), dim=-1).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    result = dict(status="COMPLETE", method=method, seed=seed, identity_sha256=signature, steps=steps,
        seconds=elapsed_before + time.perf_counter() - started, evaluations=evaluations, final=final,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        model_sha256=file_sha256(output / "model.npz"), normalization_sha256=file_sha256(output / "normalization.npz"),
        sequence_sha256=file_sha256(output / "sequence.json"), export_max_error=float(np.max(np.abs(actual - expected))),
        completed_at=shared_api.now(), runtime=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__), threads=1))
    atomic_write_json(output / "summary.json", result)
    return result


def train(run, smoke, resume, stop_after_step, output_override=None):
    config = read_json(run / "config.json")
    if config["device"] != "cpu" or config["threads"] != 1 or tuple(config["methods"]) != METHODS:
        raise ValueError("Unexpected frozen training configuration")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    views, valid, records, receipts = load_inputs(config)
    root = output_override if output_override else run / "smoke" if smoke else run
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "descriptor_records.json", records)
    atomic_write_json(root / "input_identity.json", receipts)
    identity = dict(input_sha256=shared_api.json_digest(receipts), source_hashes=source_hashes())
    folds = shared_api.make_folds(records, config["train_video_ids"], config["inner_folds"], config["fold_seed"])
    atomic_write_json(root / "grouped_folds.json", folds)
    steps = config["smoke_steps"] if smoke else config["steps"]
    checkpoints = [steps] if smoke else config["checkpoints"]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    active_folds = folds[:1] if smoke else folds
    outputs, selections = [], []
    total = len(seeds) * len(METHODS) * (len(active_folds) + 1)
    for seed in seeds:
        for method in METHODS:
            inner = []
            for fold, held in enumerate(active_folds):
                context = dict(phase="training", fold=fold, completed_jobs=len(outputs), total_jobs=total,
                               progress_path=str(root / "training_progress.json"))
                folder = root / "inner_folds" / method / f"seed{seed}" / f"fold{fold}"
                item = training_job(config, method, seed, views, valid, records,
                    sorted(set(config["train_video_ids"]) - set(held)), held, folder, steps, checkpoints,
                    identity, context, resume, stop_after_step)
                inner.append(item)
                outputs.append(dict(method=method, seed=seed, fold=fold, directory=str(folder), summary=item))
            candidates = []
            for checkpoint in checkpoints:
                rows = [row for item in inner for evaluation in item["evaluations"] if evaluation["step"] == checkpoint
                        for row in evaluation["by_video"].values()]
                candidates.append(dict(step=checkpoint, recall=float(np.mean([r["recall"] for r in rows])),
                                       eligible_procedures=len(rows)))
            selected = min(candidates, key=lambda r: (-r["recall"], r["step"]))["step"]
            selections.append(dict(method=method, seed=seed, selected_steps=selected, candidates=candidates))
            atomic_write_json(root / "checkpoint_selection.json", selections)
            context = dict(phase="training", fold="full", completed_jobs=len(outputs), total_jobs=total,
                           progress_path=str(root / "training_progress.json"))
            folder = root / "fit" / method / f"seed{seed}"
            item = training_job(config, method, seed, views, valid, records, config["train_video_ids"],
                config["validation_video_ids"], folder, selected, [selected], identity, context, resume, stop_after_step)
            outputs.append(dict(method=method, seed=seed, fold="full", directory=str(folder), summary=item))
    for seed in seeds:
        for fold in list(range(len(active_folds))) + ["full"]:
            sequences = [read_json(Path(r["directory"]) / "sequence.json") for r in outputs if r["seed"] == seed and r["fold"] == fold]
            length = min(map(len, sequences))
            if any(s[:length] != sequences[0][:length] for s in sequences):
                raise ValueError("Methods received different training inputs")
    if source_hashes() != identity["source_hashes"]:
        raise ValueError("Source changed during training")
    atomic_write_json(root / "training_summary.json", dict(status="COMPLETE", outputs=outputs, selection=selections, **identity))
    atomic_write_json(root / "training_progress.json", dict(status="COMPLETE", completed_jobs=total, total_jobs=total,
                                                           updated_at=shared_api.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train(args.run.resolve(), args.smoke, args.resume, args.stop_after_step, args.output.resolve() if args.output else None)
