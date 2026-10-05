import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_acknowledgement_sae as shared
from train_token_memory_edit import load_inputs, restore_checkpoint, save_checkpoint
from src.checkpoint_io import atomic_write_json, read_json
from src.shared_component_editor import TaskDictionary, correction_loss, edit_scores
from src.token_memory_edit import FrozenSupCon


def load_data(config):
    parent_config = read_json(config["parent_config"])
    _, records = load_inputs(parent_config)
    folder = Path(config["parent_dictionary_run"]) / "fit/sparse_edit/seed20260929"
    receipt = read_json(folder / "pool_summary.json")
    if receipt["status"] != "COMPLETE" or shared.file_sha256(folder / "pooled_views.npz") != receipt["sha256"]:
        raise ValueError("Parent pooled observations differ")
    with np.load(folder / "pooled_views.npz", allow_pickle=False) as archive:
        raw, valid = archive["raw"].copy(), archive["valid"].copy()
    if raw.shape != (336, 7, 768) or valid.shape != (336, 7) or not valid[:, 0].all():
        raise ValueError("Unexpected pooled observation dimensions")
    folds = []
    for index in range(3):
        identity = read_json(Path(config["parent_dictionary_run"]) /
                             f"inner_folds/sparse_edit/seed20260929/fold{index}/identity.json")
        head = Path(config["parent_head_run"]) / f"heads/fold{index}"
        summary = read_json(head / "summary.json")
        if summary["fit_video_ids"] != identity["fit_videos"] or summary["held_video_ids"] != identity["held_videos"]:
            raise ValueError("Task head training membership differs")
        for name in ("model.npz", "normalization.npz"):
            if shared.file_sha256(head / name) != summary["assets"][name]:
                raise ValueError("Task head asset changed")
        folds.append(dict(fold=index, fit=identity["fit_videos"], held=identity["held_videos"], head=str(head)))
    folds.append(dict(fold="full", fit=parent_config["train_video_ids"],
                      held=parent_config["validation_video_ids"], head=config["reference_fit"]))
    if sorted(v for fold in folds[:3] for v in fold["held"]) != sorted(folds[-1]["fit"]):
        raise ValueError("Outer folds do not partition the training cohort")
    for fold in folds:
        if set(fold["fit"]) & set(fold["held"]):
            raise ValueError("Task component procedure leakage")
    return records, raw, valid, folds, dict(pool_sha256=receipt["sha256"],
        records_sha256=shared.json_digest(records), source_directory=str(folder))


@torch.no_grad()
def task_embeddings(raw, valid, head, device):
    reference = FrozenSupCon(head).to(device)
    values = torch.as_tensor(raw[valid], device=device)
    encoded = reference(values)
    if not torch.isfinite(encoded).all() or torch.any(encoded.norm(dim=-1) < .999):
        raise ValueError("Invalid fixed task embedding")
    result = torch.zeros((*valid.shape, encoded.shape[-1]), device=device)
    result[torch.from_numpy(valid).to(device)] = encoded
    with np.load(Path(head) / "model.npz", allow_pickle=False) as model, np.load(Path(head) / "normalization.npz", allow_pickle=False) as norm:
        standardized = ((raw[valid].astype(np.float64) - norm["mean"]) / norm["scale"]).astype(np.float32)
        hidden = np.maximum(standardized @ model["0.weight"].T + model["0.bias"], 0)
        independent = hidden @ model["2.weight"].T + model["2.bias"]
        independent /= np.linalg.norm(independent, axis=1, keepdims=True)
    error = float(np.max(np.abs(independent - encoded.cpu().numpy())))
    if error > 2e-6:
        raise ValueError(f"Independent fixed task reproduction failed: {error}")
    return result, error


def observation_weights(records, valid, videos):
    weights = np.zeros(valid.shape, dtype=np.float64)
    for video in videos:
        selected = [i for i, row in enumerate(records) if row["video_id"] == video]
        for index in selected:
            weights[index, valid[index]] = 1 / (len(videos) * len(selected) * int(valid[index].sum()))
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    return weights


def pair_design(records, valid, videos, augmented, device):
    base = shared.chronological_pairs(records, videos)
    pairs = [dict(pair, view=int(view)) for pair in base
             for view in (np.flatnonzero(valid[pair["source_index"]]) if augmented else [0])]
    if not pairs:
        raise ValueError("No chronological identity pairs")
    labels = np.array([r["same_identity"] for r in pairs])
    weights = np.zeros(len(pairs))
    defined = [v for v in videos if len({r["same_identity"] for r in pairs if r["video_id"] == v}) == 2]
    if not defined:
        raise ValueError("No procedure with both identity classes")
    for video in defined:
        for label in [False, True]:
            selected = [i for i, r in enumerate(pairs) if r["video_id"] == video and r["same_identity"] == label]
            lesions = sorted({pairs[i]["source_lesion_id"] for i in selected})
            for lesion in lesions:
                local = [i for i in selected if pairs[i]["source_lesion_id"] == lesion]
                sources = sorted({pairs[i]["source_index"] for i in local})
                for source in sources:
                    observations = [i for i in local if pairs[i]["source_index"] == source]
                    queries = sorted({pairs[i]["query_index"] for i in observations})
                    for query in queries:
                        views = [i for i in observations if pairs[i]["query_index"] == query]
                        weights[views] = 1 / (len(defined) * 2 * len(lesions) * len(sources) * len(queries) * len(views))
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    return dict(pairs=pairs, source=np.array([r["source_index"] for r in pairs]),
        query=np.array([r["query_index"] for r in pairs]), view=np.array([r["view"] for r in pairs]),
        labels=torch.as_tensor(labels, device=device), weights=torch.as_tensor(weights, dtype=torch.float32, device=device),
        defined=defined, excluded=sorted(set(videos) - set(defined)))


def weighted_threshold(scores, design, quantile):
    weights = design["weights"].detach().cpu().numpy()
    negative = (~design["labels"].cpu().numpy()) & (weights > 0)
    return float(np.quantile(scores[negative], quantile, weights=weights[negative], method="inverted_cdf"))


def pair_values(embedding, code, design):
    first = embedding[design["source"], design["view"]]
    second = embedding[design["query"], 0]
    return first, second, code[design["source"], design["view"]], code[design["query"], 0]


def save_weights(path, model):
    shared.save_npz(path, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})


def progress(output, completed, total, label, step, steps, started, loss=None):
    value = dict(status="RUNNING", completed_jobs=completed, total_jobs=total, stage=label,
        step=step, steps=steps, elapsed_seconds=time.perf_counter() - started, updated_at=shared.now())
    if loss is not None:
        value["loss"] = float(loss)
    atomic_write_json(output / "training_progress.json", value)
    print(f"COMPONENT_JOB {completed}/{total} {label} {step}/{steps} loss={loss}", flush=True)


def optimize(model, objective, directory, config, identity, steps, lr, context, resume, stop_step, dictionary):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(context["seed"])
    checkpoint = directory / "checkpoint.pt"
    restored = restore_checkpoint(checkpoint, model, optimizer, rng, identity, config["device"], resume)
    first = int(restored["step"]) + 1 if restored else 1
    history = restored["history"] if restored else []
    elapsed = float(restored["elapsed_seconds"]) if restored else 0.
    started = time.perf_counter()
    for step in range(first, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite task-component objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite task-component gradient")
        optimizer.step()
        with torch.no_grad():
            if dictionary:
                model.normalize_decoder()
            else:
                model["gains"].clamp_(0., 1.)
        if step % config["checkpoint_every"] == 0 or step == steps or step == stop_step:
            history.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm)))
            save_checkpoint(checkpoint, model, optimizer, rng, step, history, [], identity,
                            elapsed + time.perf_counter() - started)
            progress(context["output"], context["completed"], context["total"], context["label"],
                     step, steps, context["started"], float(loss.detach()))
            if step == stop_step and step < steps:
                raise SystemExit(75)
    return dict(history=history, seconds=elapsed + time.perf_counter() - started, steps=steps)


@torch.no_grad()
def evaluate_policy(folder, embedding, code, directions, gains, designs, config, mode, seed):
    reports = {}
    for name, design in designs.items():
        first, second, a, b = pair_values(embedding, code, design)
        before = (first * second).sum(-1)
        after, delta, coefficients = edit_scores(first, second, a, b, directions, gains, config["edit_budget"], mode)
        if not torch.isfinite(after).all() or torch.any(delta.norm(dim=-1) > config["edit_budget"] + 1e-6):
            raise ValueError("Invalid bounded component effect")
        arrays = dict(before=before.cpu().numpy(), after=after.cpu().numpy(),
            source=design["source"], query=design["query"], view=design["view"],
            labels=design["labels"].cpu().numpy(), weights=design["weights"].cpu().numpy(),
            edit_norm=delta.norm(dim=-1).cpu().numpy(), shared_active=(coefficients != 0).sum(-1).cpu().numpy())
        for index in range(config["random_controls"]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 63821]))
            rotation, _ = np.linalg.qr(rng.normal(size=(config["dimension"], config["dimension"])))
            altered, rotated, _ = edit_scores(first, second, a, b, directions, gains, config["edit_budget"], mode,
                torch.tensor(rotation, dtype=embedding.dtype, device=embedding.device))
            torch.testing.assert_close(rotated.norm(dim=-1), delta.norm(dim=-1), atol=2e-7, rtol=2e-6)
            arrays[f"random{index}"] = altered.cpu().numpy()
        selected = torch.argsort(gains, descending=True, stable=True)[:8]
        component_effects = []
        for component in selected:
            reduced = gains.clone()
            reduced[component] = 0
            altered, _, _ = edit_scores(first, second, a, b, directions, reduced, config["edit_budget"], mode)
            component_effects.append((after - altered).cpu().numpy())
        arrays["component_indices"] = selected.cpu().numpy()
        arrays["component_effects"] = np.stack(component_effects)
        shared.save_npz(folder / f"{name}.npz", **arrays)
        atomic_write_json(folder / f"{name}_pairs.json", design["pairs"])
        reports[name] = dict(pairs=len(design["pairs"]), procedures=design["defined"], excluded=design["excluded"],
            before_threshold=weighted_threshold(arrays["before"], design, config["negative_quantile"]),
            after_threshold=weighted_threshold(arrays["after"], design, config["negative_quantile"]),
            mean_edit_norm=float(arrays["edit_norm"].mean()), mean_shared_active=float(arrays["shared_active"].mean()))
    atomic_write_json(folder / "evaluation.json", reports)
    return reports


def run_batch(run, smoke, resume, stop_step):
    config = read_json(run / "config.json")
    output = run / "smoke" if smoke else run
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    records, raw, valid, folds, data_identity = load_data(config)
    if smoke:
        folds = [folds[0], folds[-1]]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    code_identity = {name: shared.file_sha256(ROOT / name) for name in [
        "scripts/train_shared_task_components.py", "src/shared_component_editor.py",
        "scripts/train_token_memory_edit.py", "scripts/train_acknowledgement_sae.py", "src/token_memory_edit.py"]}
    batch_identity = dict(config=shared.file_sha256(run / "config.json"), code=code_identity, data=data_identity,
                          smoke=smoke, torch=str(torch.__version__), numpy=np.__version__, python=sys.version)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != batch_identity:
        raise ValueError("Batch identity changed")
    atomic_write_json(output / "identity.json", batch_identity)
    atomic_write_json(output / "records.json", records)
    atomic_write_json(output / "folds.json", folds)
    completed, total = 0, len(folds) * len(seeds) * (2 + len(config["basis_methods"]) * len(config["modes"]))
    started, results = time.perf_counter(), []
    for fold in folds:
        embedding, reproduction_error = task_embeddings(raw, valid, fold["head"], config["device"])
        obs_weights = observation_weights(records, valid, fold["fit"])
        fit_positions = obs_weights > 0
        inputs = embedding[torch.as_tensor(fit_positions, device=config["device"])]
        weights = torch.as_tensor(obs_weights[fit_positions], dtype=inputs.dtype, device=inputs.device)
        designs = {name: pair_design(records, valid, videos, augmented, config["device"])
                   for name, videos, augmented in [("fit", fold["fit"], True), ("held", fold["held"], False)]}
        head_hashes = {name: shared.file_sha256(Path(fold["head"]) / name) for name in ["model.npz", "normalization.npz"]}
        for seed in seeds:
            for basis in config["basis_methods"]:
                folder = output / "models" / f"fold{fold['fold']}" / f"seed{seed}" / basis
                folder.mkdir(parents=True, exist_ok=True)
                identity = shared.json_digest(dict(batch=batch_identity, fold=fold, head=head_hashes, seed=seed, basis=basis))
                spec = dict(identity_sha256=identity, fold=fold, seed=seed, basis=basis, head_hashes=head_hashes,
                    task_reproduction_max_error=reproduction_error)
                atomic_write_json(folder / "identity.json", spec)
                context = dict(output=output, completed=completed, total=total, started=started,
                               seed=seed, label=f"fold{fold['fold']} {seed} {basis} dictionary")
                if basis in ["sparse", "dense"]:
                    torch.manual_seed(seed)
                    model = TaskDictionary(config["dimension"], config["width"], config["top_k"], basis == "sparse").to(config["device"])
                    if (folder / "dictionary.json").exists():
                        receipt = read_json(folder / "dictionary.json")
                        if not resume or receipt["identity_sha256"] != identity or shared.file_sha256(folder / "dictionary.npz") != receipt["sha256"]:
                            raise ValueError("Dictionary receipt changed")
                        with np.load(folder / "dictionary.npz", allow_pickle=False) as archive:
                            model.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
                    else:
                        loss = lambda: ((model(inputs) - inputs).square().sum(-1) * weights).sum()
                        fit = optimize(model, loss, folder, config, identity,
                            config["smoke_steps"] if smoke else config["dictionary_steps"],
                            config["dictionary_learning_rate"], context, resume, stop_step, True)
                        save_weights(folder / "dictionary.npz", model)
                        atomic_write_json(folder / "dictionary.json", dict(fit, identity_sha256=identity,
                            sha256=shared.file_sha256(folder / "dictionary.npz")))
                    model.eval().requires_grad_(False)
                    with torch.no_grad():
                        code = model.encode(embedding)
                        directions = model.decoder.weight.detach()
                        reconstruction = model(embedding)
                    completed += 1
                else:
                    if basis == "pca":
                        weighted = inputs.cpu().numpy().astype(np.float64) * np.sqrt(weights.cpu().numpy())[:, None]
                        _, _, vt = np.linalg.svd(weighted, full_matrices=False)
                        directions = torch.tensor(vt.T, dtype=inputs.dtype, device=inputs.device)
                    elif basis == "coordinates":
                        directions = torch.eye(config["dimension"], device=inputs.device)
                    else:
                        raise ValueError(basis)
                    code = embedding @ directions
                    reconstruction = code @ directions.T
                with torch.no_grad():
                    nmse = (reconstruction - embedding).square().sum(-1)
                shared.save_npz(folder / "basis.npz", directions=directions.cpu().numpy(),
                    code=code.cpu().numpy(), embedding=embedding.cpu().numpy(), valid=valid, nmse=nmse.cpu().numpy())
                for mode in config["modes"]:
                    target = folder / mode
                    target.mkdir(exist_ok=True)
                    mode_identity = shared.json_digest(dict(identity=identity, mode=mode))
                    gate = nn.ParameterDict({"gains": nn.Parameter(torch.zeros(directions.shape[1], device=inputs.device))})
                    context.update(completed=completed, label=f"fold{fold['fold']} {seed} {basis} {mode}")
                    before = (embedding[designs["fit"]["source"], designs["fit"]["view"]] * embedding[designs["fit"]["query"], 0]).sum(-1)
                    threshold = weighted_threshold(before.cpu().numpy(), designs["fit"], config["negative_quantile"])
                    values = pair_values(embedding, code, designs["fit"])
                    if (target / "complete.json").exists():
                        receipt = read_json(target / "complete.json")
                        if not resume or receipt["identity_sha256"] != mode_identity or shared.file_sha256(target / "gains.npz") != receipt["gains_sha256"]:
                            raise ValueError("Completed gain policy changed")
                    else:
                        def objective():
                            scores, _, _ = edit_scores(*values, directions, gate["gains"], config["edit_budget"], mode)
                            return correction_loss(scores, designs["fit"]["labels"], designs["fit"]["weights"],
                                threshold, config["temperature"], gate["gains"], config["gain_penalty"])
                        fit = optimize(gate, objective, target, config, mode_identity,
                            config["smoke_steps"] if smoke else config["gate_steps"], config["gate_learning_rate"],
                            context, resume, stop_step, False)
                        save_weights(target / "gains.npz", gate)
                        reports = evaluate_policy(target, embedding, code, directions, gate["gains"], designs, config, mode, seed)
                        receipt = dict(status="COMPLETE", identity_sha256=mode_identity, fit=fit, reports=reports,
                            gains_sha256=shared.file_sha256(target / "gains.npz"), completed_at=shared.now())
                        atomic_write_json(target / "complete.json", receipt)
                    completed += 1
                    results.append(dict(directory=str(target), basis=basis, mode=mode, fold=fold["fold"], seed=seed,
                        fit_videos=fold["fit"], held_videos=fold["held"], receipt=receipt))
                    progress(output, completed, total, context["label"], receipt["fit"]["steps"], receipt["fit"]["steps"], started)
    atomic_write_json(output / "training_summary.json", dict(status="COMPLETE", outputs=results,
        completed_at=shared.now(), jobs=completed, seconds=time.perf_counter() - started, identity=batch_identity))
    atomic_write_json(output / "training_progress.json", dict(status="COMPLETE", completed_jobs=completed,
        total_jobs=total, updated_at=shared.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    run_batch(args.run, args.smoke, args.resume, args.stop_after_step)
