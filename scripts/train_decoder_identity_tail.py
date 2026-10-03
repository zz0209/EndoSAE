import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import discover_component_memory as parent
import discover_crossfit_component_memory as crossfit
from src.checkpoint_io import atomic_write_json, read_json
from src.conditional_component_policy import source_weights, supervised_targets
from src.token_memory_edit import BoundedGains, FrozenSupCon, residual_edit


def hashes():
    result = crossfit.source_hashes()
    result[str(Path(__file__).relative_to(ROOT))] = parent.file_sha256(__file__)
    return result


def agreement_gradient(gradients):
    stacked = torch.stack(gradients)
    agreed = (stacked > 0).all(0) | (stacked < 0).all(0)
    summed = stacked.sum(0)
    return torch.where(agreed, summed, torch.zeros_like(summed)), agreed


def design(assets, records, videos, reference, include_canonical):
    pairs = parent.shared.shared.chronological_pairs(records, videos)
    rows, excluded = [], []
    for source in sorted({p["source_index"] for p in pairs}):
        local = [p for p in pairs if p["source_index"] == source]
        positive = [p["query_index"] for p in local if p["same_identity"]]
        negative = [p["query_index"] for p in local if not p["same_identity"]]
        if not positive or not negative:
            excluded.append(dict(source_index=source, video_id=records[source]["video_id"],
                                 positives=len(positive), negatives=len(negative)))
            continue
        views = np.flatnonzero(assets["valid"][source])
        for view in views if include_canonical else views[views != 0]:
            rows.append(dict(source_index=source, view=int(view), video_id=records[source]["video_id"],
                             positives=positive, negatives=negative))
    if not rows:
        raise ValueError("No chronological source has both identity classes")
    device = assets["model"].decoder.weight.device
    sources = np.array([r["source_index"] for r in rows])
    views = np.array([r["view"] for r in rows])
    with torch.no_grad():
        queries = reference(torch.as_tensor(assets["raw"][:, 0], device=device))
    return dict(rows=rows, excluded=excluded, reference=reference, queries=queries,
        raw=torch.as_tensor(assets["raw"][sources, views], device=device),
        canonical=torch.as_tensor(assets["raw"][sources, 0], device=device),
        codes=torch.as_tensor(assets["codes"][sources, views], device=device),
        weights=torch.as_tensor(source_weights(rows), device=device))


def loss_terms(model, gains, scale, original_decoder, part, config):
    edited = residual_edit(part["raw"], part["codes"], gains(), model, scale)
    memories = part["reference"](edited)
    scores = memories.double() @ part["queries"].double().T
    tail = []
    for index, row in enumerate(part["rows"]):
        threshold = torch.quantile(scores[index, row["negatives"]], config["negative_quantile"])
        tail.append(F.softplus((threshold - scores[index, row["positives"]]) /
                               config["utility_temperature"]).mean())
    tail_loss = part["weights"] @ torch.stack(tail)
    canonical_loss = part["weights"] @ (((edited - part["canonical"]) / scale).square().mean(-1))
    drift = part["codes"] @ (model.decoder.weight - original_decoder).T
    anchor_loss = part["weights"] @ drift.double().square().mean(-1)
    return canonical_loss, tail_loss, anchor_loss


@torch.no_grad()
def measure(assets, model, gains, records, videos, include_canonical=False):
    part = design(assets, records, videos, assets["reference"], include_canonical)
    edited = residual_edit(part["raw"], part["codes"], gains(), model,
                           torch.as_tensor(assets["scale"], device=part["raw"].device))
    memories = assets["reference"](edited).cpu().numpy()
    original = assets["reference"](part["raw"]).cpu().numpy()
    queries = part["queries"].cpu().numpy()
    rows, scores_saved = [], {}
    for index, row in enumerate(part["rows"]):
        thresholds, utility, scores = supervised_targets(np.vstack([original[index], memories[index]]),
            queries, row["positives"], row["negatives"], .99, .05)
        recall = [float(np.mean(s[row["positives"]] > threshold)) for s, threshold in zip(scores, thresholds)]
        rows.append(dict(row, zero_utility=float(utility[0]), utility=float(utility[1]),
            zero_recall=recall[0], recall=recall[1], zero_threshold=float(thresholds[0]), threshold=float(thresholds[1])))
        scores_saved[f"scores_{index}"] = scores[:, row["positives"] + row["negatives"]]
    weights = source_weights(rows)
    metrics = ["zero_utility", "utility", "zero_recall", "recall"]
    means = {name: float(weights @ np.array([r[name] for r in rows])) for name in metrics}
    by_video = {}
    for video in sorted({r["video_id"] for r in rows}):
        selected = np.array([r["video_id"] == video for r in rows])
        local_weights = weights[selected] / weights[selected].sum()
        by_video[video] = {name: float(local_weights @ np.array([r[name] for r in rows])[selected]) for name in metrics}
    return dict(means=means, procedure_results=by_video, sources=rows, excluded=part["excluded"]), dict(
        original_memory=original, memory=memories, edited_raw=edited.cpu().numpy(), **scores_saved)


def fit_job(config, output, method, seed, fold, arm, steps, resume, stop_after_step, completed_jobs=0, total_jobs=1):
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    folds = crossfit.outer_folds(config)
    held = config["validation_video_ids"] if fold == "full_training" else folds[fold]
    fit = [v for v in config["train_video_ids"] if v not in held]
    assets = parent.load_assets(config, records, method, seed, fold, fit, held)
    specs = [crossfit.verify_head(config, p) for p in crossfit.head_plan(config, records, fold)]
    if any(set(s["scope_video_ids"]) & set(held) for s in specs):
        raise ValueError("Held procedures entered decoder training")
    relative = Path("fit") if fold == "full_training" else Path("inner_folds")
    folder = output / relative / method / f"seed{seed}" / ("full" if fold == "full_training" else f"fold{fold}") / arm
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(config=parent.shared.shared.json_digest(config), source_hashes=hashes(), assets=assets["identity"],
        heads=specs, method=method, seed=seed, fold=fold, arm=arm, steps=steps, fit_videos=fit, held_videos=held)
    key = parent.shared.shared.json_digest(identity)
    if (folder / "identity.json").exists():
        if not resume or read_json(folder / "identity.json") != identity:
            raise ValueError("Decoder job identity changed or resume is missing")
    else:
        atomic_write_json(folder / "identity.json", identity)
    if (folder / "summary.json").exists():
        summary = read_json(folder / "summary.json")
        if summary["identity_sha256"] != key:
            raise ValueError("Decoder summary identity changed")
        for name, signature in summary["artifacts"].items():
            if parent.file_sha256(folder / name) != signature:
                raise ValueError("Decoder output changed")
        return summary
    model = assets["model"]
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_decoder = model.decoder.weight.detach().clone()
    model.decoder.weight.requires_grad_(arm == "trainable_decoder")
    gains = BoundedGains(model.latent_dim, config["gain_bound"]).to(config["device"])
    groups = [dict(params=list(gains.parameters()), lr=config["gain_learning_rate"])]
    if arm == "trainable_decoder":
        groups.append(dict(params=[model.decoder.weight], lr=config["decoder_learning_rate"]))
    optimizer = torch.optim.Adam(groups)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    gradient_rule = config.get("gradient_rule", "mean")
    if gradient_rule not in ("mean", "agreement"):
        raise ValueError("Unknown gradient rule")
    parts = []
    for spec in specs:
        reference = FrozenSupCon(spec["directory"]).to(config["device"])
        part = design(assets, records, spec["held_video_ids"], reference, True)
        part["group_weight"] = len({r["video_id"] for r in part["rows"]})
        parts.append(part)
    count = sum(p["group_weight"] for p in parts)
    for part in parts:
        part["group_weight"] /= count
    scale = torch.as_tensor(assets["scale"], device=config["device"])
    history, first, previous_elapsed = [], 1, 0.
    checkpoint = folder / "checkpoint.pt"
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(checkpoint)
        state = torch.load(checkpoint, map_location=config["device"], weights_only=True)
        if state["identity_sha256"] != key:
            raise ValueError("Decoder checkpoint identity changed")
        model.load_state_dict(state["model"], strict=True)
        gains.load_state_dict(state["gains"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history, first, previous_elapsed = state["history"], state["step"] + 1, state["elapsed_seconds"]
    else:
        with torch.no_grad():
            if not torch.equal(residual_edit(parts[0]["raw"], parts[0]["codes"], gains(), model, scale), parts[0]["raw"]):
                raise ValueError("Zero gains must preserve source exactly")
    started = time.perf_counter()
    for step in range(first, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3)
        partition_gradients = [[] for _ in parameters]
        for part in parts:
            if gradient_rule == "agreement":
                optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, gains, scale, original_decoder, part, config)
            objective = terms[0] + config["identity_weight"] * terms[1] + config["decoder_anchor_weight"] * terms[2]
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite decoder objective")
            (objective * part["group_weight"]).backward()
            if gradient_rule == "agreement":
                for parameter, stored in zip(parameters, partition_gradients):
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError("Invalid partition gradient")
                    stored.append(parameter.grad.detach().clone())
            totals += np.array([float(value.detach()) for value in terms]) * part["group_weight"]
        agreement = {}
        if gradient_rule == "agreement":
            for index, (parameter, gradients) in enumerate(zip(parameters, partition_gradients)):
                combined, mask = agreement_gradient(gradients)
                parameter.grad = combined
                agreement[f"parameter{index}_agreement_fraction"] = float(mask.float().mean())
                agreement[f"parameter{index}_gradient_norm"] = float(torch.linalg.vector_norm(combined))
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Invalid decoder/gain gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"]))
        if not np.isfinite(gradient) or gradient <= 0:
            raise ValueError("No valid training gradient")
        optimizer.step()
        if arm == "trainable_decoder":
            model.normalize_decoder()
        row = dict(step=step, canonical_loss=float(totals[0]), tail_loss=float(totals[1]), anchor_loss=float(totals[2]), gradient=gradient, **agreement)
        history.append(row)
        stopping = stop_after_step is not None and step >= stop_after_step
        if step % config["checkpoint_every"] == 0 or step == steps or stopping:
            elapsed = previous_elapsed + time.perf_counter() - started
            parent.shared.shared.save_torch(checkpoint, dict(identity_sha256=key, step=step,
                model=model.state_dict(), gains=gains.state_dict(), optimizer=optimizer.state_dict(),
                history=history, elapsed_seconds=elapsed))
            progress = dict(status="PAUSED" if stopping else "RUNNING", phase="decoder_training",
                method=method, seed=seed, fold=fold, arm=arm, step=step, steps=steps,
                completed_jobs=completed_jobs, total_jobs=total_jobs,
                elapsed_seconds=elapsed, updated_at=parent.shared.shared.now(), losses=row)
            atomic_write_json(folder / "progress.json", progress)
            atomic_write_json(output / "training_progress.json", progress)
            print(progress, flush=True)
            if stopping:
                raise SystemExit(75)
    for name, value in model.state_dict().items():
        if name != "decoder.weight" and not torch.equal(value, original_state[name]):
            raise ValueError("Frozen encoder or bias changed")
    if arm == "fixed_decoder" and not torch.equal(model.decoder.weight, original_decoder):
        raise ValueError("Fixed decoder changed")
    held_report, arrays = measure(assets, model, gains, records, held)
    parent.shared.shared.save_npz(folder / "model.npz", **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})
    parent.shared.shared.save_npz(folder / "gains.npz", gains=gains().detach().cpu().numpy())
    parent.shared.shared.save_npz(folder / "held.npz", **arrays)
    atomic_write_json(folder / "history.json", history)
    atomic_write_json(folder / "held.json", held_report)
    weights = model.decoder.weight.detach()
    result = dict(status="COMPLETE", method=method, seed=seed, fold=fold, arm=arm, steps=steps,
        identity_sha256=key, folder=str(folder), elapsed_seconds=previous_elapsed + time.perf_counter() - started,
        held=held_report["means"], procedure_results=held_report["procedure_results"],
        decoder_cosine_mean=float((weights * original_decoder).sum(0).mean()),
        decoder_change_norm=float(torch.linalg.vector_norm(weights - original_decoder)),
        encoder_unchanged=True, gain_max=float(gains().abs().max().detach()),
        artifacts={name: parent.file_sha256(folder / name) for name in ("model.npz", "gains.npz", "held.npz", "held.json", "history.json")},
        runtime=parent.runtime(config))
    atomic_write_json(folder / "summary.json", result)
    return result


def run(config, output, smoke, resume, stop_after_step):
    output.mkdir(parents=True, exist_ok=True)
    steps = config["smoke_steps"] if smoke else config["steps"]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    folds = [0, "full_training"] if smoke else [0, 1, 2, "full_training"]
    total = len(seeds) * len(folds) * len(config["methods"]) * len(config["arms"])
    outputs, initial, started = [], hashes(), time.perf_counter()
    for seed in seeds:
        for fold in folds:
            for method in config["methods"]:
                for arm in config["arms"]:
                    summary = fit_job(config, output, method, seed, fold, arm, steps, resume, stop_after_step, len(outputs), total)
                    outputs.append(summary)
                    progress = dict(status="RUNNING", phase="decoder_training", completed_jobs=len(outputs),
                        total_jobs=total, elapsed_seconds=time.perf_counter() - started,
                        updated_at=parent.shared.shared.now())
                    atomic_write_json(output / "training_progress.json", progress)
                    print("DECODER_JOB", len(outputs), total, summary["held"], flush=True)
    if hashes() != initial:
        raise ValueError("Training source changed during execution")
    atomic_write_json(output / "training_summary.json", dict(status="COMPLETE", outputs=outputs,
        completed_jobs=len(outputs), total_jobs=total, source_hashes=initial, elapsed_seconds=time.perf_counter() - started))
    atomic_write_json(output / "training_progress.json", dict(progress, status="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read_json(args.run / "config.json")
    output = args.output or (args.run / "smoke" if args.smoke else args.run)
    run(config, output, args.smoke, args.resume, args.stop_after_step)
