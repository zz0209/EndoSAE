import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import discover_component_memory as parent
import discover_crossfit_component_memory as crossfit
from src import conditional_component_policy as policy
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.token_memory_edit import FrozenSupCon, residual_edit


def source_hashes():
    result = parent.source_hashes()
    for path in (Path(__file__), Path(policy.__file__), Path(crossfit.__file__)):
        result[str(path.relative_to(ROOT))] = parent.file_sha256(path)
    return result


def relative_job(method, seed, fold):
    return (Path("fit") / method / f"seed{seed}" if fold == "full_training" else
            Path("inner_folds") / method / f"seed{seed}" / f"fold{fold}")


def support_table(config, records):
    output = np.full((len(records), 7, len(policy.SUPPORT_NAMES)), np.nan)
    identities = {}
    for video in config["train_video_ids"]:
        folder = Path(config["token_data_root"]) / "videos" / video
        path = folder / "masks.npz"
        receipt = read_json(folder / "complete.json")
        signature = parent.file_sha256(path)
        if signature != receipt["assets"]["masks.npz"]:
            raise ValueError("Training support masks changed")
        identities[str(path)] = signature
        with np.load(path, allow_pickle=False) as archive:
            masks = archive["masks"]
            for index, record in enumerate(records):
                if record["video_id"] != video:
                    continue
                local = int(record["index"])
                for view in range(7):
                    if masks[local, view].any():
                        output[index, view] = policy.mask_features(masks[local, view])
    return output, identities


def component_actions(config, method, seed, fold, fit, held):
    folder = Path(config["parent_component_run"]) / relative_job(method, seed, fold)
    identity = read_json(folder / "identity.json")
    if identity["fit_video_ids"] != fit or identity["held_video_ids"] != held:
        raise ValueError("Parent component discovery has a different procedure scope")
    summary = read_json(folder / "discovery.json")
    if parent.file_sha256(folder / "component_effects.npz") != summary["effects_sha256"]:
        raise ValueError("Parent component effects changed")
    ordering = summary["ordered_components"]
    candidates = parent.candidates(config)
    gains = np.zeros((len(candidates), 1024), dtype=np.float32)
    for i, candidate in enumerate(candidates):
        gains[i, ordering[:candidate["k"]]] = -candidate["strength"]
    return candidates, gains, dict(path=str(folder), discovery_sha256=parent.file_sha256(folder / "discovery.json"),
                                  identity_sha256=parent.file_sha256(folder / "identity.json"))


@torch.no_grad()
def candidate_arrays(assets, reference, raw, codes, gains):
    original = torch.as_tensor(np.asarray(raw), dtype=torch.float64, device="cpu")
    code = torch.as_tensor(np.asarray(codes), dtype=torch.float32, device="cpu")
    edited = residual_edit(original[None], code[None], torch.from_numpy(gains),
                           assets["model"], torch.from_numpy(assets["scale"]))
    if not np.array_equal(edited[0].numpy(), original.numpy()):
        raise ValueError("Zero intervention changed the source")
    return edited.numpy(), reference(edited).numpy()


@torch.no_grad()
def dataset(config, assets, records, videos, reference, bank_videos, support, candidates, gains):
    pairs = parent.shared.shared.chronological_pairs(records, videos)
    bank_indices = [i for i, r in enumerate(records) if r["video_id"] in bank_videos]
    if set(videos) & set(bank_videos):
        raise ValueError("A source procedure entered its reference cohort")
    bank = reference(torch.from_numpy(assets["raw"][bank_indices, 0])).numpy()
    weights = policy.cohort_weights(records, bank_indices)
    queries = reference(torch.from_numpy(assets["raw"][:, 0])).numpy()
    rows, features, thresholds, utilities, scores, excluded = [], [], [], [], [], []
    for source in sorted({p["source_index"] for p in pairs}):
        local = [p for p in pairs if p["source_index"] == source]
        positive = [p["query_index"] for p in local if p["same_identity"]]
        negative = [p["query_index"] for p in local if not p["same_identity"]]
        if not positive or not negative:
            excluded.append(dict(source_index=source, video_id=records[source]["video_id"],
                                 positives=len(positive), negatives=len(negative)))
            continue
        for view in np.flatnonzero(assets["valid"][source, 1:]) + 1:
            raw, codes = assets["raw"][source, view], assets["codes"][source, view]
            edited, memories = candidate_arrays(assets, reference, raw, codes, gains)
            features.append(policy.candidate_features(raw, edited, codes, memories, bank, weights,
                                                       support[source, view], candidates))
            threshold, utility, full_scores = policy.supervised_targets(memories, queries, positive, negative,
                float(config["negative_quantile"]), float(config["utility_temperature"]))
            thresholds.append(threshold)
            utilities.append(utility)
            query_indices = positive + negative
            scores.append(full_scores[:, query_indices])
            rows.append(dict(source_index=source, view=int(view), video_id=records[source]["video_id"],
                lesion_id=records[source]["lesion_id"], positives=positive, negatives=negative,
                bank_video_ids=list(bank_videos)))
    if not rows:
        raise ValueError("No sources have both chronological identity classes")
    return dict(metadata=rows, features=np.asarray(features), thresholds=np.asarray(thresholds),
                utility=np.asarray(utilities), scores=scores, excluded=excluded)


def combine(parts):
    return dict(metadata=[r for p in parts for r in p["metadata"]],
        features=np.concatenate([p["features"] for p in parts]),
        thresholds=np.concatenate([p["thresholds"] for p in parts]),
        utility=np.concatenate([p["utility"] for p in parts]),
        scores=[r for p in parts for r in p["scores"]],
        excluded=[r for p in parts for r in p["excluded"]])


def score_dataset(model, data, label):
    chosen, predicted, threshold = policy.choose(model, data["features"])
    fixed = int(model["fixed_action"])
    weights = policy.source_weights(data["metadata"])
    index = np.arange(len(chosen))
    utilities = dict(zero=data["utility"][:, 0], fixed=data["utility"][:, fixed],
                     conditional=data["utility"][index, chosen], oracle=data["utility"].max(1))
    per_video = {}
    for video in sorted({r["video_id"] for r in data["metadata"]}):
        selected = np.array([r["video_id"] == video for r in data["metadata"]])
        local_weight = weights[selected] / weights[selected].sum()
        per_video[video] = {name: float(local_weight @ value[selected]) for name, value in utilities.items()}
    report = dict(scope=label, source_views=len(chosen), sources=len({r["source_index"] for r in data["metadata"]}),
        procedures=len(per_video), mean_utility={name: float(weights @ value) for name, value in utilities.items()},
        per_procedure=per_video, action_counts=np.bincount(chosen, minlength=data["utility"].shape[1]).tolist(),
        predicted_selected_advantage=float(weights @ predicted[index, chosen]),
        realized_selected_advantage=float(weights @ (utilities["conditional"] - utilities["zero"])),
        zero_threshold_mae=float(weights @ np.abs(threshold[:, 0] - data["thresholds"][:, 0])),
        selected_threshold_mae=float(weights @ np.abs(threshold[index, chosen] - data["thresholds"][index, chosen])),
        excluded_sources=data["excluded"],
        interpretation="Training-label tail-separation proxy; complete prompt outcomes determine application utility.")
    return report, dict(chosen=chosen, predicted_advantage=predicted, predicted_threshold=threshold)


def save_dataset(directory, name, data, predictions):
    parent.shared.shared.save_npz(directory / (name + ".npz"), features=data["features"],
        thresholds=data["thresholds"], utility=data["utility"], **predictions,
        **{f"scores_{i}": score for i, score in enumerate(data["scores"])})
    atomic_write_json(directory / (name + ".json"), dict(metadata=data["metadata"], excluded=data["excluded"]))


def fit_job(config, method, seed, fold, records, folds, support, support_identity, output, resume):
    held = config["validation_video_ids"] if fold == "full_training" else folds[fold]
    fit = [v for v in config["train_video_ids"] if v not in held]
    folder = output / relative_job(method, seed, fold)
    folder.mkdir(parents=True, exist_ok=True)
    assets = parent.load_assets(config, records, method, seed, fold, fit, held)
    candidates, gains, component_identity = component_actions(config, method, seed, fold, fit, held)
    discovery_config = read_json(Path(config["discovery_run"]) / "config.json")
    plans = crossfit.head_plan(discovery_config, records, fold)
    head_receipts = [crossfit.verify_head(discovery_config, plan) for plan in plans]
    identity = dict(config_sha256=parent.shared.shared.json_digest(config), source_hashes=source_hashes(),
        method=method, seed=seed, fold=fold, fit_video_ids=fit, held_video_ids=held,
        support_inputs=support_identity, assets=assets["identity"], component=component_identity, heads=head_receipts)
    key = parent.shared.shared.json_digest(identity)
    if (folder / "identity.json").exists() and read_json(folder / "identity.json") != identity:
        raise ValueError("Conditional-policy input identity changed")
    if (folder / "summary.json").exists():
        result = read_json(folder / "summary.json")
        if not resume or result["identity_sha256"] != key or result["status"] != "COMPLETE":
            raise ValueError("Completed policy requires matching resume")
        for name, signature in result["artifacts"].items():
            if parent.file_sha256(folder / name) != signature:
                raise ValueError("Conditional policy output changed")
        return result
    atomic_write_json(folder / "identity.json", identity)
    started = time.perf_counter()
    parts = []
    for receipt in head_receipts:
        if set(receipt["scope_video_ids"]) & set(held):
            raise ValueError("Outer-held procedure entered policy discovery")
        reference = FrozenSupCon(receipt["directory"])
        parts.append(dataset(config, assets, records, receipt["held_video_ids"], reference,
                             receipt["fit_video_ids"], support, candidates, gains))
    fitting = combine(parts)
    model = policy.fit_policy(fitting["features"], fitting["thresholds"], fitting["utility"],
                              fitting["metadata"], float(config["ridge_alpha"]))
    model.update(candidates=candidates, dictionary_directory=str(assets["parent_directory"].resolve()),
                 reference_directory=str(Path(assets["spec"]["reference_fit"]).resolve()),
                 bank_video_ids=fit, identity_sha256=key)
    fitting_report, fitting_predictions = score_dataset(model, fitting, "fitting_crossfit_head_labels")
    save_dataset(folder, "fitting", fitting, fitting_predictions)
    parent.shared.shared.save_npz(folder / "actions.npz", gains=gains)
    parent.shared.shared.save_npz(folder / "training_context_actions.npz",
                                  actions=fitting_predictions["chosen"])
    atomic_write_json(folder / "policy.json", model)
    report = dict(fitting=fitting_report)
    if fold != "full_training":
        held_data = dataset(config, assets, records, held, assets["reference"], fit, support, candidates, gains)
        report["held"], held_predictions = score_dataset(model, held_data, "outer_held_procedures")
        save_dataset(folder, "held", held_data, held_predictions)
    paths = [p for p in folder.iterdir() if p.name not in ("identity.json", "summary.json") and p.is_file()]
    result = dict(status="COMPLETE", identity_sha256=key, method=method, seed=seed, fold=fold,
        elapsed_seconds=time.perf_counter() - started, results=report,
        artifacts={p.name: parent.file_sha256(p) for p in paths}, runtime=parent.runtime(config))
    atomic_write_json(folder / "summary.json", result)
    pause_after_checkpoint(folder / "summary.json")
    return result


def run(config, method, smoke, resume, stop_after_jobs):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(config["run_dir"]) / "smoke" if smoke else Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    folds = read_json(Path(config["parent_dictionary_run"]) / "grouped_folds.json")["inner_held_folds"]
    support, support_identity = support_table(config, records)
    methods = config["methods"] if method == "all" else [method]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    scopes = [0, "full_training"] if smoke else [0, 1, 2, "full_training"]
    total = len(methods) * len(seeds) * len(scopes)
    results, started = [], time.perf_counter()
    initial_hashes = source_hashes()
    for selected_method in methods:
        for seed in seeds:
            for fold in scopes:
                result = fit_job(config, selected_method, seed, fold, records, folds, support,
                                 support_identity, output, resume)
                results.append(result)
                progress = dict(status="RUNNING", method=selected_method, seed=seed, fold=fold,
                    completed_jobs=len(results), total_jobs=total, elapsed_seconds=time.perf_counter() - started,
                    stage="source_condition_policy", updated_at=parent.shared.shared.now())
                atomic_write_json(output / "training_progress.json", progress)
                print(progress, flush=True)
                if stop_after_jobs is not None and len(results) >= stop_after_jobs:
                    raise SystemExit(75)
    if initial_hashes != source_hashes():
        raise ValueError("Source changed during policy fitting")
    result = dict(status="COMPLETE", completed_jobs=len(results), total_jobs=total, outputs=results,
                  elapsed_seconds=time.perf_counter() - started, source_hashes=initial_hashes)
    atomic_write_json(output / ("training_summary_" + method + ".json"), result)
    atomic_write_json(output / "training_progress.json", dict(progress, status="COMPLETE"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("all", "sparse_edit", "dense_edit"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-jobs", type=int)
    args = parser.parse_args()
    run(read_json(args.config), args.method, args.smoke, args.resume, args.stop_after_jobs)
