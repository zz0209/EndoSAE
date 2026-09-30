import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_token_memory_edit as shared
from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.component_memory_intervention import INTERVENTION_TYPE, load_predictor
from src.token_memory_edit import METHODS, FrozenSupCon, TokenDictionary, residual_edit


def source_hashes():
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "src/component_memory_intervention.py", Path(shared.__file__),
        ROOT / "src/token_memory_edit.py", ROOT / "src/region_identity_sae.py",
        ROOT / "scripts/train_acknowledgement_sae.py", ROOT / "src/acknowledgement_sae.py",
        ROOT / "src/checkpoint_io.py")}


def runtime(config):
    return dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                device=config["device"], threads=int(config["threads"]))


def candidates(config):
    result = [dict(name="zero", k=0, strength=0.)]
    for k in config["component_counts"]:
        for strength in config["intervention_strengths"]:
            result.append(dict(name=f"k{k:03d}_a{int(round(strength * 100)):03d}", k=int(k), strength=float(strength)))
    if config["component_counts"] != [4, 16, 64] or config["intervention_strengths"] != [.25, .5, 1.]:
        raise ValueError("The fixed component intervention budget differs")
    return result


def load_assets(config, records, method, seed, fold, fit_videos, held_videos):
    parent = Path(config["parent_dictionary_run"])
    relative = (Path("fit") / method / f"seed{seed}" if fold == "full_training" else
                Path("inner_folds") / method / f"seed{seed}" / f"fold{fold}")
    directory = parent / relative
    parent_identity = read_json(directory / "identity.json")
    if parent_identity["fit_videos"] != fit_videos or parent_identity["held_videos"] != held_videos:
        raise ValueError("Parent dictionary fit/held procedure scope differs")
    dictionary_summary = read_json(directory / "dictionary_summary.json")
    pool_summary = read_json(directory / "pool_summary.json")
    assets = {"model.npz": dictionary_summary["model_sha256"],
              "normalization.npz": dictionary_summary["normalization_sha256"],
              "pooled_views.npz": pool_summary["sha256"]}
    if dictionary_summary["status"] != "COMPLETE" or pool_summary["status"] != "COMPLETE":
        raise ValueError("Parent dictionary or pooling is incomplete")
    for name, value in assets.items():
        if file_sha256(directory / name) != value:
            raise ValueError(f"Parent dictionary input changed: {directory / name}")
    head = Path(config["reference_fit"]) if fold == "full_training" else Path(config["parent_head_run"]) / "heads" / f"fold{fold}"
    head_assets = {name: file_sha256(head / name) for name in ("model.npz", "normalization.npz")}
    if fold != "full_training":
        head_summary = read_json(head / "summary.json")
        if (head_summary["status"] != "COMPLETE" or head_summary["fit_video_ids"] != fit_videos
                or head_summary["held_video_ids"] != held_videos
                or any(head_assets[name] != head_summary["assets"][name] for name in head_assets)):
            raise ValueError("Crossfit head asset or procedure scope differs")
    spec = read_json(directory / "model_config.json")
    spec["reference_fit"] = str(head.resolve())
    model = TokenDictionary(spec).to(config["device"])
    with np.load(directory / "model.npz", allow_pickle=False) as data:
        model.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files}, strict=True)
    model.eval().requires_grad_(False)
    with np.load(directory / "normalization.npz", allow_pickle=False) as data:
        mean, scale = data["mean"].copy(), data["scale"].copy()
    with np.load(directory / "pooled_views.npz", allow_pickle=False) as data:
        raw, codes, valid = (data[name].copy() for name in ("raw", "mean_codes", "valid"))
    if raw.shape != (len(records), 7, 768) or codes.shape != (len(records), 7, 1024) or not valid[:, 0].all():
        raise ValueError("Unexpected parent region dimensions")
    identity = dict(parent_directory=str(directory.resolve()), parent_assets=assets,
                    head_directory=str(head.resolve()), head_assets=head_assets,
                    dictionary_summary_sha256=file_sha256(directory / "dictionary_summary.json"),
                    pool_summary_sha256=file_sha256(directory / "pool_summary.json"))
    return dict(model=model, reference=FrozenSupCon(head).to(config["device"]), mean=mean, scale=scale,
                raw=raw, codes=codes, valid=valid, spec=spec, identity=identity,
                parent_directory=directory, dictionary_summary=dictionary_summary)


@torch.no_grad()
def source_design(records, videos, raw, valid, reference, device):
    pairs = shared.shared.chronological_pairs(records, videos)
    sources = sorted({pair["source_index"] for pair in pairs})
    positive_by_source = {s: [p["query_index"] for p in pairs if p["source_index"] == s and p["same_identity"]] for s in sources}
    negative_by_source = {s: [p["query_index"] for p in pairs if p["source_index"] == s and not p["same_identity"]] for s in sources}
    sources = [s for s in sources if positive_by_source[s]]
    indices = np.array([s for s in sources for v in np.flatnonzero(valid[s, 1:]) + 1], dtype=np.int64)
    views = np.array([v for s in sources for v in np.flatnonzero(valid[s, 1:]) + 1], dtype=np.int64)
    positive = np.zeros((len(indices), len(records)), dtype=bool)
    negative = np.zeros_like(positive)
    for i, source in enumerate(indices):
        positive[i, positive_by_source[int(source)]] = True
        negative[i, negative_by_source[int(source)]] = True
    if not len(indices) or not positive.any(1).all():
        raise ValueError("No future-positive source views support component discovery")
    negative_defined = negative.any(1)
    video_ids = sorted(videos)
    weights = {}
    for label, eligible in (("benefit", negative_defined), ("harm", positive.any(1))):
        matrix = np.zeros((len(video_ids), len(indices)), dtype=np.float64)
        for v, video in enumerate(video_ids):
            eligible_sources = sorted({int(s) for i, s in enumerate(indices) if eligible[i] and records[s]["video_id"] == video})
            for source in eligible_sources:
                selected = np.flatnonzero((indices == source) & eligible)
                matrix[v, selected] = 1 / (len(eligible_sources) * len(selected))
        weights[label] = torch.from_numpy(matrix).to(device)
    raw_tensor = torch.from_numpy(raw).to(device)
    queries = reference(raw_tensor[:, 0])
    source_raw = raw_tensor[indices, views]
    before = reference(source_raw) @ queries.T
    positive_tensor, negative_tensor = torch.from_numpy(positive).to(device), torch.from_numpy(negative).to(device)
    hard_before = before.masked_fill(~negative_tensor, -torch.inf).amax(-1)
    hard_before[~torch.from_numpy(negative_defined).to(device)] = 0
    metadata = dict(videos=video_ids, sources=len(sources), source_views=len(indices),
        sources_with_both=sum(bool(negative_by_source[s]) for s in sources),
        aggregation="Equal views within source, equal sources within procedure, then equal defined procedures separately for benefit and harm.",
        views="Valid noncanonical views inherited from the parent; no additional mask deduplication.",
        by_video={video: dict(positive_sources=sum(records[s]["video_id"] == video for s in sources),
            paired_sources=sum(records[s]["video_id"] == video and bool(negative_by_source[s]) for s in sources),
            previously_seen_negative_pairs=sum(p["video_id"] == video and p["previously_seen_other"] for p in pairs)) for video in video_ids})
    return dict(indices=indices, views=views, positive=positive_tensor, negative=negative_tensor,
                negative_defined=torch.from_numpy(negative_defined).to(device), before=before, hard_before=hard_before,
                queries=queries, raw=source_raw, weights=weights, metadata=metadata)


def effects(memories, design):
    scores = memories @ design["queries"].T
    hard_after = scores.masked_fill(~design["negative"], -torch.inf).amax(-1)
    benefit = torch.where(design["negative_defined"], design["hard_before"] - hard_after, 0.)
    harm = ((design["before"] - scores).clamp_min(0) * design["positive"]).sum(-1) / design["positive"].sum(-1)
    return benefit, harm


@torch.no_grad()
def discover(config, assets, records, videos, directory, identity_hash, context, resume, stop_after_components):
    design = source_design(records, videos, assets["raw"], assets["valid"], assets["reference"], config["device"])
    atomic_write_json(directory / "discovery_sources.json", design["metadata"])
    code = torch.from_numpy(assets["codes"][design["indices"], design["views"]]).to(config["device"])
    scale = torch.from_numpy(assets["scale"]).to(config["device"])
    total, procedure_count = assets["model"].latent_dim, len(videos)
    checkpoint = directory / "discovery_checkpoint.npz"
    benefits = np.full((procedure_count, total), np.nan)
    harms = np.full_like(benefits, np.nan)
    first, elapsed_before = 0, 0.
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("Use --resume for existing component discovery")
        with np.load(checkpoint, allow_pickle=False) as data:
            if str(data["identity_sha256"]) != identity_hash:
                raise ValueError("Component checkpoint identity differs")
            benefits, harms = data["benefit"].copy(), data["harm"].copy()
            first, elapsed_before = int(data["completed"]), float(data["elapsed_seconds"])
    started = time.perf_counter()
    for begin in range(first, total, int(config["component_batch_size"])):
        end = min(total, begin + int(config["component_batch_size"]))
        coefficients = -code[:, begin:end].T
        delta = (coefficients[:, :, None] * assets["model"].decoder.weight[:, begin:end].T[:, None, :]).double() * scale
        edited = design["raw"][None] + delta
        memories = assets["reference"](edited)
        benefit, harm = effects(memories, design)
        for name, value, target in (("benefit", benefit, benefits), ("harm", harm, harms)):
            result = (design["weights"][name] @ value.double().T).cpu().numpy()
            result[design["weights"][name].sum(-1).cpu().numpy() == 0] = np.nan
            target[:, begin:end] = result
        elapsed = elapsed_before + time.perf_counter() - started
        shared.shared.save_npz(checkpoint, identity_sha256=np.array(identity_hash), completed=np.array(end),
            elapsed_seconds=np.array(elapsed), benefit=benefits, harm=harms)
        stopping = stop_after_components is not None and end >= stop_after_components
        shared.progress(context, directory, "component_discovery", end, total, elapsed,
            dict(components_per_second=(end - first) / max(time.perf_counter() - started, 1e-9)),
            "PAUSED" if stopping else "RUNNING")
        pause_after_checkpoint(checkpoint)
        if stopping:
            raise SystemExit(75)
    benefit = np.nanmean(benefits, axis=0)
    harm = np.nanmean(harms, axis=0)
    objective = benefit - harm
    if not np.isfinite(objective).all():
        raise ValueError("Component discovery has an undefined aggregate objective")
    eligible = np.flatnonzero((objective > 0) & (benefit > 0))
    ordering = eligible[np.lexsort((eligible, -objective[eligible]))]
    shared.shared.save_npz(directory / "component_effects.npz", components=np.arange(total),
        videos=np.array(design["metadata"]["videos"]), benefit=benefits, harm=harms,
        macro_benefit=benefit, macro_harm=harm, objective=objective, eligible_order=ordering)
    result = dict(status="COMPLETE", components=total, eligible_components=len(ordering), ordered_components=ordering.tolist(),
        elapsed_seconds=elapsed_before + time.perf_counter() - started,
        macro_benefit=benefit.tolist(), macro_harm=harm.tolist(), objective=objective.tolist(),
        source_counts=design["metadata"], effects_sha256=file_sha256(directory / "component_effects.npz"))
    atomic_write_json(directory / "discovery.json", result)
    return result, design


@torch.no_grad()
def group_diagnostics(gains, assets, design):
    scale = torch.from_numpy(assets["scale"]).to(gains.device)
    raw = torch.from_numpy(assets["raw"]).to(gains.device)
    codes = torch.from_numpy(assets["codes"]).to(gains.device)
    indices, views = design["indices"], design["views"]
    rows = {}
    for label, source_views in (("variant", views), ("canonical", np.zeros_like(views))):
        original = raw[indices, source_views]
        edited = residual_edit(original, codes[indices, source_views], gains, assets["model"], scale)
        local = dict(design)
        local["before"] = assets["reference"](original) @ design["queries"].T
        local["hard_before"] = local["before"].masked_fill(~design["negative"], -torch.inf).amax(-1)
        local["hard_before"][~design["negative_defined"]] = 0
        b, h = effects(assets["reference"](edited), local)
        rows[label] = {}
        for name, value in (("benefit", b), ("harm", h)):
            weights = design["weights"][name]
            values = (weights @ value.double()).cpu().numpy()
            defined = weights.sum(-1).cpu().numpy() > 0
            rows[label][name] = {video: float(values[i]) if defined[i] else None for i, video in enumerate(design["metadata"]["videos"])}
    rows["variant_minus_canonical"] = {name: {video: None if rows["variant"][name][video] is None else
        rows["variant"][name][video] - rows["canonical"][name][video] for video in rows["variant"][name]} for name in ("benefit", "harm")}
    return rows


def evaluate_candidate(config, assets, records, videos, candidate, ordering, directory, design):
    selected = ordering[:candidate["k"]]
    gains = torch.zeros(assets["model"].latent_dim, device=config["device"])
    gains[selected] = -candidate["strength"]
    report = shared.evaluate_gains(config, assets["model"], lambda: gains, assets["reference"],
        torch.from_numpy(assets["raw"]).to(config["device"]), torch.from_numpy(assets["codes"]).to(config["device"]),
        assets["valid"], torch.from_numpy(assets["scale"]).to(config["device"]), records, videos, directory,
        "held_" + candidate["name"])
    zero_report = read_json(directory / "held_zero.json")
    current_report = read_json(directory / ("held_" + candidate["name"] + ".json"))
    threshold = zero_report["protection_point"]["threshold"]
    fixed = {}
    if len(zero_report["pairs"]) != len(current_report["pairs"]):
        raise ValueError("Candidate changed the held pair population")
    for before, after in zip(zero_report["pairs"], current_report["pairs"]):
        if {k: v for k, v in before.items() if k != "score"} != {k: v for k, v in after.items() if k != "score"}:
            raise ValueError("Candidate changed a held pair identity")
        video = before["video_id"]
        row = fixed.setdefault(video, dict(positives=0, negatives=0, false_matches_before=0,
            true_matches_before=0, false_matches_corrected=0, false_matches_introduced=0,
            true_matches_damaged=0, true_matches_recovered=0))
        same, accepted_before, accepted_after = before["same_identity"], before["score"] >= threshold, after["score"] >= threshold
        row["positives" if same else "negatives"] += 1
        if same:
            row["true_matches_before"] += int(accepted_before)
            row["true_matches_damaged"] += int(accepted_before and not accepted_after)
            row["true_matches_recovered"] += int(not accepted_before and accepted_after)
        else:
            row["false_matches_before"] += int(accepted_before)
            row["false_matches_corrected"] += int(accepted_before and not accepted_after)
            row["false_matches_introduced"] += int(not accepted_before and accepted_after)
    return dict(**candidate, selected_components=selected, actual_components=len(selected), evaluation=report,
                mechanism=group_diagnostics(gains, assets, design),
                fixed_zero_threshold=dict(threshold=threshold, unit="Chronological source-view/query pairs; procedure-level counts.",
                                          used_for_selection=False, by_video=fixed))


def job(config, method, seed, fold, records, fit, held, output, context, global_identity, resume,
        stop_after_components, selected_candidate=None):
    relative = (Path("fit") / method / f"seed{seed}" if fold == "full_training" else
                Path("inner_folds") / method / f"seed{seed}" / f"fold{fold}")
    directory = output / relative
    directory.mkdir(parents=True, exist_ok=True)
    assets = load_assets(config, records, method, seed, fold, fit, held)
    identity = dict(**global_identity, configuration_sha256=shared.shared.json_digest(config),
        method=method, seed=seed, fold=fold, fit_video_ids=fit, held_video_ids=held,
        selection=selected_candidate, **assets["identity"])
    key = shared.shared.json_digest(identity)
    if (directory / "identity.json").exists() and read_json(directory / "identity.json") != identity:
        raise ValueError("Component discovery job identity changed")
    if (directory / "summary.json").exists():
        result = read_json(directory / "summary.json")
        if not resume or result["identity_sha256"] != key or result["status"] != "COMPLETE":
            raise ValueError("Completed component discovery cannot be reused")
        return result
    atomic_write_json(directory / "identity.json", identity)
    discovered, fit_design = discover(config, assets, records, fit, directory, key, context, resume, stop_after_components)
    held_design = source_design(records, held, assets["raw"], assets["valid"], assets["reference"], config["device"])
    candidate_results = []
    evaluation_started = time.perf_counter()
    for index, candidate in enumerate(candidates(config)):
        receipt = directory / ("candidate_" + candidate["name"] + ".json")
        if receipt.exists():
            if not resume:
                raise FileExistsError("Existing candidate requires --resume")
            result = read_json(receipt)
        else:
            result = evaluate_candidate(config, assets, records, held, candidate,
                discovered["ordered_components"], directory, held_design)
            result["fit_mechanism"] = group_diagnostics(
                torch.tensor([-candidate["strength"] if j in result["selected_components"] else 0. for j in range(1024)], device=config["device"]),
                assets, fit_design)
            atomic_write_json(receipt, result)
        candidate_results.append(result)
        shared.progress(context, directory, "candidate_evaluation", index + 1, 10, time.perf_counter() - evaluation_started)
        pause_after_checkpoint(receipt)
    result = dict(status="COMPLETE", identity_sha256=key, method=method, seed=seed, fold=fold,
        dictionary=assets["dictionary_summary"], discovery=discovered, candidates=candidate_results,
        completed_at=shared.shared.now(), reused_dictionary=True, reused_head=True, reused_pool=True,
        runtime=runtime(config))
    if selected_candidate is not None:
        selected = discovered["ordered_components"][:selected_candidate["k"]]
        gains = np.zeros(1024, dtype=np.float32)
        gains[selected] = -selected_candidate["strength"]
        spec = dict(assets["spec"], intervention_type=INTERVENTION_TYPE,
                    intervention_strength=selected_candidate["strength"], component_cap=selected_candidate["k"],
                    coefficient_range=[-1., 0.])
        spec.pop("gain_bound", None)
        atomic_write_json(directory / "model_config.json", spec)
        for name in ("model.npz", "normalization.npz", "dictionary_summary.json", "pool_summary.json"):
            shutil.copyfile(assets["parent_directory"] / name, directory / name)
        shared.shared.save_npz(directory / "gains.npz", gains=gains, selected_indices=np.asarray(selected, dtype=np.int64))
        predictor = load_predictor(directory, config["device"])
        source = fit_design["indices"][:8]
        view = fit_design["views"][:8]
        raw, codes = assets["raw"][source, view], assets["codes"][source, view]
        expected = residual_edit(torch.from_numpy(raw).to(config["device"]), torch.from_numpy(codes).to(config["device"]),
            torch.from_numpy(gains).to(config["device"]), assets["model"], torch.from_numpy(assets["scale"]).to(config["device"])).cpu().numpy()
        if not np.array_equal(expected, predictor.edit_raw(raw, codes)):
            raise ValueError("Exported component intervention differs")
        if not np.array_equal(raw, predictor.edit_raw(raw, codes, np.zeros_like(gains))):
            raise ValueError("Zero component intervention changed the source")
        probe = np.zeros_like(gains)
        probe[int(np.argmax(codes.mean(0)))] = -1
        direct_probe = residual_edit(torch.from_numpy(raw).to(config["device"]), torch.from_numpy(codes).to(config["device"]),
            torch.from_numpy(probe).to(config["device"]), assets["model"], torch.from_numpy(assets["scale"]).to(config["device"])).cpu().numpy()
        if not np.array_equal(direct_probe, predictor.edit_raw(raw, codes, probe)) or np.array_equal(direct_probe, raw):
            raise ValueError("Actual component removal verification failed")
        result.update(selected_candidate=selected_candidate, selected_components=selected, no_edit=not bool(selected),
                      actual_components=len(selected), model_reload_exact=True, zero_edit_exact=True, single_component_exact=True,
                      gains_sha256=file_sha256(directory / "gains.npz"))
    atomic_write_json(directory / "summary.json", result)
    return result


def run(config, phase, method, resume, stop_after_components):
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(config["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    output = Path(config["run_dir"]) / "smoke" if phase == "smoke" else Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    parent = Path(config["parent_dictionary_run"])
    parent_config = read_json(parent / "config.json")
    for key in ("train_video_ids", "validation_video_ids", "seeds", "methods", "reference_fit"):
        if config[key] != parent_config[key]:
            raise ValueError(f"Component discovery changed the fixed parent scope: {key}")
    records = read_json(parent / "descriptor_records.json")
    folds = read_json(parent / "grouped_folds.json")["inner_held_folds"]
    head_folds = read_json(Path(config["parent_head_run"]) / "grouped_folds.json")["inner_held_folds"]
    if folds != head_folds:
        raise ValueError("Head and dictionary held folds disagree")
    candidates(config)
    identity = dict(source_hashes=source_hashes(), parent_config_sha256=file_sha256(parent / "config.json"),
                    records_sha256=file_sha256(parent / "descriptor_records.json"),
                    head_config_sha256=file_sha256(Path(config["parent_head_run"]) / "config.json"))
    atomic_write_json(output / "training_config.json", config)
    atomic_write_json(output / "descriptor_records.json", records)
    atomic_write_json(output / "grouped_folds.json", dict(inner_held_folds=folds,
        training_procedures=config["train_video_ids"], validation_procedures=config["validation_video_ids"]))
    methods = config["methods"] if method == "all" else [method]
    seeds = config["seeds"][:1] if phase == "smoke" else config["seeds"]
    active_folds = folds[:1] if phase == "smoke" else folds
    total, completed = len(methods) * len(seeds) * (len(active_folds) + 1), 0
    outputs, selections = [], []
    started = time.perf_counter()
    for seed in seeds:
        for selected_method in methods:
            results = []
            for fold, held in enumerate(active_folds):
                fit = [v for v in config["train_video_ids"] if v not in held]
                context = dict(phase=phase, method=selected_method, seed=seed, fold=fold,
                    completed_jobs=completed, total_jobs=total,
                    progress_path=str((output / "training_progress.json").resolve()))
                results.append(job(config, selected_method, seed, fold, records, fit, held, output,
                    context, identity, resume, stop_after_components))
                completed += 1
            points = []
            for candidate in candidates(config):
                fold_points = [next(c for c in result["candidates"] if c["name"] == candidate["name"])["evaluation"]["protection_point"] for result in results]
                rows = [row for point in fold_points for row in point["procedure_rows"]]
                if len({row["video"] for row in rows}) != len(rows):
                    raise ValueError("A held procedure is counted more than once")
                points.append(dict(**candidate, recall=float(np.mean([row["recall"] for row in rows])),
                    procedure_rows=rows, fold_points=fold_points))
            chosen = max(points, key=lambda row: (row["recall"], -row["k"], -row["strength"]))
            selected = {key: chosen[key] for key in ("name", "k", "strength")}
            selections.append(dict(method=selected_method, seed=seed, selected_candidate=selected, candidates=points))
            atomic_write_json(output / f"component_selection_{selected_method}.json", [r for r in selections if r["method"] == selected_method])
            context = dict(phase=phase, method=selected_method, seed=seed, fold="full_training",
                completed_jobs=completed, total_jobs=total,
                progress_path=str((output / "training_progress.json").resolve()))
            summary = job(config, selected_method, seed, "full_training", records, config["train_video_ids"],
                config["validation_video_ids"], output, context, identity, resume, stop_after_components, selected)
            outputs.append(dict(method=selected_method, seed=seed,
                fit_directory=str((output / "fit" / selected_method / f"seed{seed}").resolve()), summary=summary))
            completed += 1
    if source_hashes() != identity["source_hashes"]:
        raise ValueError("Component discovery source changed during execution")
    result = dict(status="COMPLETE", phase=phase, method=method, completed_jobs=completed, total_jobs=total,
        outputs=outputs, component_selection=selections, elapsed_seconds=time.perf_counter() - started,
        runtime=runtime(config),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if str(config["device"]).startswith("cuda") else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if str(config["device"]).startswith("cuda") else 0, **identity)
    atomic_write_json(output / "training_summary.json", result)
    for selected_method in methods:
        atomic_write_json(output / f"training_summary_{selected_method}.json", dict(result,
            outputs=[r for r in outputs if r["method"] == selected_method],
            component_selection=[r for r in selections if r["method"] == selected_method]))
    atomic_write_json(output / "training_progress.json", dict(status="COMPLETE", phase=phase, method=method,
        completed_jobs=completed, total_jobs=total, updated_at=shared.shared.now()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("smoke", "fit"), required=True)
    parser.add_argument("--method", choices=("all",) + METHODS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-components", type=int)
    args = parser.parse_args()
    result = run(read_json(args.config), args.phase, args.method, args.resume, args.stop_after_components)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
