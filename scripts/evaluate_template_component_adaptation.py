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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_token_memory_edit as parent
import train_conditioned_component_policy as inherited
from src import template_component_adaptation as adaptation
from src.conditional_component_policy import cohort_weights, source_weights, supervised_targets
from src.token_memory_edit import TokenMemoryPredictor
from src.checkpoint_io import atomic_write_json, read_json


def source_hashes():
    result = inherited.source_hashes()
    for path in (Path(__file__), Path(adaptation.__file__), Path(parent.__file__),
                 Path(parent.evaluation.__file__), Path(parent.evaluation.shared.__file__),
                 Path(parent.evaluation.shared.memory.__file__)):
        result[str(path.relative_to(ROOT))] = parent.digest(path)
    return result


def bundle(config, records, method, seed, fold):
    folds = read_json(Path(config["parent_dictionary_run"]) / "grouped_folds.json")["inner_held_folds"]
    held = config["validation_video_ids"] if fold == "full_training" else folds[fold]
    fit = [v for v in config["train_video_ids"] if v not in held]
    assets = inherited.parent.load_assets(config, records, method, seed, fold, fit, held)
    predictor = TokenMemoryPredictor(assets["model"], assets["mean"], assets["scale"], np.zeros(1024), assets["reference"])
    indices = [i for i, row in enumerate(records) if row["video_id"] in fit]
    bank = predictor.encode(assets["raw"][indices, 0])
    candidates, old_gains, component_identity = inherited.component_actions(config, method, seed, fold, fit, held)
    old_action = 0
    if fold == "full_training":
        previous = read_json(Path(config["parent_component_run"]) / "fit" / method / f"seed{seed}" / "summary.json")
        old_action = next(i for i, c in enumerate(candidates) if c == previous["selected_candidate"])
    return dict(assets=assets, predictor=predictor, bank=bank, weights=cohort_weights(records, indices),
        candidates=candidates, old_gains=old_gains[old_action], held=held, fit=fit,
        identity=dict(assets=assets["identity"], component=component_identity, bank_indices=indices))


def make_identity(directory, payload, resume):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.exists():
        if not resume or read_json(path) != payload:
            raise ValueError("Existing adaptation run has different inputs or needs resume")
    else:
        atomic_write_json(path, payload)


def run_held(run, smoke, resume):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    all_results, started = [], time.perf_counter()
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    folds = [0] if smoke else range(3)
    hashes = source_hashes()
    for method in config["methods"]:
        for seed in seeds:
            for fold in folds:
                folder = root / "held" / method / f"seed{seed}" / f"fold{fold}"
                model = bundle(config, records, method, seed, fold)
                identity = dict(config=parent.digest(run / "config.json"), protocol=parent.digest(run / "protocol.json"),
                    source_hashes=hashes, model=model["identity"], seed=seed, method=method, fold=fold, smoke=smoke)
                make_identity(folder, identity, resume)
                if (folder / "summary.json").exists():
                    all_results.append(read_json(folder / "summary.json"))
                    continue
                assets, predictor = model["assets"], model["predictor"]
                pairs = inherited.parent.shared.shared.chronological_pairs(records, model["held"])
                queries = predictor.encode(assets["raw"][:, 0])
                rows = []
                sources = sorted({p["source_index"] for p in pairs})
                for source_number, source in enumerate(sources, start=1):
                    local = [p for p in pairs if p["source_index"] == source]
                    positive = [p["query_index"] for p in local if p["same_identity"]]
                    negative = [p["query_index"] for p in local if not p["same_identity"]]
                    if not positive or not negative:
                        continue
                    for view in np.flatnonzero(assets["valid"][source, 1:]) + 1:
                        path = folder / f"source{source:04d}_view{view}.json"
                        if path.exists() and resume:
                            rows.append(read_json(path))
                            continue
                        raw, codes = assets["raw"][source, view], assets["codes"][source, view]
                        adapted = adaptation.adapt_source(raw, codes, predictor, model["bank"], model["weights"], model["candidates"])
                        choice = adapted["choice"]
                        memories = np.vstack([adapted["original_memory"], adapted["candidates_memory"][choice]])
                        _, utility, scores = supervised_targets(memories, queries, positive, negative, .99, .05)
                        svm, svm_info = adaptation.fit_template_svm(memories[0], model["bank"], model["weights"])
                        svm_scores = queries.astype(float) @ svm[:128] + svm[128]
                        svm_threshold = np.quantile(svm_scores[negative], .99)
                        svm_recall = float(np.mean(svm_scores[positive] > svm_threshold))
                        recalls = [float(np.mean(score[positive] > np.quantile(score[negative], .99))) for score in scores]
                        row = dict(source_index=source, view=int(view), video_id=records[source]["video_id"],
                            choice=choice, eligible_components=len(adapted["ordered_components"]),
                            available_objective=float(adapted["candidate_objective"][choice]),
                            zero_utility=float(utility[0]), adapted_utility=float(utility[1]),
                            zero_recall=recalls[0], adapted_recall=recalls[1], svm_recall=svm_recall,
                            positives=positive, negatives=negative, svm=svm_info)
                        atomic_write_json(path, row)
                        rows.append(row)
                    print("HELD_SOURCE", method, seed, fold, source_number, "of", len(sources), "source_index", source, flush=True)
                weights = source_weights(rows)
                metrics = ["zero_utility", "adapted_utility", "zero_recall", "adapted_recall", "svm_recall"]
                per_procedure = {}
                for video in sorted({r["video_id"] for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r["video_id"] == video]
                    local_weight = weights[ids] / weights[ids].sum()
                    per_procedure[video] = {name: float(local_weight @ [rows[i][name] for i in ids]) for name in metrics}
                result = dict(status="COMPLETE", method=method, seed=seed, fold=fold, sources=rows,
                    per_procedure=per_procedure, means={name: float(weights @ [r[name] for r in rows]) for name in metrics})
                atomic_write_json(folder / "summary.json", result)
                all_results.append(result)
                atomic_write_json(root / "held_progress.json", dict(completed=len(all_results), total=len(config["methods"]) * len(seeds) * len(folds),
                    phase=f"{method} seed{seed} fold{fold}"))
    if source_hashes() != hashes:
        raise ValueError("Held analysis source changed")
    atomic_write_json(root / "held_summary.json", dict(status="COMPLETE", jobs=all_results, elapsed_seconds=time.perf_counter() - started))


def source_memories(config, records, models, reference, output, seed, resume, stop_after_sources):
    mappings, diagnostics = {}, []
    for number, record in enumerate(records):
        episode = record["episode"]["episode_id"]
        if not record["available"]:
            diagnostics.append(dict(population=record["population"], video=record["video"], episode=episode, status="UNAVAILABLE"))
            continue
        folder = output / "sources" / record["population"] / record["video"] / episode
        folder.mkdir(parents=True, exist_ok=True)
        summary_path, array_path = folder / "summary.json", folder / "adaptation.npz"
        inputs = {str(p): parent.digest(p) for p in (record["original"], record["source"] / "tokens.npz")}
        if summary_path.exists():
            diagnostic = read_json(summary_path)
            if not resume or diagnostic["inputs"] != inputs or parent.digest(array_path) != diagnostic["array_sha256"]:
                raise ValueError("Source adaptation checkpoint changed")
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
        else:
            with np.load(record["original"], allow_pickle=False) as saved:
                original = saved["raw_mean"].reshape(-1).copy()
            with np.load(record["source"] / "tokens.npz", allow_pickle=False) as saved:
                tokens, mask = saved["tokens"].copy(), saved["mask"].astype(bool)
                if not np.array_equal(original, saved["original_raw"].reshape(-1)):
                    raise ValueError("Observed source descriptor changed")
            arrays, decisions, baseline = dict(original_raw=original), {}, None
            for method, model in models.items():
                predictor = model["predictor"]
                codes = predictor.pooled_codes(tokens, mask)
                adapted = adaptation.adapt_source(original, codes, predictor, model["bank"], model["weights"], model["candidates"])
                choice = adapted["choice"]
                prefix = "sae" if method == "sparse_edit" else "dense"
                arrays["memory__" + prefix + "_adapted"] = adapted["candidates_memory"][choice]
                arrays.update({method + "__" + key: value for key, value in adapted.items() if isinstance(value, np.ndarray)})
                decisions[method] = dict(choice=choice, candidate=model["candidates"][choice],
                    eligible_components=len(adapted["ordered_components"]), active_components=len(adapted["active_components"]),
                    gain=float(adapted["candidate_objective"][choice]), component_count=int(np.count_nonzero(adapted["gains"][choice])))
                if baseline is None:
                    baseline = adapted["original_memory"]
                    arrays["memory__template_svm"], decisions["svm"] = adaptation.fit_template_svm(baseline, model["bank"], model["weights"])
                elif not np.array_equal(baseline, adapted["original_memory"]):
                    raise ValueError("Dictionary methods disagree on the original memory")
                if method == "sparse_edit":
                    arrays["memory__sae0445"] = predictor.memory_from_pooled(original, codes, model["old_gains"])
                    predictor.source_gains = adapted["gains"][choice]
                    learned = adapted["candidates_raw"][choice] - original
                    controls = []
                    for permutation in range(3):
                        delta, gains, control = parent.random_delta(predictor, tokens, mask, learned, seed, episode, method, permutation)
                        arrays[f"memory__sae_random{permutation}"] = predictor.encode(original + delta)[0]
                        arrays[f"random{permutation}__delta"] = delta
                        arrays[f"random{permutation}__gains"] = gains
                        controls.append(control)
                    decisions[method]["random_controls"] = controls
            np.savez_compressed(array_path, **arrays)
            diagnostic = dict(population=record["population"], video=record["video"], episode=episode,
                status="COMPLETE", decisions=decisions, inputs=inputs, array_sha256=parent.digest(array_path))
            atomic_write_json(summary_path, diagnostic)
        original = arrays["original_raw"]
        key = adaptation.raw_key(original)
        for name, memory in arrays.items():
            if name.startswith("memory__"):
                method = name.removeprefix("memory__")
                mapping = mappings.setdefault(method, {})
                if key in mapping and not np.array_equal(mapping[key][1], memory):
                    raise ValueError("Duplicate source has inconsistent adapted memory")
                mapping[key] = (original, memory)
        diagnostics.append(diagnostic)
        atomic_write_json(output / "source_progress.json", dict(completed=number + 1, total=len(records), phase="source adaptation"))
        print("ADAPTED_SOURCE", number + 1, len(records), episode, flush=True)
        if stop_after_sources is not None and number + 1 >= stop_after_sources:
            raise SystemExit(75)
    atomic_write_json(output / "source_diagnostics.json", dict(sources=diagnostics))
    return {name: adaptation.TemplatePredictor(reference, memory, name == "template_svm") for name, memory in mappings.items()}


def evaluate(run, seed, smoke, resume, stop_after_sources):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    settings = read_json(config["evaluation_settings_file"])
    settings.update(include_interventions=False, protocol=str(run / "protocol.json"), retention_floor=config["application_retention_floor"])
    records = parent.source_records(config, settings, smoke)
    training_records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    output = root / "evaluation" / f"seed{seed}"
    models = {method: bundle(config, training_records, method, seed, "full_training") for method in config["methods"]}
    hashes = source_hashes()
    identity = dict(config=parent.digest(run / "config.json"), protocol=parent.digest(run / "protocol.json"),
        source_hashes=hashes, models={m: b["identity"] for m, b in models.items()}, seed=seed, smoke=smoke,
        settings_sha256=parent.digest(Path(config["evaluation_settings_file"])), python=sys.version,
        numpy=np.__version__, torch=str(torch.__version__))
    make_identity(output, identity, resume)
    if (output / "summary.json").exists():
        print("REUSE_TEMPLATE_EVALUATION", seed, flush=True)
        return
    started = time.perf_counter()
    atomic_write_json(output / "evaluation_settings.json", settings)
    reference = parent.reference_api.Representations(Path(config["reference_fit"]))
    predictors = source_memories(config, records, models, reference, output, seed, resume, stop_after_sources)
    parent.evaluation.evaluate_phase(settings, output, "development", predictors, None, smoke, resume)
    points_path = output / "operating_points.json"
    points = read_json(points_path)["methods"] if points_path.exists() else parent.evaluation.calibrate(output,
        list(predictors) + ["reference_supcon"], settings["retention_floor"])
    parent.evaluation.evaluate_phase(settings, output, "extension", predictors, points, smoke, resume)
    if source_hashes() != hashes:
        raise ValueError("Evaluation source changed")
    atomic_write_json(output / "summary.json", dict(status="REAL_SMOKE_COMPLETE" if smoke else "COMPLETE", seed=seed,
        operating_points=points, elapsed_seconds=time.perf_counter() - started, conditions=list(predictors) + ["reference_supcon"]))
    atomic_write_json(output / "progress.json", dict(completed=2 if smoke else 10, total=2 if smoke else 10, phase="COMPLETE"))


def summarize(run, smoke, resume):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    rows, inputs = [], {}
    for seed in seeds:
        directory = root / "evaluation" / f"seed{seed}"
        summary = read_json(directory / "summary.json")
        inputs[str(directory / "summary.json")] = parent.digest(directory / "summary.json")
        for method, point in summary["operating_points"].items():
            for phase in ("development", "extension"):
                path = directory / phase / (method + "__score_procedure_curve.npz")
                inputs[str(path)] = parent.digest(path)
                with np.load(path, allow_pickle=False) as curve:
                    column = int(np.flatnonzero(curve["threshold"] == point["threshold"])[0]) if phase == "development" else 0
                    for i, video in enumerate(curve["videos"].tolist()):
                        values = {metric: float(curve[key][i, column]) for metric, key in parent.METRICS.items()}
                        rows.append(dict(seed=seed, method=method, population=phase, video=video, threshold=point["threshold"],
                            **{k: v if np.isfinite(v) else None for k, v in values.items()}))
    if (root / "summary.json").exists():
        if not resume or read_json(root / "summary.json")["inputs"] != inputs:
            raise ValueError("Completed summary changed")
        return
    aggregates = []
    for population, method in sorted({(r["population"], r["method"]) for r in rows}):
        by_seed = {seed: {metric: parent.finite_mean([r[metric] for r in rows if r["population"] == population
            and r["method"] == method and r["seed"] == seed]) for metric in parent.METRICS} for seed in seeds}
        aggregates.append(dict(population=population, method=method, seed_results=by_seed,
            **{metric: parent.finite_mean([r[metric] for r in by_seed.values()]) for metric in parent.METRICS}))
    atomic_write_json(root / "summary.json", dict(status="REAL_SMOKE_COMPLETE" if smoke else "COMPLETE", inputs=inputs,
        procedure_rows=rows, aggregates=aggregates, seeds=seeds, independent_unit="procedure"))
    lines = ["# Source-template component adaptation", "", "Both application populations have been examined previously.", "",
        "| Population | Method | Repeat removal | Other retention | First prompt |", "|---|---|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['population']} | {row['method']} | " + " | ".join(f"{100 * row[m]:.4f}%" for m in parent.METRICS) + " |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(root / "summary_progress.json", dict(completed=1, total=1, phase="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("held", "evaluate", "summary"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "held":
        run_held(args.run, args.smoke, args.resume)
    elif args.phase == "evaluate":
        evaluate(args.run, args.seed, args.smoke, args.resume, args.stop_after_sources)
    else:
        summarize(args.run, args.smoke, args.resume)
