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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_template_component_adaptation as previous
from src import observed_view_adaptation as adaptation
from src.checkpoint_io import atomic_write_json, read_json
from src.conditional_component_policy import source_weights, supervised_targets

parent = previous.parent


def hashes():
    values = previous.source_hashes()
    for path in (Path(__file__), Path(adaptation.__file__), Path(previous.__file__)):
        values[str(path.relative_to(ROOT))] = parent.digest(path)
    return values


def prepare_views(run, smoke, resume):
    config = read_json(run / "config.json")
    output = run / "smoke_views" if smoke else Path(config["view_cache"])
    output.mkdir(parents=True, exist_ok=True)
    videos = config["train_video_ids"][:1] if smoke else config["train_video_ids"]
    completed, started = [], time.perf_counter()
    for number, video in enumerate(videos):
        source = Path(config["token_data_root"]) / "videos" / video
        receipt = read_json(source / "complete.json")
        if receipt["status"] != "COMPLETE":
            raise ValueError("Training token cache is incomplete")
        identity = dict(config=parent.digest(run / "config.json"), source_hashes=hashes(),
            receipt=parent.digest(source / "complete.json"), video=video,
            token_bytes=(source / "tokens.npy").stat().st_size, token_sha256=receipt["assets"]["tokens.npy"])
        folder = output / video
        previous.make_identity(folder, identity, resume)
        ready = folder / "summary.json"
        if ready.exists():
            complete = read_json(ready)
            if parent.digest(folder / "frames.npz") != complete["sha256"]:
                raise ValueError("Observed-frame cache changed")
        else:
            tokens = np.load(source / "tokens.npy", mmap_mode="r", allow_pickle=False)
            indices = np.load(source / "record_token_index.npy", allow_pickle=False)
            with np.load(source / "masks.npz", allow_pickle=False) as saved:
                masks = saved["masks"]
            valid = np.load(source / "valid.npy", allow_pickle=False)
            raw = np.full((len(indices), 7, 8, 768), np.nan, dtype=np.float64)
            for record, token_index in enumerate(indices):
                for view in np.flatnonzero(valid[record]):
                    frames, positions = adaptation.frame_views(tokens[token_index], masks[record, view])
                    raw[record, view, positions] = frames
            canonical = np.load(source / "canonical_raw.npy", allow_pickle=False)
            for record in range(len(indices)):
                counts = masks[record, 0].sum(axis=1)
                recovered = (raw[record, 0] * counts[:, None]).sum(axis=0) / counts.sum()
                if not np.allclose(recovered, canonical[record], rtol=0, atol=1e-10):
                    raise ValueError("Frame pooling does not reproduce the canonical source")
            np.savez_compressed(folder / "frames.npz", raw=raw, valid=valid)
            complete = dict(status="COMPLETE", video=video, records=len(indices), sha256=parent.digest(folder / "frames.npz"))
            atomic_write_json(ready, complete)
        completed.append(complete)
        atomic_write_json(run / ("smoke_views_progress.json" if smoke else "views_progress.json"),
            dict(completed=number + 1, total=len(videos), phase=video))
        print("OBSERVED_VIEW_CACHE", number + 1, len(videos), video, flush=True)
    atomic_write_json(run / ("smoke_views_summary.json" if smoke else "views_summary.json"),
        dict(status="COMPLETE", videos=completed, elapsed_seconds=time.perf_counter() - started))


def run_held(run, smoke, resume, stop_after_sources):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    seed_values = config["seeds"][:1] if smoke else config["seeds"]
    folds = [0] if smoke else range(3)
    source_hashes, jobs, stopped_count, started = hashes(), [], 0, time.perf_counter()
    for method in config["methods"]:
        for seed in seed_values:
            for fold in folds:
                model = previous.bundle(config, records, method, seed, fold)
                folder = root / "held" / method / f"seed{seed}" / f"fold{fold}"
                old_path = Path(config["predecessor_run"]) / "held" / method / f"seed{seed}" / f"fold{fold}" / "summary.json"
                old_rows = {(r["source_index"], r["view"]): r for r in read_json(old_path)["sources"]}
                view_signatures = {video: parent.digest(Path(config["view_cache"]) / video / "summary.json") for video in model["held"]}
                previous.make_identity(folder, dict(config=parent.digest(run / "config.json"), protocol=parent.digest(run / "protocol.json"),
                    source_hashes=source_hashes, model=model["identity"], views=view_signatures,
                    previous=parent.digest(old_path), seed=seed, method=method, fold=fold, smoke=smoke), resume)
                if (folder / "summary.json").exists():
                    jobs.append(read_json(folder / "summary.json"))
                    continue
                predictor, assets = model["predictor"], model["assets"]
                query = predictor.encode(assets["raw"][:, 0])
                pairs = previous.inherited.parent.shared.shared.chronological_pairs(records, model["held"])
                views = {}
                for video in model["held"]:
                    with np.load(Path(config["view_cache"]) / video / "frames.npz", allow_pickle=False) as saved:
                        views[video] = saved["raw"].copy()
                rows = []
                source_indices = sorted({p["source_index"] for p in pairs})
                for ordinal, source in enumerate(source_indices):
                    local = [p for p in pairs if p["source_index"] == source]
                    positive = [p["query_index"] for p in local if p["same_identity"]]
                    negative = [p["query_index"] for p in local if not p["same_identity"]]
                    if not positive or not negative:
                        continue
                    record = records[source]
                    for view in np.flatnonzero(assets["valid"][source, 1:]) + 1:
                        target = folder / f"source{source:04d}_view{view}.json"
                        if target.exists() and resume:
                            rows.append(read_json(target))
                            continue
                        positive_raw = views[record["video_id"]][record["index"], view]
                        positive_raw = positive_raw[np.isfinite(positive_raw).all(axis=1)]
                        adapted = adaptation.adapt(assets["raw"][source, view], assets["codes"][source, view], predictor,
                            positive_raw, model["bank"], model["weights"], model["candidates"])
                        choice = adapted["choice"]
                        mean = adapted["positive_memory"].mean(axis=0, dtype=np.float64)
                        mean /= np.linalg.norm(mean)
                        memories = np.vstack([adapted["original_memory"], adapted["candidates_memory"][choice], mean])
                        _, utility, scores = supervised_targets(memories, query, positive, negative, .99, .05)
                        recall = [float(np.mean(s[positive] > np.quantile(s[negative], .99))) for s in scores]
                        old = old_rows[(source, int(view))]
                        if abs(utility[0] - old["zero_utility"]) > 1e-7 or recall[0] != old["zero_recall"]:
                            raise ValueError("Inherited held baseline differs")
                        svm = adaptation.fit_view_svm(adapted["positive_memory"], model["bank"], model["weights"])
                        svm_scores = query.astype(float) @ svm[:128] + svm[128]
                        svm_recall = float(np.mean(svm_scores[positive] > np.quantile(svm_scores[negative], .99)))
                        row = dict(source_index=source, view=int(view), video_id=record["video_id"], positive_views=len(positive_raw),
                            choice=choice, component_count=int(np.count_nonzero(adapted["gains"][choice])),
                            available_gain=float(adapted["candidate_objective"][choice]),
                            positive_change=float(adapted["candidate_positive_change"][choice]),
                            negative_change=float(adapted["candidate_negative_change"][choice]),
                            zero_utility=float(utility[0]), adapted_utility=float(utility[1]), mean_utility=float(utility[2]),
                            zero_recall=recall[0], adapted_recall=recall[1], mean_recall=recall[2], svm_recall=svm_recall,
                            self_utility=old["adapted_utility"], self_recall=old["adapted_recall"])
                        atomic_write_json(target, row)
                        rows.append(row)
                        stopped_count += 1
                        if stop_after_sources is not None and stopped_count >= stop_after_sources:
                            raise SystemExit(75)
                    print("HELD_OBSERVED_SOURCE", method, seed, fold, ordinal + 1, len(source_indices), flush=True)
                weights = source_weights(rows)
                metrics = [name for name in rows[0] if name.endswith(("_utility", "_recall"))]
                procedures = {}
                for video in sorted({r["video_id"] for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r["video_id"] == video]
                    w = weights[ids] / weights[ids].sum()
                    procedures[video] = {m: float(w @ [rows[i][m] for i in ids]) for m in metrics}
                result = dict(status="COMPLETE", method=method, seed=seed, fold=fold, sources=rows, per_procedure=procedures,
                    means={m: float(weights @ [row[m] for row in rows]) for m in metrics})
                atomic_write_json(folder / "summary.json", result)
                jobs.append(result)
                atomic_write_json(root / "held_progress.json", dict(completed=len(jobs), total=2 * len(seed_values) * len(folds), phase="held"))
    if hashes() != source_hashes:
        raise ValueError("Observed-view source changed during evaluation")
    atomic_write_json(root / "held_summary.json", dict(status="COMPLETE", jobs=jobs, elapsed_seconds=time.perf_counter() - started))


def source_memories(run, config, records, models, reference, output, seed, resume, stop_after_sources):
    mappings, rows = {}, []
    for index, record in enumerate(records):
        episode = record["episode"]["episode_id"]
        if not record["available"]:
            rows.append(dict(population=record["population"], video=record["video"], episode=episode, status="UNAVAILABLE"))
            continue
        folder = output / "sources" / record["population"] / record["video"] / episode
        folder.mkdir(parents=True, exist_ok=True)
        old_folder = Path(config["predecessor_run"]) / "evaluation" / f"seed{seed}" / "sources" / record["population"] / record["video"] / episode
        inputs = {str(p): parent.digest(p) for p in (record["original"], record["source"] / "tokens.npz", old_folder / "adaptation.npz")}
        target, array_path = folder / "summary.json", folder / "adaptation.npz"
        if target.exists():
            row = read_json(target)
            if not resume or row["inputs"] != inputs or parent.digest(array_path) != row["array_sha256"]:
                raise ValueError("Observed-view source checkpoint changed")
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
        else:
            with np.load(record["source"] / "tokens.npz", allow_pickle=False) as saved:
                tokens, mask, original = saved["tokens"].copy(), saved["mask"].astype(bool), saved["original_raw"].reshape(-1).copy()
                frames = saved["frame_indices"].copy()
            with np.load(record["original"], allow_pickle=False) as saved:
                if not np.array_equal(original, saved["raw_mean"].reshape(-1)):
                    raise ValueError("Actual source identity differs")
            if frames[-1] != record["episode"]["click"]["input_frame"] or np.any(np.diff(frames) != 1):
                raise ValueError("Source views are not the actual causal clip")
            raw_views, positions = adaptation.frame_views(tokens, mask)
            arrays = dict(original_raw=original, observed_raw=raw_views, observed_frames=frames[positions])
            decisions = {}
            for method, model in models.items():
                predictor = model["predictor"]
                codes = predictor.pooled_codes(tokens, mask)
                adapted = adaptation.adapt(original, codes, predictor, raw_views, model["bank"], model["weights"], model["candidates"])
                choice = adapted["choice"]
                prefix = "sae" if method == "sparse_edit" else "dense"
                arrays["memory__" + prefix + "_views"] = adapted["candidates_memory"][choice]
                arrays.update({method + "__" + k: v for k, v in adapted.items() if isinstance(v, np.ndarray)})
                decisions[method] = dict(choice=choice, components=int(np.count_nonzero(adapted["gains"][choice])),
                    objective=float(adapted["candidate_objective"][choice]), positive_change=float(adapted["candidate_positive_change"][choice]),
                    negative_change=float(adapted["candidate_negative_change"][choice]))
                if method == "sparse_edit":
                    mean = adapted["positive_memory"].mean(axis=0, dtype=np.float64)
                    arrays["memory__view_mean"] = mean / np.linalg.norm(mean)
                    arrays["memory__view_svm"] = adaptation.fit_view_svm(adapted["positive_memory"], model["bank"], model["weights"])
                    with np.load(old_folder / "adaptation.npz", allow_pickle=False) as saved:
                        arrays["memory__sae_self"] = saved["memory__sae_adapted"].copy()
                        arrays["memory__sae0445"] = saved["memory__sae0445"].copy()
                    predictor.source_gains = adapted["gains"][choice]
                    learned = adapted["candidates_raw"][choice] - original
                    controls = []
                    for permutation in range(3):
                        delta, gains, info = parent.random_delta(predictor, tokens, mask, learned, seed, episode, method, permutation)
                        arrays[f"memory__sae_random{permutation}"] = predictor.encode(original + delta)[0]
                        arrays[f"random{permutation}__delta"] = delta
                        arrays[f"random{permutation}__gains"] = gains
                        controls.append(info)
                    decisions[method]["random_controls"] = controls
            np.savez_compressed(array_path, **arrays)
            row = dict(population=record["population"], video=record["video"], episode=episode, status="COMPLETE",
                observed_frames=frames[positions].tolist(), decisions=decisions, inputs=inputs, array_sha256=parent.digest(array_path))
            atomic_write_json(target, row)
        key, original = previous.adaptation.raw_key(arrays["original_raw"]), arrays["original_raw"]
        for name, memory in arrays.items():
            if name.startswith("memory__"):
                method = name.removeprefix("memory__")
                mapping = mappings.setdefault(method, {})
                if key in mapping and not np.array_equal(mapping[key][1], memory):
                    raise ValueError("Duplicate source has inconsistent view memory")
                mapping[key] = (original, memory)
        rows.append(row)
        atomic_write_json(output / "source_progress.json", dict(completed=index + 1, total=len(records), phase="observed-view source"))
        print("OBSERVED_SOURCE", index + 1, len(records), episode, flush=True)
        if stop_after_sources is not None and index + 1 >= stop_after_sources:
            raise SystemExit(75)
    atomic_write_json(output / "source_diagnostics.json", dict(sources=rows))
    return {name: previous.adaptation.TemplatePredictor(reference, value, name == "view_svm") for name, value in mappings.items()}


def evaluate(run, seed, smoke, resume, stop_after_sources):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    settings = read_json(config["evaluation_settings_file"])
    settings.update(include_interventions=False, protocol=str(run / "protocol.json"), retention_floor=config["application_retention_floor"])
    records = parent.source_records(config, settings, smoke)
    training = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    models = {m: previous.bundle(config, training, m, seed, "full_training") for m in config["methods"]}
    output = root / "evaluation" / f"seed{seed}"
    signature = hashes()
    previous.make_identity(output, dict(config=parent.digest(run / "config.json"), protocol=parent.digest(run / "protocol.json"),
        source_hashes=signature, models={m: b["identity"] for m, b in models.items()}, seed=seed, smoke=smoke,
        settings=parent.digest(Path(config["evaluation_settings_file"])), python=sys.version, numpy=np.__version__, torch=str(torch.__version__)), resume)
    if (output / "summary.json").exists():
        print("REUSE_OBSERVED_EVALUATION", seed, flush=True)
        return
    started = time.perf_counter()
    atomic_write_json(output / "evaluation_settings.json", settings)
    reference = parent.reference_api.Representations(Path(config["reference_fit"]))
    predictors = source_memories(run, config, records, models, reference, output, seed, resume, stop_after_sources)
    parent.evaluation.evaluate_phase(settings, output, "development", predictors, None, smoke, resume)
    points_path = output / "operating_points.json"
    points = read_json(points_path)["methods"] if points_path.exists() else parent.evaluation.calibrate(output,
        list(predictors) + ["reference_supcon"], settings["retention_floor"])
    parent.evaluation.evaluate_phase(settings, output, "extension", predictors, points, smoke, resume)
    if hashes() != signature:
        raise ValueError("Source changed during evaluation")
    atomic_write_json(output / "summary.json", dict(status="REAL_SMOKE_COMPLETE" if smoke else "COMPLETE", seed=seed,
        operating_points=points, elapsed_seconds=time.perf_counter() - started, conditions=list(predictors) + ["reference_supcon"]))
    atomic_write_json(output / "progress.json", dict(completed=2 if smoke else 10, total=2 if smoke else 10, phase="COMPLETE"))


def summarize(run, smoke, resume):
    previous.summarize(run, smoke, resume)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    root = run / "smoke" if smoke else run
    summary = read_json(root / "summary.json")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(summary["procedure_rows"])
    table.to_csv(output / "procedure_outcomes.csv", index=False)
    labels = dict(reference_supcon="Original", sae0445="Global SAE", sae_self="Self-only SAE", sae_views="Observed-view SAE",
        dense_views="Observed-view dense", view_mean="View mean", view_svm="View SVM",
        sae_random0="Random 1", sae_random1="Random 2", sae_random2="Random 3")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for i, phase in enumerate(("development", "extension")):
        for j, metric in enumerate(("removal", "retention")):
            ax = axes[i, j]
            for number, method in enumerate(labels):
                values = table[(table.population == phase) & (table.method == method)].groupby("seed")[metric].mean()
                ax.scatter(100 * values.values, np.full(len(values), number), c=["#0072B2", "#D55E00", "#009E73"][:len(values)], s=24)
                ax.scatter(100 * values.mean(), number, color="black", marker="|")
            ax.set_yticks(range(len(labels)), labels.values(), fontsize=9)
            ax.set(xlabel="Percent", title=phase + " | " + metric)
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Observed-view component adaptation" + (" — real smoke only" if smoke else ""))
    fig.text(.02, .02, "All seeds; black marks show means. Equal procedure weights. Panel ranges differ. Extension retains its previously examined status.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("application_outcomes." + suffix), dpi=180)
    plt.close(fig)
    if (root / "held_summary.json").exists():
        held = read_json(root / "held_summary.json")
        pd.DataFrame([dict(method=j["method"], seed=j["seed"], fold=j["fold"], video=v, **metrics)
            for j in held["jobs"] for v, metrics in j["per_procedure"].items()]).to_csv(output / "held_procedures.csv", index=False)
    lines = ["# Observed-view component adaptation", "", "Both application populations have been examined previously.", "",
        "| Population | Method | Repeat removal | Other retention | First prompt |", "|---|---|---:|---:|---:|"]
    for row in summary["aggregates"]:
        lines.append(f"| {row['population']} | {row['method']} | " + " | ".join(f"{100 * row[m]:.4f}%" for m in parent.METRICS) + " |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(output / "manifest.json", dict(status="COMPLETE", source_sha256=parent.digest(__file__),
        summary_sha256=parent.digest(root / "summary.json"), files={p.name: parent.digest(p) for p in output.iterdir() if p.suffix in (".png", ".pdf", ".csv")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("views", "held", "evaluate", "summary"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "views":
        prepare_views(args.run, args.smoke, args.resume)
    elif args.phase == "held":
        run_held(args.run, args.smoke, args.resume, args.stop_after_sources)
    elif args.phase == "evaluate":
        evaluate(args.run, args.seed, args.smoke, args.resume, args.stop_after_sources)
    else:
        summarize(args.run, args.smoke, args.resume)
