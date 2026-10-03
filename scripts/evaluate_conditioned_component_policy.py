import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_token_memory_edit as parent
import train_conditioned_component_policy as training
from src import conditional_component_policy as policy
from src.checkpoint_io import atomic_write_json, read_json
from src.token_memory_edit import TokenMemoryPredictor


def source_hashes():
    result = training.source_hashes()
    for path in (Path(__file__), Path(parent.__file__), Path(parent.evaluation.__file__),
                 Path(parent.evaluation.shared.__file__), Path(parent.evaluation.shared.memory.__file__)):
        result[str(path.relative_to(ROOT))] = parent.digest(path)
    return result


def load_model(config, root, method, seed, records):
    directory = root / "fit" / method / f"seed{seed}"
    summary = read_json(directory / "summary.json")
    for name, signature in summary["artifacts"].items():
        if parent.digest(directory / name) != signature:
            raise ValueError("Fitted policy asset changed")
    model = read_json(directory / "policy.json")
    assets = training.parent.load_assets(config, records, method, seed, "full_training",
                                       config["train_video_ids"], config["validation_video_ids"])
    with np.load(directory / "actions.npz", allow_pickle=False) as saved:
        gains = saved["gains"].copy()
    with np.load(directory / "training_context_actions.npz", allow_pickle=False) as saved:
        donors = saved["actions"].copy()
    predictor = TokenMemoryPredictor(assets["model"], assets["mean"], assets["scale"],
                                     gains[0], assets["reference"])
    indices = [i for i, row in enumerate(records) if row["video_id"] in config["train_video_ids"]]
    weights = policy.cohort_weights(records, indices)
    bank = predictor.encode(assets["raw"][indices, 0])
    original = read_json(Path(config["parent_component_run"]) / "fit" / method / f"seed{seed}" / "summary.json")
    action = next(i for i, c in enumerate(model["candidates"]) if c == original["selected_candidate"])
    return dict(policy=model, assets=assets, gains=gains, donors=donors, predictor=predictor,
                bank=bank, weights=weights, original_action=action)


def add_memory(mappings, name, original, memory, threshold):
    key = policy.raw_key(original)
    content = (np.asarray(original).reshape(-1).copy(), memory.copy(), float(threshold))
    mapping = mappings.setdefault(name, {})
    if key in mapping:
        previous = mapping[key]
        if not all(np.array_equal(a, b) for a, b in zip(previous, content)):
            raise ValueError("Identical source descriptors received different policies")
    mapping[key] = content


def source_memories(config, records, models, reference, output, seed):
    mappings, diagnostics = {}, []
    for number, record in enumerate(records):
        episode, source = record["episode"], record["source"]
        if not record["available"]:
            diagnostics.append(dict(population=record["population"], video=record["video"],
                episode_id=episode["episode_id"], status="UNAVAILABLE"))
            continue
        with np.load(record["original"], allow_pickle=False) as archive:
            original = archive["raw_mean"].reshape(-1).copy()
            if not bool(archive["available"]):
                raise ValueError("Available token source has no original descriptor")
        with np.load(source / "tokens.npz", allow_pickle=False) as archive:
            tokens, mask = archive["tokens"].copy(), archive["mask"].astype(bool)
            if not np.array_equal(original, archive["original_raw"].reshape(-1)):
                raise ValueError("Original source identity differs from token receipt")
        support = policy.mask_features(mask)
        arrays, decisions = dict(support=support, original_raw=original), {}
        reference_threshold = None
        for method, bundle in models.items():
            model, predictor = bundle["policy"], bundle["predictor"]
            codes = predictor.pooled_codes(tokens, mask)
            edited, memories = training.candidate_arrays(bundle["assets"], predictor.reference,
                                                         original, codes, bundle["gains"])
            features = policy.candidate_features(original, edited, codes, memories, bundle["bank"],
                                                 bundle["weights"], support, model["candidates"])
            action, advantages, thresholds = policy.choose(model, features)
            action = int(action)
            fixed = int(model["fixed_action"])
            if reference_threshold is None:
                reference_threshold = float(thresholds[0])
                add_memory(mappings, "reference_calibrated", original, memories[0], thresholds[0])
            elif not np.isclose(reference_threshold, thresholds[0], atol=1e-10, rtol=0):
                raise ValueError("Reference calibration differs between dictionary methods")
            prefix = "sae" if method == "sparse_edit" else "dense"
            for choice_name, choice in (("conditional", action), ("fixed", fixed)):
                add_memory(mappings, prefix + "_" + choice_name + "_calibrated", original,
                           memories[choice], thresholds[choice])
                if choice_name == "conditional":
                    add_memory(mappings, prefix + "_conditional_shared", original, memories[choice], 0.)
            decisions[method] = dict(conditional_action=action, fixed_action=fixed,
                original_action=bundle["original_action"], predicted_advantage=advantages.tolist(),
                predicted_threshold=thresholds.tolist(), candidates=model["candidates"])
            arrays.update({method + "__features": features, method + "__candidate_raw": edited,
                           method + "__candidate_memories": memories, method + "__codes": codes,
                           method + "__advantages": advantages, method + "__thresholds": thresholds})
            if method != "sparse_edit":
                continue
            original_action = bundle["original_action"]
            for suffix in ("shared", "calibrated"):
                add_memory(mappings, "sae0445_" + suffix, original, memories[original_action], thresholds[original_action])
            stream = int.from_bytes(hashlib.sha256((str(seed) + policy.raw_key(original)).encode()).digest()[:8], "little")
            donor = int(bundle["donors"][stream % len(bundle["donors"])])
            add_memory(mappings, "sae_training_context_calibrated", original, memories[donor], thresholds[donor])
            decisions[method]["training_context_action"] = donor
            decisions[method]["training_context_seed"] = stream
            predictor.source_gains = bundle["gains"][action]
            learned = edited[action] - original
            controls = []
            for permutation in range(int(config["random_controls"])):
                delta, gains, control = parent.random_delta(predictor, tokens, mask, learned,
                    seed, policy.raw_key(original), method, permutation)
                random_raw = np.stack([original, original + delta])
                random_memory = predictor.encode(random_raw)
                scaled_strength = float(model["candidates"][action]["strength"] * (control["raw_scale"] or 0.))
                candidates = [model["candidates"][0], dict(k=int(np.count_nonzero(gains)), strength=scaled_strength)]
                random_features = policy.candidate_features(original, random_raw, codes, random_memory,
                    bundle["bank"], bundle["weights"], support, candidates)
                _, _, random_thresholds = policy.choose(model, random_features)
                add_memory(mappings, f"sae_random{permutation}_calibrated", original, random_memory[1], random_thresholds[1])
                arrays[f"random{permutation}__delta"] = delta
                arrays[f"random{permutation}__features"] = random_features
                controls.append(control)
            decisions[method]["random_controls"] = controls
        directory = output / "source_policies" / record["population"] / record["video"] / episode["episode_id"]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "predictions.npz", **arrays)
        diagnostic = dict(population=record["population"], video=record["video"], episode_id=episode["episode_id"],
            status="COMPLETE", decisions=decisions, tokens=int(mask.sum()), frames=int(mask.any(1).sum()),
            inputs={str(path): parent.digest(path) for path in (source / "tokens.npz", record["original"])})
        atomic_write_json(directory / "summary.json", diagnostic)
        diagnostics.append(diagnostic)
        print("SOURCE_POLICY", number + 1, len(records), episode["episode_id"], flush=True)
    atomic_write_json(output / "source_policies.json", dict(sources=diagnostics,
        definition="Prediction uses confirmation-time source support and a training-only reference cohort.",
        context_control="Deterministic donor action from fixed training source contexts; actual source calibration retained."))
    return {name: policy.SourcePolicyPredictor(reference, values, name.endswith("_calibrated"))
            for name, values in mappings.items()}


def evaluate(run, seed, smoke, resume):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    settings = read_json(config["evaluation_settings_file"])
    settings.update(include_interventions=False, protocol=str(run / "protocol.json"),
                    retention_floor=config["application_retention_floor"])
    records = parent.source_records(config, settings, smoke)
    training_records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    output = root / "evaluation" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=resume)
    identity = dict(config_sha256=parent.digest(run / "config.json"), protocol_sha256=parent.digest(run / "protocol.json"),
        seed=seed, smoke=smoke, source_hashes=source_hashes(),
        policies={method: parent.digest(root / "fit" / method / f"seed{seed}" / "summary.json") for method in config["methods"]},
        settings_sha256=parent.digest(Path(config["evaluation_settings_file"])),
        python=sys.version, numpy=np.__version__, torch=str(torch.__version__))
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("Application evaluation inputs changed")
    expected = "REAL_SMOKE_COMPLETE" if smoke else "COMPLETE"
    if (output / "summary.json").exists():
        if not resume or read_json(output / "summary.json")["status"] != expected:
            raise ValueError("Completed application needs matching resume")
        print("REUSE_CONDITIONAL_EVALUATION", seed, flush=True)
        return
    atomic_write_json(output / "identity.json", identity)
    atomic_write_json(output / "evaluation_settings.json", settings)
    started = time.perf_counter()
    reference = parent.reference_api.Representations(Path(config["reference_fit"]))
    models = {method: load_model(config, root, method, seed, training_records) for method in config["methods"]}
    predictors = source_memories(config, records, models, reference, output, seed)
    parent.evaluation.evaluate_phase(settings, output, "development", predictors, None, smoke, resume)
    path = output / "operating_points.json"
    points = read_json(path)["methods"] if path.exists() else parent.evaluation.calibrate(
        output, list(predictors) + ["reference_supcon"], settings["retention_floor"])
    parent.evaluation.evaluate_phase(settings, output, "extension", predictors, points, smoke, resume)
    if source_hashes() != identity["source_hashes"]:
        raise ValueError("Evaluation source changed during execution")
    result = dict(status=expected, seed=seed, elapsed_seconds=time.perf_counter() - started,
        operating_points=points, conditions=list(predictors) + ["reference_supcon"],
        extension_population="Previously examined extension", source_conditions="Actual available support only")
    atomic_write_json(output / "summary.json", result)
    atomic_write_json(output / "progress.json", dict(completed=2 if smoke else 10, total=2 if smoke else 10, phase="COMPLETE"))
    print("CONDITIONAL_EVALUATION_COMPLETE", seed, result["elapsed_seconds"], flush=True)


def summarize(run, smoke, resume):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    rows, fitting, inputs = [], [], {}
    for seed in seeds:
        folder = root / "evaluation" / f"seed{seed}"
        summary = read_json(folder / "summary.json")
        inputs[str(folder / "summary.json")] = parent.digest(folder / "summary.json")
        for method, point in summary["operating_points"].items():
            for phase in ("development", "extension"):
                path = folder / phase / (method + "__score_procedure_curve.npz")
                inputs[str(path)] = parent.digest(path)
                with np.load(path, allow_pickle=False) as curve:
                    column = int(np.flatnonzero(curve["threshold"] == point["threshold"])[0]) if phase == "development" else 0
                    for i, video in enumerate(curve["videos"].tolist()):
                        metrics = {name: float(curve[key][i, column]) for name, key in parent.METRICS.items()}
                        rows.append(dict(seed=seed, method=method, population=phase, video=video, threshold=point["threshold"],
                            **{k: v if np.isfinite(v) else None for k, v in metrics.items()}))
        for method in config["methods"]:
            for fold in ([0] if smoke else range(3)):
                path = root / training.relative_job(method, seed, fold) / "summary.json"
                result = read_json(path)
                inputs[str(path)] = parent.digest(path)
                fitting.append(dict(seed=seed, method=method, fold=fold, **result["results"]["held"]))
    aggregates = []
    for population, method in sorted({(r["population"], r["method"]) for r in rows}):
        selected = [r for r in rows if (r["population"], r["method"]) == (population, method)]
        by_seed = {seed: {metric: parent.finite_mean([r[metric] for r in selected if r["seed"] == seed])
                          for metric in parent.METRICS} for seed in seeds}
        aggregates.append(dict(population=population, method=method, seed_results=by_seed,
            **{metric: parent.finite_mean([r[metric] for r in by_seed.values()]) for metric in parent.METRICS}))
    if (root / "summary.json").exists():
        if not resume or read_json(root / "summary.json")["inputs"] != inputs:
            raise ValueError("Summary inputs changed")
        return
    atomic_write_json(root / "summary.json", dict(status="REAL_SMOKE_COMPLETE" if smoke else "COMPLETE", inputs=inputs,
        seeds=seeds, procedure_rows=rows, aggregates=aggregates, held_policy_results=fitting,
        independent_unit="procedure", extension_population="Previously examined extension"))
    lines = ["# Confirmation-conditioned component policy", "",
        "Every method has the same development protection requirement and first-prompt constraint. Extension uses the development offset unchanged.", "",
        "| Population | Method | Repeat removal | Other retention | First prompt |",
        "|---|---|---:|---:|---:|"]
    for row in aggregates:
        values = ["NA" if row[name] is None else f"{100 * row[name]:.4f}%" for name in parent.METRICS]
        lines.append(f"| {row['population']} | {row['method']} | " + " | ".join(values) + " |")
    lines += ["", "All source-action predictions, assigned controls, seed outcomes and held procedure proxy results are saved. Both application populations have been examined previously.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(root / "summary_progress.json", dict(completed=1, total=1, phase="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("evaluate", "summary"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "evaluate":
        evaluate(args.run, args.seed, args.smoke, args.resume)
    else:
        summarize(args.run, args.smoke, args.resume)
