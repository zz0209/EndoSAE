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
import train_decoder_identity_tail as training
from src.checkpoint_io import atomic_write_json, read_json
from src.template_component_adaptation import TemplatePredictor, raw_key
from src.token_memory_edit import TokenDictionary, TokenMemoryPredictor

parent = previous.parent
LABELS = dict(reference_supcon="Original", sae0445="Global SAE intervention",
    sae_fixed="SAE | fixed decoder", sae_refined="SAE | trained decoder",
    dense_fixed="Ordinary | fixed decoder", dense_refined="Ordinary | trained decoder",
    sae_random0="Norm-matched random 1", sae_random1="Norm-matched random 2", sae_random2="Norm-matched random 3")


def hashes():
    result = dict(previous.source_hashes(), **training.hashes())
    result[str(Path(__file__).relative_to(ROOT))] = parent.digest(__file__)
    return result


def bundles(config, root, seed):
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    models, identities = {}, {}
    for method in config["methods"]:
        base = previous.bundle(config, records, method, seed, "full_training")
        prefix = "sae" if method == "sparse_edit" else "dense"
        if method == "sparse_edit":
            base["predictor"].source_gains = base["old_gains"]
            models["sae0445"] = base["predictor"]
            identities["sae0445"] = base["identity"]
        for arm in config["arms"]:
            folder = root / "fit" / method / f"seed{seed}" / "full" / arm
            summary, identity = read_json(folder / "summary.json"), read_json(folder / "identity.json")
            if summary["status"] != "COMPLETE" or summary["identity_sha256"] != training.parent.shared.shared.json_digest(identity):
                raise ValueError("Decoder fit identity differs")
            for name, signature in summary["artifacts"].items():
                if parent.digest(folder / name) != signature:
                    raise ValueError("Decoder fit asset changed")
            model = TokenDictionary(base["assets"]["spec"])
            with np.load(folder / "model.npz", allow_pickle=False) as data:
                model.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files}, strict=True)
            for name, value in model.state_dict().items():
                if name != "decoder.weight" and not torch.equal(value, base["assets"]["model"].state_dict()[name]):
                    raise ValueError("Encoder or bias differs from the parent dictionary")
            with np.load(folder / "gains.npz", allow_pickle=False) as data:
                gains = data["gains"].copy()
            label = prefix + ("_fixed" if arm == "fixed_decoder" else "_refined")
            models[label] = TokenMemoryPredictor(model, base["assets"]["mean"], base["assets"]["scale"],
                                                 gains, base["assets"]["reference"])
            identities[label] = dict(training_summary_sha256=parent.digest(folder / "summary.json"), assets=base["identity"])
    return models, identities


def source_memories(records, models, reference, output, seed, resume, stop_after_sources):
    mappings, diagnostics = {}, []
    for number, record in enumerate(records):
        episode = record["episode"]["episode_id"]
        if not record["available"]:
            diagnostics.append(dict(population=record["population"], video=record["video"], episode=episode, status="UNAVAILABLE"))
            continue
        folder = output / "sources" / record["population"] / record["video"] / episode
        folder.mkdir(parents=True, exist_ok=True)
        summary_path, array_path = folder / "summary.json", folder / "memories.npz"
        inputs = {str(p): parent.digest(p) for p in (record["original"], record["source"] / "tokens.npz")}
        if summary_path.exists():
            diagnostic = read_json(summary_path)
            if not resume or diagnostic["inputs"] != inputs or parent.digest(array_path) != diagnostic["array_sha256"]:
                raise ValueError("Decoder source checkpoint changed")
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
        else:
            with np.load(record["original"], allow_pickle=False) as saved:
                original = saved["raw_mean"].reshape(-1).copy()
            with np.load(record["source"] / "tokens.npz", allow_pickle=False) as saved:
                tokens, mask = saved["tokens"].copy(), saved["mask"].astype(bool)
                if not np.array_equal(original, saved["original_raw"].reshape(-1)):
                    raise ValueError("Original source changed")
            arrays, decisions = dict(original_raw=original), {}
            for label, predictor in models.items():
                codes = predictor.pooled_codes(tokens, mask)
                edited = predictor.edit_raw(original, codes).reshape(-1)
                memory = predictor.encode(edited)[0]
                arrays["memory__" + label] = memory
                arrays[label + "__codes"], arrays[label + "__edited_raw"] = codes, edited
                decisions[label] = dict(active_components=int(np.count_nonzero(codes)),
                    effective_gains=int(np.count_nonzero(codes * predictor.source_gains)),
                    edit_norm=float(np.linalg.norm(edited - original)))
                if label == "sae_refined":
                    for permutation in range(3):
                        delta, gains, control = parent.random_delta(predictor, tokens, mask, edited - original,
                                                                   seed, episode, "sparse_edit", permutation)
                        if control["status"] == "UNDEFINED_ZERO_DIRECTION":
                            raise ValueError("Norm-matched random direction is undefined")
                        arrays[f"memory__sae_random{permutation}"] = predictor.encode(original + delta)[0]
                        arrays[f"random{permutation}__delta"], arrays[f"random{permutation}__gains"] = delta, gains
                        decisions[f"sae_random{permutation}"] = control
            if not np.array_equal(arrays["sae_fixed__codes"], arrays["sae_refined__codes"]):
                raise ValueError("Decoder refinement changed sparse token codes")
            if not np.array_equal(arrays["dense_fixed__codes"], arrays["dense_refined__codes"]):
                raise ValueError("Decoder refinement changed ordinary token codes")
            np.savez_compressed(array_path, **arrays)
            diagnostic = dict(population=record["population"], video=record["video"], episode=episode,
                status="COMPLETE", decisions=decisions, inputs=inputs, array_sha256=parent.digest(array_path))
            atomic_write_json(summary_path, diagnostic)
        original = arrays["original_raw"]
        key = raw_key(original)
        for name, memory in arrays.items():
            if name.startswith("memory__"):
                mapping = mappings.setdefault(name.removeprefix("memory__"), {})
                if key in mapping and not np.array_equal(mapping[key][1], memory):
                    raise ValueError("Duplicate source has different memory")
                mapping[key] = (original, memory)
        diagnostics.append(diagnostic)
        atomic_write_json(output / "source_progress.json", dict(completed=number + 1, total=len(records), phase="source memories"))
        print("DECODER_SOURCE", number + 1, len(records), episode, flush=True)
        if stop_after_sources is not None and number + 1 >= stop_after_sources:
            raise SystemExit(75)
    atomic_write_json(output / "source_diagnostics.json", dict(sources=diagnostics))
    return {name: TemplatePredictor(reference, values) for name, values in mappings.items()}


def evaluate(run, seed, smoke, resume, stop_after_sources, output_override=None):
    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    settings = read_json(config["evaluation_settings_file"])
    settings.update(include_interventions=False, protocol=str(run / "protocol.json"), retention_floor=config["application_retention_floor"])
    records = parent.source_records(config, settings, smoke)
    models, identities = bundles(config, root, seed)
    output = output_override or root / "evaluation" / f"seed{seed}"
    source_hashes = hashes()
    previous.make_identity(output, dict(config=parent.digest(run / "config.json"), protocol=parent.digest(run / "protocol.json"),
        source_hashes=source_hashes, models=identities, seed=seed, smoke=smoke,
        settings_sha256=parent.digest(Path(config["evaluation_settings_file"])),
        python=sys.version, torch=str(torch.__version__), numpy=np.__version__), resume)
    if (output / "summary.json").exists():
        return
    started = time.perf_counter()
    atomic_write_json(output / "evaluation_settings.json", settings)
    reference = parent.reference_api.Representations(Path(config["reference_fit"]))
    predictors = source_memories(records, models, reference, output, seed, resume, stop_after_sources)
    parent.evaluation.evaluate_phase(settings, output, "development", predictors, None, smoke, resume)
    points_path = output / "operating_points.json"
    points = read_json(points_path)["methods"] if points_path.exists() else parent.evaluation.calibrate(
        output, list(predictors) + ["reference_supcon"], settings["retention_floor"])
    parent.evaluation.evaluate_phase(settings, output, "extension", predictors, points, smoke, resume)
    if hashes() != source_hashes:
        raise ValueError("Decoder evaluator changed during execution")
    atomic_write_json(output / "summary.json", dict(status="REAL_SMOKE_COMPLETE" if smoke else "COMPLETE", seed=seed,
        operating_points=points, elapsed_seconds=time.perf_counter() - started, conditions=list(predictors) + ["reference_supcon"]))
    atomic_write_json(output / "progress.json", dict(completed=2 if smoke else 10, total=2 if smoke else 10, phase="COMPLETE"))


def summarize(run, smoke, resume):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    previous.summarize(run, smoke, resume)
    root = run / "smoke" if smoke else run
    summary = read_json(root / "summary.json")
    config = read_json(run / "config.json")
    title = ("Shared-gradient decoder training" if config.get("gradient_rule") == "agreement"
             else "Decoder directions and identity-tail training")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(summary["procedure_rows"])
    table.to_csv(output / "procedure_outcomes.csv", index=False)
    fitted = read_json(root / "training_summary.json")
    rows = [dict(**{key: r[key] for key in ("method", "seed", "fold", "arm", "decoder_change_norm", "decoder_cosine_mean")},
                 **r["held"]) for r in fitted["outputs"]]
    pd.DataFrame(rows).to_csv(output / "held_jobs.csv", index=False)
    pd.DataFrame([dict(method=r["method"], seed=r["seed"], fold=r["fold"], arm=r["arm"], video_id=video, **values)
        for r in fitted["outputs"] for video, values in r["procedure_results"].items()]).to_csv(output / "held_procedures.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for i, population in enumerate(("development", "extension")):
        for j, metric in enumerate(("removal", "retention")):
            ax = axes[i, j]
            subset = table[table.population == population]
            for number, (method, label) in enumerate(LABELS.items()):
                values = subset[subset.method == method].groupby("seed")[metric].mean()
                for seed_index, (seed, value) in enumerate(values.items()):
                    ax.scatter(100 * value, number, color=["#0072B2", "#D55E00", "#009E73"][seed_index],
                               label=str(seed) if number == i == j == 0 else None)
                ax.scatter(100 * values.mean(), number, marker="|", color="black")
            ax.set_yticks(range(len(LABELS)), list(LABELS.values()), fontsize=9)
            ax.set_xlabel("Percent")
            ax.set_title(("Development" if i == 0 else "Examined extension") + " | " +
                         ("Repeat removal" if j == 0 else "Other-prompt retention"))
            ax.grid(axis="x", alpha=.25)
            ax.spines[["right", "top"]].set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, bbox_to_anchor=(.65, .96))
    fig.suptitle(title + (" | real smoke only" if smoke else ""), fontsize=16)
    fig.text(.02, .025, "All seeds; black marks show means. Equal procedure weighting; panel scales differ.\n"
        "Common development protection requirement. Extension uses frozen thresholds; both populations were previously examined.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .93))
    for extension in ("png", "pdf"):
        fig.savefig(output / ("application_outcomes." + extension), dpi=180)
    plt.close(fig)
    lines = ["# " + title, "", "Both application populations have been examined previously.", "",
        "| Population | Method | Repeat removal | Other retention | First prompt |", "|---|---|---:|---:|---:|"]
    for row in summary["aggregates"]:
        lines.append(f"| {row['population']} | {row['method']} | " + " | ".join(f"{100 * row[m]:.4f}%" for m in parent.METRICS) + " |")
    if config.get("gradient_rule") == "agreement":
        mean_run = Path(config["mean_gradient_run"]) / ("smoke" if smoke else "")
        mean = pd.DataFrame(read_json(mean_run / "summary.json")["procedure_rows"])
        paired = table.merge(mean, on=["population", "method", "seed", "video"],
                             suffixes=("", "_mean"), validate="one_to_one")
        if len(paired) != len(table):
            raise ValueError("Mean-gradient reference is incomplete")
        for metric in parent.METRICS:
            paired[metric + "_change"] = paired[metric] - paired[metric + "_mean"]
        original = paired[paired.method == "reference_supcon"]
        if any((original[m + "_change"] != 0).any() for m in parent.METRICS):
            raise ValueError("Original reference differs between gradient rules")
        paired.to_csv(output / "agreement_minus_mean_procedures.csv", index=False)
        contrast = paired.groupby(["population", "method", "seed"])[[m + "_change" for m in parent.METRICS]].mean()
        contrast.to_csv(output / "agreement_minus_mean_seeds.csv")
        lines.extend(["", "## Paired change from mean-gradient training", "",
            "Percentage-point changes; procedures then seeds receive equal weight.", "",
            "| Population | Method | Repeat removal | Other retention | First prompt |", "|---|---|---:|---:|---:|"])
        for (population, method), row in contrast.groupby(["population", "method"]).mean().iterrows():
            lines.append(f"| {population} | {method} | " + " | ".join(f"{100 * row[m + '_change']:+.4f}" for m in parent.METRICS) + " |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(output / "manifest.json", dict(status="COMPLETE", summary_sha256=parent.digest(root / "summary.json"),
        training_sha256=parent.digest(root / "training_summary.json"), source_sha256=parent.digest(__file__),
        files={p.name: parent.digest(p) for p in output.iterdir() if p.suffix in (".png", ".pdf", ".csv")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("evaluate", "summary"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase == "evaluate":
        evaluate(args.run, args.seed, args.smoke, args.resume, args.stop_after_sources, args.output)
    else:
        summarize(args.run, args.smoke, args.resume)
