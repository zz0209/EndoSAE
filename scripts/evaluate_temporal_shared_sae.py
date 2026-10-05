import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_acknowledgement_sae as application
from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, read_json
from src.temporal_shared_sae import METHODS, load_predictor


METRICS = {
    "removal": "source_suppression__procedure_values",
    "retention": "all_other__baseline_qualified_retention__procedure_values",
    "first_prompt": "all_other__first_baseline_frame_retention__procedure_values",
}


def evaluate(run, seed, smoke, resume, output_override=None):
    config = read_json(run / "config.json")
    if seed not in config["seeds"]:
        raise ValueError("Unspecified seed")
    torch.set_num_threads(1)
    root = run / "smoke" if smoke else run
    output = output_override if output_override else root / "evaluation" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=resume)
    settings = read_json(config["evaluation_settings_file"])
    settings.update(include_interventions=False, retention_floor=config["application_retention_floor"])
    models = {method: root / "fit" / method / f"seed{seed}" for method in METHODS}
    paths = [run / "config.json", run / "protocol.json", Path(__file__), ROOT / "src/temporal_shared_sae.py",
             Path(application.__file__), Path(application.shared.__file__), Path(application.shared.memory.__file__),
             ROOT / "src/acknowledgement_sae.py", Path(config["evaluation_settings_file"])]
    paths += [folder / name for folder in models.values() for name in ("model.npz", "normalization.npz", "model_config.json")]
    identity = dict(seed=seed, smoke=smoke, files={str(p): file_sha256(p) for p in paths})
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("Application inputs or source files changed")
    if (output / "summary.json").exists():
        if not resume or read_json(output / "summary.json")["status"] != "COMPLETE":
            raise ValueError("Cannot reuse incomplete evaluation")
        return
    atomic_write_json(output / "identity.json", identity)
    atomic_write_json(output / "evaluation_settings.json", settings)
    predictors = {method: load_predictor(folder) for method, folder in models.items()}
    started = time.perf_counter()
    application.evaluate_phase(settings, output, "development", predictors, None, smoke, resume)
    points_path = output / "operating_points.json"
    points = read_json(points_path)["methods"] if points_path.exists() else application.calibrate(
        output, list(predictors) + ["reference_supcon"], config["application_retention_floor"])
    application.evaluate_phase(settings, output, "extension", predictors, points, smoke, resume)
    if identity["files"] != {str(p): file_sha256(p) for p in paths}:
        raise ValueError("Application source or model changed during evaluation")
    atomic_write_json(output / "summary.json", dict(status="COMPLETE", smoke=smoke, seed=seed,
        operating_points=points, seconds=time.perf_counter() - started,
        population="Previously examined development and extension procedures; no independent confirmation."))
    atomic_write_json(output / "progress.json", dict(completed=2 if smoke else 10, total=2 if smoke else 10, phase="COMPLETE"))


def summarize(run, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    config = read_json(run / "config.json")
    root = run / "smoke" if smoke else run
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    methods = ["reference_supcon"] + list(METHODS)
    rows, inputs = [], {}
    for seed in seeds:
        output = root / "evaluation" / f"seed{seed}"
        complete = read_json(output / "summary.json")
        if complete["status"] != "COMPLETE":
            raise ValueError("Incomplete application evaluation")
        inputs[str(output / "summary.json")] = file_sha256(output / "summary.json")
        for population in ("development", "extension"):
            for method in methods:
                path = output / population / (method + "__score_procedure_curve.npz")
                inputs[str(path)] = file_sha256(path)
                with np.load(path, allow_pickle=False) as curve:
                    threshold = complete["operating_points"][method]["threshold"]
                    index = int(np.argmin(np.abs(curve["threshold"] - threshold))) if population == "development" else 0
                    if population == "development" and not np.isclose(curve["threshold"][index], threshold, atol=1e-10):
                        raise ValueError("Selected threshold missing from saved curve")
                    for number, video in enumerate(curve["videos"].tolist()):
                        rows.append(dict(seed=seed, population=population, method=method, video=video,
                            **{name: float(curve[key][number, index]) for name, key in METRICS.items()}))
    table = pd.DataFrame(rows)
    per_seed = table.groupby(["population", "method", "seed"])[list(METRICS)].mean()
    aggregates = per_seed.groupby(["population", "method"]).mean().reset_index().to_dict("records")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "application_procedures.csv", index=False)
    per_seed.to_csv(output / "application_seeds.csv")
    mechanism, interventions = [], []
    fitted = read_json(root / "training_summary.json")
    inputs[str(root / "training_summary.json")] = file_sha256(root / "training_summary.json")
    for item in fitted["outputs"]:
        folder = Path(item["directory"])
        population = "examined_validation" if item["fold"] == "full" else "training_held"
        for video, values in item["summary"]["final"]["mechanism_by_video"].items():
            mechanism.append(dict(method=item["method"], seed=item["seed"], fold=item["fold"], video=video,
                                  population=population, **values))
        pairs = read_json(root / "descriptor_records.json")
        with np.load(folder / "held_final_pairs.npz", allow_pickle=False) as saved:
            videos = np.asarray([pairs[i]["video_id"] for i in saved["source"]])
            cross = saved["same_identity"] & ~saved["same_interval"]
            negative = ~saved["same_identity"]
            for video in sorted(set(videos)):
                for intervention in ["selected"] + [f"random{i}" for i in range(config["random_controls"])]:
                    finite = np.isfinite(saved[intervention])
                    same_mask = (videos == video) & cross & finite
                    other_mask = (videos == video) & negative & finite
                    change = saved["scores"] - saved[intervention]
                    positive_drop = float(change[same_mask].mean()) if same_mask.any() else None
                    negative_drop = float(change[other_mask].mean()) if other_mask.any() else None
                    interventions.append(dict(method=item["method"], seed=item["seed"], fold=item["fold"], video=video,
                        population=population, intervention=intervention, same_pairs=int(same_mask.sum()),
                        other_pairs=int(other_mask.sum()), same_drop=positive_drop, other_drop=negative_drop,
                        selective_drop=positive_drop - negative_drop if positive_drop is not None and negative_drop is not None else None))
    mechanisms = pd.DataFrame(mechanism)
    effects = pd.DataFrame(interventions)
    mechanisms.to_csv(output / "component_procedures.csv", index=False)
    effects.to_csv(output / "intervention_procedures.csv", index=False)
    contrast_rows = []
    for left, right, label in [("sparse_shared", "dense_shared", "sparse_shared_minus_dense_shared"),
                               ("sparse_shared", "sparse_self", "shared_minus_self_sparse"),
                               ("dense_shared", "dense_self", "shared_minus_self_dense"),
                               ("sparse_shared", "reference_supcon", "sparse_shared_minus_reference")]:
        merged = table[table.method == left].merge(table[table.method == right], on=["population", "seed", "video"],
                                                   suffixes=("_left", "_right"), validate="one_to_one")
        for row in merged.to_dict("records"):
            contrast_rows.append(dict(contrast=label, population=row["population"], seed=row["seed"], video=row["video"],
                                      **{key: row[key + "_left"] - row[key + "_right"] for key in METRICS}))
    pd.DataFrame(contrast_rows).to_csv(output / "paired_application_differences.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for i, population in enumerate(("development", "extension")):
        for j, metric in enumerate(("removal", "retention")):
            ax = axes[i, j]
            for number, method in enumerate(methods):
                values = per_seed.loc[(population, method)][metric]
                for k, (_, value) in enumerate(values.items()):
                    ax.scatter(value * 100, number, color=colors[k], label=str(seeds[k]) if i == j == number == 0 else None)
                ax.scatter(values.mean() * 100, number, color="black", marker="|")
            ax.set_yticks(range(len(methods)), methods)
            ax.set_title(("Development" if i == 0 else "Examined extension") + " | " + metric)
            ax.set_xlabel("Percent")
            ax.grid(axis="x", alpha=.2)
            ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle("Cross-observation shared components" + (" | real smoke" if smoke else ""))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.65, .94), ncol=len(seeds), frameon=False)
    fig.text(.02, .025, "Equal procedure weighting. All seeds shown; black marks are means. Panels use separate scales.\n"
             "Common development protection requirement; fixed extension thresholds. Both populations were previously examined.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .92))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("application_outcomes." + suffix), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, population in enumerate(("training_held", "examined_validation")):
        subset = mechanisms[mechanisms.population == population].copy()
        subset["identity_gap"] = subset.shared_cosine_same - subset.shared_cosine_other
        for number, method in enumerate(METHODS):
            values = subset[subset.method == method].groupby("seed").identity_gap.mean()
            for k, (_, value) in enumerate(values.items()):
                axes[i].scatter(value, number, color=colors[k])
        axes[i].set_yticks(range(len(METHODS)), METHODS)
        axes[i].set_title(population.replace("_", " "))
        axes[i].set_xlabel("Shared-code cosine: same identity minus other identity")
        axes[i].axvline(0, color="gray", linewidth=.8)
        axes[i].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cross-interval component discrimination" + (" | real smoke" if smoke else ""))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("component_discrimination." + suffix), dpi=170)
    plt.close(fig)
    atomic_write_json(root / "summary.json", dict(status="COMPLETE", smoke=smoke, aggregates=aggregates,
        procedure_rows=rows, paired_differences=contrast_rows, input_sha256=inputs, source_sha256=file_sha256(__file__)))
    lines = ["# Cross-observation shared components", "", "Both application populations were previously examined.", "",
             "| Population | Method | Repeat removal | Other retention | First prompt |", "|---|---|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['population']} | {row['method']} | " + " | ".join(f"{row[key] * 100:.4f}%" for key in METRICS) + " |")
    lines += ["", "Component stability and matched latent-norm interventions are saved per procedure in figures/. "
              "Code sharing and swapping are established objectives. These outputs assess this configuration, with no guaranteed semantic separation."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(output / "manifest.json", dict(status="COMPLETE", input_sha256=inputs,
        files={p.name: file_sha256(p) for p in output.iterdir() if p.suffix in (".png", ".pdf", ".csv")}))
    atomic_write_json(root / "summary_progress.json", dict(completed=1, total=1, phase="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("evaluate", "summary"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "evaluate":
        evaluate(args.run.resolve(), args.seed, args.smoke, args.resume, args.output.resolve() if args.output else None)
    else:
        summarize(args.run.resolve(), args.smoke)
