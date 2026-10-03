import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def analyze(run):
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError("Completed analysis is immutable")
    held = read_json(run / "held_summary.json")
    summary = read_json(run / "summary.json")
    if held["status"] != "COMPLETE" or summary["status"] != "COMPLETE":
        raise ValueError("Both full result sets are required")
    jobs, units, sources = [], [], []
    for job in held["jobs"]:
        identity = {name: job[name] for name in ("method", "seed", "fold")}
        jobs.append(dict(**identity, **job["means"]))
        for video, values in job["per_procedure"].items():
            units.append(dict(**identity, video=video, **values))
        for row in job["sources"]:
            sources.append(dict(**identity, **{k: v for k, v in row.items() if k not in ("positives", "negatives", "svm")}))
    job_table, unit_table, source_table = map(pd.DataFrame, (jobs, units, sources))
    for table in (job_table, unit_table, source_table):
        table["utility_change"] = table.adapted_utility - table.zero_utility
        table["recall_change"] = table.adapted_recall - table.zero_recall
        table["svm_recall_change"] = table.svm_recall - table.zero_recall
    app = pd.DataFrame(summary["procedure_rows"])
    baseline = app[app.method == "reference_supcon"].drop(columns="method")
    contrasted = app.merge(baseline, on=["population", "seed", "video"], suffixes=("", "_original"), validate="many_to_one")
    for metric in ("removal", "retention", "first_prompt"):
        contrasted[metric + "_change"] = contrasted[metric] - contrasted[metric + "_original"]
    for name, table in (("held_jobs", job_table), ("held_procedures", unit_table),
                        ("held_sources", source_table), ("application_contrasts", contrasted)):
        table.to_csv(output / (name + ".csv"), index=False)
    effects = pd.read_csv(run / "figures/source_component_effects.csv")
    effect_summary = effects.groupby(["method", "population"]).agg(sources=("choice", "size"),
        mean_components=("component_count", "mean"), mean_gain=("gain", "mean"),
        zero_fraction=("choice", lambda x: float((x == 0).mean())))
    effect_summary.to_csv(output / "application_actions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = dict(sparse_edit="#0072B2", dense_edit="#D55E00")
    for number, (method, color) in enumerate(colors.items()):
        selected = unit_table[unit_table.method == method].groupby("video").mean(numeric_only=True)
        axes[0].scatter(np.full(len(selected), number), 100 * selected.recall_change, color=color, alpha=.8)
        axes[0].scatter(number, 100 * selected.recall_change.mean(), marker="_", s=250, color="black")
        selected_sources = source_table[source_table.method == method]
        axes[1].scatter(selected_sources.available_objective, selected_sources.utility_change,
            color=color, alpha=.3, s=10, label=method)
    axes[0].set_xticks([0, 1], ["SAE", "Dense"])
    axes[0].set(ylabel="Recall change (percentage points)", title="Held procedures, averaged over seeds")
    axes[1].set(xlabel="Available-context objective gain", ylabel="Future identity utility change", title="Local proxy versus future identity")
    axes[1].legend(fontsize=8)
    labels = dict(sae_adapted="SAE", dense_adapted="Dense", template_svm="SVM")
    for index, (method, label) in enumerate(labels.items()):
        for population, marker in (("development", "o"), ("extension", "s")):
            points = contrasted[(contrasted.method == method) & (contrasted.population == population)].groupby("video").mean(numeric_only=True)
            axes[2].scatter(100 * points.retention_change, 100 * points.removal_change,
                marker=marker, color=["#0072B2", "#D55E00", "#009E73"][index], label=label + " / " + population)
    axes[2].set(xlabel="Other-prompt retention change (pp)", ylabel="Repeat removal change (pp)", title="Application trade-off by procedure")
    axes[2].legend(fontsize=6)
    for ax in axes:
        ax.axhline(0, color="grey", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].axvline(0, color="grey", linewidth=.8)
    fig.text(.02, .02, "Training-held and application procedures are separate. All seeds and eligible procedures are retained. Application data were previously examined.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("mechanism_transfer." + suffix), dpi=180)
    plt.close(fig)
    atomic_write_json(output / "complete.json", dict(status="COMPLETE", source_sha256=file_sha256(__file__),
        inputs={str(run / name): file_sha256(run / name) for name in ("held_summary.json", "summary.json")},
        jobs=len(jobs), held_procedures=len(set(unit_table.video)), held_sources=len(sources),
        application_rows=len(app)))
    print("HELD_SEED_FOLD_MEANS\n", job_table.to_string(index=False))
    print("HELD_PROCEDURE_MEANS\n", unit_table.groupby(["method", "video"])[["utility_change", "recall_change", "svm_recall_change"]].mean().to_string())
    print("APPLICATION_PROCEDURE_CHANGES\n", contrasted.groupby(["population", "method", "video"])[["removal_change", "retention_change", "first_prompt_change"]].mean().to_string())
    print("SOURCE_ACTIONS\n", effect_summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
