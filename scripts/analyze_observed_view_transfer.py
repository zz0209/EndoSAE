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
    output = run / "analysis_v2"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError("Completed analysis is immutable")
    summary = read_json(run / "summary.json")
    held = read_json(run / "held_summary.json")
    inventory = read_json(run / "source_support_inventory.json")["sources"]
    if summary["status"] != "COMPLETE" or held["status"] != "COMPLETE":
        raise ValueError("Full results are required")
    job_rows, held_rows = [], []
    for job in held["jobs"]:
        identity = {key: job[key] for key in ("method", "seed", "fold")}
        job_rows.append(dict(**identity, **job["means"]))
        held_rows.extend(dict(**identity, **row) for row in job["sources"])
    jobs, held_sources = pd.DataFrame(job_rows), pd.DataFrame(held_rows)
    for table in (jobs, held_sources):
        for prefix in ("adapted", "self", "mean"):
            for metric in ("recall", "utility"):
                table[prefix + "_" + metric + "_change"] = table[prefix + "_" + metric] - table["zero_" + metric]
    jobs.to_csv(output / "held_jobs.csv", index=False)
    held_sources.to_csv(output / "held_sources.csv", index=False)
    procedure = pd.DataFrame(summary["procedure_rows"])
    base = procedure[procedure.method == "reference_supcon"].drop(columns="method")
    contrasts = procedure.merge(base, on=["population", "seed", "video"], suffixes=("", "_original"), validate="many_to_one")
    for metric in ("removal", "retention", "first_prompt"):
        contrasts[metric + "_change"] = contrasts[metric] - contrasts[metric + "_original"]
    contrasts.to_csv(output / "application_procedures.csv", index=False)
    rows, checks = [], []
    for seed in summary["seeds"]:
        root = run / "evaluation" / f"seed{seed}"
        points = read_json(root / "operating_points.json")["methods"]
        for source in inventory:
            population, video, episode = (source[k] for k in ("population", "video", "episode"))
            count = int(np.count_nonzero(source["frame_counts"]))
            folder = root / population / video / "sources" / episode
            memories = root / "sources" / population / video / episode / "adaptation.npz"
            with np.load(memories, allow_pickle=False) as data:
                positives = data["sparse_edit__positive_memory"].astype(float)
                minimum = float(np.min(positives @ positives.T))
                distance = float(np.max(np.abs(data["memory__sae_views"] - data["memory__sae_self"])))
                decisions = read_json(memories.parent / "summary.json")["decisions"]
            checks.append(dict(seed=seed, population=population, video=video, episode=episode,
                frames=count, minimum_view_cosine=minimum, self_view_memory_max_difference=distance,
                **{method + "_" + k: value for method, d in decisions.items()
                    for k, value in d.items() if k in ("choice", "components", "objective", "positive_change", "negative_change")}))
            for method, point in points.items():
                with np.load(folder / (method + "__score_curve.npz"), allow_pickle=False) as curve:
                    column = int(np.searchsorted(curve["threshold"], point["threshold"], side="right") - 1) if population == "development" else 0
                    if column < 0:
                        raise ValueError("Operating point is outside the source curve")
                    other = curve["all_other_lesion_columns"]
                    retained = curve["baseline_qualified_retention"][column, other]
                    defined = retained[np.isfinite(retained)]
                    retention = float(defined.mean()) if len(defined) else np.nan
                    rows.append(dict(seed=seed, population=population, video=video, episode=episode, method=method,
                        frames=count, support="multi" if count > 1 else "single", threshold=point["threshold"],
                        removal=float(curve["acknowledged_suppression_fraction"][column]), retention=retention))
    source_table, mechanism = pd.DataFrame(rows), pd.DataFrame(checks)
    recovered = source_table.groupby(["population", "method", "seed", "video"])[["removal", "retention"]].mean().reset_index()
    verification = recovered.merge(procedure, on=["population", "method", "seed", "video"], suffixes=("_source", "_procedure"), validate="one_to_one")
    errors = {metric: float(np.nanmax(np.abs(verification[metric + "_source"] - verification[metric + "_procedure"]))) for metric in ("removal", "retention")}
    if len(verification) != len(procedure) or max(errors.values()) > 1e-12:
        raise ValueError("Source aggregation differs from the formal procedure results")
    source_table.to_csv(output / "application_sources.csv", index=False)
    mechanism.to_csv(output / "source_mechanism.csv", index=False)
    grouped = source_table.groupby(["population", "support", "method", "seed", "video"])[["removal", "retention"]].mean()
    grouped = grouped.groupby(["population", "support", "method", "seed"]).mean()
    grouped.to_csv(output / "support_group_seed_means.csv")
    grouped_mean = grouped.groupby(["population", "support", "method"]).mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    names = ["reference_supcon", "sae_self", "sae_views", "dense_views", "view_mean", "view_svm"]
    labels = ["Original", "Self SAE", "View SAE", "Dense", "Mean", "SVM"]
    for i, population in enumerate(("development", "extension")):
        for support, marker in (("single", "o"), ("multi", "s")):
            values = grouped_mean[(grouped_mean.population == population) & (grouped_mean.support == support)].set_index("method")
            axes[i].plot(range(len(names)), 100 * values.loc[names, "removal"], marker=marker, label=support + " frame")
        axes[i].set_xticks(range(len(names)), labels, rotation=30, ha="right")
        axes[i].set(title=population, ylabel="Repeat removal (%)")
        axes[i].legend()
    per_source = mechanism.groupby(["population", "episode"]).mean(numeric_only=True)
    multi = per_source[per_source.frames > 1]
    axes[2].scatter(multi.minimum_view_cosine, multi.sparse_edit_positive_change,
        c=["#0072B2" if p == "development" else "#D55E00" for p, _ in multi.index])
    axes[2].axhline(0, color="grey", linewidth=.8)
    axes[2].set(xlabel="Minimum pairwise cosine of observed frames", ylabel="Change in protected-view similarity", title="Nine multiframe sources, all seeds averaged")
    fig.text(.015, .015, "Previously examined data. Same full-population development calibration; within-support means weight procedures equally. No support-group threshold fitting.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("support_mechanism." + suffix), dpi=180)
    plt.close(fig)
    atomic_write_json(output / "complete.json", dict(status="COMPLETE", source_sha256=file_sha256(__file__),
        inputs={name: file_sha256(run / name) for name in ("summary.json", "held_summary.json", "source_support_inventory.json")},
        source_rows=len(rows), mechanism_rows=len(checks), jobs=len(job_rows), procedure_reconstruction_errors=errors))
    print("HELD_JOBS", jobs.to_string(index=False))
    print("PROCEDURES", contrasts.groupby(["population", "method", "video"])[["removal_change", "retention_change"]].mean().to_string())
    print("SUPPORT", grouped_mean.to_string(index=False))
    print("SOURCE_MECHANISM", per_source.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
