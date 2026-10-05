import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json


def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run, training_root, output):
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(run / "config.json")
    training = read_json(training_root / "training_summary.json")
    if training["status"] != "COMPLETE":
        raise ValueError("Training is incomplete")
    procedures, clips = [], []
    selected = {(row["method"], row["seed"]): row["step"] for row in training["selection"]}
    for item in training["outputs"]:
        step = selected[item["method"], item["seed"]]
        evaluation = next(row for row in item["summary"]["evaluations"] if row["step"] == step)
        common = dict(method=item["method"], seed=item["seed"], fold=item["fold"], step=step,
                      partition="validation" if item["fold"] == "full" else "training_oof")
        for video, row in evaluation["by_video"].items():
            procedures.append(dict(**common, video_id=video, threshold=evaluation["threshold"], **row))
        for row in evaluation["by_clip"]:
            clips.append(dict(**common, **row))
    table(output / "procedures.csv", procedures)
    clip_fields = sorted({key for row in clips for key in row})
    table(output / "clips.csv", [{key: row.get(key) for key in clip_fields} for row in clips])
    summary = []
    for partition in ["training_oof", "validation"]:
        for method in config["methods"]:
            rows = [row for row in procedures if row["partition"] == partition and row["method"] == method]
            temporal = [row for row in rows if row["cross_interval_recall"] is not None]
            local = [row for row in clips if row["partition"] == partition and row["method"] == method]
            summary.append(dict(partition=partition, method=method,
                procedure_count=len({r["video_id"] for r in rows}), seeds=len({r["seed"] for r in rows}),
                recall=float(np.mean([row["recall"] for row in rows])),
                negative_retention=float(np.mean([row["negative_retention"] for row in rows])),
                auroc=float(np.mean([row["auroc"] for row in rows])),
                cross_interval_recall=float(np.mean([row["cross_interval_recall"] for row in temporal])) if temporal else None,
                cross_interval_procedures=len({r["video_id"] for r in temporal}),
                reconstruction_nmse=float(np.mean([row["reconstruction_nmse"] for row in local])) if method != "raw_supcon" else None,
                local_active=float(np.mean([row["local_active"] for row in local])) if method != "raw_supcon" else None,
                pooled_active=float(np.mean([row["pooled_active"] for row in local])) if method != "raw_supcon" else None,
                noncommutation=float(np.mean([row["noncommutation"] for row in local])) if method != "raw_supcon" else None))
    table(output / "summary.csv", summary)
    contrasts = []
    for first, second in [("token_sparse", "mean_sparse"), ("token_dense", "mean_dense"),
                          ("token_sparse", "token_dense"), ("token_sparse", "raw_supcon")]:
        reference = {(r["partition"], r["seed"], r["video_id"]): r for r in procedures if r["method"] == second}
        for row in procedures:
            if row["method"] != first:
                continue
            other = reference[row["partition"], row["seed"], row["video_id"]]
            contrasts.append(dict(partition=row["partition"], seed=row["seed"], video_id=row["video_id"],
                first=first, second=second, recall_difference=row["recall"] - other["recall"],
                auroc_difference=row["auroc"] - other["auroc"],
                retention_difference=row["negative_retention"] - other["negative_retention"]))
    table(output / "paired_differences.csv", contrasts)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    labels = ["Token\nSAE", "Mean\nSAE", "Token\ndense", "Mean\ndense", "Ordinary\nSupCon"]
    for axis, (metric, title) in zip(axes, [("recall", "Same-lesion recall"),
            ("cross_interval_recall", "Cross-interval recall"), ("auroc", "Within-procedure AUROC")], strict=True):
        for i, method in enumerate(config["methods"]):
            data = [row for row in procedures if row["partition"] == "validation" and row["method"] == method and row[metric] is not None]
            videos = sorted({row["video_id"] for row in data})
            per_video = [np.mean([row[metric] for row in data if row["video_id"] == video]) for video in videos]
            axis.bar(i, np.mean(per_video), color="#176B87" if method == "token_sparse" else "#BAC6CE", width=.65)
            axis.scatter(i + np.linspace(-.18, .18, len(videos)), per_video, s=19, color="#303A45", zorder=3)
        axis.set_xticks(range(5), labels)
        axis.set_ylim(0, 1.04)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Local evidence before pooling — exposed validation procedures\nBars: procedure means; dots: individual procedures averaged across seeds", fontsize=12)
    fig.savefig(output / "identity_comparison.png", dpi=170)
    plt.close(fig)
    lines = ["# Local token evidence experiment", "", "All validation procedures have prior development exposure. The training-fold rows include checkpoint selection. These are GT-region identity results; no complete-video prompt benefit is inferred.", "",
        "| Population | Method | Recall | Other-identity retention | AUROC | Cross-interval recall |",
        "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        cross = "undefined" if row["cross_interval_recall"] is None else f"{100 * row['cross_interval_recall']:.2f}%"
        lines.append(f"| {row['partition']} | {row['method']} | {100 * row['recall']:.2f}% | {100 * row['negative_retention']:.2f}% | {row['auroc']:.4f} | {cross} |")
    lines.extend(["", "Per-procedure values, paired contrasts and feature measurements are saved beside this report. ROC thresholds are measured independently in each population for descriptive comparisons.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(output / "summary.json", dict(status="COMPLETE", results=summary, training_root=str(training_root)))
    print("TOKEN_RESULTS_COMPLETE", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.run, args.training_root or args.run, args.output or args.run / "analysis")
